from copy import deepcopy
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from main_model import CSDI_base
from baseline5.data import BASE_WEATHER_COLUMNS, PV_WEATHER_COLUMNS, TARGET_COLUMNS
from baseline5.model import ConditionalCSDI, generate_scenarios
from baseline5.runtime import load_config
from baseline5.sweep import sweep, trial_configs
from baseline5.train import make_dataset, train


@pytest.fixture
def tiny_config(tmp_path):
    torch.set_num_threads(1)
    config = load_config("baseline5/configs/heew.yaml")
    config["model"].update(timeemb=16, featureemb=4)
    config["diffusion"].update(num_steps=4, channels=8, layers=1, nheads=2, diffusion_embedding_dim=16)
    config["training"].update(epochs=1, batch_size=2, val_every=1)
    config["evaluation"].update(batch_size=2, val_scenarios=2, test_scenarios=3, sample_chunk_size=4)
    config["run"].update(device="cpu", output_root=str(tmp_path / "runs"))
    timestamps = pd.DatetimeIndex([
        ts for year in (2014, 2020, 2021, 2022)
        for ts in pd.date_range(f"{year}-01-01", periods=48, freq="h")
    ])
    cols = dict(Year=timestamps.year, Month=timestamps.month, Day=timestamps.day, Hour=timestamps.hour)
    rng = np.random.default_rng(12)
    energy = pd.DataFrame(cols)
    weather = pd.DataFrame(cols)
    for i, label in enumerate(TARGET_COLUMNS):
        energy[label] = rng.uniform(10, 50, len(timestamps)) * (i + 1)
    for label in BASE_WEATHER_COLUMNS + PV_WEATHER_COLUMNS:
        weather[label] = rng.uniform(0, 1, len(timestamps))
    for kind, frame in (("energy", energy), ("weather", weather)):
        path = tmp_path / f"{kind}.csv"
        frame.to_csv(path, index=False)
        config["data"][f"{kind}_path"] = str(path)
    return config


def test_exact_upstream_loss_and_sampler_parity(tiny_config):
    model = ConditionalCSDI(tiny_config, 18).eval()
    upstream = CSDI_base(23, deepcopy(tiny_config), torch.device("cpu")).eval()
    upstream.load_state_dict({k: v for k, v in model.state_dict().items() if k != "alpha_torch"})
    cond, year, target = torch.randn(2, 18, 24), torch.randn(2, 1, 24), torch.randn(2, 4, 24)
    observed, mask, side = model.pack_conditions(cond, year, target)
    assert mask[:, :4].count_nonzero() == 0 and mask[:, 4:].min() == 1
    torch.manual_seed(13)
    expected = upstream.calc_loss(observed, mask, torch.ones_like(observed), side, 1)
    torch.manual_seed(13)
    actual = model(target, cond, year)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # For one ensemble member the same random draws reproduce original impute.
    torch.manual_seed(22)
    original = upstream.impute(observed, mask, side, 1)[:, 0, :4]
    adapted = model.sample(cond, year, torch.Generator().manual_seed(22))
    torch.testing.assert_close(adapted, original, rtol=0, atol=0)
    # Targets change the supervised noisy target, never the known-condition lane.
    observed2, _, _ = model.pack_conditions(cond, year, target + 999)
    noisy = torch.randn_like(observed)
    torch.testing.assert_close(model.set_input_to_diffmodel(noisy, observed, mask),
                               model.set_input_to_diffmodel(noisy, observed2, mask))


def test_joint_condition_gradients_and_repeatable_sampling(tiny_config):
    model = ConditionalCSDI(tiny_config, 18)
    cond, year, target = torch.randn(2, 18, 24), torch.randn(2, 1, 24), torch.randn(2, 4, 24)
    # Upstream zero-initializes its final projection. Take one optimization step
    # before checking information propagation to the earlier attention layers.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model(target, cond, year).backward()
    optimizer.step()
    optimizer.zero_grad()
    model(target, cond.requires_grad_(), year).backward()
    assert cond.grad.abs().sum() > 0
    assert model.diffmodel.residual_layers[0].feature_layer.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0
    model.eval()
    def sample(c):
        return generate_scenarios(model, c, year, 3, torch.Generator().manual_seed(42), 4)
    first = sample(cond)
    assert first.shape == (2, 3, 4, 24) and torch.isfinite(first).all()
    torch.testing.assert_close(first, sample(cond), rtol=0, atol=0)
    assert not torch.allclose(first, sample(cond + 2))
    assert not torch.allclose(first[:, 0], first[:, 1])
    # Fixed-seed samples do not consume the global training RNG.
    state = torch.random.get_rng_state()
    sample(cond)
    assert torch.equal(state, torch.random.get_rng_state())


def test_train_only_scaling_and_splits(tiny_config):
    train_data = make_dataset(tiny_config, "train")
    val_data = make_dataset(tiny_config, "val")
    test_data = make_dataset(tiny_config, "test")
    assert len(train_data) == 4 and len(val_data) == len(test_data) == 2
    assert all(d.startswith("2021") for d in val_data.dates)
    assert all(d.startswith("2022") for d in test_data.dates)
    frame = pd.read_csv(tiny_config["data"]["energy_path"])
    train_values = frame.loc[frame.Year <= 2020, TARGET_COLUMNS].to_numpy(np.float32)
    np.testing.assert_allclose(train_data.target_mean, train_values.mean(0), rtol=1e-6)
    np.testing.assert_array_equal(train_data.target_mean, test_data.target_mean)
    np.testing.assert_array_equal(train_data.target_std, val_data.target_std)


def test_six_trials_and_reject_duplicates(tiny_config):
    search = yaml.safe_load(open("baseline5/configs/search_space.yaml"))
    assert len(trial_configs(tiny_config, search)) == 6
    bad = deepcopy(search)
    bad["configurations"].pop()
    with pytest.raises(ValueError, match="six"):
        trial_configs(tiny_config, bad)
    bad = deepcopy(search)
    bad["configurations"][1]["overrides"] = bad["configurations"][0]["overrides"]
    with pytest.raises(ValueError, match="Duplicate"):
        trial_configs(tiny_config, bad)


def test_checkpoint_to_test_outputs(tiny_config, tmp_path, monkeypatch):
    checkpoint = train(tiny_config, "smoke")
    from baseline5.evaluate import main
    out = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", str(checkpoint),
                                      "--outdir", str(out), "--device", "cpu"])
    main()
    payload = np.load(out / "baseline5_scenarios.npz")
    assert payload["scenarios"].shape == (2, 3, 4, 24)
    metrics = json.loads((out / "global_metrics.json").read_text())
    for label in TARGET_COLUMNS:
        for suffix in ("RMSE_Z", "MAE_Z", "Precision_Z", "Recall_Z", "CR", "IW", "R2", "IS", "IS_Z"):
            assert np.isfinite(metrics[f"{label}_{suffix}"])
    assert metrics["sample_seed"] == 90042
    assert all(np.isfinite(metrics[k]) for k in ("ES", "VS", "ES_Z", "VS_Z"))
    assert len(pd.read_csv(out / "channel_metrics.csv")) == 4
    assert len(pd.read_csv(out / "daily_scores.csv")) == 2
    assert (out / "metric_definitions.json").exists()
    assert metrics["sampling_steps"] == 4
    assert len(list((out / "random_timeseries_50").glob("*.png"))) == 2
    assert (out / "pearson" / "generated_global_pearson.png").exists()
    from baseline5.score_npz import main as score_main
    rescored = tmp_path / "rescored"
    monkeypatch.setattr(sys, "argv", ["score_npz", "--npz", str(out / "baseline5_scenarios.npz"), "--outdir", str(rescored)])
    score_main()
    rescored_metrics = json.loads((rescored / "global_metrics.json").read_text())
    for key in ("ES", "VS", "Electricity_R2", "PV_IS", "Heat_IS_Z"):
        assert rescored_metrics[key] == metrics[key]
    with pytest.raises(FileExistsError):
        train(tiny_config, "smoke")


def test_sweep_exports_usable_winner(tiny_config, tmp_path):
    search = {"configurations": [
        {"id": f"trial{i}", "overrides": {"diffusion.channels": 8,
                                         "training.learning_rate": (i + 1) * 1e-5}}
        for i in range(6)
    ]}
    out = tmp_path / "sweep"
    best = sweep(tiny_config, search, out)
    winner = yaml.safe_load((out / "best_config.yaml").read_text())
    assert winner["training"]["learning_rate"] == best["learning_rate"]
    assert len(pd.read_csv(out / "tuning_results.csv")) == 6
    assert sweep(tiny_config, search, out, skip_completed=True) == best
    winner["run"]["seed"] = 123
    assert train(winner).exists()


def test_typo_overrides_rejected():
    with pytest.raises(KeyError, match="Unknown"):
        load_config("baseline5/configs/heew.yaml", ["diffusion.sampling_steps=100"])
