"""python -m step_study.run --help"""
import argparse
import copy
import csv
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .condition_data import TARGETS, load_daily_csv, prepare_daily, synthetic_daily
from .metrics import crps, energy_score, ncrps_from_sums, relative_loss_rows, step_summary
from .models import ConditionalDLinear, ExogenousCSDI
from .plotting import plot_curves, select_plot_variables


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit(model, datasets, config, device, seed, path, baseline=None, stage1=False):
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(datasets["train"], batch_size=config["batch_size"], shuffle=True,
                       generator=generator)
    valid = DataLoader(datasets["val"], batch_size=config["batch_size"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    epochs = config["dlinear_epochs" if stage1 else "diffusion_epochs"]
    best, log = float("inf"), []
    def objective(conditions, truth, pv_year):
        if stage1:
            return torch.nn.functional.mse_loss(model(conditions, pv_year), truth)
        with torch.no_grad():
            skeleton = None if baseline is None else baseline(conditions, pv_year)
            target = truth if skeleton is None else model.normalize_residual(truth, skeleton)
        return model.loss(model.pack_conditions(conditions, skeleton, pv_year), target)
    for epoch in range(epochs):
        model.train()
        train_sum, seen = 0., 0
        for conditions, truth, pv_year in train:
            conditions, truth, pv_year = conditions.to(device), truth.to(device), pv_year.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(conditions, truth, pv_year)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            train_sum += loss.item() * len(conditions)
            seen += len(conditions)
        model.eval()
        val_sum = 0.
        # Fixed validation timesteps/noise, isolated from training's RNG stream.
        devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed + 10000)
            for conditions, truth, pv_year in valid:
                val_sum += objective(conditions.to(device), truth.to(device), pv_year.to(device)).item() * len(conditions)
        val_loss = val_sum / len(datasets["val"])
        if not np.isfinite(val_loss):
            raise RuntimeError("Nonfinite validation loss")
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), path)
        log.append({"epoch": epoch + 1, "train_loss": train_sum / seen, "val_loss": val_loss})
        print(f"{path.stem} epoch={epoch+1}/{epochs} train={train_sum/seen:.5f} val={val_loss:.5f}", flush=True)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    dump_json(path.with_suffix(".history.json"), log)


@torch.no_grad()
def evaluate(models, baseline, datasets, config, scaler, names, device, seed):
    rows, joint = [], []
    scale = torch.tensor(scaler["scale"], device=device)
    mean = torch.tensor(scaler["mean"], device=device)
    for split in ("val", "test"):
        loader = DataLoader(datasets[split], batch_size=config["batch_size"])
        for method, model in models.items():
            steps = model.num_steps
            # Separate streams: initial states remain identical across T, even
            # though full chains consume different numbers of reverse innovations.
            split_seed = seed + (20000 if split == "val" else 30000)
            generator = torch.Generator().manual_seed(split_seed)
            reverse_generator = torch.Generator().manual_seed(split_seed + 100000)
            sums = torch.zeros(len(names), device=device, dtype=torch.float64)
            squares, denominator = torch.zeros_like(sums), torch.zeros_like(sums)
            es_sum, count, windows, elapsed = 0., 0, 0, 0.
            for conditions, truth, pv_year in loader:
                conditions, truth, pv_year = conditions.to(device), truth.to(device), pv_year.to(device)
                noise = torch.randn((len(conditions), config["samples"], *truth.shape[1:]),
                                    generator=generator).to(device)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                skeleton = baseline(conditions, pv_year) if method == "residual" else None
                packed = model.pack_conditions(conditions, skeleton, pv_year)
                samples = model.sample_ddpm(packed, noise, reverse_generator)
                if skeleton is not None:
                    samples = model.reconstruct(samples, skeleton)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed += time.perf_counter() - start
                if not torch.isfinite(samples).all():
                    raise RuntimeError(f"Nonfinite samples: {method}, T={steps}")
                sums += crps(samples, truth).double().sum((0, 1))
                denominator += (truth.double() * scale + mean).abs().sum((0, 1))
                squares += (samples.mean(1) - truth).double().square().sum((0, 1))
                es_sum += energy_score(samples, truth).sum().item()
                windows += len(conditions)
                count += len(conditions) * truth.shape[1]
            normalized = ncrps_from_sums(sums * scale, denominator, epsilon=1e-8)
            for i, name in enumerate(names):
                rows.append({"seed": seed, "split": split, "method": method,
                             "diffusion_steps": steps, "sampling_steps": steps, "sampler": "DDPM",
                             "variable": name, "ncrps": normalized[i].item(),
                             "ncrps_denominator_mean_abs_target": (denominator[i] / count).item(),
                             "crps_scaled": (sums[i] / count).item(),
                             "crps_original": (sums[i] / count * scale[i]).item(),
                             "rmse_scaled": (squares[i] / count).sqrt().item()})
            joint.append({"seed": seed, "split": split, "method": method, "diffusion_steps": steps,
                          "energy_score_scaled": es_sum / windows,
                          "sampling_seconds": elapsed, "windows": windows,
                          "samples_per_window": config["samples"]})
            print(f"eval seed={seed} {split} {method} T={steps} nCRPS={normalized.mean().item():.5f}", flush=True)
    return rows, joint


@torch.no_grad()
def residual_diagnostics(baseline, dataset, batch_size, device, names):
    targets, residuals = [], []
    for conditions, truth, pv_year in DataLoader(dataset, batch_size=batch_size):
        mu = baseline(conditions.to(device), pv_year.to(device)).cpu()
        targets.append(truth)
        residuals.append(truth - mu)
    target, residual = torch.cat(targets).flatten(0, 1), torch.cat(residuals).flatten(0, 1)
    return [{"variable": name, "target_variance_scaled": target[:, i].var(unbiased=False).item(),
             "residual_variance_scaled": residual[:, i].var(unbiased=False).item(),
             "dlinear_mse_scaled": residual[:, i].square().mean().item()}
            for i, name in enumerate(names)]


@torch.no_grad()
def fit_residual_scaler(baseline, train_dataset, energy_scale, batch_size, device,
                        mode="train_residual"):
    """Fit physical residual mean/sample std on training days only.

    Buffers use target-standardized units for common physical-unit evaluation.
    `target_scale` disables residual re-standardization for a controlled ablation.
    """
    scale = torch.tensor(energy_scale, dtype=torch.float32)
    if mode == "target_scale":
        center, spread = torch.zeros_like(scale), scale.clone()
    elif mode == "train_residual":
        residuals = []
        for conditions, truth, year in DataLoader(train_dataset, batch_size=batch_size):
            mu = baseline(conditions.to(device), year.to(device)).cpu()
            residuals.append(((truth - mu) * scale).reshape(-1, len(scale)))
        residuals = torch.cat(residuals)
        center, spread = residuals.mean(0), residuals.std(0, unbiased=True) + 1e-6
    else:
        raise ValueError("residual_normalization must be train_residual or target_scale")
    if not torch.isfinite(center).all() or not torch.isfinite(spread).all() or (spread <= 0).any():
        raise ValueError("Invalid training residual statistics")
    return {"mode": mode, "center_physical": center.tolist(), "scale_physical": spread.tolist(),
            "center_target_units": (center/scale).tolist(), "scale_target_units": (spread/scale).tolist(),
            "training_days": len(train_dataset)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/step_study.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--energy-path", help="CN03_energy_cleaned.csv with Year/Month/Day/Hour and Electricity/Heat/Cooling/PV")
    source.add_argument("--synthetic", action="store_true", help="Pipeline demonstration only")
    parser.add_argument("--weather-path", help="weather_cleaned.csv aligned by Year/Month/Day/Hour (required with --energy-path)")
    parser.add_argument("--plot-variables", help="Optional display order for Electricity,Heat,Cooling,PV")
    parser.add_argument("--diffusion-steps", nargs="+", type=int, help="Total training/ancestral DDPM steps; train a fresh pair for each T")
    parser.add_argument("--output", required=True, help="New, empty run directory")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--smoke", action="store_true", help="Small model/one epoch; not scientific evidence")
    parser.add_argument("--evaluate-only", action="store_true", help="Reuse checkpoints in --output, no retraining")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if bool(args.energy_path) != bool(args.weather_path):
        parser.error("Provide both --energy-path and --weather-path, or use --synthetic")
    if not (args.device == "cpu" or args.device.startswith("cuda")):
        parser.error("Use cpu or cuda:N for reproducible validation RNG isolation")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 - 30000 for s in args.seeds):
        parser.error("Seeds must be distinct nonnegative integers below 2**32-30000")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    config = yaml.safe_load(Path(args.config).read_text())
    if config["data"].get("task") != "daily_exogenous" or config["data"].get("seq_len") != 24:
        raise ValueError("Use the paper's daily_exogenous configuration with seq_len=24; history forecasts are not this task")
    if "train_fraction" in config["data"] or "train_years" not in config["data"]:
        raise ValueError("Use the current paper config with fixed natural-year splits, not the old 80/10/10 config")
    if config["data"]["residual_normalization"] not in ("train_residual", "target_scale"):
        raise ValueError("Unknown residual_normalization")
    if "study" not in config or "steps" in config["evaluation"] or "num_steps" in config["diffusion"]:
        raise ValueError("Use the new total-T config/step_study.yaml, not a legacy DDIM-sweep config")
    if args.smoke:
        config = copy.deepcopy(config)
        config["dlinear"]["kernel"] = 3
        config["train"].update(dlinear_epochs=1, diffusion_epochs=1, batch_size=8)
        config["diffusion"].update(layers=1, channels=8, nheads=2, diffusion_embedding_dim=16)
        config["model"].update(timeemb=8, featureemb=4)
        config["evaluation"].update(samples=4, batch_size=2)
        config["study"]["diffusion_steps"] = [10, 20, 40]
    if args.diffusion_steps is not None:
        config["study"]["diffusion_steps"] = args.diffusion_steps
    weather_set = config["data"]["weather_feature_set"]
    raw = synthetic_daily(weather_feature_set=weather_set) if args.synthetic else load_daily_csv(
        args.energy_path, args.weather_path, weather_set)
    names = list(TARGETS)
    plot_names = select_plot_variables(names, args.plot_variables)
    train_config, eval_config = config["train"], config["evaluation"]
    if min(train_config["dlinear_epochs"], train_config["diffusion_epochs"], train_config["batch_size"],
           eval_config["batch_size"], eval_config["samples"]) < 1:
        raise ValueError("Epochs, batch sizes and sample count must be positive")
    steps = config["study"]["diffusion_steps"]
    if len(steps) < 2 or len(set(steps)) != len(steps) or any(type(k) is not int or k < 2 for k in steps):
        raise ValueError("Use at least two distinct integer total diffusion steps >= 2")
    config["study"]["diffusion_steps"] = sorted(steps)
    if config["diffusion"]["schedule"] != "vp_continuous":
        raise ValueError("Total-T study requires the matched vp_continuous schedule")
    if config["study"]["relative_loss_epsilon"] <= 0:
        raise ValueError("relative_loss_epsilon must be positive")
    if eval_config["near_optimal_tolerance"] < 0:
        raise ValueError("Near-optimal tolerance must be nonnegative")
    datasets, scaler = prepare_daily(raw, weather_set, config["data"]["train_years"],
                                    config["data"]["val_years"], config["data"]["test_years"], smoke=args.smoke)
    condition_dim = scaler["condition_dim"]
    output = Path(args.output)
    if args.evaluate_only:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest.get("experiment") != "paper_daily_total_diffusion_steps_v4":
            raise ValueError("Legacy history/ratio-split checkpoints cannot be reused for the fixed-year PV-year task")
        if manifest["config"] != config or manifest["scaler"] != scaler or manifest["variables"] != names or manifest["seeds"] != args.seeds:
            raise ValueError("Evaluation-only requires original config, data, variable order and seeds")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Refusing to overwrite a nonempty run directory")
        output.mkdir(parents=True, exist_ok=True)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dump_json(output / "manifest.json", {"experiment": "paper_daily_total_diffusion_steps_v4",
                  "paper_task_source": "wenzi0731/wenzia:2-stages/dataset.py",
                  "paper_task_reference_commit": "4dcb890a35c6597faeeecdcdfd47192c9a737e65",
                  "config": config, "scaler": scaler, "variables": names,
                  "plot_variables": plot_names, "sampler": "DDPM_full_chain",
                  "ncrps_definition": "sum empirical CRPS(original units) / (sum abs(target in original units) + 1e-8)",
                  "seeds": args.seeds, "synthetic": args.synthetic, "smoke": args.smoke,
                  "device": args.device, "torch": torch.__version__, "numpy": np.__version__,
                  "python": platform.python_version(), "base_commit": commit})
    rows, joint = [], []
    for seed in args.seeds:
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        seed_everything(seed)
        baseline = ConditionalDLinear(config["data"]["seq_len"], condition_dim, len(names), **config["dlinear"]).to(args.device)
        if args.evaluate_only:
            baseline.load_state_dict(torch.load(seed_dir / "dlinear.pt", map_location=args.device, weights_only=True))
        else:
            fit(baseline, datasets, train_config, args.device, seed, seed_dir / "dlinear.pt", stage1=True)
        baseline.eval().requires_grad_(False)
        residual_stats_path = seed_dir / "residual_scaler.json"
        if args.evaluate_only:
            residual_stats = json.loads(residual_stats_path.read_text())
        else:
            residual_stats = fit_residual_scaler(baseline, datasets["train"], scaler["scale"],
                train_config["batch_size"], args.device, config["data"]["residual_normalization"])
            dump_json(residual_stats_path, residual_stats)
        dump_json(seed_dir / "residual_diagnostics.json", residual_diagnostics(baseline, datasets["val"], train_config["batch_size"], args.device, names))
        for total_steps in config["study"]["diffusion_steps"]:
            step_dir = seed_dir / f"T_{total_steps}"
            step_dir.mkdir(exist_ok=True)
            resolved = copy.deepcopy(config)
            resolved["diffusion"]["num_steps"] = total_steps
            models = {}
            for method in ("direct", "residual"):
                # Fresh pair at every T; matched initialization/data/noise within pair.
                seed_everything(seed)
                model = ExogenousCSDI(len(names), resolved, args.device, condition_dim).to(args.device)
                if method == "residual":
                    model.residual_center.copy_(torch.tensor(residual_stats["center_target_units"], device=args.device))
                    model.residual_scale.copy_(torch.tensor(residual_stats["scale_target_units"], device=args.device))
                if args.evaluate_only:
                    model.load_state_dict(torch.load(step_dir / f"{method}.pt", map_location=args.device, weights_only=True))
                    model.eval()
                else:
                    print(f"Training seed={seed} T={total_steps} {method}", flush=True)
                    fit(model, datasets, train_config, args.device, seed, step_dir / f"{method}.pt",
                        baseline=baseline if method == "residual" else None)
                models[method] = model
            assert sum(p.numel() for p in models["direct"].parameters()) == sum(p.numel() for p in models["residual"].parameters())
            dump_json(step_dir / "diffusion_config.json", {"config": resolved["diffusion"],
                "terminal_alpha_bar": models["direct"].alpha_bar[-1].item(),
                "sampling_steps": total_steps, "sampler": "DDPM_full_chain"})
            scores, joint_scores = evaluate(models, baseline, datasets, eval_config, scaler, names, args.device, seed)
            rows.extend(scores)
            joint.extend(joint_scores)
            write_csv(output / "per_variable.csv", rows)
            write_csv(output / "joint_scores.csv", joint)
    relative = relative_loss_rows(rows, config["study"]["relative_loss_epsilon"])
    write_csv(output / "per_variable_relative_loss.csv", relative)
    summary = step_summary(rows, eval_config["near_optimal_tolerance"], config["study"]["relative_loss_epsilon"])
    dump_json(output / "summary.json", summary)
    plot_curves(rows, relative, summary, output, plot_names,
                demonstration=args.synthetic or args.smoke)
    print(f"Finished: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
