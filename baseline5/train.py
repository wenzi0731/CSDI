from __future__ import annotations

# Support both python -m baseline5.train and python baseline5/train.py.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from baseline5.data import HEEWConditionDataset
from baseline5.metrics import crps_map
from baseline5.model import ConditionalCSDI, generate_scenarios
from baseline5.runtime import choose_device, load_config, resolve_path, seed_everything, seed_worker


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def data_manifest(config, dataset):
    result = {"normalization": {}, "files": {}}
    for key in ("target_mean", "target_std", "weather_mean", "weather_std"):
        result["normalization"][key] = getattr(dataset, key).tolist()
    result["target_columns"] = dataset.target_cols
    result["weather_columns"] = dataset.weather_cols
    for key in ("energy_path", "weather_path"):
        digest = hashlib.sha256()
        with resolve_path(config["data"][key]).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result["files"][key] = digest.hexdigest()
    return result


def make_dataset(config, split):
    data = config["data"]
    return HEEWConditionDataset(
        resolve_path(data["energy_path"]), resolve_path(data["weather_path"]),
        split=split, weather_feature_set=data["weather_feature_set"],
    )


@torch.no_grad()
def validate(model, loader, dataset, config, device):
    model.eval()
    rng = torch.Generator(device=device).manual_seed(int(config["run"]["seed"]) + 50000)
    scores = np.zeros(4, dtype=np.float64)
    denominators = np.zeros(4, dtype=np.float64)
    for conditions, pv_year, target, _ in loader:
        samples = generate_scenarios(
            model, conditions.to(device), pv_year.to(device),
            int(config["evaluation"]["val_scenarios"]), rng,
            int(config["evaluation"]["sample_chunk_size"]),
        )
        physical = dataset.denormalize(samples).cpu().numpy()
        physical[:, :, 3] = np.maximum(physical[:, :, 3], 0)
        truth = dataset.denormalize(target).numpy()
        scores += crps_map(physical, truth).sum(axis=(0, 2))
        denominators += np.abs(truth).sum(axis=(0, 2))
    result = {"val_macro_nCRPS": float((scores / (denominators + 1e-8)).mean())}
    result.update({f"val_{label}_nCRPS": float(scores[i] / (denominators[i] + 1e-8))
                   for i, label in enumerate(dataset.target_cols)})
    return result


def train(config, run_name=None):
    seed = int(config["run"]["seed"])
    seed_everything(seed)
    device = choose_device(config["run"]["device"])
    name = run_name or config["run"]["name"]
    run_dir = resolve_path(config["run"]["output_root"]) / f"{name}_seed{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_dir}. Use a new --run-name.")
    training = config["training"]
    if min(int(training[k]) for k in ("epochs", "batch_size", "val_every", "early_stopping_patience")) < 1:
        raise ValueError("Training counts must be positive")
    train_data = make_dataset(config, "train")
    val_data = make_dataset(config, "val")
    manifest = data_manifest(config, train_data)
    train_loader = DataLoader(
        train_data, batch_size=int(training["batch_size"]), shuffle=True,
        num_workers=int(config["data"]["num_workers"]), worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_data, batch_size=int(config["evaluation"]["batch_size"]), shuffle=False,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    model = ConditionalCSDI(config, train_data.condition_channels).to(device)
    ema = deepcopy(model).eval().requires_grad_(False)
    decay = float(training["ema_decay"])
    if not 0 <= decay < 1:
        raise ValueError("ema_decay must be in [0,1)")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training["learning_rate"]), betas=tuple(training["betas"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[max(1, int(training["epochs"] * fraction)) for fraction in (0.75, 0.9)],
        gamma=0.1,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "device": str(device), "parameters": sum(p.numel() for p in model.parameters())}
    environment["packages"] = {name: importlib.metadata.version(name) for name in
                               ("numpy", "pandas", "scipy", "scikit-learn", "PyYAML", "matplotlib")}
    environment["upstream_commit"] = "7f24a436f08d98853a6b43d4f7f04e5a65ecdf27"
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2))
    best_score, stale = float("inf"), 0
    best_path = run_dir / "best.pt"
    started = time.perf_counter()
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        loss_sum, seen = 0.0, 0
        for conditions, pv_year, target, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(target.to(device), conditions.to(device), pv_year.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss in epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]),
                                          error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for average, current in zip(ema.parameters(), model.parameters()):
                    if decay == 0:
                        average.copy_(current)
                    else:
                        average.lerp_(current, 1 - decay)
                for average, current in zip(ema.buffers(), model.buffers()):
                    average.copy_(current)
            loss_sum += loss.item() * len(target)
            seen += len(target)
        record = {"epoch": epoch, "train_loss": loss_sum / seen}
        record["learning_rate"] = optimizer.param_groups[0]["lr"]
        scheduler.step()
        if epoch % int(training["val_every"]) == 0 or epoch == int(training["epochs"]):
            metrics = validate(ema, val_loader, val_data, config, device)
            record.update(metrics)
            score = metrics["val_macro_nCRPS"]
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite validation score")
            if score < best_score:
                best_score, stale = score, 0
                payload = {
                    "format_version": 1, "method": "conditional-csdi-baseline5",
                    "config": deepcopy(config), "epoch": epoch,
                    "model": ema.state_dict(), "raw_model": model.state_dict(),
                    "metrics": metrics, "data_manifest": manifest,
                    "condition_channels": train_data.condition_channels,
                }
                torch.save(payload, best_path)
                (run_dir / "best_metrics.json").write_text(json.dumps({"epoch": epoch, **metrics}, indent=2))
            else:
                stale += 1
        with (run_dir / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if stale >= int(training["early_stopping_patience"]):
            break
    if not best_path.exists():
        raise RuntimeError("No validated checkpoint was saved")
    (run_dir / "completed.json").write_text(json.dumps({
        "config_hash": config_hash(config), "training_seconds": time.perf_counter() - started,
        "best_checkpoint": str(best_path), "best_val_macro_nCRPS": best_score,
    }, indent=2))
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Train condition-only joint CSDI")
    parser.add_argument("--config", default="baseline5/configs/heew.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    print(train(load_config(args.config, args.overrides), args.run_name))


if __name__ == "__main__":
    main()
