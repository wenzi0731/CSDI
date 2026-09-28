from __future__ import annotations

# Support both python -m baseline5.evaluate and python baseline5/evaluate.py.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline5.data import HEEWConditionDataset
from baseline5.metrics import (
    save_global_pearson_comparison,
    save_global_pearson_plot,
    save_random_timeseries_plots,
    summarize,
    save_additional_scores,
)
from baseline5.model import ConditionalCSDI, generate_scenarios
from baseline5.runtime import choose_device, load_checkpoint, resolve_path, seed_everything
from baseline5.train import data_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate conditional joint CSDI.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--energy-path", default=None)
    parser.add_argument("--weather-path", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scenarios", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-days", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("method") != "conditional-csdi-baseline5":
        raise ValueError("Expected a baseline-5 CSDI checkpoint")
    config = checkpoint["config"]
    seed = int(args.seed if args.seed is not None else config["run"]["seed"])
    seed_everything(seed)
    args.scenarios = int(config["evaluation"]["test_scenarios"]) if args.scenarios is None else args.scenarios
    args.batch_size = int(config["evaluation"]["batch_size"]) if args.batch_size is None else args.batch_size
    if args.scenarios < 1 or args.batch_size < 1 or (args.max_days is not None and args.max_days < 1):
        raise ValueError("Counts must be positive")
    device = choose_device(args.device)
    data_config = config["data"]
    dataset = HEEWConditionDataset(
        energy_path=resolve_path(args.energy_path or data_config["energy_path"]),
        weather_path=resolve_path(args.weather_path or data_config["weather_path"]),
        split="test",
        weather_feature_set=data_config.get("weather_feature_set", "pv10"),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    manifest_config = json.loads(json.dumps(config))
    manifest_config["data"]["energy_path"] = str(args.energy_path or data_config["energy_path"])
    manifest_config["data"]["weather_path"] = str(args.weather_path or data_config["weather_path"])
    if data_manifest(manifest_config, dataset) != checkpoint["data_manifest"]:
        raise ValueError("Data or normalization differs from training checkpoint")
    generator = ConditionalCSDI(config, dataset.condition_channels)
    generator.load_state_dict(checkpoint["model"])
    generator.to(device).eval()
    output_dir = (
        resolve_path(args.outdir)
        if args.outdir
        else checkpoint_path.parent / "evaluation_test"
    )
    if (output_dir / "global_metrics.json").exists():
        raise FileExistsError("Evaluation already exists; choose a new --outdir")
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    condition_batches: list[np.ndarray] = []
    dates: list[str] = []
    noise_generator = torch.Generator(device=device).manual_seed(seed + 90_000)
    elapsed = 0.0
    processed = 0
    with torch.no_grad():
        for conditions, pv_year, target, batch_dates in loader:
            if args.max_days is not None and processed >= args.max_days:
                break
            take = len(batch_dates)
            if args.max_days is not None:
                take = min(take, args.max_days - processed)
            conditions = conditions[:take].to(device)
            pv_year = pv_year[:take].to(device)
            target = target[:take].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            generated_z = generate_scenarios(
                generator,
                conditions,
                pv_year,
                args.scenarios,
                noise_generator,
                int(config["evaluation"]["sample_chunk_size"]),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - start
            generated = dataset.denormalize(generated_z)
            target_physical = dataset.denormalize(target)
            generated[:, :, 3].clamp_(min=0.0)
            generated_batches.append(generated.cpu().numpy())
            target_batches.append(target_physical.cpu().numpy())
            condition_batches.append(conditions.cpu().numpy())
            dates.extend(list(batch_dates[:take]))
            processed += take

    generated_values = np.concatenate(generated_batches)
    target_values = np.concatenate(target_batches)
    condition_values = np.concatenate(condition_batches)
    labels = list(dataset.target_cols)
    metrics = summarize(
        generated_values,
        target_values,
        labels,
        dataset.target_mean,
        dataset.target_std,
        seed=seed,
        precision_recall_k=int(config["evaluation"]["precision_recall_k"]),
        max_precision_samples=int(config["evaluation"]["max_precision_samples"]),
    )
    metrics.update(save_additional_scores(generated_values, target_values,
                   dataset.target_mean, dataset.target_std, dates, labels, output_dir))
    if not args.no_plots:
        metrics.update(
            save_global_pearson_comparison(
                target_values, generated_values, output_dir / "pearson", labels
            )
        )
    metrics["sampling_seconds"] = elapsed
    metrics["seconds_per_1000_scenarios"] = (
        elapsed * 1000.0 / (len(target_values) * args.scenarios)
    )
    metrics["num_test_days"] = int(len(target_values))
    metrics["scenarios_per_day"] = int(args.scenarios)
    metrics["seed"] = seed
    metrics["sampler"] = "CSDI_DDPM_full_steps"
    metrics["training_timesteps"] = generator.num_steps
    metrics["sampling_steps"] = generator.num_steps
    metrics["sample_seed"] = seed + 90000
    metrics["sampling_chunk_size"] = int(config["evaluation"]["sample_chunk_size"])
    metrics["sampling_batch_size"] = args.batch_size
    metrics["checkpoint_epoch"] = checkpoint["epoch"]
    metrics["partial_test"] = len(target_values) != len(dataset)
    metrics["interval_alpha"] = 0.05
    metrics["VS_p"] = 0.5

    if not args.no_plots:
        save_global_pearson_plot(
            target_values, output_dir / "global_pearson.png", labels
        )
        save_random_timeseries_plots(
            target_values,
            generated_values,
            dates,
            output_dir / "random_timeseries_50",
            labels,
            seed=seed,
            n_plots=50,
        )
    np.savez_compressed(
        output_dir / "baseline5_scenarios.npz",
        scenarios=generated_values,
        targets=target_values,
        conditions=condition_values,
        dates=np.asarray(dates),
        target_mean=dataset.target_mean,
        target_std=dataset.target_std,
    )
    # Valid JSON for degenerate channels (e.g. constant target -> undefined R2).
    metrics = {key: (None if isinstance(value, float) and not np.isfinite(value) else value)
               for key, value in metrics.items()}
    suffixes = ("MAE", "RMSE", "MAE_Z", "RMSE_Z", "R2", "CRPS", "nCRPS", "Precision_Z", "Recall_Z", "CR", "IW", "IS", "IS_Z")
    with (output_dir / "channel_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["channel", *suffixes])
        writer.writeheader()
        for label in labels:
            writer.writerow({"channel": label, **{key: metrics[f"{label}_{key}"] for key in suffixes}})
    with (output_dir / "global_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_dir / "global_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
