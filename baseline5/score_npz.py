"""Rescore compatible saved physical scenarios, without retraining/resampling."""
from __future__ import annotations

import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json

import numpy as np

from baseline5.data import TARGET_COLUMNS
from baseline5.metrics import summarize, save_additional_scores
from baseline5.runtime import resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, out = resolve_path(args.npz), resolve_path(args.outdir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty output directory")
    with np.load(source, allow_pickle=False) as payload:
        required = {"scenarios", "targets", "dates", "target_mean", "target_std"}
        if not required.issubset(payload.files):
            raise ValueError(f"NPZ must contain {sorted(required)}; do not infer normalization from test data")
        samples, truth = payload["scenarios"], payload["targets"]
        mean, std, dates = payload["target_mean"], payload["target_std"], payload["dates"]
    if samples.ndim != 4 or samples.shape[2:] != (4, 24) or len(dates) != len(samples):
        raise ValueError("Require physical [N,S,4,24], fixed channel order, and N dates")
    result = summarize(samples, truth, TARGET_COLUMNS, mean, std, seed=args.seed)
    result.update(save_additional_scores(samples, truth, mean, std, dates.tolist(), TARGET_COLUMNS, out))
    result.update(source_npz=str(source), num_test_days=len(truth), scenarios_per_day=samples.shape[1], seed=args.seed)
    result = {key: (None if isinstance(value, float) and not np.isfinite(value) else value)
              for key, value in result.items()}
    (out / "global_metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    with (out / "global_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(result.items())
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
