"""Chronological splits with training-only scaling and no crossing target windows."""
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def synthetic_series(length=600, variables=3, seed=31415):
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    common = rng.normal(size=length)
    return np.stack([
        (1 + i) * np.sin(2 * np.pi * t / (12 + 7 * i))
        + .002 * (i + 1) * t + (.1 + .2 * i) * rng.normal(size=length)
        + .15 * common for i in range(variables)
    ], axis=-1).astype(np.float32)


def load_series(path, columns=None):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
        names = [f"variable_{i}" for i in range(values.shape[1])] if values.ndim == 2 else []
    elif path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not columns:
                raise ValueError("CSV input requires --columns: explicitly exclude timestamp/ID fields")
            names = columns.split(",")
            if not set(names).issubset(reader.fieldnames or []):
                raise ValueError(f"Missing requested columns: {names}")
            values = np.asarray([[float(row[name]) for name in names] for row in reader], dtype=np.float32)
    else:
        raise ValueError("Use a numeric .npy [time, variable] or a headered .csv")
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Input must be a nonempty, finite [time, variable] array")
    if len(set(names)) != len(names):
        raise ValueError("Variable names must be unique")
    return values.astype(np.float32), names


class Windows(Dataset):
    def __init__(self, values, start, stop, history, horizon, stride):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.history, self.horizon = history, horizon
        self.starts = list(range(max(start, history), stop - horizon + 1, stride))
        if not self.starts:
            raise ValueError("Split too short for the selected history/horizon")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        return self.values[start - self.history:start], self.values[start:start + self.horizon]


def prepare(values, history, horizon, stride, train_fraction=.6, val_fraction=.2):
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 - train_fraction:
        raise ValueError("Invalid chronological split fractions")
    if min(history, horizon, stride) < 1:
        raise ValueError("history, horizon and stride must be positive")
    first = int(len(values) * train_fraction)
    second = int(len(values) * (train_fraction + val_fraction))
    mean = values[:first].mean(0)
    scale = values[:first].std(0)
    scale = np.where(scale < 1e-6, 1., scale)
    normalized = (values - mean) / scale
    splits = {
        "train": Windows(normalized, history, first, history, horizon, stride),
        "val": Windows(normalized, first, second, history, horizon, stride),
        "test": Windows(normalized, second, len(values), history, horizon, stride),
    }
    metadata = {"mean": mean.tolist(), "scale": scale.tolist(),
                "train_end": first, "val_end": second, "length": len(values),
                "data_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                "windows": {key: len(value) for key, value in splits.items()}}
    return splits, metadata
