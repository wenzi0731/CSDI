from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def choose_device(value: str = "auto") -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("Configuration root must be a mapping.")
    for raw in overrides:
        if "=" not in raw:
            raise ValueError(f"Invalid override {raw!r}; expected key=value.")
        key, value = raw.split("=", 1)
        cursor = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                raise KeyError(f"Unknown configuration key: {key}")
            cursor = cursor[part]
        if parts[-1] not in cursor:
            raise KeyError(f"Unknown configuration key: {key}")
        cursor[parts[-1]] = yaml.safe_load(value)
    return config


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = resolve_path(path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a mapping.")
    return checkpoint
