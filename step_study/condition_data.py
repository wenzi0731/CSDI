"""Paper task: target-day weather/calendar -> 24 h Electricity, Heat, Cooling, PV.

Matches HEEWDailyStage1Dataset's task, pv11 feature names, cyclical calendar,
complete-day split and train-only standardization. No energy history is exposed.
"""
import calendar
import csv
from datetime import datetime, timedelta
import hashlib
import json

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


TARGETS = ["Electricity", "Heat", "Cooling", "PV"]
BASE_WEATHER = ["Temperature", "Dew Point", "Humidity", "Wind Speed", "Wind Gust", "Pressure", "Precip"]
WEATHER_SETS = {
    "base7": BASE_WEATHER,
    "pv9": BASE_WEATHER + ["ALLSKY_SFC_SW_DWN", "PV_CLEARNESS_RATIO"],
    "pv11": BASE_WEATHER + ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "PV_CLEARNESS_RATIO", "PV_IS_DAYLIGHT"],
}
TIME_FEATURES = ["month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos",
                 "weekday_sin", "weekday_cos", "hour_sin", "hour_cos"]


def calendar_features(timestamps):
    result = []
    for t in timestamps:
        angles = [2*np.pi*(t.month-1)/12,
                  2*np.pi*(t.timetuple().tm_yday-1)/(366 if calendar.isleap(t.year) else 365),
                  2*np.pi*t.weekday()/7, 2*np.pi*t.hour/24]
        result.append([value for angle in angles for value in (np.sin(angle), np.cos(angle))])
    return np.asarray(result, dtype=np.float32)


def read_timestamped_csv(path, columns):
    rows = {}
    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = ["Year", "Month", "Day", "Hour"] + columns
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line, row in enumerate(reader, 2):
            try:
                timestamp = datetime(*(int(row[key]) for key in required[:4]))
                values = [float(row[key]) for key in columns]
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}: invalid timestamp or numeric value on line {line}") from error
            if timestamp in rows:
                raise ValueError(f"{path}: duplicate timestamp {timestamp}")
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: nonfinite values on line {line}")
            rows[timestamp] = values
    if not rows:
        raise ValueError(f"{path}: empty data")
    return rows


def load_daily_csv(energy_path, weather_path, weather_feature_set="pv11"):
    if weather_feature_set not in WEATHER_SETS:
        raise ValueError(f"Unknown weather feature set: {weather_feature_set}")
    weather_cols = WEATHER_SETS[weather_feature_set]
    energy = read_timestamped_csv(energy_path, TARGETS)
    weather = read_timestamped_csv(weather_path, weather_cols)
    if energy.keys() != weather.keys():
        raise ValueError("Energy/weather timestamps differ; align files explicitly (no silent truncation)")
    groups = {}
    for t in sorted(energy):
        groups.setdefault(t.date(), []).append(t)
    complete = [group for group in groups.values() if len(group) == 24 and [t.hour for t in group] == list(range(24))]
    if not complete:
        raise ValueError("No complete 24-hour days found")
    timestamps = [t for group in complete for t in group]
    days = [group[0].date().isoformat() for group in complete]
    targets = np.asarray([energy[t] for t in timestamps], dtype=np.float32).reshape(-1, 24, 4)
    meteo = np.asarray([weather[t] for t in timestamps], dtype=np.float32).reshape(-1, 24, len(weather_cols))
    times = calendar_features(timestamps).reshape(-1, 24, len(TIME_FEATURES))
    return targets, meteo, times, days, len(groups)-len(complete)


def synthetic_daily(days=60, weather_feature_set="pv11", seed=31415):
    """Demo-only, condition-driven complete days; independent of training seed."""
    rng = np.random.default_rng(seed)
    timestamps = [datetime(2020, 1, 1) + timedelta(hours=i) for i in range(days*24)]
    hour = np.arange(days*24) % 24
    daylight = np.maximum(0., np.sin(np.pi*(hour-6)/12))
    temperature = 18 + 7*np.sin(2*np.pi*(hour-8)/24) + rng.normal(size=len(hour))
    all_weather = np.column_stack([temperature, temperature-5, 60-10*daylight,
        3+rng.random(len(hour)), 5+rng.random(len(hour)), 1010+rng.normal(size=len(hour)),
        rng.random(len(hour))*.1, 700*daylight, 900*daylight,
        .7+.05*rng.normal(size=len(hour)), (daylight>.01).astype(float)])
    common = rng.normal(size=len(hour))
    targets = np.column_stack([100+15*daylight+temperature+common,
        np.maximum(0., 35-temperature+common), 10+2*np.maximum(temperature-20, 0)+common,
        np.maximum(0., 30*daylight+daylight*common)])
    indices = [WEATHER_SETS["pv11"].index(name) for name in WEATHER_SETS[weather_feature_set]]
    return (targets.astype(np.float32).reshape(days, 24, 4),
            all_weather[:, indices].astype(np.float32).reshape(days, 24, -1),
            calendar_features(timestamps).reshape(days, 24, -1),
            [timestamps[i*24].date().isoformat() for i in range(days)], 0)


class DailyConditions(Dataset):
    def __init__(self, conditions, targets, dates):
        self.conditions = torch.as_tensor(conditions, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.dates = list(dates)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        # Conditions contain only weather and calendar, never historical/target energy.
        return self.conditions[index], self.targets[index]


def prepare_daily(raw, weather_feature_set="pv11", train_fraction=.8, val_fraction=.1, smoke=False):
    targets, meteo, times, dates, dropped = raw
    # Fix reduction order across CSV and synthetic/array-backed inputs.
    targets, meteo, times = (np.ascontiguousarray(array, dtype=np.float32)
                             for array in (targets, meteo, times))
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1-train_fraction:
        raise ValueError("Invalid daily split fractions")
    n = len(dates)
    first, second = int(n*train_fraction), int(n*(train_fraction+val_fraction))
    if not 0 < first < second < n:
        raise ValueError("Need enough complete days for nonempty train/val/test splits")
    mean, scale = targets[:first].mean((0, 1)), targets[:first].std((0, 1)) + 1e-6
    weather_mean = meteo[:first].mean((0, 1))
    weather_scale = meteo[:first].std((0, 1)) + 1e-6
    conditions = np.concatenate([(meteo-weather_mean)/weather_scale, times], axis=-1)
    normalized = (targets-mean)/scale
    bounds = {"train": (0, first), "val": (first, second), "test": (second, n)}
    splits, selected_dates = {}, {}
    for split, (start, stop) in bounds.items():
        dataset = DailyConditions(conditions[start:stop], normalized[start:stop], dates[start:stop])
        selected_dates[split] = list(dataset.dates)
        if smoke:
            count = min(len(dataset), 16 if split == "train" else 4)
            indices = np.linspace(0, len(dataset)-1, count).astype(int).tolist()
            selected_dates[split] = [dataset.dates[i] for i in indices]
            dataset = Subset(dataset, indices)
        splits[split] = dataset
    fingerprint = hashlib.sha256()
    for array in (targets, meteo, times):
        fingerprint.update(np.ascontiguousarray(array).tobytes())
    fingerprint.update(json.dumps(dates).encode())
    metadata = {"mean": mean.tolist(), "scale": scale.tolist(),
        "weather_mean": weather_mean.tolist(), "weather_scale": weather_scale.tolist(),
        "condition_columns": WEATHER_SETS[weather_feature_set] + TIME_FEATURES,
        "condition_dim": conditions.shape[-1], "target_columns": TARGETS,
        "complete_days": n, "dropped_incomplete_days": dropped,
        "split_day_counts": {key: stop-start for key, (start, stop) in bounds.items()},
        "split_date_ranges": {key: [dates[start], dates[stop-1]] for key, (start, stop) in bounds.items()},
        "used_samples": {key: len(value) for key, value in splits.items()},
        "selected_dates": selected_dates, "data_sha256": fingerprint.hexdigest()}
    return splits, metadata
