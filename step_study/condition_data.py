"""Paper task: target-day weather/calendar -> 24 h Electricity, Heat, Cooling, PV.

Matches the paper's current pv10 features, natural-year splits, cyclical calendar
and PV-only year coordinate. No energy history is exposed.
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
BASE_WEATHER = ["Temperature", "Dew Point", "Humidity", "Wind Speed", "Pressure", "Precip"]
WEATHER_SETS = {
    "base6": BASE_WEATHER,
    "pv10": BASE_WEATHER + ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "PV_CLEARNESS_RATIO", "PV_IS_DAYLIGHT"],
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


def load_daily_csv(energy_path, weather_path, weather_feature_set="pv10"):
    if weather_feature_set not in WEATHER_SETS:
        raise ValueError(f"Unknown weather feature set: {weather_feature_set}")
    weather_cols = WEATHER_SETS[weather_feature_set]
    energy = read_timestamped_csv(energy_path, TARGETS)
    weather = read_timestamped_csv(weather_path, weather_cols)
    # Paper protocol: timestamp intersection followed by complete-day filtering.
    common = energy.keys() & weather.keys()
    groups = {}
    for t in sorted(common):
        groups.setdefault(t.date(), []).append(t)
    complete = [group for group in groups.values() if len(group) == 24 and [t.hour for t in group] == list(range(24))]
    if not complete:
        raise ValueError("No complete 24-hour days found")
    timestamps = [t for group in complete for t in group]
    days = [group[0].date().isoformat() for group in complete]
    targets = np.asarray([energy[t] for t in timestamps], dtype=np.float32).reshape(-1, 24, 4)
    meteo = np.asarray([weather[t] for t in timestamps], dtype=np.float32).reshape(-1, 24, len(weather_cols))
    times = calendar_features(timestamps).reshape(-1, 24, len(TIME_FEATURES))
    audit = {"dropped_incomplete_days": len(groups)-len(complete),
             "energy_hours_without_weather": len(energy.keys()-weather.keys()),
             "weather_hours_without_energy": len(weather.keys()-energy.keys())}
    return targets, meteo, times, days, audit


def synthetic_daily(days=63, weather_feature_set="pv10", seed=31415):
    """Demo-only, condition-driven complete days; independent of training seed."""
    rng = np.random.default_rng(seed)
    if days < 9:
        raise ValueError("Synthetic demo needs at least one day in each year 2014-2022")
    dates = [datetime(year, 1, 1) + timedelta(days=day)
             for year in range(2014, 2023)
             for day in range(days//9 + (year-2014 < days%9))]
    timestamps = [date + timedelta(hours=hour) for date in dates for hour in range(24)]
    hour = np.arange(days*24) % 24
    daylight = np.maximum(0., np.sin(np.pi*(hour-6)/12))
    temperature = 18 + 7*np.sin(2*np.pi*(hour-8)/24) + rng.normal(size=len(hour))
    all_weather = np.column_stack([temperature, temperature-5, 60-10*daylight,
        3+rng.random(len(hour)), 1010+rng.normal(size=len(hour)),
        rng.random(len(hour))*.1, 700*daylight, 900*daylight,
        .7+.05*rng.normal(size=len(hour)), (daylight>.01).astype(float)])
    common = rng.normal(size=len(hour))
    targets = np.column_stack([100+15*daylight+temperature+common,
        np.maximum(0., 35-temperature+common), 10+2*np.maximum(temperature-20, 0)+common,
        np.maximum(0., 30*daylight+daylight*common)])
    indices = [WEATHER_SETS["pv10"].index(name) for name in WEATHER_SETS[weather_feature_set]]
    return (targets.astype(np.float32).reshape(days, 24, 4),
            all_weather[:, indices].astype(np.float32).reshape(days, 24, -1),
            calendar_features(timestamps).reshape(days, 24, -1),
            [timestamps[i*24].date().isoformat() for i in range(days)],
            {"dropped_incomplete_days": 0, "energy_hours_without_weather": 0,
             "weather_hours_without_energy": 0})


class DailyConditions(Dataset):
    def __init__(self, conditions, targets, pv_year, dates):
        self.conditions = torch.as_tensor(conditions, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.pv_year = torch.as_tensor(pv_year, dtype=torch.float32)
        self.dates = list(dates)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        # Conditions contain only weather and calendar, never historical/target energy.
        return self.conditions[index], self.targets[index], self.pv_year[index]


def prepare_daily(raw, weather_feature_set="pv10", train_years=tuple(range(2014, 2021)),
                  val_years=(2021,), test_years=(2022,), smoke=False):
    targets, meteo, times, dates, dropped = raw
    # Fix reduction order across CSV and synthetic/array-backed inputs.
    targets, meteo, times = (np.ascontiguousarray(array, dtype=np.float32)
                             for array in (targets, meteo, times))
    n = len(dates)
    if len(set(dates)) != n or dates != sorted(dates):
        raise ValueError("Daily dates must be unique and chronologically sorted")
    if targets.shape != (n, 24, 4) or meteo.shape != (n, 24, len(WEATHER_SETS[weather_feature_set])) or times.shape != (n, 24, 8):
        raise ValueError("Expected complete days: targets [N,24,4], weather [N,24,Cw], calendar [N,24,8]")
    if not all(np.isfinite(a).all() for a in (targets, meteo, times)):
        raise ValueError("Daily arrays must contain only finite values")
    years = {"train": list(train_years), "val": list(val_years), "test": list(test_years)}
    flat_years = sum(years.values(), [])
    if any(not y for y in years.values()) or len(set(flat_years)) != len(flat_years):
        raise ValueError("Year splits must be nonempty and disjoint without duplicate years")
    span = max(train_years)-min(train_years)
    if span <= 0:
        raise ValueError("PV year coordinate requires a positive training-year span")
    date_years = np.array([datetime.fromisoformat(date).year for date in dates])
    positions = {key: np.flatnonzero(np.isin(date_years, value)) for key, value in years.items()}
    if any(len(value) == 0 for value in positions.values()):
        raise ValueError("Each fixed-year split needs at least one complete day")
    train_targets = targets[positions["train"]].reshape(-1, 4)
    train_weather = meteo[positions["train"]].reshape(-1, meteo.shape[-1])
    mean, scale = train_targets.mean(0), train_targets.std(0) + 1e-6
    weather_mean = train_weather.mean(0)
    weather_scale = train_weather.std(0) + 1e-6
    conditions = np.concatenate([(meteo-weather_mean)/weather_scale, times], axis=-1)
    normalized = (targets-mean)/scale
    pv_year = np.broadcast_to(((date_years-min(train_years))/span)[:, None, None], (n, 24, 1)).copy().astype(np.float32)
    splits, selected_dates = {}, {}
    for split, selected in positions.items():
        dataset = DailyConditions(conditions[selected], normalized[selected], pv_year[selected], [dates[i] for i in selected])
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
        "complete_days": n, "alignment_audit": dropped,
        "excluded_year_days": int((~np.isin(date_years, flat_years)).sum()),
        "year_splits": years, "pv_year_origin": min(train_years), "pv_year_span": span,
        "pv_year_shared_condition": False,
        "split_day_counts": {key: len(value) for key, value in positions.items()},
        "split_date_ranges": {key: [dates[value[0]], dates[value[-1]]] for key, value in positions.items()},
        "used_samples": {key: len(value) for key, value in splits.items()},
        "selected_dates": selected_dates, "data_sha256": fingerprint.hexdigest()}
    return splits, metadata
