from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TARGET_COLUMNS = ["Electricity", "Heat", "Cooling", "PV"]
BASE_WEATHER_COLUMNS = [
    "Temperature",
    "Dew Point",
    "Humidity",
    "Wind Speed",
    "Pressure",
    "Precip",
]
PV_WEATHER_COLUMNS = [
    "ALLSKY_SFC_SW_DWN",
    "CLRSKY_SFC_SW_DWN",
    "PV_CLEARNESS_RATIO",
    "PV_IS_DAYLIGHT",
]
TIME_COLUMNS = [
    "month_sin",
    "month_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    "weekday_sin",
    "weekday_cos",
    "hour_sin",
    "hour_cos",
]
SPLIT_YEARS = {
    "train": tuple(range(2014, 2021)),
    "val": (2021,),
    "test": (2022,),
}


class HEEWConditionDataset(Dataset):
    """Target-day exogenous conditions and four targets, without energy history."""

    def __init__(
        self,
        energy_path: str | Path,
        weather_path: str | Path,
        split: str,
        weather_feature_set: str = "pv10",
    ) -> None:
        if split not in SPLIT_YEARS:
            raise ValueError(f"Unknown split {split!r}.")
        if weather_feature_set == "base6":
            weather_columns = BASE_WEATHER_COLUMNS
        elif weather_feature_set == "pv10":
            weather_columns = BASE_WEATHER_COLUMNS + PV_WEATHER_COLUMNS
        else:
            raise ValueError("weather_feature_set must be base6 or pv10")

        self.split = split
        self.target_cols = list(TARGET_COLUMNS)
        self.weather_cols = list(weather_columns)
        self.time_cols = list(TIME_COLUMNS)
        energy = pd.read_csv(energy_path)
        weather = pd.read_csv(weather_path)
        timestamp_columns = ["Year", "Month", "Day", "Hour"]
        for name, frame, required in (
            ("energy", energy, timestamp_columns + self.target_cols),
            ("weather", weather, timestamp_columns + self.weather_cols),
        ):
            missing = [column for column in required if column not in frame.columns]
            if missing:
                raise ValueError(f"{name} file missing columns:" f" {missing}")

        rename = {"Year": "year", "Month": "month", "Day": "day", "Hour": "hour"}
        energy["_timestamp"] = pd.to_datetime(
            energy[timestamp_columns].rename(columns=rename), errors="raise"
        )
        weather["_timestamp"] = pd.to_datetime(
            weather[timestamp_columns].rename(columns=rename), errors="raise"
        )
        for name, frame in (("energy", energy), ("weather", weather)):
            if frame["_timestamp"].duplicated().any():
                raise ValueError(f"{name} file contains duplicate timestamps")

        common = pd.Index(
            sorted(set(energy["_timestamp"]).intersection(weather["_timestamp"])),
            name="_timestamp",
        )
        energy = energy.set_index("_timestamp").loc[common].reset_index()
        weather = weather.set_index("_timestamp").loc[common].reset_index()
        if not energy["_timestamp"].equals(weather["_timestamp"]):
            raise RuntimeError("Energy and weather timestamps are not aligned.")

        target_values = energy[self.target_cols].to_numpy(np.float32)
        weather_values = weather[self.weather_cols].to_numpy(np.float32)
        if not np.isfinite(target_values).all() or not np.isfinite(weather_values).all():
            raise ValueError("Energy/weather arrays contain NaN or infinite values.")

        timestamps = energy["_timestamp"]
        time_values = self._cyclical_time_features(timestamps)
        daily = pd.DataFrame(
            {
                "date": timestamps.dt.floor("D"),
                "hour": timestamps.dt.hour,
                "index": np.arange(len(timestamps)),
            }
        )
        day_indices: list[np.ndarray] = []
        day_dates: list[pd.Timestamp] = []
        for date, group in daily.groupby("date", sort=True):
            ordered = group.sort_values("hour")
            if len(ordered) == 24 and np.array_equal(
                ordered["hour"].to_numpy(), np.arange(24)
            ):
                day_indices.append(ordered["index"].to_numpy())
                day_dates.append(date)
        if not day_indices:
            raise ValueError("No complete 24-hour days found.")

        keep = [index for index, date in enumerate(day_dates) if 2014 <= date.year <= 2022]
        indices = np.stack([day_indices[index] for index in keep])
        dates = [day_dates[index] for index in keep]
        years = np.asarray([date.year for date in dates])
        train_positions = np.flatnonzero((years >= 2014) & (years <= 2020))
        split_positions = np.flatnonzero(np.isin(years, SPLIT_YEARS[split]))
        if train_positions.size == 0 or split_positions.size == 0:
            raise ValueError(f"Missing complete days for split={split}.")

        train_flat = indices[train_positions].reshape(-1)
        self.target_mean = target_values[train_flat].mean(0)
        self.target_std = target_values[train_flat].std(0) + 1e-6
        self.weather_mean = weather_values[train_flat].mean(0)
        self.weather_std = weather_values[train_flat].std(0) + 1e-6

        selected = indices[split_positions]
        target_norm = (target_values - self.target_mean) / self.target_std
        weather_norm = (weather_values - self.weather_mean) / self.weather_std
        self.targets = torch.from_numpy(target_norm[selected].transpose(0, 2, 1))
        self.weather = torch.from_numpy(weather_norm[selected].transpose(0, 2, 1))
        self.calendar = torch.from_numpy(time_values[selected].transpose(0, 2, 1))
        pv_year = (timestamps.dt.year.to_numpy(np.float32) - 2014.0) / 6.0
        self.pv_year = torch.from_numpy(pv_year[selected][:, None, :])
        self.dates = [dates[index].strftime("%Y-%m-%d") for index in split_positions]

    @staticmethod
    def _cyclical_time_features(timestamps: pd.Series) -> np.ndarray:
        month = timestamps.dt.month.to_numpy(np.float32)
        day = timestamps.dt.dayofyear.to_numpy(np.float32)
        weekday = timestamps.dt.weekday.to_numpy(np.float32)
        hour = timestamps.dt.hour.to_numpy(np.float32)
        days_in_year = timestamps.dt.is_leap_year.to_numpy(np.float32) + 365.0
        angles = (
            2 * np.pi * (month - 1) / 12,
            2 * np.pi * (day - 1) / days_in_year,
            2 * np.pi * weekday / 7,
            2 * np.pi * hour / 24,
        )
        return np.column_stack(
            [component for angle in angles for component in (np.sin(angle), np.cos(angle))]
        ).astype(np.float32)

    @property
    def condition_channels(self) -> int:
        return len(self.weather_cols) + len(self.time_cols)

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, index: int):
        conditions = torch.cat([self.weather[index], self.calendar[index]], dim=0)
        return conditions, self.pv_year[index], self.targets[index], self.dates[index]

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.target_mean, dtype=values.dtype, device=values.device)
        std = torch.as_tensor(self.target_std, dtype=values.dtype, device=values.device)
        shape = [1] * values.ndim
        shape[-2] = len(self.target_cols)
        return values * std.view(shape) + mean.view(shape)

