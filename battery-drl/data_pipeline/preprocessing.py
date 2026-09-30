from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from data_pipeline.loader import load_raw_directory


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"


HEALTH_FEATURE_COLUMNS = [
    "voltage",
    "current",
    "temperature",
    "soc",
    "cycle_normalized",
    "prev_soh",
    "temp_trend",
    "current_trend",
]


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _safe_group_ffill(series: pd.Series, fallback: float) -> pd.Series:
    filled = series.ffill().bfill()
    return filled.fillna(fallback)


def add_soh_features(dataframe: pd.DataFrame, rated_capacity_ah: float) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe["capacity_ah"] = dataframe.groupby("battery_id")["capacity_ah"].transform(
        lambda series: _safe_group_ffill(series, rated_capacity_ah)
    )
    dataframe["soh_actual"] = (dataframe["capacity_ah"] / rated_capacity_ah).clip(0.5, 1.05)
    dataframe["prev_soh"] = dataframe.groupby("battery_id")["soh_actual"].shift(1)
    dataframe["prev_soh"] = dataframe["prev_soh"].fillna(dataframe["soh_actual"])
    return dataframe


def add_soc_estimate(dataframe: pd.DataFrame, nominal_capacity_ah: float) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe["soc"] = 0.0

    for _, index in dataframe.groupby(["battery_id", "cycle_index"]).groups.items():
        group = dataframe.loc[index].sort_values("elapsed_time_s")
        dt_hours = group["elapsed_time_s"].diff().fillna(0.0) / 3600.0
        current_ah = group["current"].abs() * dt_hours
        start_soc = 0.2 if group["operation"].iloc[0] == "charge" else 1.0
        direction = 1.0 if group["operation"].iloc[0] == "charge" else -1.0
        soc = start_soc + direction * (current_ah.cumsum() / nominal_capacity_ah)
        dataframe.loc[group.index, "soc"] = soc.clip(0.0, 1.0)

    return dataframe


def add_trend_features(dataframe: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    dataframe = dataframe.copy()
    grouped = dataframe.groupby("battery_id", group_keys=False)
    dataframe["temp_trend"] = grouped["temperature"].transform(
        lambda series: series.diff().rolling(window, min_periods=1).mean().fillna(0.0)
    )
    dataframe["current_trend"] = grouped["current"].transform(
        lambda series: series.diff().rolling(window, min_periods=1).mean().fillna(0.0)
    )
    dataframe["cycle_normalized"] = grouped["cycle_index"].transform(
        lambda series: series / max(series.max(), 1)
    )
    return dataframe


def add_risk_target(dataframe: pd.DataFrame, temp_limit_c: float) -> pd.DataFrame:
    dataframe = dataframe.copy()
    temp_component = ((dataframe["temperature"] - temp_limit_c) / 12.0).clip(-1.0, 2.0)
    current_component = dataframe["current"].abs() / 4.0
    soh_component = 1.0 - dataframe["soh_actual"]
    risk = 0.35 * current_component + 0.4 * temp_component.clip(lower=0.0) + 0.25 * soh_component
    dataframe["degradation_risk_target"] = risk.clip(0.0, 1.0)
    return dataframe


def preprocess_raw_dataset(config: dict) -> pd.DataFrame:
    raw_dir = ROOT / config["paths"]["raw_dir"]
    preferred = set(config["data"]["selected_batteries"])
    dataframe = load_raw_directory(raw_dir)
    dataframe = dataframe[dataframe["battery_id"].isin(preferred)].copy()
    dataframe = dataframe.sort_values(["battery_id", "cycle_index", "elapsed_time_s"]).reset_index(drop=True)
    dataframe = add_soh_features(dataframe, config["project"]["rated_capacity_ah"])
    dataframe = add_soc_estimate(dataframe, config["environment"]["nominal_capacity_ah"])
    dataframe = add_trend_features(dataframe)
    dataframe = add_risk_target(dataframe, config["environment"]["temp_limit_c"])
    dataframe = dataframe.dropna(subset=["voltage", "current", "temperature", "soc", "soh_actual"])
    return dataframe


def split_by_battery(dataframe: pd.DataFrame, train_ratio: float, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    batteries = sorted(dataframe["battery_id"].unique().tolist())
    if len(batteries) < 3:
        total = len(dataframe)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        return (
            dataframe.iloc[:train_end].copy(),
            dataframe.iloc[train_end:val_end].copy(),
            dataframe.iloc[val_end:].copy(),
        )

    train_cut = max(1, int(len(batteries) * train_ratio))
    val_cut = max(train_cut + 1, int(len(batteries) * (train_ratio + val_ratio)))
    train_ids = batteries[:train_cut]
    val_ids = batteries[train_cut:val_cut]
    test_ids = batteries[val_cut:]

    return (
        dataframe[dataframe["battery_id"].isin(train_ids)].copy(),
        dataframe[dataframe["battery_id"].isin(val_ids)].copy(),
        dataframe[dataframe["battery_id"].isin(test_ids)].copy(),
    )


def save_processed_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> None:
    processed_dir = ROOT / config["paths"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(ROOT / config["paths"]["train_csv"], index=False)
    val_df.to_csv(ROOT / config["paths"]["val_csv"], index=False)
    test_df.to_csv(ROOT / config["paths"]["test_csv"], index=False)


def main() -> None:
    config = load_config()
    dataframe = preprocess_raw_dataset(config)
    split_cfg = config["data"]["split"]
    train_df, val_df, test_df = split_by_battery(
        dataframe,
        split_cfg["train"],
        split_cfg["val"],
    )
    save_processed_splits(train_df, val_df, test_df, config)
    print("Preprocessing complete")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Feature columns: {HEALTH_FEATURE_COLUMNS}")


if __name__ == "__main__":
    main()