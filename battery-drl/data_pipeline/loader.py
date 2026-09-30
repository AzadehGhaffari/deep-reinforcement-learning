from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


@dataclass
class BatteryFileSummary:
    battery_id: str
    rows: int
    cycles: int
    operations: list[str]


def _maybe_datetime(value: Any) -> str | None:
    if value is None:
        return None
    try:
        values = np.asarray(value).astype(int).flatten().tolist()
    except (TypeError, ValueError):
        return None
    if len(values) < 6:
        return None
    try:
        return datetime(*values[:6]).isoformat()
    except ValueError:
        return None


def _to_1d_float_array(value: Any, length: int | None = None) -> np.ndarray:
    if value is None:
        base = np.array([], dtype=float)
    elif isinstance(value, np.ndarray):
        base = value.astype(float).reshape(-1)
    elif np.isscalar(value):
        base = np.array([float(value)], dtype=float)
    else:
        base = np.asarray(value, dtype=float).reshape(-1)

    if length is None:
        return base
    if base.size == length:
        return base
    if base.size == 0:
        return np.full(length, np.nan, dtype=float)
    if base.size == 1:
        return np.full(length, float(base[0]), dtype=float)
    if base.size > length:
        return base[:length]
    padded = np.full(length, np.nan, dtype=float)
    padded[: base.size] = base
    return padded


def _iter_cycles(cycles: Any) -> list[Any]:
    if isinstance(cycles, np.ndarray):
        return cycles.reshape(-1).tolist()
    return [cycles]


def _rows_from_cycle(battery_id: str, cycle_index: int, cycle: Any) -> list[dict[str, Any]]:
    operation = str(getattr(cycle, "type", "")).strip()
    if operation not in {"charge", "discharge"}:
        return []

    data = getattr(cycle, "data", None)
    if data is None:
        return []

    times = _to_1d_float_array(getattr(data, "Time", None))
    if times.size == 0:
        return []

    row_count = len(times)
    capacity = getattr(data, "Capacity", np.nan)
    capacity_array = _to_1d_float_array(capacity, row_count)
    voltage = _to_1d_float_array(getattr(data, "Voltage_measured", None), row_count)
    current = _to_1d_float_array(getattr(data, "Current_measured", None), row_count)
    temperature = _to_1d_float_array(getattr(data, "Temperature_measured", None), row_count)
    current_charge = _to_1d_float_array(getattr(data, "Current_charge", None), row_count)
    voltage_charge = _to_1d_float_array(getattr(data, "Voltage_charge", None), row_count)
    ambient_temperature = float(getattr(cycle, "ambient_temperature", np.nan))
    timestamp = _maybe_datetime(getattr(cycle, "time", None))

    rows: list[dict[str, Any]] = []
    for idx in range(row_count):
        rows.append(
            {
                "battery_id": battery_id,
                "cycle_index": cycle_index,
                "operation": operation,
                "cycle_timestamp": timestamp,
                "ambient_temperature": ambient_temperature,
                "elapsed_time_s": float(times[idx]),
                "voltage": float(voltage[idx]),
                "current": float(current[idx]),
                "temperature": float(temperature[idx]),
                "charger_current": float(current_charge[idx]),
                "charger_voltage": float(voltage_charge[idx]),
                "capacity_ah": float(capacity_array[idx]) if not np.isnan(capacity_array[idx]) else np.nan,
            }
        )
    return rows


def load_nasa_battery_file(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)
    mat = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    battery_id = file_path.stem
    root = mat[battery_id]
    cycles = _iter_cycles(root.cycle)

    rows: list[dict[str, Any]] = []
    for cycle_index, cycle in enumerate(cycles):
        rows.extend(_rows_from_cycle(battery_id, cycle_index, cycle))

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No charge/discharge rows were parsed from {file_path.name}.")
    return frame


def load_raw_directory(raw_dir: str | Path = RAW_DIR) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.mat"))
    if not files:
        raise FileNotFoundError(
            f"No .mat files were found in {raw_dir}. Place NASA files in data/raw/."
        )

    frames = [load_nasa_battery_file(file_path) for file_path in files]
    return pd.concat(frames, ignore_index=True)


def summarize_available_batteries(dataframe: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for battery_id, group in dataframe.groupby("battery_id"):
        summary_rows.append(
            BatteryFileSummary(
                battery_id=battery_id,
                rows=len(group),
                cycles=group["cycle_index"].nunique(),
                operations=sorted(group["operation"].dropna().unique().tolist()),
            ).__dict__
        )
    return pd.DataFrame(summary_rows).sort_values("battery_id")


def main() -> None:
    dataframe = load_raw_directory()
    summary = summarize_available_batteries(dataframe)
    print("Parsed NASA battery data summary")
    print(summary.to_string(index=False))
    print("\nColumns")
    print(dataframe.columns.tolist())


if __name__ == "__main__":
    main()