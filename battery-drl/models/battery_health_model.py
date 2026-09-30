from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn


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


class BatteryHealthGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(inputs)
        last_hidden = outputs[:, -1, :]
        raw = self.head(last_hidden)
        soh = torch.sigmoid(raw[:, :1])
        risk = torch.sigmoid(raw[:, 1:2])
        return torch.cat([soh, risk], dim=1)


def pad_feature_window(window: pd.DataFrame, sequence_length: int) -> np.ndarray:
    values = window[HEALTH_FEATURE_COLUMNS].astype(float).to_numpy()
    if len(values) >= sequence_length:
        return values[-sequence_length:]
    if len(values) == 0:
        return np.zeros((sequence_length, len(HEALTH_FEATURE_COLUMNS)), dtype=np.float32)
    pad_rows = np.repeat(values[:1], sequence_length - len(values), axis=0)
    return np.vstack([pad_rows, values]).astype(np.float32)


def predict_health_from_window(
    model: BatteryHealthGRU | None,
    history_window: pd.DataFrame,
    sequence_length: int,
    device: str = "cpu",
) -> tuple[float, float]:
    if model is None:
        latest = history_window.iloc[-1]
        soh = float(latest.get("soh_actual", latest.get("prev_soh", 1.0)))
        risk = float(latest.get("degradation_risk_target", 0.0))
        return soh, risk

    model.eval()
    features = pad_feature_window(history_window, sequence_length)
    with torch.no_grad():
        batch = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        output = model(batch).squeeze(0).cpu().numpy()
    return float(output[0]), float(output[1])


def save_health_model(model: BatteryHealthGRU, checkpoint_path: str | Path) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


def load_health_model(
    checkpoint_path: str | Path,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    device: str = "cpu",
) -> BatteryHealthGRU:
    model = BatteryHealthGRU(input_size, hidden_size, num_layers, dropout)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model