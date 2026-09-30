from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from models.battery_health_model import BatteryHealthGRU, HEALTH_FEATURE_COLUMNS, save_health_model


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class BatterySequenceDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, sequence_length: int, stride: int = 1):
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        grouped = dataframe.groupby(["battery_id", "cycle_index"])
        for _, group in grouped:
            group = group.sort_values("elapsed_time_s")
            features = group[HEALTH_FEATURE_COLUMNS].astype(float).to_numpy(dtype=np.float32)
            targets = group[["soh_actual", "degradation_risk_target"]].astype(float).to_numpy(dtype=np.float32)
            if len(group) < sequence_length:
                continue
            last_start = len(group) - sequence_length
            starts = list(range(0, last_start + 1, stride))
            if starts[-1] != last_start:
                starts.append(last_start)
            for start in starts:
                end = start + sequence_length
                self.samples.append((features[start:end], targets[end - 1]))

        if not self.samples:
            raise ValueError("No training sequences were created. Check preprocessing output and sequence length.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features, target = self.samples[index]
        return torch.tensor(features), torch.tensor(target)


def evaluate_model(model: BatteryHealthGRU, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    y_true: list[float] = []
    y_pred: list[float] = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            total_loss += float(loss.item())
            y_true.extend(targets[:, 0].cpu().numpy().tolist())
            y_pred.extend(outputs[:, 0].cpu().numpy().tolist())

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    avg_loss = total_loss / max(len(loader), 1)
    return float(avg_loss), float(mae), float(rmse)


def main() -> None:
    config = load_config()
    train_df = pd.read_csv(ROOT / config["paths"]["train_csv"])
    val_df = pd.read_csv(ROOT / config["paths"]["val_csv"])

    sequence_length = int(config["data"]["sequence_length"])
    stride = int(config["health_model"].get("sequence_stride", 1))
    train_dataset = BatterySequenceDataset(train_df, sequence_length, stride=stride)
    val_dataset = BatterySequenceDataset(val_df, sequence_length, stride=stride)

    batch_size = int(config["health_model"]["batch_size"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device(config["project"]["device"])
    model = BatteryHealthGRU(
        input_size=len(HEALTH_FEATURE_COLUMNS),
        hidden_size=int(config["health_model"]["hidden_size"]),
        num_layers=int(config["health_model"]["num_layers"]),
        dropout=float(config["health_model"]["dropout"]),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["health_model"]["learning_rate"]))
    criterion = nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in range(int(config["health_model"]["epochs"])):
        model.train()
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss, mae, rmse = evaluate_model(model, val_loader, device)
        print(f"epoch={epoch + 1}, val_loss={val_loss:.4f}, val_mae={mae:.4f}, val_rmse={rmse:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_health_model(model, ROOT / config["paths"]["health_model_checkpoint"])

    metrics = {"best_val_loss": best_val_loss, "val_mae": mae, "val_rmse": rmse}
    metrics_path = ROOT / config["paths"]["artifacts_dir"] / "health_model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()