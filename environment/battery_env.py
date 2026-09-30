from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
from gymnasium import spaces

from models.battery_health_model import predict_health_from_window


ACTION_NAMES = {
    0: "MAINTAIN",
    1: "REDUCE",
    2: "INCREASE",
    3: "PAUSE",
    4: "STOP",
}


def load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class BatteryChargingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dataframe: pd.DataFrame,
        config: dict | None = None,
        health_model: Any | None = None,
    ) -> None:
        super().__init__()
        self.config = config or load_config()
        self.env_cfg = self.config["environment"]
        self.reward_cfg = self.config["rewards"]
        self.sequence_length = self.config["data"]["sequence_length"]
        self.health_model = health_model
        self.source_df = dataframe.sort_values("elapsed_time_s").reset_index(drop=True).copy()
        if self.source_df.empty:
            raise ValueError("BatteryChargingEnv requires non-empty episode data.")

        self.state_columns = [
            "soc",
            "voltage",
            "current",
            "temperature",
            "estimated_soh",
            "degradation_risk",
            "temp_trend",
            "current_trend",
        ]
        self.observation_space = spaces.Box(low=-self.env_cfg["state_clip"], high=self.env_cfg["state_clip"], shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.current_index = 0
        self.current_state: dict[str, float] = {}
        self.last_action = 0
        self.steps_taken = 0

    def _normalization_bounds(self, key: str) -> tuple[float, float]:
        lower, upper = self.env_cfg["normalization"][key]
        return float(lower), float(upper)

    def _normalize_value(self, key: str, value: float) -> float:
        lower, upper = self._normalization_bounds(key)
        if upper == lower:
            return 0.0
        normalized = 2.0 * ((value - lower) / (upper - lower)) - 1.0
        return float(np.clip(normalized, -self.env_cfg["state_clip"], self.env_cfg["state_clip"]))

    def _history_window(self) -> pd.DataFrame:
        end = self.current_index + 1
        start = max(0, end - self.sequence_length)
        history = self.source_df.iloc[start:end].copy()
        if self.current_state:
            history.loc[history.index[-1], "soc"] = self.current_state["soc"]
            history.loc[history.index[-1], "temperature"] = self.current_state["temperature"]
            history.loc[history.index[-1], "current"] = self.current_state["current"]
        return history

    def _estimate_health(self) -> tuple[float, float]:
        window = self._history_window()
        return predict_health_from_window(
            model=self.health_model,
            history_window=window,
            sequence_length=self.sequence_length,
            device=self.config["project"]["device"],
        )

    def _build_state(self, row: pd.Series, soc: float, temperature: float, current: float) -> dict[str, float]:
        estimated_soh, degradation_risk = self._estimate_health() if self.current_state else (
            float(row.get("soh_actual", 1.0)),
            float(row.get("degradation_risk_target", 0.0)),
        )
        risk_from_limits = max(0.0, (temperature - self.env_cfg["temp_limit_c"]) / 10.0)
        degradation_risk = float(np.clip(0.7 * degradation_risk + 0.3 * risk_from_limits, 0.0, 1.0))

        return {
            "soc": float(np.clip(soc, 0.0, 1.0)),
            "voltage": float(row["voltage"]),
            "current": float(current),
            "temperature": float(temperature),
            "estimated_soh": float(np.clip(estimated_soh, 0.5, 1.05)),
            "degradation_risk": degradation_risk,
            "temp_trend": float(row.get("temp_trend", 0.0)),
            "current_trend": float(row.get("current_trend", 0.0)),
        }

    def _state_to_vector(self, state: dict[str, float]) -> np.ndarray:
        keys = [
            ("soc", "soc"),
            ("voltage", "voltage"),
            ("current", "current"),
            ("temperature", "temperature"),
            ("estimated_soh", "soh"),
            ("degradation_risk", "risk"),
            ("temp_trend", "temp_trend"),
            ("current_trend", "current_trend"),
        ]
        values = [self._normalize_value(norm_key, state[state_key]) for state_key, norm_key in keys]
        return np.asarray(values, dtype=np.float32)

    def _current_multiplier(self, action: int) -> float:
        mapping = self.env_cfg["action_current_multipliers"]
        return float(
            {
                0: mapping["maintain"],
                1: mapping["reduce"],
                2: mapping["increase"],
                3: mapping["pause"],
                4: mapping["stop"],
            }[action]
        )

    def _temperature_delta(self, action: int) -> float:
        mapping = self.env_cfg["action_temperature_delta_c"]
        return float(
            {
                0: mapping["maintain"],
                1: mapping["reduce"],
                2: mapping["increase"],
                3: mapping["pause"],
                4: mapping["stop"],
            }[action]
        )

    def _compute_reward(self, previous_soc: float, action: int, state: dict[str, float]) -> float:
        reward = self.reward_cfg["soc_progress_weight"] * (state["soc"] - previous_soc)
        temp_excess = max(0.0, state["temperature"] - self.env_cfg["temp_limit_c"])
        reward -= self.reward_cfg["temp_penalty_weight"] * temp_excess
        reward -= self.reward_cfg["degradation_penalty_weight"] * state["degradation_risk"]

        if action == 3:
            reward -= self.reward_cfg["pause_penalty"]
        elif action == 4:
            reward -= self.reward_cfg["stop_penalty"]

        unsafe = (
            state["temperature"] >= self.env_cfg["critical_temp_c"]
            or state["degradation_risk"] >= 1.0
        )
        if unsafe:
            reward -= self.reward_cfg["unsafe_penalty"]

        if state["soc"] >= self.env_cfg["target_soc"]:
            reward += self.reward_cfg["target_soc_bonus"]

        return float(reward)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_index = 0
        self.steps_taken = 0
        self.last_action = 0
        row = self.source_df.iloc[self.current_index]
        soc = float(row.get("soc", 0.2))
        temperature = float(row["temperature"])
        current = float(row["current"])
        self.current_state = self._build_state(row, soc=soc, temperature=temperature, current=current)
        return self._state_to_vector(self.current_state), {"action_name": ACTION_NAMES[self.last_action]}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is outside the discrete action space.")

        previous_state = self.current_state.copy()
        previous_soc = previous_state["soc"]
        self.last_action = action
        self.steps_taken += 1

        next_index = min(self.current_index + 1, len(self.source_df) - 1)
        next_row = self.source_df.iloc[next_index]
        dt_hours = max((float(next_row["elapsed_time_s"]) - float(self.source_df.iloc[self.current_index]["elapsed_time_s"])) / 3600.0, 0.0)
        multiplier = self._current_multiplier(action)
        simulated_current = float(next_row["current"] * multiplier)
        soc_delta = abs(simulated_current) * dt_hours / self.env_cfg["nominal_capacity_ah"]
        direction = 1.0 if next_row["operation"] == "charge" else -1.0
        simulated_soc = previous_soc + direction * soc_delta
        simulated_temperature = float(next_row["temperature"] + self._temperature_delta(action) + abs(simulated_current) * 0.25)

        self.current_index = next_index
        self.current_state = self._build_state(
            next_row,
            soc=simulated_soc,
            temperature=simulated_temperature,
            current=simulated_current,
        )
        reward = self._compute_reward(previous_soc, action, self.current_state)

        terminated = self.current_state["temperature"] >= self.env_cfg["critical_temp_c"]
        truncated = self.current_index >= len(self.source_df) - 1 or self.steps_taken >= self.env_cfg["episode_horizon"]

        info = {
            "action_name": ACTION_NAMES[action],
            "soc": self.current_state["soc"],
            "voltage": self.current_state["voltage"],
            "current": self.current_state["current"],
            "temperature": self.current_state["temperature"],
            "estimated_soh": self.current_state["estimated_soh"],
            "degradation_risk": self.current_state["degradation_risk"],
        }
        return self._state_to_vector(self.current_state), reward, terminated, truncated, info

    def render(self) -> None:
        print(self.current_state)