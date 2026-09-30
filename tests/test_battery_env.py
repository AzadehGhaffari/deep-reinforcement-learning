from __future__ import annotations

import pandas as pd

from environment.battery_env import BatteryChargingEnv, load_config


def build_sample_episode() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "battery_id": ["B0005"] * 6,
            "cycle_index": [1] * 6,
            "operation": ["charge"] * 6,
            "elapsed_time_s": [0, 10, 20, 30, 40, 50],
            "voltage": [3.5, 3.55, 3.6, 3.65, 3.7, 3.75],
            "current": [1.5, 1.5, 1.55, 1.55, 1.6, 1.6],
            "temperature": [25.0, 25.3, 25.6, 26.0, 26.5, 27.0],
            "capacity_ah": [2.0] * 6,
            "soc": [0.20, 0.24, 0.29, 0.34, 0.40, 0.47],
            "soh_actual": [0.99] * 6,
            "prev_soh": [0.99] * 6,
            "temp_trend": [0.0, 0.1, 0.12, 0.18, 0.2, 0.22],
            "current_trend": [0.0, 0.0, 0.05, 0.0, 0.05, 0.0],
            "cycle_normalized": [0.1] * 6,
            "degradation_risk_target": [0.1, 0.11, 0.12, 0.15, 0.17, 0.2],
        }
    )


def test_reset_returns_state_vector() -> None:
    env = BatteryChargingEnv(build_sample_episode(), config=load_config())
    observation, info = env.reset()
    assert observation.shape == (8,)
    assert info["action_name"] == "MAINTAIN"


def test_increase_action_heats_more_than_reduce() -> None:
    config = load_config()
    env_reduce = BatteryChargingEnv(build_sample_episode(), config=config)
    env_increase = BatteryChargingEnv(build_sample_episode(), config=config)
    env_reduce.reset()
    env_increase.reset()

    _, _, _, _, reduce_info = env_reduce.step(1)
    _, _, _, _, increase_info = env_increase.step(2)
    assert increase_info["temperature"] > reduce_info["temperature"]


def test_stop_action_applies_penalty_without_terminating() -> None:
    env = BatteryChargingEnv(build_sample_episode(), config=load_config())
    env.reset()
    _, reward, terminated, truncated, _ = env.step(4)
    assert terminated is False
    assert truncated is False
    assert reward < 0.0


def build_hot_episode() -> pd.DataFrame:
    episode = build_sample_episode()
    episode.loc[1, ["temperature", "current"]] = [45.0, 1.6]
    return episode


def test_unsafe_temperature_terminates_episode() -> None:
    env = BatteryChargingEnv(build_hot_episode(), config=load_config())
    env.reset()
    _, reward, terminated, truncated, info = env.step(2)
    assert terminated is True
    assert truncated is False
    assert info["temperature"] >= load_config()["environment"]["critical_temp_c"]