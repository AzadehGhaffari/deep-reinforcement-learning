from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from environment.battery_env import ACTION_NAMES, BatteryChargingEnv
from models.battery_health_model import HEALTH_FEATURE_COLUMNS, load_health_model
from rl.dqn_agent import DQNAgent


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_processed_data(config: dict) -> pd.DataFrame:
    test_path = ROOT / config["paths"]["test_csv"]
    train_path = ROOT / config["paths"]["train_csv"]
    source_path = test_path if test_path.exists() else train_path
    if not source_path.exists():
        raise FileNotFoundError("No processed CSV was found. Run preprocessing first.")
    return pd.read_csv(source_path)


def load_health_model_if_available(config: dict):
    checkpoint_path = ROOT / config["paths"]["health_model_checkpoint"]
    if not checkpoint_path.exists():
        return None
    model_cfg = config["health_model"]
    return load_health_model(
        checkpoint_path=checkpoint_path,
        input_size=len(HEALTH_FEATURE_COLUMNS),
        hidden_size=int(model_cfg["hidden_size"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        device=config["project"]["device"],
    )


def load_agent_if_available(config: dict) -> DQNAgent:
    agent = DQNAgent(state_size=8, action_size=5, config=config)
    checkpoint_path = ROOT / config["paths"]["dqn_checkpoint"]
    if checkpoint_path.exists():
        agent.load(checkpoint_path)
    return agent


def get_episode_dataframe(dataframe: pd.DataFrame, battery_id: str, cycle_index: int, operation: str) -> pd.DataFrame:
    episode_df = dataframe[
        (dataframe["battery_id"] == battery_id)
        & (dataframe["cycle_index"] == cycle_index)
        & (dataframe["operation"] == operation)
    ].copy()
    if episode_df.empty:
        raise ValueError("The selected battery/cycle combination has no rows for the chosen operation.")
    return episode_df.sort_values("elapsed_time_s").reset_index(drop=True)


def init_episode(dataframe: pd.DataFrame, config: dict, battery_id: str, cycle_index: int, operation: str) -> None:
    health_model = load_health_model_if_available(config)
    env = BatteryChargingEnv(
        get_episode_dataframe(dataframe, battery_id, cycle_index, operation),
        config=config,
        health_model=health_model,
    )
    observation, _ = env.reset()
    st.session_state.env = env
    st.session_state.agent = load_agent_if_available(config)
    st.session_state.observation = observation
    st.session_state.done = False
    st.session_state.history = [
        {
            "step": 0,
            "action": "MAINTAIN",
            "soc": env.current_state["soc"],
            "voltage": env.current_state["voltage"],
            "current": env.current_state["current"],
            "temperature": env.current_state["temperature"],
            "estimated_soh": env.current_state["estimated_soh"],
            "degradation_risk": env.current_state["degradation_risk"],
        }
    ]
    st.session_state.last_q_values = st.session_state.agent.get_q_values(observation)
    st.session_state.last_action = 0


def advance_one_step() -> None:
    if st.session_state.done:
        return

    agent: DQNAgent = st.session_state.agent
    observation = st.session_state.observation
    q_values = agent.get_q_values(observation)
    action = int(q_values.argmax())
    next_observation, reward, terminated, truncated, info = st.session_state.env.step(action)

    st.session_state.last_q_values = q_values
    st.session_state.last_action = action
    st.session_state.observation = next_observation
    st.session_state.done = bool(terminated or truncated)
    st.session_state.history.append(
        {
            "step": len(st.session_state.history),
            "action": ACTION_NAMES[action],
            "reward": reward,
            **info,
        }
    )


def render_q_values(q_values) -> None:
    action_items = []
    for action_id, action_name in ACTION_NAMES.items():
        action_items.append({"Action": action_name, "Q-Value": float(q_values[action_id])})
    q_df = pd.DataFrame(action_items)
    st.dataframe(q_df, use_container_width=True, hide_index=True)


def render_history(history_df: pd.DataFrame) -> None:
    metric_columns = [
        "voltage",
        "current",
        "temperature",
        "soc",
        "estimated_soh",
        "degradation_risk",
    ]
    for metric in metric_columns:
        st.plotly_chart(
            px.line(history_df, x="step", y=metric, title=f"{metric.replace('_', ' ').title()} vs Time"),
            use_container_width=True,
        )
    st.plotly_chart(
        px.scatter(history_df, x="step", y="action", title="Agent Actions Over Time"),
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="Battery DRL Dashboard", layout="wide")
    st.title("Real-Time EV Battery Health & Charging Optimization Agent")

    config = load_config()
    dataframe = load_processed_data(config)
    preferred_operation = config["data"]["preferred_operation"]

    batteries = sorted(dataframe["battery_id"].unique().tolist())
    battery_id = st.sidebar.selectbox("Battery", batteries)
    cycles = sorted(
        dataframe[
            (dataframe["battery_id"] == battery_id)
            & (dataframe["operation"] == preferred_operation)
        ]["cycle_index"].unique().tolist()
    )
    cycle_index = st.sidebar.selectbox("Cycle", cycles)
    replay_seconds = st.sidebar.slider(
        "Replay speed (seconds)",
        min_value=0.1,
        max_value=2.0,
        value=float(config["dashboard"]["default_replay_seconds"]),
        step=0.1,
    )
    run_simulation = st.sidebar.toggle("Start / Stop simulation", value=False)

    if "env" not in st.session_state or st.sidebar.button("Reset episode"):
        init_episode(dataframe, config, battery_id, cycle_index, preferred_operation)

    if st.button("Advance one step"):
        advance_one_step()

    if run_simulation and not st.session_state.done:
        advance_one_step()
        time.sleep(replay_seconds)
        st.rerun()

    history_df = pd.DataFrame(st.session_state.history)
    latest = history_df.iloc[-1]
    q_values = st.session_state.last_q_values

    metric_cols = st.columns(4)
    metric_cols[0].metric("SOC", f"{latest['soc']:.3f}")
    metric_cols[1].metric("Voltage", f"{latest['voltage']:.3f} V")
    metric_cols[2].metric("Current", f"{latest['current']:.3f} A")
    metric_cols[3].metric("Temperature", f"{latest['temperature']:.2f} C")

    health_cols = st.columns(3)
    health_cols[0].metric("Estimated SOH", f"{latest['estimated_soh']:.3f}")
    health_cols[1].metric("Degradation Risk", f"{latest['degradation_risk']:.3f}")
    health_cols[2].metric("Current Action", ACTION_NAMES[st.session_state.last_action])

    st.subheader("DQN Q-values")
    render_q_values(q_values)

    st.subheader("Streaming History")
    render_history(history_df)


if __name__ == "__main__":
    main()