from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from environment.battery_env import BatteryChargingEnv
from models.battery_health_model import HEALTH_FEATURE_COLUMNS, load_health_model
from rl.dqn_agent import DQNAgent


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_episodes(dataframe: pd.DataFrame, preferred_operation: str) -> list[pd.DataFrame]:
    episodes = []
    for _, group in dataframe.groupby(["battery_id", "cycle_index"]):
        if group["operation"].iloc[0] == preferred_operation and len(group) > 5:
            episodes.append(group.sort_values("elapsed_time_s").reset_index(drop=True))
    return episodes


def choose_rule_based_action(info: dict, config: dict) -> int:
    env_cfg = config["environment"]
    if info["temperature"] >= env_cfg["temp_limit_c"] + 3 or info["degradation_risk"] >= env_cfg["risk_limit"]:
        return 3
    if info["temperature"] >= env_cfg["temp_limit_c"] or info["degradation_risk"] >= env_cfg["risk_limit"] * 0.8:
        return 1
    if info["soc"] < env_cfg["target_soc"] * 0.9 and info["temperature"] < env_cfg["temp_limit_c"] - 4:
        return 2
    return 0


def run_policy(name: str, episodes: list[pd.DataFrame], config: dict, agent: DQNAgent | None, health_model) -> dict:
    totals = []
    unsafe_events = 0
    target_hits = 0
    mean_risks = []
    temperature_exposure = 0.0

    for episode_df in episodes:
        env = BatteryChargingEnv(episode_df, config=config, health_model=health_model)
        state, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        last_info = {
            "soc": env.current_state["soc"],
            "temperature": env.current_state["temperature"],
            "degradation_risk": env.current_state["degradation_risk"],
        }

        while not done and not truncated:
            if name == "fixed":
                action = 0
            elif name == "rule_based":
                action = choose_rule_based_action(last_info, config)
            else:
                if agent is None:
                    raise ValueError("DQN policy evaluation requires a trained agent.")
                action = agent.select_action(state, exploit=True)

            state, reward, done, truncated, last_info = env.step(action)
            total_reward += reward
            mean_risks.append(last_info["degradation_risk"])
            temperature_exposure += max(0.0, last_info["temperature"] - config["environment"]["temp_limit_c"])
            if last_info["temperature"] >= config["environment"]["temp_limit_c"]:
                unsafe_events += 1
            if last_info["soc"] >= config["environment"]["target_soc"]:
                target_hits += 1

        totals.append(total_reward)

    return {
        "policy": name,
        "episodes": len(episodes),
        "mean_total_reward": sum(totals) / max(len(totals), 1),
        "unsafe_events": unsafe_events,
        "temperature_exposure": temperature_exposure,
        "mean_risk": sum(mean_risks) / max(len(mean_risks), 1),
        "soc_target_hits": target_hits,
    }


def main() -> None:
    config = load_config()
    test_df = pd.read_csv(ROOT / config["paths"]["test_csv"])
    episodes = select_episodes(test_df, config["data"]["preferred_operation"])
    if not episodes:
        raise ValueError("No evaluation episodes were found in the test split.")

    health_checkpoint = ROOT / config["paths"]["health_model_checkpoint"]
    health_model = None
    if health_checkpoint.exists():
        model_cfg = config["health_model"]
        health_model = load_health_model(
            checkpoint_path=health_checkpoint,
            input_size=len(HEALTH_FEATURE_COLUMNS),
            hidden_size=int(model_cfg["hidden_size"]),
            num_layers=int(model_cfg["num_layers"]),
            dropout=float(model_cfg["dropout"]),
            device=config["project"]["device"],
        )

    agent = DQNAgent(state_size=8, action_size=5, config=config)
    dqn_checkpoint = ROOT / config["paths"]["dqn_checkpoint"]
    if dqn_checkpoint.exists():
        agent.load(dqn_checkpoint)
    else:
        agent = None

    results = [
        run_policy("fixed", episodes, config, agent=None, health_model=health_model),
        run_policy("rule_based", episodes, config, agent=None, health_model=health_model),
        run_policy("dqn", episodes, config, agent=agent, health_model=health_model),
    ]

    results_df = pd.DataFrame(results)
    output_path = ROOT / config["paths"]["evaluation_csv"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()