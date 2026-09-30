from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from environment.battery_env import BatteryChargingEnv
from models.battery_health_model import HEALTH_FEATURE_COLUMNS, load_health_model
from rl.dqn_agent import DQNAgent


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_episode_groups(dataframe: pd.DataFrame, preferred_operation: str) -> list[pd.DataFrame]:
    groups = []
    grouped = dataframe.groupby(["battery_id", "cycle_index"])
    for _, group in grouped:
        if group["operation"].iloc[0] == preferred_operation and len(group) > 5:
            groups.append(group.sort_values("elapsed_time_s").reset_index(drop=True))
    if not groups:
        raise ValueError("No suitable training episodes were found in the processed data.")
    return groups


def maybe_load_health_model(config: dict):
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    config = load_config()
    set_seed(int(config["project"]["random_seed"]))
    train_df = pd.read_csv(ROOT / config["paths"]["train_csv"])
    episodes = select_episode_groups(train_df, config["data"]["preferred_operation"])
    health_model = maybe_load_health_model(config)

    agent = DQNAgent(state_size=8, action_size=5, config=config)
    episode_rewards: list[float] = []
    action_counts = {action: 0 for action in range(5)}
    unsafe_events = 0
    soc_targets = 0

    for episode_idx in range(int(config["dqn"]["episodes"])):
        episode_df = random.choice(episodes)
        env = BatteryChargingEnv(episode_df, config=config, health_model=health_model)
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        while not done and not truncated:
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done or truncated)
            agent.learn()

            state = next_state
            total_reward += reward
            action_counts[action] += 1
            if info["temperature"] >= config["environment"]["temp_limit_c"]:
                unsafe_events += 1
            if info["soc"] >= config["environment"]["target_soc"]:
                soc_targets += 1

        episode_rewards.append(total_reward)
        if (episode_idx + 1) % 10 == 0:
            recent = np.mean(episode_rewards[-10:])
            print(
                f"episode={episode_idx + 1}, avg_reward_last_10={recent:.3f}, epsilon={agent.epsilon:.3f}"
            )

    checkpoint_path = ROOT / config["paths"]["dqn_checkpoint"]
    agent.save(checkpoint_path)

    summary = {
        "episodes": len(episode_rewards),
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
        "action_counts": action_counts,
        "unsafe_events": int(unsafe_events),
        "soc_target_hits": int(soc_targets),
    }
    summary_path = ROOT / config["paths"]["artifacts_dir"] / "dqn_training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()