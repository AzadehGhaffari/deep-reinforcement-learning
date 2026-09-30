from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Battery DRL term-project command helper."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("inspect-data", help="Inspect raw NASA battery files")
    subparsers.add_parser("preprocess", help="Preprocess raw NASA battery files")
    subparsers.add_parser("train-health", help="Train the GRU battery-health model")
    subparsers.add_parser("train-dqn", help="Train the DQN agent")
    subparsers.add_parser("evaluate", help="Evaluate baseline and DQN policies")
    subparsers.add_parser("dashboard", help="Show the Streamlit launch command")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "inspect-data": ("script", ".\\data_pipeline\\loader.py"),
        "preprocess": ("module", "data_pipeline.preprocessing"),
        "train-health": ("module", "training.train_health_model"),
        "train-dqn": ("module", "rl.train_dqn"),
        "evaluate": ("module", "evaluation.evaluate_agent"),
    }

    if args.command == "dashboard":
        print("Run: ..\\.venv\\Scripts\\python.exe -m streamlit run .\\dashboard\\app.py")
        return

    kind, target = commands[args.command]
    if kind == "module":
        print(f"Run: ..\\.venv\\Scripts\\python.exe -m {target}")
    else:
        print(f"Run: ..\\.venv\\Scripts\\python.exe {target}")
    print(f"Project root: {PROJECT_ROOT}")


if __name__ == "__main__":
    main()