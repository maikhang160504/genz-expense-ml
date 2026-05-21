"""Huấn luyện encoder intent + action (không train sentiment — Chitchat dùng LLM)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TRAIN = Path(__file__).resolve().parent


def run(name: str) -> None:
    p = TRAIN / name
    print(f"\n>>> {p.name}")
    subprocess.run([sys.executable, str(p)], cwd=str(TRAIN), check=True)


def main() -> None:
    run("train_intent_encoder.py")
    run("train_action_type_encoder.py")
    print("\nChitchat: không train sentiment — dùng Gemini/LLM (xem src/nlg/prompt.py).")


if __name__ == "__main__":
    main()
