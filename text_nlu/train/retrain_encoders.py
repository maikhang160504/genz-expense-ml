"""Train encoder models (PhoBERT) for A/B comparison vs TF-IDF — NOT production default.

Production inference always uses TF-IDF from retrain_all / Kaggle unless you set:
  NLU_USE_ENCODER=1

Run manually after dataset changes when benchmarking:
  python text_nlu/train/retrain_encoders.py
  python text_nlu/tools/compare_tfidf_encoder.py --samples "mua cà phê 50k"
"""
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
    from encoder_metrics import finalize_metrics, reset_metrics

    reset_metrics()
    run("train_intent_encoder.py")
    run("train_action_type_encoder.py")
    run("train_category_encoder.py")
    run("train_record_type_encoder.py")
    metrics_path = finalize_metrics()
    print(f"\nEncoder metrics saved → {metrics_path}")
    print(
        "\nEncoder weights saved."
        "\nCompare: python text_nlu/tools/compare_tfidf_encoder.py"
        "\nChitchat: LLM only (no encoder sentiment)."
    )


if __name__ == "__main__":
    main()
