"""Compare TF-IDF vs encoder predictions on sample texts (offline benchmark)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config.env import load_env_file
from src.config import settings
from src.nlu import models
from src.nlu.pipeline import classify_intent, MONEY_RE


def _intent_label(intent_model: dict, text: str) -> str:
    intent, _, _ = classify_intent(text, intent_model)
    return intent


def main() -> None:
    load_env_file(settings.ENV_PATH)
    parser = argparse.ArgumentParser(description="Compare TF-IDF vs encoder NLU backends")
    parser.add_argument(
        "--samples",
        nargs="*",
        default=[
            "mua cà phê 50k",
            "tổng chi tháng này bao nhiêu",
            "chào mimo",
            "đặt hạn mức ăn uống 2 triệu",
        ],
    )
    args = parser.parse_args()

    tfidf_intent = models.load_intent_model()
    try:
        enc_intent = models.load_encoder_intent_model()
    except FileNotFoundError as exc:
        print(f"Encoder not trained: {exc}")
        enc_intent = None

    print("Production backend (default loaders):", tfidf_intent.get("backend"))
    print("-" * 72)
    for text in args.samples:
        tf = _intent_label(tfidf_intent, text)
        enc = _intent_label(enc_intent, text) if enc_intent else "N/A"
        match = "OK" if tf == enc else "DIFF"
        print(f"[{match}] {text!r}")
        print(f"      TF-IDF: {tf}  |  Encoder: {enc}")
    print("-" * 72)
    print("To run full pipeline with encoder: NLU_USE_ENCODER=1 (experimental only)")


if __name__ == "__main__":
    main()
