"""So sánh category TF-IDF vs encoder (TASK-16) — sau train_category_encoder.py."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "text_nlu"))

from src.config import settings
from src.nlu.models import (
    load_action_type_model,
    load_chitchat_sentiment_model,
    load_intent_model,
    load_record_type_model,
)
from src.nlu.pipeline import run_nlu

TESTS = [
    ("mua trên shopee 120k", "Shopping"),
    ("tiktok shop 89k", "Shopping"),
    ("gạo 50k", "Essentials"),
    ("mua quà cho mẹ 200k", "Essentials"),
    ("ăn phở 45k", "Food"),
]


def _cat_model(backend: str):
    if backend == "encoder" and settings.CATEGORY_ENCODER_PATH.exists():
        return {"backend": "encoder", "bundle": joblib.load(settings.CATEGORY_ENCODER_PATH)}
    payload = joblib.load(settings.CATEGORY_MODEL_PATH)
    return {"backend": "tfidf", "vectorizer": payload["vectorizer"], "model": payload["model"]}


def _run(backend: str) -> list[str]:
    if backend == "encoder" and not settings.CATEGORY_ENCODER_PATH.exists():
        return ["SKIP encoder: category_encoder.joblib missing"]
    cat_m = _cat_model(backend)
    intent_m = load_intent_model()
    rec_m = load_record_type_model()
    act_m = load_action_type_model()
    sent_m = load_chitchat_sentiment_model()
    lines = [f"[{backend}]"]
    ok = 0
    for text, exp in TESTS:
        r = run_nlu(text, intent_m, cat_m, act_m, rec_m, sent_m, None)
        got = r.get("category")
        status = "OK" if got == exp else "FAIL"
        if status == "OK":
            ok += 1
        lines.append(f"  {status} {text!r} -> {got} (exp {exp})")
    lines.append(f"  {ok}/{len(TESTS)}")
    return lines


def main() -> None:
    lines = ["TASK-16 category backend compare", ""]
    lines.extend(_run("tfidf"))
    lines.append("")
    lines.extend(_run("encoder"))
    out = Path(__file__).parent / "_task16_verify.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
