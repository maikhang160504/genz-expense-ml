"""So sánh record_type TF-IDF vs encoder (TASK-13) — chạy sau train_record_type_encoder.py."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "text_nlu"))

from src.config import settings
from src.nlu.models import (
    load_action_type_model,
    load_category_model,
    load_chitchat_sentiment_model,
    load_intent_model,
    load_record_type_model,
)
from src.nlu.pipeline import run_nlu

TESTS = [
    ("me cho 1tr", "Income"),
    ("mẹ ck 500k", "Income"),
    ("ăn phở 45k", "Expense"),
    ("mua quà cho mẹ 200k", "Expense"),
    ("hoàn tiền 50k", "Income"),
]


def _run_with_record_backend(backend: str) -> list[str]:
    enc_path = settings.RECORD_TYPE_ENCODER_PATH
    tfidf_path = settings.RECORD_TYPE_MODEL_PATH
    if backend == "encoder":
        if not enc_path.exists():
            return ["SKIP encoder: file missing"]
        os.environ["RECORD_TYPE_FORCE"] = "encoder"
        # Temporarily hide tfidf priority by renaming not needed — load uses encoder first
    rec_m = load_record_type_model()
    if backend == "tfidf" and rec_m.get("backend") == "encoder":
        # Force tfidf only
        import joblib

        payload = joblib.load(tfidf_path)
        rec_m = {"backend": "tfidf", "vectorizer": payload["vectorizer"], "model": payload["model"]}
    elif backend == "encoder" and enc_path.exists():
        import joblib

        rec_m = {"backend": "encoder", "bundle": joblib.load(enc_path)}

    intent_m = load_intent_model()
    cat_m = load_category_model()
    act_m = load_action_type_model()
    sent_m = load_chitchat_sentiment_model()
    lines = []
    ok = 0
    for text, exp in TESTS:
        r = run_nlu(text, intent_m, cat_m, act_m, rec_m, sent_m, None)
        got = r.get("record_type")
        status = "OK" if got == exp else "FAIL"
        if status == "OK":
            ok += 1
        hint = None  # hints removed; model-only
        lines.append(f"  {status} {text!r} -> {got} (exp {exp}, hint={hint})")
    lines.insert(0, f"[{backend}] {ok}/{len(TESTS)}")
    return lines


def main() -> None:
    out_lines = ["TASK-13 record_type backend compare", ""]
    out_lines.extend(_run_with_record_backend("tfidf"))
    out_lines.append("")
    out_lines.extend(_run_with_record_backend("encoder"))
    out_lines.append("")
    out_lines.append("Chọn backend tốt hơn; encoder cần record_type_encoder.joblib")
    out = Path(__file__).parent / "_task13_verify.txt"
    out.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
