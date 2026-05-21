"""Smoke: multi_record_task từ câu có hai món."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "text_nlu"))

from src.config import settings
from src.config.env import load_env_file
from src.nlu.models import (
    load_action_type_model,
    load_category_model,
    load_chitchat_sentiment_model,
    load_intent_model,
    load_record_type_model,
)
from src.nlu.ner import load_ner_model
from src.nlu.pipeline import run_nlu

if __name__ == "__main__":
    load_env_file(settings.ENV_PATH)
    text = "bánh ngọt 29k, bàn chải 18k"
    r = run_nlu(
        text,
        load_intent_model(),
        load_category_model(),
        load_action_type_model(),
        load_record_type_model(),
        load_chitchat_sentiment_model(),
        load_ner_model(settings.NER_MODEL_DIR),
    )
    out = Path(__file__).resolve().parent / "_smoke_multi_record.json"
    out.write_text(
        json.dumps(
            {k: r[k] for k in ("intent", "multi_record_task", "multi_records", "amount_spent")},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out}")
