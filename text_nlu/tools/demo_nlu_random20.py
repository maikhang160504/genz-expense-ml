"""
Demo NLU (encoder nếu có): 20 câu ngẫu nhiên từ 3 nguồn (Record / Action / Chitchat),
ghi JSON để kiểm tra.

Chạy từ thư mục repo:
  python text_nlu/tools/demo_nlu_random20.py

Biến môi trường:
  DEMO_NLU_JSON   đường dẫn file JSON đầu ra (mặc định text_nlu/tools/demo_nlu_random20_output.json)
  DEMO_NLU_SEED   seed (mặc định 42)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "text_nlu"))

from src.config import settings  # noqa: E402
from src.config.env import load_env_file  # noqa: E402
from src.nlu.json_sanitize import json_sanitize  # noqa: E402
from src.nlu.models import (  # noqa: E402
    load_action_type_model,
    load_category_model,
    load_chitchat_sentiment_model,
    load_intent_model,
    load_record_type_model,
)
from src.nlu.ner import load_ner_model  # noqa: E402
from src.nlu.pipeline import run_nlu  # noqa: E402

DATA = settings.TEXT_NLU_DIR / "datasets"
DEFAULT_OUT = Path(__file__).resolve().parent / "demo_nlu_random20_output.json"


def _sample(df: pd.DataFrame, n: int, seed: int, start: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed + start, replace=False).reset_index(drop=True)


def main() -> None:
    load_env_file(settings.ENV_PATH)
    seed = int(os.environ.get("DEMO_NLU_SEED", "42"))
    out_path = Path(os.environ.get("DEMO_NLU_JSON", str(DEFAULT_OUT)))

    rec = pd.read_csv(DATA / "intent_record.csv", encoding="utf-8-sig")[["text"]].dropna()
    act = pd.read_csv(DATA / "intent_action.csv", encoding="utf-8-sig")[
        ["text", "intent", "action_type"]
    ].dropna()
    chat = pd.read_csv(DATA / "intent_chitchat.csv", encoding="utf-8-sig")[
        ["text", "intent", "sentiment"]
    ].dropna()

    # 7 + 7 + 6 = 20 câu, 3 loại
    n_rec, n_act, n_chat = 7, 7, 6
    s_rec = _sample(rec, n_rec, seed, 0)
    s_act = _sample(act, n_act, seed, 100)
    s_chat = _sample(chat, n_chat, seed, 200)

    intent_m = load_intent_model()
    cat_m = load_category_model()
    act_m = load_action_type_model()
    rec_m = load_record_type_model()
    sent_m = load_chitchat_sentiment_model()
    ner = load_ner_model(settings.NER_MODEL_DIR)

    rows: list[dict] = []

    for _, r in s_rec.iterrows():
        text = str(r["text"]).strip()
        if not text:
            continue
        nlu = run_nlu(text, intent_m, cat_m, act_m, rec_m, sent_m, ner)
        rows.append(
            {
                "gold_source": "intent_record.csv",
                "gold_intent": "Record",
                "gold_action_type": None,
                "gold_sentiment": None,
                "intent_match": nlu.get("intent") == "Record",
                "nlu": json_sanitize(nlu),
            }
        )

    for _, r in s_act.iterrows():
        text = str(r["text"]).strip()
        if not text:
            continue
        nlu = run_nlu(text, intent_m, cat_m, act_m, rec_m, sent_m, ner)
        g_at = str(r["action_type"]) if pd.notna(r.get("action_type")) else None
        pred_at = nlu.get("action_type")
        rows.append(
            {
                "gold_source": "intent_action.csv",
                "gold_intent": str(r.get("intent", "Action")),
                "gold_action_type": g_at,
                "gold_sentiment": None,
                "intent_match": nlu.get("intent") == str(r.get("intent", "Action")),
                "action_type_match": (pred_at == g_at) if g_at and pred_at else None,
                "nlu": json_sanitize(nlu),
            }
        )

    for _, r in s_chat.iterrows():
        text = str(r["text"]).strip()
        if not text:
            continue
        nlu = run_nlu(text, intent_m, cat_m, act_m, rec_m, sent_m, ner)
        rows.append(
            {
                "gold_source": "intent_chitchat.csv",
                "gold_intent": str(r.get("intent", "Chitchat")),
                "gold_action_type": None,
                "gold_sentiment": None,
                "intent_match": nlu.get("intent") == str(r.get("intent", "Chitchat")),
                "sentiment_match": None,
                "chitchat_via": "llm",
                "nlu": json_sanitize(nlu),
            }
        )

    payload = {
        "meta": {
            "seed": seed,
            "count": len(rows),
            "split_record_action_chitchat": [n_rec, n_act, n_chat],
            "paths": {
                "intent_encoder": str(settings.INTENT_ENCODER_PATH),
                "action_type_encoder": str(settings.ACTION_TYPE_ENCODER_PATH),
                "chitchat_encoder": str(settings.CHITCHAT_ENCODER_PATH),
            },
            "backends": {
                "intent": intent_m.get("backend"),
                "action_type": act_m.get("backend"),
                "sentiment": sent_m.get("backend"),
                "category": cat_m.get("backend"),
                "record_type": rec_m.get("backend"),
            },
        },
        "samples": rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(rows)} samples)")


if __name__ == "__main__":
    main()
