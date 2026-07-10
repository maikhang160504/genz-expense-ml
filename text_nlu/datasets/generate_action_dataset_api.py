"""
Tạo lại intent_action.csv (~20k mẫu) bằng Gemini/OpenAI API — nhãn slot đầy đủ, không rule.

Yêu cầu .env (expense-ocr-nlu/.env hoặc app/ai-service/.env):
  gemini_API_v1=...
  GEMINI_MODEL=gemini-2.5-flash

Chạy:
  python text_nlu/datasets/generate_action_dataset_api.py
  python text_nlu/datasets/generate_action_dataset_api.py --total 500 --dry-run
  python text_nlu/datasets/generate_action_dataset_api.py --resume

Sau khi xong:
  python text_nlu/train/train_action_slots.py
  python text_nlu/train/retrain_all.py   # hoặc Kaggle retrain
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "text_nlu"))
sys.path.insert(0, str(ROOT))

from action_slot_columns import (  # noqa: E402
    ACTION_QUOTAS_20K,
    ACTION_TYPES,
    ALL_COLUMNS,
    CATEGORY_CODES,
    SLOT_COLUMNS,
    SLOTS_BY_ACTION,
    THEMES,
    VERBAL_STYLES,
    VERB_LABELS,
)
from src.config.env import load_env_file  # noqa: E402

OUTPUT_CSV = ROOT / "intent_action.csv"
CHECKPOINT = ROOT / "intent_action_api_checkpoint.jsonl"
NER_JSONL = ROOT / "ner_dataset.jsonl"

SYSTEM_PROMPT = """Bạn là chuyên gia gán nhãn NLU cho app quản lý chi tiêu Mimo (tiếng Việt).
Sinh câu chat tự nhiên (gen Z, teencode, có/không dấu) và gán nhãn CHÍNH XÁC theo action_type.
Mỗi câu phải khớp action_type và điền đủ slot bắt buộc; slot không dùng để null/empty string.
Không copy ví dụ — tạo câu mới đa dạng."""

FEW_SHOT = [
    {
        "text": "thêm 1tr vào giới hạn di chuyển",
        "action_type": "SET_LIMIT",
        "verb": "ADD",
        "category_code": "Transport",
        "value": 1000000,
    },
    {
        "text": "bù 2tr vào mục tiêu mua nhà",
        "action_type": "ADD_GOAL",
        "verb": "ADD",
        "goal_name": "mua nhà",
        "value": 2000000,
    },
]


def _load_env() -> None:
    repo = PROJECT.parent
    for p in (
        PROJECT / ".env",
        repo / "app" / "ai-service" / ".env",
    ):
        if p.is_file():
            load_env_file(p)
            print(f"Loaded env: {p}")
            return
    print("Warning: no .env found — set gemini_API_v1 in environment")


def _build_user_prompt(action_type: str, n: int) -> str:
    slots = SLOTS_BY_ACTION.get(action_type, [])
    slot_doc = ", ".join(slots) if slots else "(không có slot)"
    cats = ", ".join(sorted(CATEGORY_CODES))
    return f"""action_type bắt buộc: {action_type}
Slot bắt buộc cho loại này: {slot_doc}

Quy tắc slot:
- verb: chỉ SET | ADD | SUB (SET_LIMIT, ADD_GOAL)
- category_code: một trong [{cats}] hoặc để trống nếu không áp dụng
- value: số nguyên VND (50000, 1000000), không chữ "k"/"tr" trong JSON
- goal_name: tên mục tiêu tiết kiệm (chuỗi tiếng Việt)
- enabled: "true" hoặc "false" (SET_ALERT)
- theme: dark | light (SYSTEM_SETTING)
- verbal_style: funny | gentle | serious | sarcastic | strict (SET_TONE)
- time_range: cụm thời gian tiếng Việt (REPORT_GENERAL, SUGGEST_BUDGET)
- query: từ khóa tìm kiếm (SEARCH_RECORD)
- note: ghi chú mới (UPDATE_RECORD)

Sinh đúng {n} mẫu JSON trong mảng "samples".
Mỗi phần tử:
{{"text","action_type","verb","category_code","value","goal_name","enabled","theme","verbal_style","time_range","query","note","ner"}}

"ner" (tuỳ chọn, có thể bỏ qua): mảng [start, end, label]

Ví dụ tham khảo (không copy):
{json.dumps(FEW_SHOT[:1], ensure_ascii=False)}"""


def _call_llm(prompt: str) -> list[dict]:
    from src.llm.gemini_keys import call_gemini_with_key_fallback

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    payload = {
        "systemInstruction": SYSTEM_PROMPT,
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.9,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "object",
                "properties": {
                    "samples": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "action_type": {"type": "string"},
                                "verb": {"type": "string"},
                                "category_code": {"type": "string"},
                                "value": {"type": "number"},
                                "goal_name": {"type": "string"},
                                "enabled": {"type": "string"},
                                "theme": {"type": "string"},
                                "verbal_style": {"type": "string"},
                                "time_range": {"type": "string"},
                                "query": {"type": "string"},
                                "note": {"type": "string"},
                            },
                            "required": ["text", "action_type"],
                        },
                    }
                },
                "required": ["samples"],
            },
        },
    }
    raw = call_gemini_with_key_fallback(model, payload)
    text = ""
    for c in raw.get("candidates") or []:
        parts = (c.get("content") or {}).get("parts") or []
        for p in parts:
            if p.get("text"):
                text += p["text"]
    if not text.strip():
        raise RuntimeError("Empty LLM response")
    data = json.loads(text)
    return data.get("samples") or []


def _empty_row() -> dict:
    return {c: "" for c in ALL_COLUMNS}


def _normalize_row(raw: dict, expected_action: str) -> dict | None:
    text = str(raw.get("text") or "").strip()
    if len(text) < 4 or len(text) > 200:
        return None
    at = str(raw.get("action_type") or expected_action).strip().upper()
    if at != expected_action:
        at = expected_action

    row = _empty_row()
    row["text"] = text
    row["intent"] = "Action"
    row["action_type"] = at

    allowed = set(SLOTS_BY_ACTION.get(at, []))
    for col in SLOT_COLUMNS:
        val = raw.get(col)
        if val is None or val == "" or (isinstance(val, float) and pd.isna(val)):
            continue
        if col not in allowed:
            continue
        if col == "verb" and str(val).upper() not in VERB_LABELS:
            continue
        if col == "category_code" and str(val) not in CATEGORY_CODES:
            continue
        if col == "theme" and str(val).lower() not in THEMES:
            continue
        if col == "verbal_style" and str(val).lower() not in VERBAL_STYLES:
            continue
        if col == "value":
            try:
                row[col] = str(int(float(val)))
            except (TypeError, ValueError):
                row[col] = str(val).strip()
        else:
            row[col] = str(val).strip()

    # Bắt buộc có slot chính
    for req in allowed:
        if req == "value":
            if not row.get("value"):
                return None
        elif req in ("verb", "goal_name", "enabled", "theme", "verbal_style", "time_range", "query"):
            if not str(row.get(req) or "").strip():
                return None
        elif req == "category_code" and not str(row.get("category_code") or "").strip():
            # category có thể optional cho SET_LIMIT tổng — cho phép trống
            pass

    row["_ner"] = raw.get("ner")
    return row


def _load_checkpoint() -> tuple[set[str], list[dict]]:
    seen: set[str] = set()
    rows: list[dict] = []
    if not CHECKPOINT.is_file():
        return seen, rows
    with CHECKPOINT.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = row.get("text", "").strip()
            if t and t not in seen:
                seen.add(t)
                rows.append(row)
    return seen, rows


def _append_checkpoint(row: dict) -> None:
    out = {k: v for k, v in row.items() if not k.startswith("_")}
    with CHECKPOINT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")


def _write_csv(rows: list[dict], path: Path) -> None:
    clean_rows = [{c: row.get(c, "") for c in ALL_COLUMNS} for row in rows]
    df = pd.DataFrame(clean_rows, columns=ALL_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _merge_ner(rows: list[dict]) -> int:
    existing: dict[str, dict] = {}
    if NER_JSONL.is_file():
        with NER_JSONL.open(encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                existing[o["text"].strip()] = o
    added = 0
    for row in rows:
        ner = row.pop("_ner", None)
        if not ner or not isinstance(ner, list):
            continue
        text = row["text"]
        if text in existing and existing[text].get("label"):
            continue
        existing[text] = {"text": text, "label": ner}
        added += 1
    with NER_JSONL.open("w", encoding="utf-8") as f:
        for o in existing.values():
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    return added


def _scale_quotas(total: int) -> dict[str, int]:
    if total <= len(ACTION_TYPES):
        return {k: (1 if i < total else 0) for i, k in enumerate(ACTION_TYPES)}

    base_sum = sum(ACTION_QUOTAS_20K.values())
    out: dict[str, int] = {}
    allocated = 0
    keys = list(ACTION_QUOTAS_20K.keys())
    for k in keys[:-1]:
        n = max(1, round(ACTION_QUOTAS_20K[k] * total / base_sum))
        out[k] = n
        allocated += n
    out[keys[-1]] = max(0, total - allocated)
    return out


STOP_FILE = ROOT / "GENERATE_API_STOPPED"


def main() -> None:
    if STOP_FILE.is_file():
        print(f"Stopped: {STOP_FILE} exists. Remove it to resume API generation.")
        print("Use merge_intent_action_sources.py to merge existing CSVs instead.")
        raise SystemExit(0)

    parser = argparse.ArgumentParser(description="Generate intent_action.csv via LLM API")
    parser.add_argument("--total", type=int, default=20000, help="Tổng số mẫu mục tiêu")
    parser.add_argument("--batch-size", type=int, default=40, help="Mẫu mỗi lần gọi API")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in quota, không gọi API")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục từ checkpoint")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Xóa checkpoint cũ và bắt đầu lại (không ghi đè CSV cho đến khi xong)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Đường dẫn CSV (mặc định: intent_action_generated.csv; chỉ ghi intent_action.csv khi đủ --total)",
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="Giây nghỉ giữa các batch")
    args = parser.parse_args()

    quotas = _scale_quotas(args.total)
    print("Quotas:", quotas, "sum=", sum(quotas.values()))

    if args.dry_run:
        return

    out_path = args.output or (ROOT / "intent_action_generated.csv")
    if args.fresh and CHECKPOINT.is_file():
        CHECKPOINT.unlink()
        print(f"Cleared checkpoint: {CHECKPOINT}")

    _load_env()
    seen, all_rows = _load_checkpoint() if args.resume or args.fresh else (set(), [])
    counts = {k: sum(1 for r in all_rows if r.get("action_type") == k) for k in ACTION_TYPES}
    consecutive_errors = 0
    max_consecutive_errors = 8

    for action_type, target in quotas.items():
        while counts.get(action_type, 0) < target:
            need = min(args.batch_size, target - counts.get(action_type, 0))
            print(f"\n[{action_type}] {counts.get(action_type, 0)}/{target} — requesting {need}...")
            try:
                batch = _call_llm(_build_user_prompt(action_type, need))
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                print(f"API error ({consecutive_errors}/{max_consecutive_errors}): {exc}")
                if consecutive_errors >= max_consecutive_errors:
                    print("Too many consecutive API errors — saving checkpoint and exiting.")
                    break
                time.sleep(args.sleep * 3)
                continue

            added = 0
            for raw in batch:
                row = _normalize_row(raw, action_type)
                if not row:
                    continue
                t = row["text"]
                if t in seen:
                    continue
                seen.add(t)
                all_rows.append(row)
                _append_checkpoint(row)
                counts[action_type] = counts.get(action_type, 0) + 1
                added += 1
            print(f"  accepted {added}/{len(batch)}")
            if all_rows:
                _write_csv(all_rows, out_path)
            time.sleep(args.sleep)

        if consecutive_errors >= max_consecutive_errors:
            break

    # Ghi CSV + NER
    ner_added = _merge_ner(all_rows)

    _write_csv(all_rows, out_path)
    print(f"\nSaved {len(all_rows)} rows → {out_path}")

    if len(all_rows) >= args.total and out_path != OUTPUT_CSV:
        _write_csv(all_rows, OUTPUT_CSV)
        print(f"Promoted → {OUTPUT_CSV}")

    print(f"NER entries updated: {ner_added}")
    print("Next: python text_nlu/datasets/label_action_slots.py")
    print("      python text_nlu/train/train_action_slots.py")


if __name__ == "__main__":
    main()
