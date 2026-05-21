"""
Sinh >30 câu nhập chi tiêu kiểu sinh viên (Gemini) + chạy NLU demo.

Cần .env: gemini_API, GEMINI_MODEL (mặc định gemini-2.5-flash)

Chạy:
  python text_nlu/tools/gemini_demo_student_expenses.py
  python text_nlu/tools/gemini_demo_student_expenses.py --count 40 --nlu
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
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
from src.nlu.pipeline import run_nlu

OUT_JSON = Path(__file__).resolve().parent / "demo_student_expenses.json"
OUT_TXT = Path(__file__).resolve().parent / "demo_student_expenses.txt"

SYSTEM = (
    "Bạn là sinh viên Việt Nam đang nhập chi tiêu vào app. "
    "Chỉ trả về danh sách câu, mỗi dòng một câu, không đánh số, không giải thích."
)


def _get_gemini():
    from pipeline.llm_module import call_gemini

    api_key = os.environ.get("gemini_API") or os.environ.get("GEMINI_API")
    if not api_key:
        raise RuntimeError("Thiếu gemini_API trong .env")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    return call_gemini, api_key, model


def _extract_text(resp: dict) -> str:
    for c in resp.get("candidates") or []:
        for p in (c.get("content") or {}).get("parts") or []:
            if p.get("text"):
                return str(p["text"])
    return ""


_LOCAL_TEMPLATES = [
    "ăn phở sáng {a}",
    "trà sữa full topping {a}",
    "grab đi học {a}",
    "đổ xăng {a}",
    "mua trên shopee {a}",
    "tiktok shop áo {a}",
    "tiền điện KTX {a}",
    "wifi tháng {a}",
    "photocopy tài liệu {a}",
    "netflix tháng {a}",
    "cf sáng {a}",
    "mì gói siêu thị {a}",
    "ăn vặt căng tin {a}",
    "xe ôm về KTX {a}",
    "in poster {a}",
]
_LOCAL_AMOUNTS = ["15k", "22k", "35k", "48k", "55k", "68k", "88k", "120k", "150k", "199k", "250k", "1.2tr"]


def _parse_lines(raw: str) -> list[str]:
    lines: list[str] = []
    for line in raw.splitlines():
        line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
        line = line.strip("-•* ")
        if len(line) >= 4 and re.search(r"\d", line):
            lines.append(line)
    return lines


def generate_sentences_local(count: int) -> list[str]:
    """Không cần API — mẫu GenZ cố định + ngẫu nhiên."""
    random.seed()
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < count:
        tpl = random.choice(_LOCAL_TEMPLATES)
        text = tpl.format(a=random.choice(_LOCAL_AMOUNTS))
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def generate_sentences(count: int, *, batch_size: int = 20, use_local: bool = False) -> list[str]:
    if use_local:
        return generate_sentences_local(count)

    call_fn, api_key, model = _get_gemini()
    all_lines: list[str] = []
    seen: set[str] = set()
    themes_pool = [
        "ăn uống căng tin, trà sữa, cf, ăn vặt",
        "grab, xe ôm, xăng, bus",
        "shopee, tiktok shop, đồ điện tử nhỏ",
        "tiền điện, wifi KTX, nước",
        "học phí, photocopy, in tài liệu",
        "giải trí: phim, game, spotify",
        "tiệm tạp: mì, gạo, đồ vệ sinh",
    ]

    while len(all_lines) < count:
        need = min(batch_size, count - len(all_lines))
        themes = random.sample(themes_pool, k=min(4, len(themes_pool)))
        user = (
            f"Viết đúng {need} câu TIẾNG VIỆT ngắn (1–12 từ + số tiền: k, tr, xị, vnd). "
            f"GenZ/sinh viên, chủ đề: {', '.join(themes)}. "
            "Chỉ câu GHI CHI TIÊU (mua/chi/order/grab/ăn…), KHÔNG hỏi báo cáo, KHÔNG chitchat. "
            "Mỗi dòng 1 câu."
        )
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.95, "maxOutputTokens": 2048},
        }
        try:
            resp = call_fn(api_key, model, payload)
            for line in _parse_lines(_extract_text(resp)):
                if line not in seen:
                    seen.add(line)
                    all_lines.append(line)
                    if len(all_lines) >= count:
                        break
        except Exception as exc:
            print(f"Gemini batch failed: {exc}", file=sys.stderr)
            if all_lines:
                break
            print("Falling back to local templates (no API).", file=sys.stderr)
            return generate_sentences_local(count)
        if len(all_lines) < count:
            time.sleep(float(os.environ.get("GEMINI_BATCH_SLEEP", "1.5")))

    if len(all_lines) < count:
        for line in generate_sentences_local(count - len(all_lines)):
            if line not in seen:
                all_lines.append(line)
    return all_lines[:count]


def run_nlu_on(lines: list[str]) -> list[dict]:
    load_env_file(settings.ENV_PATH)
    intent_m = load_intent_model()
    cat_m = load_category_model()
    act_m = load_action_type_model()
    rec_m = load_record_type_model()
    sent_m = load_chitchat_sentiment_model()
    out: list[dict] = []
    for text in lines:
        r = run_nlu(text, intent_m, cat_m, act_m, rec_m, sent_m, None)
        out.append(
            {
                "text": text,
                "intent": r.get("intent"),
                "category": r.get("category"),
                "amount": r.get("amount_spent"),
                "record_type": r.get("record_type"),
            }
        )
    return out


def main() -> int:
    load_env_file(settings.ENV_PATH)
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=35)
    ap.add_argument("--nlu", action="store_true", help="Chạy pipeline NLU trên từng câu")
    ap.add_argument("--local", action="store_true", help="Không gọi Gemini (mẫu local)")
    ap.add_argument("--batch-size", type=int, default=20, help="Số câu / request Gemini")
    args = ap.parse_args()

    src = "local" if args.local else "gemini"
    print(f"Generating {args.count} student expense lines via {src}...")
    lines = generate_sentences(args.count, batch_size=args.batch_size, use_local=args.local)
    if len(lines) < args.count:
        print(f"Warning: only got {len(lines)} lines from API")

    payload: dict = {"source": src, "count": len(lines), "lines": lines}
    if args.nlu:
        print("Running NLU...")
        payload["nlu"] = run_nlu_on(lines)
        ok = sum(1 for x in payload["nlu"] if x.get("intent") == "Record")
        print(f"Record intent: {ok}/{len(lines)}")

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_TXT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
