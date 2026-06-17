"""
Rút gọn record_to_Need_Fix.csv: rule-based trước, Gemini cho câu còn dài.

Chạy:
  python text_nlu/datasets/clean_need_fix.py              # rules + Gemini (câu >45 ký tự)
  python text_nlu/datasets/clean_need_fix.py --rules-only # chỉ rule, không API
  python text_nlu/datasets/clean_need_fix.py --min-len 50
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(__file__).resolve().parent
CSV_PATH = DATASETS / "record_to_Need_Fix.csv"
BACKUP_PATH = DATASETS / "record_to_Need_Fix.raw.bak.csv"
RECORD_CSV = DATASETS / "intent_record.csv"
STATE_PATH = DATASETS / "clean_need_fix_state.json"

# --- Rule-based patterns ---
PREFIX_RE = re.compile(
    r"^(?:"
    r"nay\s+[^,]{0,40}?(?:có vụ|co vu)\s*|"
    r"story time\s*\d+\s*:\s*|"
    r"case văn phòng\s*:\s*|case van phong\s*:\s*|"
    r"english boi\s*:\s*|"
    r"teencode mode on\s*:\s*|"
    r"an sang khong dau\s*:\s*|"
    r"lương thưởng style genz\s*:\s*|luong thuong style genz\s*:\s*|"
    r"boi tien roi ne\s*:\s*|"
    r"đòi được tiền\s*:\s*|doi duoc tien\s*:\s*|"
    r"cash-in from\s*|"
    r"gia đình ở\s*|gia dinh o\s*|"
    r"nhận tiền từ\s+|nhan tien tu\s+"
    r")",
    re.I | re.UNICODE,
)

LOCATION_RE = re.compile(
    r"(?:"
    r"\s*(?:ở|o|tai|tại|in|at)\s+"
    r"(?:food court(?: trung tâm thương mại)?|pantry văn phòng|pantry van phong|"
    r"mall lunch zone|coworking|khu trọ|khu tro|sảnh tòa nhà|sanh toa nha|"
    r"chung cư|chung cu|nhà ngoại|nha ngoai|nhà nội|nha noi|quán cóc|quan coc|"
    r"food court trung tâm thương mại|food court trung tam thuong mai)"
    r"|\s+food court trung tâm thương mại"
    r")",
    re.I | re.UNICODE,
)

SUFFIX_RE = re.compile(
    r"(?:"
    r",?\s*(?:về ví|ve vi|về tk|ve tk)\s*(?:qua|bang|bằng)?\s*"
    r"(?:scan qr|quẹt thẻ|quet the|tiền mặt|tien mat|chuyển khoản|chuyen khoan|banking app)[^,]*"
    r"|,?\s*(?:thanh toán|thanh toan|xử lý|xu ly|tra|trả)\s*"
    r"(?:bằng|bang|qua)?\s*(?:banking app|scan qr|quẹt thẻ|quet the|tiền mặt|tien mat|chuyển khoản)[^,]*"
    r"|,?\s*(?:order trên|order tren|mua trên|mua tren)\s*"
    r"(?:Shopee|Lazada|Tiki|Grab|Netflix)[^,]*"
    r"|,?\s*hóa đơn định kỳ[^,]*"
    r"|,?\s*(?:okela|xỉu up xỉu down|xiu up xiu down|vibe ổn|chill|căng phết|flex nhẹ)[^,]*"
    r")+$",
    re.I | re.UNICODE,
)

MONEY_INLINE = re.compile(
    r"(\+?\d+(?:[\.,]\d+)?\s?(?:k|K|tr|triệu|trieu|củ|cu|ngàn|ngan|nghìn|nghin|vnđ|vnd|đ))",
    re.I,
)

NOISE_WORDS = re.compile(
    r"\b(?:tổng|tong|mất|mat|hết|het|chốt|chot|pay|combo|làm|lam|xong)\b",
    re.I,
)


def load_env() -> None:
    for env_path in (ROOT / ".env", ROOT.parent / ".env"):
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def get_gemini_client():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "text_nlu"))
    from pipeline.llm_module import call_gemini

    keys = [
        os.environ.get("gemini_API_v1"),
        os.environ.get("gemini_API_v2"),
        os.environ.get("gemini_API"),
        os.environ.get("GEMINI_API_KEY"),
    ]
    api_key = next((k for k in keys if k), None)
    if not api_key:
        raise RuntimeError("Thiếu gemini_API_v1/v2 trong .env")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    return call_gemini, api_key, model


def _extract_money(text: str) -> str | None:
    m = MONEY_INLINE.search(text)
    return m.group(1).strip() if m else None


def rule_based_shorten(text: str, label: str, typ: str) -> str:
    t = str(text).strip()
    if len(t) <= 40:
        return t

    # "Nay có khoản về từ X, Y chuyển khoản cái rẹt" -> "X Y"
    m_nay = re.match(
        r"^nay\s+có\s+khoản\s+về\s+từ\s+(.+?),\s*(.+?)\s+chuyển\s+khoản.*$",
        t,
        re.I | re.UNICODE,
    )
    if m_nay:
        core, money = m_nay.group(1).strip(), m_nay.group(2).strip()
        money = re.sub(r"\s*(?:đồng|vnđ|vnd)\s*$", "", money, flags=re.I)
        return f"{core} {money}".strip()[:72]

    money = _extract_money(t)
    low = t.lower()

    if typ == "income" or "nhan tien" in low or "nhận tiền" in low:
        core = PREFIX_RE.sub("", t, count=1)
        core = LOCATION_RE.sub("", core)
        core = SUFFIX_RE.sub("", core)
        core = re.sub(r"^[^:]*:\s*", "", core)
        core = re.sub(r"^\+\s*", "", core.strip())
        core = NOISE_WORDS.sub(" ", core)
        core = re.sub(r"\s+", " ", core).strip(" ,.")
        if money and money not in core:
            # Keep product/service noun chunk + money
            chunk = re.sub(r"[\+\d][\d\.,kKtrcủcu\s]*", "", core).strip()
            chunk = chunk[:40].strip() if chunk else label.lower()
            return f"{chunk} {money}".strip()[:72]
        return core[:72] if core else t

    # Expense prefixes
    t = PREFIX_RE.sub("", t, count=1)
    t = LOCATION_RE.sub("", t)
    t = SUFFIX_RE.sub("", t)

    # "mua X tổng Y" / "trả Y qua..." -> "mua X Y"
    t = re.sub(
        r"^(mua|chi|order|thanh toán|trả|đóng|đăng ký|dang ky)\s+(.+?)\s+"
        r"(?:tổng|tong|mất|mat|hết|het|chốt|chot)\s+",
        r"\1 \2 ",
        t,
        flags=re.I,
    )
    t = re.sub(r"\s+(?:bằng|bang|qua)\s+(?:tiền mặt|quet the|quẹt thẻ)[^,]*", "", t, flags=re.I)
    t = NOISE_WORDS.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip(" ,.")

    if len(t) > 50 and money:
        verb = "mua"
        for v in ("mua", "chi", "order", "thanh toán", "trả", "đóng", "đi", "ăn"):
            if low.startswith(v):
                verb = v
                break
        chunk = re.sub(r"[\d\+][\d\.,kKtrcủcu\s]*", "", t)
        chunk = re.sub(r"^(mua|chi|order|thanh toán|trả|đóng|đi|ăn)\s+", "", chunk, flags=re.I)
        chunk = chunk.strip()[:35]
        if chunk:
            t = f"{verb} {chunk} {money}".strip()

    return t[:72] if t else str(text).strip()[:72]


SYSTEM_PROMPT = """Bạn rút gọn câu ghi chi tiêu/thu nhập tiếng Việt về CỐT LÕI: hành động/sản phẩm + số tiền (1-8 từ + giá).
Loại bỏ: địa điểm, story/teencode prefix, app thanh toán, hóa đơn định kỳ, cảm xúc genz.
Input/output: index | cleaned_text | label | type | is_money (pipe-separated, không markdown)."""


LMSTUDIO_ROW_PROMPT = """Rút gọn câu ghi chi tiêu/thu nhập tiếng Việt.
Chỉ trả về MỘT câu ngắn (tối đa 8 từ + số tiền). Không giải thích.
Ví dụ: "Case văn phòng: mua đồ tech trả 35tr qua quẹt thẻ" -> "mua đồ tech 35tr"
"""


def clean_one_lmstudio(base_url: str, model: str, text: str) -> str:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "text_nlu"))
    from pipeline.llm_module import call_lmstudio, extract_chat_text

    user = f'Câu: "{text}"\nCâu rút gọn:'
    resp = call_lmstudio(
        base_url, model, LMSTUDIO_ROW_PROMPT, user,
        temperature=0.05, max_tokens=64, timeout=90.0,
    )
    out = extract_chat_text(resp).strip().strip('"').strip("'")
    # Lấy dòng đầu, bỏ reasoning
    for line in out.splitlines():
        line = line.strip().lstrip("-•*0123456789. ")
        if not line or line.lower().startswith(("câu", "output", "answer", "rút gọn")):
            continue
        if "->" in line:
            line = line.split("->")[-1].strip().strip('"')
        line = line.replace("**", "").strip()
        if len(line) >= 3:
            return line[:72]
    return out.splitlines()[0][:72] if out else text


def clean_batch_llm(
    backend: str,
    batch: list[dict],
    *,
    lmstudio_url: str = "",
    lmstudio_model: str = "",
) -> list[dict]:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "text_nlu"))
    from pipeline.llm_module import call_gemini, call_lmstudio, extract_chat_text

    lines = [
        f"{item['index']} | {item['text']} | {item['label']} | {item['type']} | {item['is_money']}"
        for item in batch
    ]
    user_prompt = "\n".join(lines)

    if backend == "lmstudio":
        out = []
        url = lmstudio_url or os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234")
        mdl = lmstudio_model or os.environ.get(
            "LMSTUDIO_MODEL",
            "gemma-3-1b-it-glm-4.7-flash-heretic-uncensored-thinking_gguf",
        )
        for item in batch:
            try:
                short = clean_one_lmstudio(url, mdl, item["text"])
                short = rule_based_shorten(short, item["label"], item["type"])
                if len(short) < len(str(item["text"])):
                    item = {**item, "text": short}
                else:
                    item = {**item, "text": rule_based_shorten(item["text"], item["label"], item["type"])}
            except Exception as exc:
                print(f"    row {item['index']} lmstudio err: {exc}", flush=True)
                item = {**item, "text": rule_based_shorten(item["text"], item["label"], item["type"])}
            out.append(item)
        return out

    # Gemini batch
    if backend == "gemini":
        call_fn, api_key, model = get_gemini_client()
        payload = {
            "systemInstruction": SYSTEM_PROMPT,
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.15, "maxOutputTokens": 8192},
        }
        resp = call_fn(api_key, model, payload)
        raw_text = extract_chat_text(resp)
        results: dict[int, str] = {}
        for line in raw_text.splitlines():
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 5:
                continue
            try:
                results[int(parts[0])] = parts[1].strip('"').strip("'").strip()
            except ValueError:
                continue
        out = []
        for item in batch:
            text = results.get(item["index"], item["text"])
            out.append({**item, "text": text.strip()[:72]})
        return out

    raise ValueError(f"Unknown backend: {backend}")


def clean_batch_gemini(call_fn, api_key: str, model: str, batch: list[dict]) -> list[dict]:
    lines = [
        f"{item['index']} | {item['text']} | {item['label']} | {item['type']} | {item['is_money']}"
        for item in batch
    ]
    payload = {
        "systemInstruction": SYSTEM_PROMPT,
        "contents": [{"role": "user", "parts": [{"text": "\n".join(lines)}]}],
        "generationConfig": {"temperature": 0.15, "maxOutputTokens": 8192},
    }
    resp = call_fn(api_key, model, payload)
    raw_text = ""
    for c in resp.get("candidates") or []:
        for p in (c.get("content") or {}).get("parts") or []:
            if p.get("text"):
                raw_text = str(p["text"])
                break
    results: dict[int, str] = {}
    for line in raw_text.splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            continue
        try:
            results[int(parts[0])] = parts[1].strip('"').strip("'").strip()
        except ValueError:
            continue
    out = []
    for item in batch:
        text = results.get(item["index"], item["text"])
        out.append({**item, "text": text.strip()[:72]})
    return out


def sync_cleaned_to_intent_record() -> tuple[int, int]:
    """Thay câu dài (backup) trong intent_record bằng bản đã rút gọn."""
    if not BACKUP_PATH.is_file() or not CSV_PATH.is_file() or not RECORD_CSV.is_file():
        print("Skip sync: missing backup, cleaned, or intent_record")
        return 0, 0
    import pandas as pd

    bak = pd.read_csv(BACKUP_PATH, header=None, names=["text", "label", "type", "is_money"], encoding="utf-8-sig")
    clean = pd.read_csv(CSV_PATH, header=None, names=["text", "label", "type", "is_money"], encoding="utf-8-sig")
    rec = pd.read_csv(RECORD_CSV, encoding="utf-8-sig")

    replace_map: dict[str, str] = {}
    for i in range(min(len(bak), len(clean))):
        old_t = str(bak.iloc[i]["text"]).strip()
        new_t = str(clean.iloc[i]["text"]).strip()
        if old_t and old_t != new_t:
            replace_map[old_t] = new_t

    replaced = 0
    texts = rec["text"].astype(str).str.strip()
    for old_t, new_t in replace_map.items():
        mask = texts == old_t
        if mask.any():
            rec.loc[mask, "text"] = new_t
            replaced += int(mask.sum())

    existing = set(rec["text"].astype(str).str.strip())
    new_rows = []
    for _, r in clean.iterrows():
        t = str(r["text"]).strip()
        if t and t not in existing:
            existing.add(t)
            new_rows.append({"text": t, "label": r["label"], "type": r["type"], "is_money": int(r["is_money"])})
    if new_rows:
        rec = pd.concat([rec, pd.DataFrame(new_rows)], ignore_index=True)

    rec.to_csv(RECORD_CSV, index=False, encoding="utf-8-sig")
    print(f"Sync intent_record: replaced {replaced}, appended {len(new_rows)}")
    return replaced, len(new_rows)


def read_rows() -> list[dict]:
    rows = []
    with CSV_PATH.open("r", encoding="utf-8-sig") as f:
        for idx, row in enumerate(csv.reader(f)):
            if len(row) < 4:
                continue
            rows.append({
                "index": idx,
                "text": row[0],
                "label": row[1],
                "type": row[2],
                "is_money": row[3],
            })
    return rows


def save_rows(rows: list[dict]) -> None:
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([r["text"], r["label"], r["type"], r["is_money"]])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["gemini", "lmstudio", "rules"], default="gemini")
    parser.add_argument("--lmstudio-url", default=os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234"))
    parser.add_argument("--lmstudio-model", default=os.environ.get(
        "LMSTUDIO_MODEL",
        "gemma-3-1b-it-glm-4.7-flash-heretic-uncensored-thinking_gguf",
    ))
    parser.add_argument("--rules-only", action="store_true", help="Chỉ dùng rule-based, không gọi LLM")
    parser.add_argument("--min-len", type=int, default=45, help="Chỉ gọi LLM nếu câu dài hơn N ký tự sau rule")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--sync-record", action="store_true", help="Đồng bộ bản clean vào intent_record.csv")
    args = parser.parse_args()

    if not CSV_PATH.exists():
        print(f"File not found: {CSV_PATH}")
        return 1

    load_env()
    rows = read_rows()
    if not BACKUP_PATH.exists():
        BACKUP_PATH.write_text(CSV_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
        print(f"Backup -> {BACKUP_PATH}")

    # Pass 1: rule-based
    for r in rows:
        r["text"] = rule_based_shorten(r["text"], r["label"], r["type"])

    long_after_rule = [r for r in rows if len(str(r["text"])) > args.min_len]
    print(f"After rules: {len(rows)} rows, {len(long_after_rule)} still >{args.min_len} chars", flush=True)

    if not args.rules_only and args.backend != "rules" and long_after_rule:
        bs = 1 if args.backend == "lmstudio" else min(args.batch_size, 40)
        try:
            for i in range(0, len(long_after_rule), bs):
                batch = long_after_rule[i : i + bs]
                print(f"{args.backend} batch {i // bs + 1}/{(len(long_after_rule) + bs - 1) // bs}...", flush=True)
                try:
                    cleaned = clean_batch_llm(
                        args.backend,
                        batch,
                        lmstudio_url=args.lmstudio_url,
                        lmstudio_model=args.lmstudio_model,
                    )
                    for c in cleaned:
                        for r in rows:
                            if r["index"] == c["index"]:
                                r["text"] = c["text"]
                                break
                except Exception as exc:
                    print(f"  LLM error: {exc}. Rule fallback for batch.")
                    for item in batch:
                        for r in rows:
                            if r["index"] == item["index"]:
                                r["text"] = rule_based_shorten(r["text"], r["label"], r["type"])
                time.sleep(0.5 if args.backend == "lmstudio" else 2)
        except RuntimeError as exc:
            print(f"LLM skipped: {exc}")

    # Final pass: enforce max length
    for r in rows:
        if len(str(r["text"])) > 72:
            r["text"] = rule_based_shorten(r["text"], r["label"], r["type"])

    save_rows(rows)
    lens = [len(str(r["text"])) for r in rows]
    over60 = sum(1 for L in lens if L > 60)
    print(f"Saved {len(rows)} rows. mean len={sum(lens)/len(lens):.1f}, >60 chars: {over60}")

    if args.sync_record:
        sync_cleaned_to_intent_record()
    return 0


if __name__ == "__main__":
    sys.exit(main())
