"""
Dùng Gemini 2.5 Flash (.env: gemini_API, GEMINI_MODEL) để:
  1) audit — 1 request: thống kê + mẫu ngắn → gợi ý lỗ hổng (tiết kiệm token)
  2) generate — nhiều batch nhỏ, output CSV tối giản, có checkpoint

Mục tiêu: câu Record GenZ / sinh viên (ăn uống, grab, ký túc xá, học phí…).

Chạy:
  python text_nlu/datasets/gemini_augment_record.py audit
  python text_nlu/datasets/gemini_augment_record.py generate --batches 12 --rows 80
  python text_nlu/datasets/gemini_augment_record.py all --batches 12 --rows 80

Sau generate: python text_nlu/datasets/improve_datasets.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(__file__).resolve().parent
CSV_PATH = DATASETS / "intent_record.csv"
TOP_MD = ROOT / "mô tả dữ liệu.md"
STATE_PATH = DATASETS / "gemini_record_augment_state.json"
AUDIT_PATH = DATASETS / "gemini_record_audit.json"

LABELS = [
    "Food", "Transport", "Housing", "Essentials", "Shopping", "Beauty",
    "Health", "Education", "Entertainment", "Social", "Salary", "Bonus",
    "Business", "Investment", "Debt", "Charity", "Savings", "Others",
]
TYPES = ("expense", "income")

# Batch themes — mỗi batch 1 theme, ít token hơn gửi cả dataset
GENERATION_THEMES: list[dict[str, str]] = [
    {
        "id": "genz_food_short",
        "hint": "Ăn uống SV: ts,cf,phở,cơm,ăn vặt,GrabFood; câu 1-6 từ+giá(k,xị,tr,vnd); expense",
        "labels": "Food",
        "type": "expense",
    },
    {
        "id": "genz_transport",
        "hint": "Grab/grb/xăng/xe bus; câu cực ngắn; expense Transport",
        "labels": "Transport",
        "type": "expense",
    },
    {
        "id": "dorm_essentials",
        "hint": "KTX: giặt,gạo,mì,gvs,dầu gội,wifi,điện,nước; Essentials/Housing; expense",
        "labels": "Essentials,Housing",
        "type": "expense",
    },
    {
        "id": "student_income",
        "hint": "Thu: lg,thg,lì xì,fl,ck mẹ,hoàn tiền; KHÔNG dùng mua/chi đầu câu; income",
        "labels": "Salary,Bonus,Business",
        "type": "income",
    },
    {
        "id": "subs_digital",
        "hint": "Netflix,Spotify,Cursor,4G,data; Entertainment/Others; expense",
        "labels": "Entertainment,Others",
        "type": "expense",
    },
    {
        "id": "health_edu",
        "hint": "Thuốc,khám,photocopy,sách,học phí; Health/Education; expense",
        "labels": "Health,Education",
        "type": "expense",
    },
    {
        "id": "social_fun",
        "hint": "Bida,kara,nhậu,quà sinh nhật,hẹn hò; Social/Entertainment; expense",
        "labels": "Social,Entertainment",
        "type": "expense",
    },
    {
        "id": "shopping_gadget",
        "hint": "Shopee,tiktok,sạc,áo,giày; Shopping; expense",
        "labels": "Shopping",
        "type": "expense",
    },
    {
        "id": "debt_charity",
        "hint": "Trả nợ,trả góp,từ thiện,donate; Debt/Charity; expense",
        "labels": "Debt,Charity",
        "type": "expense",
    },
    {
        "id": "slang_typos",
        "hint": "Không dấu,viết tắt,genz; vẫn đúng type income/expense; mixed labels",
        "labels": "Food,Transport,Essentials,Business,Salary",
        "type": "mixed",
    },
    {
        "id": "invest_savings",
        "hint": "Lãi tk,cổ tức,gửi tiết kiệm; Investment/Savings; income",
        "labels": "Investment,Savings",
        "type": "income",
    },
    {
        "id": "medium_natural",
        "hint": "Câu 8-15 từ tự nhiên, có ngữ cảnh SV; phân bố label hợp lý",
        "labels": "Food,Transport,Housing,Essentials,Education",
        "type": "expense",
    },
]

# Biên mua/bán, đi cafe vs mua cafe — dùng generate_dataset_15k.py
DISAMBIGUATION_GEMINI_THEMES: list[dict[str, str]] = [
    {
        "id": "buy_sell_meat",
        "hint": "Cặp mua vs bán cùng món: mua thịt heo 188k expense Food; bán thịt heo 299k income Business",
        "labels": "Food,Business",
        "type": "mixed",
    },
    {
        "id": "cafe_go_vs_buy",
        "hint": "đi cà phê/cafe 18k Entertainment expense; mua cà phê/hạt cf Food hoặc Essentials expense",
        "labels": "Entertainment,Food,Essentials",
        "type": "expense",
    },
    {
        "id": "grocery_essentials",
        "hint": "gạo,mì gói,dầu gội,gvs — Essentials; không nhầm Shopping",
        "labels": "Essentials",
        "type": "expense",
    },
    {
        "id": "gadget_shopping",
        "hint": "mua sạc,chuột,áo — Shopping expense",
        "labels": "Shopping",
        "type": "expense",
    },
    {
        "id": "family_income_bonus",
        "hint": "mẹ cho,ck về,lì xì — income Bonus/Salary; KHÔNG expense",
        "labels": "Bonus,Salary",
        "type": "income",
    },
    {
        "id": "sell_income_variants",
        "hint": "bán đồ cũ,bán hàng online,thu tiền bán — income Business",
        "labels": "Business",
        "type": "income",
    },
    {
        "id": "spa_beauty_go",
        "hint": "đi spa,làm nails,cắt tóc — Beauty hoặc Entertainment expense",
        "labels": "Beauty,Entertainment",
        "type": "expense",
    },
    {
        "id": "supermarket_brands",
        "hint": "Coopmart,BHX,Big C + món từ topics; label đúng Food/Essentials",
        "labels": "Food,Essentials",
        "type": "expense",
    },
]

SYSTEM_PROMPT = """Bạn tạo dữ liệu huấn luyện NLU ghi chi tiêu tiếng Việt (sinh viên GenZ).
Chỉ trả về các dòng CSV mini, KHÔNG markdown, KHÔNG giải thích.
Định dạng MỖI DÒNG (dùng | làm separator, text không chứa ký tự |):
text|label|type|is_money

label ∈ {labels}
type ∈ expense,income
is_money ∈ 0,1

Quy tắc:
- expense: chi tiêu (mua,ăn,grab,trả,order…)
- income: thu nhập — câu phải thể hiện NHẬN tiền (lương,thưởng,ck về,hoàn,lãi,mẹ gửi…); KHÔNG bắt đầu bằng mua/chi/order/thanh toán
- Câu ngắn 1-8 từ hoặc vừa; giá đa dạng: k,K,xị,tr,củ,ngàn,vnd,đ
- Không trùng nghĩa các dòng trong batch
""".format(labels=",".join(LABELS))


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
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
    from pipeline.llm_module import call_gemini  # noqa: E402

    api_key = os.environ.get("gemini_API") or os.environ.get("GEMINI_API")
    if not api_key:
        raise RuntimeError("Thiếu gemini_API trong .env")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    return call_gemini, api_key, model


def extract_text_from_gemini(resp: dict) -> str:
    """Lấy text thuần từ response Gemini."""
    if not resp:
        return ""
    candidates = resp.get("candidates") or []
    for c in candidates:
        content = c.get("content") or {}
        for p in content.get("parts") or []:
            t = p.get("text")
            if t:
                return str(t)
    return str(resp.get("text") or "")


def call_gemini_text(call_fn, api_key: str, model: str, user: str, max_tokens: int = 8192) -> str:
    payload = {
        "systemInstruction": SYSTEM_PROMPT,
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": 0.85,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "text/plain",
        },
    }
    resp = call_fn(api_key, model, payload)
    return extract_text_from_gemini(resp)


def load_topics_compact(max_lines: int = 50) -> str:
    if not TOP_MD.is_file():
        return ""
    lines = []
    for line in TOP_MD.read_text(encoding="utf-8").splitlines():
        t = line.strip().rstrip(",").strip()
        if t and "{" not in t and t != ".......":
            lines.append(t[:40])
        if len(lines) >= max_lines:
            break
    return ", ".join(lines)


def dataset_stats(df) -> dict:
    import pandas as pd

    vc = df["label"].value_counts()
    weak = vc[vc < 400].index.tolist() if len(vc) else []
    return {
        "rows": len(df),
        "expense": int((df["type"] == "expense").sum()),
        "income": int((df["type"] == "income").sum()),
        "label_top": vc.head(6).to_dict(),
        "label_weak": {k: int(vc[k]) for k in weak[:10]},
    }


def sample_lines(df, n_per_label: int = 2, max_labels: int = 8) -> dict[str, list[str]]:
    import pandas as pd

    weak = df["label"].value_counts().sort_values().head(max_labels).index
    out: dict[str, list[str]] = {}
    for lab in weak:
        sub = df[df["label"] == lab]["text"].astype(str)
        if len(sub) == 0:
            continue
        k = min(n_per_label, len(sub))
        out[str(lab)] = sub.sample(n=k, random_state=42).tolist()
    return out


def run_audit(df) -> dict:
    call_fn, api_key, model = get_gemini_client()
    stats = dataset_stats(df)
    samples = sample_lines(df, n_per_label=2, max_labels=10)
    topics = load_topics_compact(40)
    user = (
        f"AUDIT (JSON ngắn, không markdown):\n"
        f"stats={json.dumps(stats, ensure_ascii=False)}\n"
        f"samples_weak_labels={json.dumps(samples, ensure_ascii=False)}\n"
        f"topics_md={topics}\n"
        "Trả ĐÚNG 1 dòng JSON (không xuống dòng): "
        '{"gaps":["..."],"bad_patterns":["..."],"themes_next":["..."]} '
        "mỗi list tối đa 6 phần tử, string <=40 ký tự."
    )
    raw = call_gemini_text(call_fn, api_key, model, user, max_tokens=2048)
    audit = {"raw": raw, "stats": stats}
    try:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            audit["parsed"] = json.loads(m.group(0))
    except json.JSONDecodeError:
        audit["parsed"] = None
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Audit saved: {AUDIT_PATH}")
    return audit


def parse_pipe_lines(raw: str, existing: set[str]) -> list[dict]:
    rows: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^```\w*\s*", "", line).strip()
        line = re.sub(r"```\s*$", "", line).strip()
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        text, label, typ, im = parts[0], parts[1], parts[2], parts[3]
        if label not in LABELS or typ not in TYPES:
            continue
        try:
            is_money = int(im)
        except ValueError:
            continue
        if is_money not in (0, 1):
            continue
        t = re.sub(r"\s+", " ", text).strip()
        if not t or len(t) > 120 or t in existing:
            continue
        if typ == "income" and re.match(r"^(mua|chi|order|thanh toán|trả|đóng)\b", t, re.I):
            continue
        existing.add(t)
        rows.append({"text": t, "label": label, "type": typ, "is_money": is_money})
    return rows


def run_generate(
    df,
    *,
    batches: int,
    rows_per_batch: int,
    sleep_s: float,
) -> int:
    call_fn, api_key, model = get_gemini_client()
    existing = set(df["text"].astype(str).str.strip())
    state = {}
    if STATE_PATH.is_file():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    done_ids = set(state.get("done_theme_ids") or [])

    themes = GENERATION_THEMES[:]
    random.seed(42)
    random.shuffle(themes)
    added_total = 0
    fieldnames = ["text", "label", "type", "is_money"]

    with CSV_PATH.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        ran = 0
        for theme in themes:
            if ran >= batches:
                break
            tid = theme["id"]
            if tid in done_ids:
                continue
            user = (
                f"Tạo đúng {rows_per_batch} dòng pipe-format.\n"
                f"theme={theme['hint']}\n"
                f"labels_allowed={theme['labels']}\n"
                f"type_bias={theme['type']}\n"
                f"avoid_duplicates_with_existing: không cần danh sách — chỉ cần câu mới.\n"
                f"topics_gợi_ý: {load_topics_compact(25)}"
            )
            print(f"  batch {ran + 1}/{batches}: {tid} …", flush=True)
            try:
                raw = call_gemini_text(call_fn, api_key, model, user, max_tokens=8192)
            except Exception as exc:
                print(f"    API error: {exc}", file=sys.stderr)
                time.sleep(sleep_s * 2)
                continue
            new_rows = parse_pipe_lines(raw, existing)
            for r in new_rows:
                writer.writerow(r)
            added_total += len(new_rows)
            done_ids.add(tid)
            state["done_theme_ids"] = sorted(done_ids)
            state["last_added"] = len(new_rows)
            state["total_added"] = state.get("total_added", 0) + len(new_rows)
            STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"    +{len(new_rows)} rows (parsed)")
            ran += 1
            if ran < batches:
                time.sleep(sleep_s)

    print(f"Generate done: +{added_total} rows appended.")
    return added_total


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser(description="Gemini augment intent_record.csv")
    parser.add_argument("mode", choices=["audit", "generate", "all"], help="audit | generate | all")
    parser.add_argument("--batches", type=int, default=12, help="Số batch generate (mỗi batch ~1 API call)")
    parser.add_argument("--rows", type=int, default=80, help="Số dòng yêu cầu / batch")
    parser.add_argument("--sleep", type=float, default=2.0, help="Giây nghỉ giữa các request")
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    if args.mode in ("audit", "all"):
        run_audit(df)

    if args.mode in ("generate", "all"):
        run_generate(df, batches=args.batches, rows_per_batch=args.rows, sleep_s=args.sleep)

    return 0


if __name__ == "__main__":
    sys.exit(main())
