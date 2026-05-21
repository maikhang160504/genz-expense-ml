"""
Sinh ~15k dòng Record đa nhãn (mua vs bán, đi cafe vs mua cafe…) từ mô tả dữ liệu.md + Gemini.

Chạy:
  python text_nlu/datasets/generate_dataset_15k.py local --target 15000
  python text_nlu/datasets/generate_dataset_15k.py gemini --batches 80 --rows 100
  python text_nlu/datasets/generate_dataset_15k.py all --target 15000
  python text_nlu/datasets/generate_dataset_15k.py fix

Sau đó: python text_nlu/datasets/improve_datasets.py
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "text_nlu"))

CSV_PATH = DATASETS / "intent_record.csv"
TOP_MD = ROOT / "mô tả dữ liệu.md"

from expand_record_topics import money_vn  # noqa: E402
from gemini_augment_record import (  # noqa: E402
    DISAMBIGUATION_GEMINI_THEMES,
    load_topics_compact,
    run_generate,
    run_audit,
)
from fix_disambiguation_labels import main as fix_labels_main  # noqa: E402

AMOUNTS = ["15k", "18k", "22k", "35k", "48k", "55k", "68k", "88k", "99k", "120k", "150k", "188k", "199k", "250k", "299k", "350k", "420k", "500k", "680k", "1.2tr", "1.8tr"]
BUY_VERBS = ["mua", "order", "chi", "thanh toán", "trả"]
SELL_VERBS = ["bán", "ban", "thu tiền bán", "nhận tiền bán"]
GO_VERBS = ["đi", "di", "ghé", "ghe", "tới", "toi"]
GO_PLACES = [
    ("cà phê", "Entertainment"),
    ("cafe", "Entertainment"),
    ("cf", "Entertainment"),
    ("bar", "Entertainment"),
    ("pub", "Entertainment"),
    ("spa", "Entertainment"),
    ("salon", "Beauty"),
    ("massage", "Entertainment"),
    ("karaoke", "Entertainment"),
    ("bida", "Entertainment"),
    ("xem phim", "Entertainment"),
    ("nhậu", "Social"),
]
COFFEE_BUY = [
    ("mua cà phê", "Food"),
    ("mua cafe", "Food"),
    ("mua cf hạt", "Food"),
    ("mua bột cà phê", "Essentials"),
    ("mua ly cf mang về", "Food"),
]


def load_topics_from_md() -> list[str]:
    if not TOP_MD.is_file():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for line in TOP_MD.read_text(encoding="utf-8").splitlines():
        t = line.strip().rstrip(",").strip()
        if not t or "{" in t or t == ".......":
            continue
        key = t.lower()
        if key in seen or len(t) < 2:
            continue
        seen.add(key)
        out.append(t)
    return out


def label_for_topic(item: str) -> str:
    t = item.lower()
    if any(x in t for x in ("thịt", "phở", "bún", "cơm", "bánh", "trà", "sữa", "mì", "gạo", "cá", "tôm", "trứng", "chuối", "cam")):
        return "Food"
    if any(x in t for x in ("grab", "xăng", "xe", "bus")):
        return "Transport"
    if any(x in t for x in ("điện", "nước", "wifi", "internet", "4g", "data")):
        return "Housing"
    if any(x in t for x in ("shopee", "lazada", "tiktok", "tiki", "sạc", "chuột", "bàn phím", "áo", "quần", "giày", "balo", "túi")):
        return "Shopping"
    if any(x in t for x in ("son", "serum", "kem", "mỹ phẩm", "make up", "nails", "tóc", "cắt tóc")):
        return "Beauty"
    if any(x in t for x in ("thuốc", "khám")):
        return "Health"
    if any(x in t for x in ("học", "sách", "photocopy")):
        return "Education"
    if any(x in t for x in ("phim", "netflix", "spotify", "game", "du lịch", "cafe", "bar", "pub", "club", "spa", "massage", "bida", "karaoke")):
        return "Entertainment"
    if any(x in t for x in ("quà", "sinh nhật")):
        return "Social"
    if any(x in t for x in ("dầu gội", "giấy", "bột giặt", "nước rửa", "muối", "đường", "tỏi", "gas", "gạo")):
        return "Essentials"
    if any(x in t for x in ("máy giặt", "tủ", "kệ", "đèn", "giặt")):
        return "Housing"
    return "Essentials"


def generate_local_rows(target: int, seed: int = 42) -> list[dict]:
    random.seed(seed)
    topics = load_topics_from_md()
    if not topics:
        topics = ["thịt heo", "cà phê", "gạo", "trà sữa", "phở", "sạc điện thoại", "áo khoác", "vé xem phim"]
    rows: list[dict] = []
    n = 0

    def add(text: str, label: str, typ: str) -> bool:
        nonlocal n
        t = " ".join(text.split())
        if len(t) > 120 or len(t) < 3:
            return False
        rows.append({"text": t, "label": label, "type": typ, "is_money": 1})
        n += 1
        return len(rows) < target

    # 1) mua vs bán cùng sản phẩm (từ md)
    for item in topics:
        lab = label_for_topic(item)
        for i, amt in enumerate(AMOUNTS):
            if not add(f"mua {item} {amt}", lab, "expense"):
                return rows
            if not add(f"bán {item} {amt}", "Business", "income"):
                return rows
            if not add(f"{random.choice(BUY_VERBS)} {item} {money_vn(n + i)}", lab, "expense"):
                return rows
            n += 1

    # 2) đi cafe vs mua cafe
    for place, lab in GO_PLACES:
        for amt in AMOUNTS[:12]:
            if not add(f"{random.choice(GO_VERBS)} {place} {amt}", lab, "expense"):
                return rows
    for tpl, lab in COFFEE_BUY:
        for amt in AMOUNTS[:10]:
            if not add(f"{tpl} {amt}", lab, "expense"):
                return rows

    # 3) biên gạo / sạc / quà
    edge = [
        ("gạo 50k", "Essentials", "expense"),
        ("mua gạo 50k", "Essentials", "expense"),
        ("mua sạc 120k", "Shopping", "expense"),
        ("mua quà cho mẹ 200k", "Essentials", "expense"),
        ("me cho 1tr", "Bonus", "income"),
        ("ck về 500k", "Salary", "income"),
    ]
    for text, lab, typ in edge:
        for _ in range(80):
            if not add(text, lab, typ):
                return rows

    # 4) fill đến target — biến thể ngắn
    while len(rows) < target:
        item = random.choice(topics)
        lab = label_for_topic(item)
        amt = random.choice(AMOUNTS)
        kind = random.randint(0, 4)
        if kind == 0:
            text = f"mua {item} {amt}"
            typ, rlab = "expense", lab
        elif kind == 1:
            text = f"bán {item} {amt}"
            typ, rlab = "income", "Business"
        elif kind == 2:
            text = f"đi {item} hết {amt}"
            typ, rlab = "expense", "Entertainment" if "cafe" in item.lower() or "phim" in item.lower() else lab
        else:
            text = f"{item} {amt}"
            typ, rlab = "expense", lab
        if not add(text, rlab, typ):
            break

    return rows[:target]


def append_rows(new_rows: list[dict]) -> int:
    if not new_rows:
        return 0
    existing: set[str] = set()
    if CSV_PATH.is_file():
        import pandas as pd

        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        existing = set(df["text"].astype(str).str.strip())
    to_write = [r for r in new_rows if r["text"] not in existing]
    if not to_write:
        return 0
    write_header = not CSV_PATH.is_file() or CSV_PATH.stat().st_size == 0
    with CSV_PATH.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "type", "is_money"])
        if write_header:
            w.writeheader()
        w.writerows(to_write)
    return len(to_write)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["local", "gemini", "fix", "all", "audit"])
    parser.add_argument("--target", type=int, default=15000)
    parser.add_argument("--batches", type=int, default=80)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=2.0)
    args = parser.parse_args()

    if args.mode == "fix":
        return fix_labels_main()

    import pandas as pd

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    print(f"Record rows before: {len(df)}")

    if args.mode in ("local", "all"):
        local = generate_local_rows(args.target)
        n = append_rows(local)
        print(f"Local: generated {len(local)}, appended {n} new")
        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    if args.mode == "audit":
        run_audit(df)
        return 0

    if args.mode in ("gemini", "all"):
        import gemini_augment_record as gar

        gar.GENERATION_THEMES = gar.GENERATION_THEMES + DISAMBIGUATION_GEMINI_THEMES
        if args.mode == "all":
            remaining = max(0, args.target - len(df))
            batches = max(1, (remaining + args.rows - 1) // args.rows)
        else:
            batches = args.batches
        run_generate(df, batches=batches, rows_per_batch=args.rows, sleep_s=args.sleep)
        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    if args.mode in ("local", "gemini", "all"):
        fix_labels_main()
        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        print(f"Record rows after fix: {len(df)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
