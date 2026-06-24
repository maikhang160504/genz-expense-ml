"""
Sửa nhãn intent_record.csv theo quy tắc biên (mua/bán, đi cafe/mua cafe, gạo, sạc…).

Chạy: python text_nlu/datasets/fix_disambiguation_labels.py
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from family_bonus_labels import is_family_gift_income

ROOT = Path(__file__).resolve().parent
RECORD_CSV = ROOT / "intent_record.csv"
ACTION_CSV = ROOT / "intent_action.csv"

LABELS = {
    "Food", "Transport", "Housing", "Essentials", "Shopping", "Beauty",
    "Health", "Education", "Entertainment", "Social", "Salary", "Bonus",
    "Business", "Investment", "Debt", "Charity", "Savings", "Others",
}


def _norm(s: str) -> str:
    nfd = unicodedata.normalize("NFD", str(s).lower().strip())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


_ACTION_IN_RECORD = re.compile(
    r"(?:"
    r"tổng\s+chi|tong\s+chi|"
    r"(?:tháng|thang|tuần|tuan|hôm|hom|quý|quy)\s+nay\b.+(?:tiêu|tieu|chi\b).*(?:bao\s+nhiêu|bao\s+nhieu|hết|het|mấy|may|roi|rồi)|"
    r"(?:tiêu|tieu)\s+bao\s+nhiêu|"
    r"(?:đã|da)\s+(?:tiêu|tieu)|"
    r"thống\s+kê\s+chi|thong\s+ke\s+chi|"
    r"xem\s+(?:tổng\s+chi|tong\s+chi)|"
    r"chi\s+tiêu\s+(?:tháng|thang|tuần|tuan)|"
    r"mình\s+đã\s+tiêu|minh\s+da\s+tieu|"
    r"tiêu\s+hết\s+bao\s+nhiêu|tieu\s+het\s+bao\s+nhiêu"
    r")",
    re.I,
)

_SELL_INCOME = re.compile(r"^(ban|thu\s+tien\s+tu\s+ban|nhan\s+tien\s+ban)\b", re.I)
_BUY_EXPENSE = re.compile(r"^(mua|order|chi|thanh\s+toan|tra|dong)\b", re.I)
_GO_ENTERTAIN_CAFE = re.compile(
    r"\b(?:di|hen|tu\s+tap|gap|uong|di\s+uong)\s+(?:ca\s+phe|cafe|cf)\b|\b(?:ca\s+phe|cafe|cf)\b.*\b(?:voi|ban|be|bo|ny|crush|nguoi\s+yeu)\b",
    re.I,
)
_GO_ENTERTAIN_OTHER = re.compile(
    r"\b(?:bar|pub|club|spa|salon|massage|karaoke|kaoke|bida|phim|cinema|rap|du\s+lich|restaurant|nhau)\b",
    re.I,
)
_COFFEE_ANY = re.compile(
    r"\b(?:ca\s+phe|cafe|cf|hat\s+ca\s+phe|bot\s+ca\s+phe)\b",
    re.I,
)
_SHOPPING_GADGET = re.compile(
    r"\bmua\s+(?:sac|cap|tai\s+nghe|airpods|case|op|"
    r"chuot|ban\s+phim|webcam|hub|o\s+cung|ao|quan|giay|dep|balo|hoodie|sneaker)\b",
    re.I,
)
_ESSENTIALS_GROCERY = re.compile(
    r"(?:^|\b)(gao|mi\s+goi|giay\s+ve\s+sinh|nuoc\s+rua|bot\s+giat)(?:\s|$|\d)",
    re.I,
)
_GIFT_ESSENTIALS = re.compile(r"\bmua\s+qua\s+cho\b", re.I)
_MEAT = re.compile(r"\bthit\s+(?:heo|lon|bo|ga|ca|tom)\b", re.I)


_MONEY_PATTERN = re.compile(
    r"\d+(?:[\.,]\d+)?\s?(k|đ|d|vnđ|vnd|ngan|nghin|tr|triệu|trieu|củ|cu)\b",
    re.I,
)


def _category_for_item(text: str) -> str | None:
    t = _norm(text).replace("đ", "d")
    if _GIFT_ESSENTIALS.search(t):
        return "Essentials"
    if _SHOPPING_GADGET.search(t):
        return "Shopping"
    if _ESSENTIALS_GROCERY.search(t):
        return "Essentials"
    if _GO_ENTERTAIN_CAFE.search(t) or _GO_ENTERTAIN_OTHER.search(t):
        return "Entertainment"
    if _COFFEE_ANY.search(t):
        return "Food"
    if _MEAT.search(t) and not _SELL_INCOME.search(t):
        return "Food"
    return None


def fix_record_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {"type": 0, "label": 0, "removed_action_like": 0}
    out = df.copy()
    
    def is_action_like(text_val: str) -> bool:
        t = _norm(text_val)
        if _MONEY_PATTERN.search(t):
            return False
        return bool(_ACTION_IN_RECORD.search(t))
        
    mask_action = out["text"].astype(str).map(is_action_like)
    removed = out[mask_action]
    out = out[~mask_action].reset_index(drop=True)
    stats["removed_action_like"] = len(removed)

    for idx, row in out.iterrows():
        text = str(row["text"])
        t = _norm(text).replace("đ", "d")
        typ = str(row["type"]).lower()
        label = str(row["label"])
        changed = False

        if _SELL_INCOME.search(t):
            if typ != "income":
                out.at[idx, "type"] = "income"
                typ = "income"
                changed = True
            if label not in ("Business", "Salary", "Bonus", "Investment"):
                out.at[idx, "label"] = "Business"
                changed = True
        elif _BUY_EXPENSE.search(t) and typ == "income" and not re.search(
            r"\b(hoàn|hoan|refund|lương|luong|thưởng|thuong|lãi|lai|ck\s+về|cho\s+\d)\b", t
        ):
            out.at[idx, "type"] = "expense"
            typ = "expense"
            changed = True

        if is_family_gift_income(text):
            if typ != "income" or label != "Bonus":
                out.at[idx, "type"] = "income"
                out.at[idx, "label"] = "Bonus"
                changed = True

        cat_hint = _category_for_item(text)
        if cat_hint and label != cat_hint and typ == "expense":
            out.at[idx, "label"] = cat_hint
            changed = True

        if changed:
            if typ != str(row["type"]).lower():
                stats["type"] += 1
            if label != str(out.at[idx, "label"]):
                stats["label"] += 1

    return out, stats, removed


def merge_action_rows(removed: pd.DataFrame, action_df: pd.DataFrame) -> pd.DataFrame:
    if removed.empty:
        return action_df
    existing = set(action_df["text"].astype(str).str.strip())
    add = []
    for text in removed["text"].astype(str):
        t = text.strip()
        if t in existing:
            continue
        existing.add(t)
        add.append({"text": t, "intent": "Action", "action_type": "REPORT_GENERAL"})
    if not add:
        return action_df
    return pd.concat([action_df, pd.DataFrame(add)], ignore_index=True)


def main() -> int:
    df = pd.read_csv(RECORD_CSV, encoding="utf-8-sig")
    action_df = pd.read_csv(ACTION_CSV, encoding="utf-8-sig")
    before = len(df)
    df, stats, removed = fix_record_df(df)
    action_df = merge_action_rows(removed, action_df)
    df.to_csv(RECORD_CSV, index=False, encoding="utf-8-sig")
    action_df.to_csv(ACTION_CSV, index=False, encoding="utf-8-sig")
    print(f"Record: {before} -> {len(df)} rows")
    print(f"  fixed type~{stats['type']}, label~{stats['label']}, removed action-like={stats['removed_action_like']}")
    print(f"Action: {len(action_df)} rows (+{len(removed)} candidates merged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
