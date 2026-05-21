"""
Sinh câu Record ngắn / vừa — đa dạng cấu trúc, nhiều kiểu tiền (k, tr, xị, vnd, đ…).

Income: chỉ dùng mẫu có dấu hiệu thu (nhận, lương, hoàn, ck về…) — không đảo thành «mua/chi …».
Expense: mua/chi/order/đi … — không gắn nhãn income.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

TOP_MD = Path(__file__).resolve().parents[2] / "mô tả dữ liệu.md"

_TOPIC_LABEL: list[tuple[str, str]] = [
    ("grab", "Transport"),
    ("xăng", "Transport"),
    ("grabfood", "Food"),
    ("grab bike", "Transport"),
    ("tiền điện", "Housing"),
    ("tiền nước", "Housing"),
    ("tiền internet", "Housing"),
    ("tiền wifi", "Housing"),
    ("tiền wif", "Housing"),
    ("tiền data", "Housing"),
    ("tiền 4g", "Housing"),
    ("gas", "Housing"),
    ("netflix", "Entertainment"),
    ("spotify", "Entertainment"),
    ("xem phim", "Entertainment"),
    ("shopee", "Shopping"),
    ("shoppe", "Shopping"),
    ("lazada", "Shopping"),
    ("tiktok", "Shopping"),
    ("spa", "Beauty"),
    ("nail", "Beauty"),
    ("cắt tóc", "Beauty"),
    ("mỹ phẩm", "Beauty"),
    ("mua thuốc", "Health"),
    ("học phí", "Education"),
]


def strip_accents(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def load_topics_from_md() -> list[str]:
    if not TOP_MD.is_file():
        return []
    topics: list[str] = []
    for line in TOP_MD.read_text(encoding="utf-8").splitlines():
        t = line.strip().rstrip(",").strip()
        if not t or t.startswith("#") or "{" in t or t == ".......":
            continue
        t = re.sub(
            r"\s+\d+[\.,]?\d*\s*(k|tr|củ|ngan|nghin|đ|xị)\s*$", "", t, flags=re.I
        ).strip()
        if t and len(t) <= 48:
            topics.append(t)
    return topics


def _tokens(low: str) -> set[str]:
    return set(re.findall(r"[a-zA-Zà-ỹ0-9]+", low))


def label_for_topic(topic: str) -> str:
    low = topic.lower()
    for key, lab in _TOPIC_LABEL:
        if key in low:
            return lab
    tok = _tokens(low)
    food_tok = {
        "ăn", "phở", "bún", "cơm", "mì", "bánh", "trà", "sữa", "thịt", "cá", "tôm",
        "trứng", "chuối", "cam", "gạo", "vặt", "nui", "chả", "xoài", "ổi",
    }
    if "cà phê" in low or "cf" in tok or "hủ tiếu" in low or "hu" in tok and "tieu" in tok:
        return "Food"
    if tok & food_tok:
        return "Food"
    if any(k in low for k in ("spa", "nail", "tóc", "kem", "son", "mặt nạ")):
        return "Beauty"
    return "Essentials"


def money_variant(i: int) -> str:
    """Một chuỗi tiền — xoay qua k / tr / xị / vnd / đồng / củ / ngan (giới hạn theo quy mô)."""
    base = 6_000 + (i * 8353) % 2_500_000
    k = max(1, base // 1000)
    rem = base % 1000
    kind = (i * 29) % 12
    if kind == 0:
        return f"{k}k"
    if kind == 1:
        return f"{k}K"
    if kind == 2:
        return f"{k} xị"
    if kind == 3:
        return f"{k} nghìn"
    if kind == 4:
        return f"{k} ngàn"
    if kind == 5:
        return f"{k} ngan"
    if kind == 6:
        return f"{k} nghin"
    if kind == 7:
        return f"{base:,}".replace(",", ".") + "đ"
    if kind == 8:
        return f"{base:,}".replace(",", ".") + " vnd"
    if kind == 9:
        return f"{k}k VND"
    if kind == 10 and k >= 80:
        tr = base / 1_000_000
        s = f"{tr:.2f}".rstrip("0").rstrip(".")
        return f"{s} tr"
    if kind == 11 and k >= 500:
        return f"{k // 100} củ"
    if rem:
        return f"{k}.{rem // 100}k"
    return f"{k}k"


_ABBREV: dict[str, list[str]] = {
    "grab": ["grab", "grb", "di grab", "bắt grab", "book grab", "cuốc grab"],
    "trà sữa": ["ts", "tra sua", "trà sữa"],
    "cà phê": ["cf", "ca phe", "cafe", "cà phê"],
    "phở": ["phở", "pho"],
    "hủ tiếu": ["hủ tiếu", "hu tieu", "ht"],
    "ăn sáng": ["ăn sáng", "an sang", "bk sáng"],
    "ăn vặt": ["ăn vặt", "an vat", "snack"],
    "siêu thị": ["st", "siêu thị", "sieu thi"],
    "shopee": ["shopee", "shoppe", "spx"],
    "tiền điện": ["điện", "tien dien", "evn"],
    "tiền nước": ["nước", "tien nuoc"],
    "xăng": ["xăng", "xang", "đổ xăng"],
}


def topic_variants(topic: str) -> list[str]:
    out = [topic.strip()]
    low = topic.lower()
    for key, alts in _ABBREV.items():
        if key in low or low == key:
            out.extend(alts)
            break
    na = strip_accents(topic).lower()
    if na and na not in {x.lower() for x in out}:
        out.append(na)
    if "sữa" in topic:
        out.append(topic.replace("sữa", "sửa"))
    return list(dict.fromkeys(x for x in out if x))


# Mẫu expense — {t}=topic, {m}=money, {p}=place (optional)
_EXPENSE_TEMPLATES: list[str] = [
    "{t} {m}",
    "{m} cho {t}",
    "{t} hết {m}",
    "{t},{m}",
    "mua {t} {m}",
    "chi {t} {m}",
    "order {t} {m}",
    "trả {t} {m}",
    "đi {t} {m}",
    "{t} {m} nha",
    "{t} tầm {m}",
    "vừa {t} {m}",
    "{t} ~{m}",
    "spend {t} {m}",
]

_EXPENSE_WITH_PLACE: list[str] = [
    "mua {t} ở {p} {m}",
    "{t} {p} {m}",
    "order {t} {p} {m}",
    "{p}: {t} {m}",
    "quẹt {p} {t} {m}",
]

_PLACES = [
    "Grab", "GF", "ShopeeFood", "st", "chợ", "BHX", "Coopmart",
    "Tiki", "Lazada", "quán", "tiktok shop", "Momo",
]

# Income — luôn giữ dấu hiệu thu TRƯỚC / cạnh tiền (không hoán «mua»)
_INCOME_SPECS: list[tuple[str, str]] = [
    ("lg {m}", "Salary"),
    ("lương {m}", "Salary"),
    ("lương về {m}", "Salary"),
    ("nhận lương {m}", "Salary"),
    ("thg {m}", "Bonus"),
    ("thưởng {m}", "Bonus"),
    ("lì xì {m}", "Bonus"),
    ("{m} về tk", "Salary"),
    ("ck về {m}", "Business"),
    ("mẹ ck {m}", "Bonus"),
    ("me cho {m}", "Bonus"),
    ("mẹ cho {m}", "Bonus"),
    ("gia đình cho {m}", "Bonus"),
    ("nhận {m}", "Business"),
    ("hoàn {m}", "Business"),
    ("refund {m}", "Business"),
    ("được hoàn {m}", "Business"),
    ("fl nhận {m}", "Business"),
    ("cổ tức {m}", "Investment"),
    ("lãi tk {m}", "Investment"),
    ("lãi {m}", "Investment"),
    ("tip {m}", "Business"),
    ("bán đồ cũ {m}", "Business"),
]

# Câu ngắn từ mô tả (grab + số)
_FIXED_SHORT: list[tuple[str, str]] = [
    ("đi grab {m}", "Transport"),
    ("bắt grab {m}", "Transport"),
    ("book grab {m}", "Transport"),
    ("grab {m}", "Transport"),
    ("grb {m}", "Transport"),
    ("ăn sáng {m}", "Food"),
    ("an sang {m}", "Food"),
    ("cf đen {m}", "Food"),
    ("cf sữa {m}", "Food"),
    ("cà phê sửa {m}", "Food"),
    ("ts {m}", "Food"),
    ("phở {m}", "Food"),
    ("ăn vặt {m}", "Food"),
    ("điện {m}", "Housing"),
    ("net {m}", "Housing"),
    ("4g {m}", "Housing"),
    ("xăng {m}", "Transport"),
]


def generate_rows(
    target: int,
    existing: set[str],
    *,
    seed: int = 0,
    allow_bare: bool = True,
) -> list[tuple[str, str, str, int]]:
    """Trả về list (text, label, type, is_money)."""
    out: list[tuple[str, str, str, int]] = []
    topics = load_topics_from_md()
    if not topics:
        topics = ["grab", "phở", "trà sữa", "cà phê", "gạo", "tiền điện", "xăng", "shopee"]

    n = seed

    def push(text: str, label: str, typ: str, is_money: int) -> bool:
        t = re.sub(r"\s+", " ", text.strip())
        if not t or t in existing or len(t) > 72:
            return False
        existing.add(t)
        out.append((t, label, typ, is_money))
        return True

    while len(out) < target:
        progressed = False
        # Fixed ultra-short
        for tpl, lab in _FIXED_SHORT:
            if len(out) >= target:
                break
            m = money_variant(n)
            if push(tpl.format(m=m), lab, "expense", 1):
                progressed = True
            n += 1

        # Topics × templates
        for topic in topics:
            if len(out) >= target:
                break
            lab = label_for_topic(topic)
            for var in topic_variants(topic):
                if len(out) >= target:
                    break
                if allow_bare and n % 17 == 0:
                    if push(var, lab, "expense", 0):
                        progressed = True
                for tpl in _EXPENSE_TEMPLATES:
                    if len(out) >= target:
                        break
                    m = money_variant(n)
                    text = tpl.format(t=var, m=m)
                    if push(text, lab, "expense", 1):
                        progressed = True
                    n += 1
                if n % 3 == 0:
                    p = _PLACES[n % len(_PLACES)]
                    for tpl in _EXPENSE_WITH_PLACE:
                        if len(out) >= target:
                            break
                        m = money_variant(n)
                        text = tpl.format(t=var, p=p, m=m)
                        if push(text, lab, "expense", 1):
                            progressed = True
                        n += 1

        # Income
        for tpl, lab in _INCOME_SPECS:
            if len(out) >= target:
                break
            for _ in range(3):
                m = money_variant(n + 500_000)
                if push(tpl.format(m=m), lab, "income", 1):
                    progressed = True
                n += 1

        # GenZ (expense only)
        genz_items = ["ts", "cf", "pho", "grab", "st", "bida", "xăng", "điện", "nf"]
        for item in genz_items:
            if len(out) >= target:
                break
            lab = label_for_topic(item)
            for tpl in ("{item} chill {m}", "{item} xịn {m}", "{item} hơi đau ví {m}", "{item} oke {m}"):
                m = money_variant(n)
                if push(tpl.format(item=item, m=m), lab, "expense", 1):
                    progressed = True
                n += 1

        if not progressed:
            m = money_variant(n + 99_000)
            if push(f"ghi chú #{n} {m}", "Others", "expense", 1):
                n += 1
            else:
                n += 1
                if n > seed + 500_000:
                    break

    return out[:target]
