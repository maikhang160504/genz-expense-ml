import re

TYPO_MAP = {
    "ca pe": "ca phe",
    "cà pê": "cà phê",
    "cuc dang": "cuc vang",
    "cục dàng": "cục vàng",
}

# Amount patterns: supports k, ngan/nghin, trieu/triệu, cu/củ, d/đ/vnd/vnđ.
AMOUNT_RE = re.compile(
    r"(?P<num>\d+(?:[\.,]\d+)?)\s*(?P<unit>k|ngan|nghin|tr|trieu|triệu|cu|củ|d|đ|vnd|vnđ)",
    re.IGNORECASE,
)

# Composite format: 2tr400 => 2,400,000
TR_COMPOSITE_RE = re.compile(
    r"(?P<million>\d+)\s*(?:tr|trieu|triệu)\s*(?P<thousand>\d{1,3})",
    re.IGNORECASE,
)

# Số thuần: 69000, 69,000, 69.000 (không cần k/tr/triệu)
PLAIN_AMOUNT_RE = re.compile(
    r"(?<![\d.])(?P<raw>\d{1,3}(?:[.,]\d{3})+|\d{4,})(?!\s*(?:k|ngan|nghin|tr|trieu|triệu|cu|củ)\b)",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_content(text: str) -> str:
    cleaned = AMOUNT_RE.sub(" ", text)
    cleaned = normalize_text(cleaned)
    for src, dst in TYPO_MAP.items():
        cleaned = cleaned.replace(src, dst)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _parse_plain_amount(raw: str) -> int | None:
    token = raw.strip()
    if not token:
        return None
    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", "")
        else:
            token = token.replace(",", "")
    elif "." in token and re.fullmatch(r"\d{1,3}(?:\.\d{3})+", token):
        token = token.replace(".", "")
    else:
        token = token.replace(",", "").replace(".", "")
    try:
        value = int(token)
    except ValueError:
        return None
    return value if value > 0 else None


def extract_amounts(text: str) -> list[int]:
    amounts = []
    for m in TR_COMPOSITE_RE.finditer(text):
        million = int(m.group("million"))
        thousand = int(m.group("thousand"))
        amounts.append(million * 1_000_000 + thousand * 1_000)

    for m in AMOUNT_RE.finditer(text):
        raw = m.group("num").replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            continue
        unit = m.group("unit").lower()
        if unit in {"k", "ngan", "nghin"}:
            value *= 1000
        elif unit in {"tr", "trieu", "triệu"}:
            value *= 1_000_000
        elif unit in {"cu", "củ"}:
            value *= 1_000_000
        elif unit in {"d", "đ", "vnd", "vnđ"}:
            value *= 1
        amounts.append(int(round(value)))

    for m in PLAIN_AMOUNT_RE.finditer(text):
        parsed = _parse_plain_amount(m.group("raw"))
        if parsed is not None:
            amounts.append(parsed)
    return amounts
