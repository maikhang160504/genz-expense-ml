import re

TYPO_MAP = {
    "ca pe": "ca phe",
    "cà phê": "cà phê",
    "cuc dang": "cuc vang",
    "cục dàng": "cục vàng",
}

# Text-based number to digit mapping (case-insensitive checks)
WORD_TO_DIGIT = {
    "một": "1", "mot": "1",
    "hai": "2",
    "ba": "3",
    "bốn": "4", "bon": "4",
    "năm": "5", "nam": "5",
    "sáu": "6", "sau": "6",
    "bảy": "7", "bay": "7",
    "tám": "8", "tam": "8",
    "chín": "9", "chin": "9",
    "mười": "10", "muoi": "10",
}

# Match text number followed by a financial slang/unit
TEXT_NUM_RE = re.compile(
    r"\b(mot|một|hai|ba|bon|bốn|nam|năm|sau|sáu|bay|bảy|tam|tám|chin|chín|muoi|mười)\b(?=\s*(?:k|cành|canh|lít|lit|xị|xi|sị|si|loét|loet|củ|cu|quả|qua|mâm|mam|chục|chuc|triệu|trieu|tr|ngàn|ngan|nghìn|nghin|đ|d|vnd|vnđ)\b)",
    re.IGNORECASE
)

# Amount patterns: supports k, ngan/nghin, trieu/triệu, cu/củ, d/đ/vnd/vnđ, and slang units.
AMOUNT_RE = re.compile(
    r"(?P<num>\d+(?:[\.,]\d+)?)\s*(?P<unit>k|ngan|nghin|tr|trieu|triệu|cu|củ|d|đ|vnd|vnđ|cành|canh|lít|lit|xị|xi|sị|si|loét|loet|quả|qua|mâm|mam|chục|chuc)",
    re.IGNORECASE,
)

# Composite: 1tr200 => 1,200,000 ; 1tr2 => 1,200,000 (1.2 triệu)
TR_COMPOSITE_RE = re.compile(
    r"(?P<million>\d+)\s*(?:tr|trieu|triệu)\s*(?P<thousand>\d{1,3})(?!\d)",
    re.IGNORECASE,
)

# Số thuần: 69000, 69,000, 69.000 (không cần k/tr/triệu)
PLAIN_AMOUNT_RE = re.compile(
    r"(?<![\d.])(?P<raw>\d{1,3}(?:[.,]\d{3})+|\d{4,})(?!\s*(?:k|ngan|nghin|tr|trieu|triệu|cu|củ|cành|canh|lít|lit|xị|xi|sị|si|loét|loet|quả|qua|mâm|mam|chục|chuc)\b)",
    re.IGNORECASE,
)


def preprocess_slang(text: str) -> str:
    text = text.lower().strip()
    
    # 1. Thay thế số bằng chữ (nếu đứng trước đơn vị tài chính)
    def replace_word(m):
        w = m.group(1).lower()
        return WORD_TO_DIGIT.get(w, w)
    
    text = TEXT_NUM_RE.sub(replace_word, text)
    
    # 2. Replace composite phrases like "nửa triệu" / "nửa củ"
    text = re.sub(r"\b(?:nửa\s+triệu|nửa\s+củ|nua\s+trieu|nua\s+cu)\b", "500k", text)
    
    # 3. X củ rưỡi / X triệu rưỡi -> X.5 triệu
    text = re.sub(r"\b(?P<num>\d+(?:[\.,]\d+)?)\s*(?:triệu|trieu|củ|cu)\s*(?:rưỡi|ruoi)\b", r"\g<num>.5 triệu", text)
    
    # 4. triệu rưỡi / củ rưỡi (khi đứng riêng lẻ) -> 1.5 triệu
    text = re.sub(r"\b(?:triệu|trieu|củ|cu)\s*(?:rưỡi|ruoi)\b", "1.5 triệu", text)
    
    return text


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_content(text: str) -> str:
    normalized_text = preprocess_slang(text)
    cleaned = AMOUNT_RE.sub(" ", normalized_text)
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
    normalized_text = preprocess_slang(text)
    amounts = []
    
    for m in TR_COMPOSITE_RE.finditer(normalized_text):
        million = int(m.group("million"))
        thousand_raw = m.group("thousand")
        if len(thousand_raw) == 1:
            amounts.append(int(round((million + int(thousand_raw) / 10.0) * 1_000_000)))
        else:
            amounts.append(million * 1_000_000 + int(thousand_raw) * 1_000)

    for m in AMOUNT_RE.finditer(normalized_text):
        raw = m.group("num").replace(",", ".") if "," in m.group("num") and "." not in m.group("num") else m.group("num").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        unit = m.group("unit").lower()
        if unit in {"k", "ngan", "nghin", "cành", "canh"}:
            value *= 1000
        elif unit in {"tr", "trieu", "triệu", "cu", "củ", "quả", "qua", "mâm", "mam"}:
            value *= 1_000_000
        elif unit in {"lít", "lit", "xị", "xi", "sị", "si", "loét", "loet"}:
            value *= 100000
        elif unit in {"chục", "chuc"}:
            value *= 10000
        elif unit in {"d", "đ", "vnd", "vnđ"}:
            value *= 1
        amounts.append(int(round(value)))

    for m in PLAIN_AMOUNT_RE.finditer(normalized_text):
        parsed = _parse_plain_amount(m.group("raw"))
        if parsed is not None:
            amounts.append(parsed)
    return amounts
