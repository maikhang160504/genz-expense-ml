import re
from pyvi import ViTokenizer


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    # Remove special characters but keep underscores (ViTokenizer output uses underscores)
    text = re.sub(r"[^0-9a-zA-Z_\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def vi_tokenize(text: str) -> list[str]:
    # Tokenize Vietnamese then split into tokens
    tokenized = ViTokenizer.tokenize(text)
    tokenized = normalize_text(tokenized)
    return tokenized.split()


def clean_category_text(text: str) -> str:
    text = str(text).lower().strip()
    # Remove money/amount tokens to focus on category semantics
    text = re.sub(
        r"\d+(?:[\.,]\d+)?\s?(k|đ|d|vnđ|vnd|ngan|nghin|triệu|trieu|củ|cu)",
        " ",
        text,
        flags=re.I,
    )
    # Remove payment methods and currency words
    text = re.sub(
        r"\b(vnđ|vnd|banking|momo|zalopay|vnpay|qr|thẻ|the|tiền mặt|cash|chuyển khoản|pos)\b",
        " ",
        text,
        flags=re.I,
    )
    # Remove receipt boilerplate headers/footers (hóa đơn, phiếu thanh toán, tổng tiền, thu ngân...)
    text = re.sub(
        r"\b(hóa đơn|hoa don|phiếu thanh toán|phieu thanh toan|thanh toán|tổng tiền|tong tien|tổng cộng|tong cong|thu ngân|thu ngan|tiền thối|tiền thừa|tiền khách|khách cần trả|khách thanh toán|mã số thuế|mst|sđt|hotline|đơn giá|thành tiền)\b",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text
