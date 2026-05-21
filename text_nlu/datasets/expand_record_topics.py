"""
Append >=3000 diverse, non-duplicate rows to intent_record.csv
based on topics in ../../mô tả dữ liệu.md (project root).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP_MDS = [ROOT.parent / "mô tả dữ liệu.md", ROOT.parent / "mo ta du lieu.md"]
CSV_PATH = Path(__file__).resolve().parent / "intent_record.csv"
MIN_NEW = 4000


def money_vn(i: int) -> str:
    """Pseudo-random VND snippets, deterministic and spread out."""
    base = 12000 + (i * 9973) % 9_500_000
    k = base // 1000
    rem = base % 1000
    roll = (i * 17) % 8
    if roll == 0 and base >= 1_000_000:
        tr = base / 1_000_000
        s = f"{tr:.1f}".rstrip("0").rstrip(".") + "tr"
        return s
    if roll == 1 and k >= 100:
        return f"{k//100}tr{k%100:02d}k" if k % 100 else f"{k//100}tr"
    if roll == 2:
        return f"{k} ngan"
    if roll == 3 and k >= 200:
        return f"{k//100} củ"
    if roll == 4:
        return f"{k}.{rem//100:01d}k" if rem // 100 else f"{k}k"
    return f"{k}k"


def collect_rows() -> list[tuple[str, str, str, int]]:
    """Return (text, label, type, is_money)."""
    rows: list[tuple[str, str, str, int]] = []
    n = 0

    # --- Expense buckets (topics from mô tả dữ liệu) ---
    foods = [
        "bánh ngọt",
        "bánh ăn vặt",
        "hủ tiếu",
        "phở bò",
        "bún chả",
        "miến lươn",
        "bánh mì",
        "bánh xèo",
        "bánh cuốn",
        "nem rán",
        "gỏi cuốn",
        "cháo lòng",
        "bún riêu",
        "bánh canh",
        "mì quảng",
        "com tấm",
        "xôi mặn",
    ]
    drinks = ["nước đá", "nước ngọt coca", "pepsi lon", "sting", "7up", "nước suối"]
    fruits = ["táo", "cam", "chanh", "nho", "nhãn", "dừa", "ổi", "xoài", "mít", "bưởi"]
    home_items = ["nồi áp suất", "chảo chống dính", "đũa tre", "gia vị", "thớt gỗ", "hộp đựng gạo"]
    furniture = ["giường foam", "tủ quần áo", "bàn làm việc", "ghế gaming", "kệ sách"]
    appliances = ["tivi 4k", "tủ lạnh mini", "máy giặt", "máy sấy quần áo"]
    personal = ["bàn chải điện", "kem dưỡng", "sửa tắm", "sữa tắm", "dao cạo", "kem đánh răng", "kem đanh răng"]
    telecom = ["gói 4g", "mạng wifi", "internet cáp quang", "data viettel", "fpt truyền hình"]
    laundry = ["giặt ủi", "giặt hấp áo vest", "giặt chăn ga", "ủi đồ công sở"]
    vehicle = ["đổ xăng RON 95", "đổ xăng e5", "sửa xe", "thay nhớt", "thay lọc gió", "phụ tùng phanh"]
    fun = ["karaoke", "kaoke", "karaok box", "bida", "bowling", "escape room", "phòng thoát hiểm"]
    edu = ["học phí", "tiền sách", "photo tài liệu", "đăng ký thi", "lệ phí lab"]
    health = ["khám bệnh", "mua thuốc", "xét nghiệm", "tiêm vaccine"]
    vices = ["thuốc lá", "bia lon", "rượu vang", "bia hơi"]
    misc_spend = [
        "ship hàng",
        "tiền ship",
        "văn phòng phẩm",
        "in ấn",
        "tiền in",
        "Netflix",
        "Spotify",
        "YouTube Premium",
        "Cursor Pro",
        "ChatGPT Plus",
        "GitHub Copilot",
        "thức ăn thú cưng",
        "phụ kiện thú cưng",
        "hẹn hò cafe",
        "ăn uống với bạn",
        "lễ tết cúng",
        "đồ cúng kiếng",
        "cây cảnh",
        "decor phòng",
        "khóa học online",
        "hosting VPS",
        "domain com",
        "cloud server AWS",
        "tiền phạt giao thông",
        "vé số",
        "donate streamer",
        "từ thiện ATM",
    ]

    places = [
        "Grab",
        "ShopeeFood",
        "siêu thị",
        "chợ gần nhà",
        "VinMart",
        "Coopmart",
        "Bách Hóa Xanh",
        "online",
        "Tiki",
        "Lazada",
    ]
    verbs = ["mua", "order", "chi", "trả", "thanh toán", "đóng", "nạp", "book"]

    def add_expense(prefix_templates: list[str], label: str, *, per_tpl: int = 12):
        """Ít vòng lặp / nhiều mẫu — tránh khối «mua X ở Y hết Z» lặp hàng nghìn dòng."""
        nonlocal n
        for tpl in prefix_templates:
            for j in range(per_tpl):
                p = places[(n + j) % len(places)]
                v = verbs[(n + j * 3) % len(verbs)]
                m = money_vn(n + j * 97)
                t = tpl.format(v=v, p=p, m=m)
                rows.append((t, label, "expense", 1))
                n += 1

    for item in foods:
        add_expense([f"{item} {v} {{p}} hết {{m}}" for v in verbs], "Food")
    for item in drinks + fruits:
        add_expense([f"{item} {v} {{p}} {{m}}" for v in verbs], "Food")
    for item in home_items:
        add_expense([f"{item} mua tại {{p}} {{m}}"], "Essentials")
    for item in furniture + appliances:
        add_expense([f"{item} thanh toán {{p}} tổng {{m}}"], "Housing")
    for item in personal:
        add_expense([f"{item} {{p}} {{m}}"], "Beauty")
    for item in telecom:
        add_expense([f"{item} tháng này {{m}}"], "Housing")
    for item in laundry:
        add_expense([f"{item} tiệm gần nhà {{m}}"], "Essentials")
    for item in vehicle:
        add_expense([f"{item} cây xăng / gara {{m}}"], "Transport")
    for item in fun:
        add_expense([f"đi {item} cuối tuần hết {{m}}"], "Entertainment")
    for item in edu:
        add_expense([f"{item} kỳ này {{m}}"], "Education")
    for item in health:
        add_expense([f"{item} phòng khám {{m}}"], "Health")
    for item in vices:
        add_expense([f"{item} tiệm tạp hóa {{m}}"], "Social")
    for item in misc_spend:
        if item in ("Netflix", "Spotify", "YouTube Premium"):
            lbl = "Entertainment"
        elif item in ("Cursor Pro", "ChatGPT Plus", "GitHub Copilot"):
            lbl = "Others"
        elif "ship" in item:
            lbl = "Essentials"
        elif item in ("văn phòng phẩm", "in ấn", "tiền in"):
            lbl = "Shopping"
        elif "thú cưng" in item:
            lbl = "Essentials"
        elif item in ("hẹn hò cafe", "ăn uống với bạn"):
            lbl = "Social"
        elif item in ("lễ tết cúng", "đồ cúng kiếng"):
            lbl = "Others"
        elif item in ("cây cảnh", "decor phòng"):
            lbl = "Shopping"
        elif item == "khóa học online":
            lbl = "Education"
        elif item in ("hosting VPS", "domain com", "cloud server AWS"):
            lbl = "Others"
        elif item == "tiền phạt giao thông":
            lbl = "Transport"
        elif item == "vé số":
            lbl = "Entertainment"
        elif item in ("donate streamer", "từ thiện ATM"):
            lbl = "Charity"
        else:
            lbl = "Others"
        add_expense([f"{item} {{m}}"], lbl)

    # Chat-style & typo-heavy (explicit lines)
    chat_style = [
        'ăn vặt {m}',
        'đổ xăng hết {m}',
        'cf sáng {m}',
        'trà sữa {m}',
        'wifi tháng này {m}',
        'net nhà {m}',
        'bida tối {m}',
        'karaoke vs bạn {m}',
        'kaoke hết {m}',
        'xăng xe {m}',
        'kem đanh răng siêu thị {m}',
        'sửa tắm walmart vn {m}',
        'thue phong escape room {m}',
        'order grabbike đi học {m}',
        'tiktok shop mua linh kiện {m}',
        'ship cod {m}',
        'in poster {m}',
        'mua vé số {m}',
        'hoan tien app {m}',  # typo style -> still expense context ambiguous - use as shopping refund? Actually "hoàn tiền" could be income - skip ambiguous
    ]
    for tpl in chat_style:
        if "hoan tien" in tpl:
            continue
        for i in range(160):
            m = money_vn(n + i * 11)
            if "xăng" in tpl or "grab" in tpl:
                lab = "Transport"
            elif any(x in tpl for x in ("ăn", "cf", "trà", "kem đanh", "sửa tắm")):
                lab = "Food"
            elif "kara" in tpl or "bida" in tpl or "escape" in tpl:
                lab = "Entertainment"
            elif "wifi" in tpl or "net" in tpl:
                lab = "Housing"
            else:
                lab = "Shopping"
            rows.append((tpl.format(m=m), lab, "expense", 1))
        n += 1

    # --- Income (topics from same md) ---
    income_templates = [
        ("lg tháng {m}", "Salary"),
        ("lương tháng {m}", "Salary"),
        ("lương part-time {m}", "Salary"),
        ("lương overtime {m}", "Salary"),
        ("thg KPI {m}", "Bonus"),
        ("thưởng Tết {m}", "Bonus"),
        ("thưởng dự án {m}", "Bonus"),
        ("lì xì tết {m}", "Bonus"),
        ("hb học bổng {m}", "Bonus"),
        ("cổ tức {m}", "Investment"),
        ("lãi tiết kiệm {m}", "Investment"),
        ("lãi ngân hàng {m}", "Investment"),
        ("lãi crypto {m}", "Investment"),
        ("rút tiết kiệm {m}", "Investment"),
        ("fl design nhận {m}", "Business"),
        ("freelance viết code {m}", "Business"),
        ("part-time ship hàng {m}", "Business"),
        ("OT cuối tuần {m}", "Business"),
        ("hoa hồng bán hàng {m}", "Business"),
        ("khách ck {m}", "Business"),
        ("refund hoàn tiền {m}", "Business"),
        ("được hoàn tiền {m}", "Business"),
        ("chuyển khoản trả nợ nhận {m}", "Business"),
        ("mẹ gửi {m}", "Business"),
        ("chu cấp từ ba mẹ {m}", "Business"),
        ("bán đồ cũ {m}", "Business"),
        ("bán iphone cũ {m}", "Business"),
        ("thu affiliate {m}", "Business"),
        ("youtube payout {m}", "Business"),
        ("tiktok creator fund {m}", "Business"),
        ("tip donate stream {m}", "Business"),
        ("thuê nhà cho thuê nhận {m}", "Business"),
        ("thu tiền góp nhóm về {m}", "Business"),
        ("bạn bè share bill ck {m}", "Business"),
        ("thu hồi công nợ {m}", "Business"),
        ("momo nhận từ shop {m}", "Business"),
    ]
    for tpl, income_label in income_templates:
        for i in range(120):
            m = money_vn(8000 + n + i * 31)
            rows.append((tpl.format(m=m), income_label, "income", 1))
            n += 1

    return rows


def main() -> int:
    rows = collect_rows()
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            print("Missing header", file=sys.stderr)
            return 1
        existing = {r["text"].strip() for r in reader if r.get("text")}

    def is_income_type(typ: str) -> bool:
        return typ.lower() == "income"

    income_count = 0
    expense_count = 0
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            typ = str(row.get("type", "")).strip()
            if not typ:
                continue
            if is_income_type(typ):
                income_count += 1
            else:
                expense_count += 1

    total_count = income_count + expense_count
    income_ratio = income_count / total_count if total_count else 0.0
    target_income = max(1 if income_ratio > 0 else 0, int(round(MIN_NEW * income_ratio)))
    target_expense = MIN_NEW - target_income

    to_add: list[dict[str, str | int]] = []
    added_income = 0
    added_expense = 0
    for text, label, typ, im in rows:
        t = text.strip()
        if t in existing:
            continue
        if is_income_type(typ) and added_income >= target_income:
            continue
        if not is_income_type(typ) and added_expense >= target_expense:
            continue
        existing.add(t)
        to_add.append({"text": text, "label": label, "type": typ, "is_money": im})
        if is_income_type(typ):
            added_income += 1
        else:
            added_expense += 1
        if len(to_add) >= MIN_NEW:
            break

    # If template collision depleted early, pad with indexed unique lines
    pad_i = 0
    bases = [
        ("Snack cay ShoppeeFood {}", "Food"),
        ("Tiền tip shipper {}", "Food"),
        ("Mua {} ở chợ tạm", "Food"),
        ("Gói data ngày {}", "Housing"),
        ("Phí cầu BOT {}", "Transport"),
        ("Giặt sneaker {}", "Essentials"),
        ("Lightroom preset {}", "Shopping"),
        ("Khóa Udemy giảm {}", "Education"),
    ]
    while len(to_add) < MIN_NEW:
        amt = money_vn(20000 + pad_i * 13)
        base, lab = bases[pad_i % len(bases)]
        text = base.format(amt) if "{}" in base else base + " " + amt
        if "{}" in base:
            text = base.format(amt)
        else:
            text = f"{base} {amt}"
        t = text.strip()
        if t not in existing:
            existing.add(t)
            to_add.append({"text": text, "label": lab, "type": "expense", "is_money": 1})
        pad_i += 1
        if pad_i > 500_000:
            print("Could not reach MIN_NEW", file=sys.stderr)
            return 1

    with CSV_PATH.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        for r in to_add:
            w.writerow(r)

    print(f"Appended {len(to_add)} rows to {CSV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
