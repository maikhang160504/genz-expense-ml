import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILE_PATH = ROOT / "datasets" / "intent_record.csv"
TARGET_TOTAL_LINES = 5000  # includes header
MIN_NEW_ROWS = 500
RANDOM_SEED = 20260512

LABELS = [
    "Food",
    "Transport",
    "Housing",
    "Essentials",
    "Health",
    "Beauty",
    "Shopping",
    "Education",
    "Entertainment",
    "Social",
    "Charity",
    "Investment",
    "Savings",
    "Debt",
    "Others",
]

EXPENSE_LABELS = [
    "Food",
    "Transport",
    "Housing",
    "Essentials",
    "Health",
    "Beauty",
    "Shopping",
    "Education",
    "Entertainment",
    "Social",
    "Charity",
    "Investment",
    "Savings",
    "Debt",
    "Others",
]

INCOME_LABELS = ["Savings", "Debt", "Investment", "Social", "Others", "Education"]
NONE_LABELS = ["Others", "Social", "Entertainment"]

UNITS = ["k", "tr", "ngan", "nghin", "vnđ", "củ", "dong"]

STREET_PLACES = [
    "vỉa hè",
    "xe đẩy đầu ngõ",
    "quán cóc",
    "chợ đêm",
    "hẻm nhỏ",
]

OFFICE_PLACES = [
    "pantry văn phòng",
    "sảnh tòa nhà",
    "mall lunch zone",
    "food court trung tâm thương mại",
    "coworking",
]

FAMILY_PLACES = ["ở nhà", "chung cư", "khu trọ", "nhà ngoại", "nhà nội"]

MONEY_ACTIONS = [
    "tiền mặt",
    "chuyển khoản",
    "quẹt thẻ",
    "scan qr",
    "banking app",
]

GENZ_FILLERS = [
    "chill",
    "okela",
    "lụm",
    "mlem",
    "flex nhẹ",
    "xỉu up xỉu down",
    "vibe ổn",
    "khum",
    "hơi cấn",
    "gấp gấp",
    "căng phết",
]

FOOD_ITEMS = [
    "phở gà",
    "bún bò",
    "cơm tấm",
    "trà sữa",
    "bánh mì",
    "mì cay",
    "gà rán combo",
    "sushi",
    "lẩu mini",
]

TRANSPORT_ITEMS = [
    "đổ xăng",
    "grab bike",
    "gửi xe",
    "taxi",
    "be car",
    "phí cầu đường",
    "ship hàng",
    "thuê xe",
    "thay nhớt",
]

HOUSING_ITEMS = [
    "tiền thuê nhà",
    "tiền điện",
    "tiền nước",
    "sửa ống nước",
    "mua đồ gia dụng",
    "dọn nhà",
    "sửa máy lạnh",
    "mua đèn",
    "wifi tháng",
]

ESSENTIAL_ITEMS = [
    "mua rau",
    "mua thịt",
    "mua gạo",
    "mua sữa",
    "siêu thị tuần",
    "đồ ăn dự trữ",
    "gia vị",
    "trái cây",
    "đồ pet",
]

HEALTH_ITEMS = [
    "mua thuốc",
    "khám tổng quát",
    "viện phí",
    "bảo hiểm y tế",
    "vitamin",
    "khám nha",
    "xét nghiệm",
    "tiêm ngừa",
    "khám mắt",
]

BEAUTY_ITEMS = [
    "đi spa",
    "làm nail",
    "mua serum",
    "mua kem chống nắng",
    "gội dưỡng sinh",
    "nhuộm tóc",
    "mua toner",
    "wax",
    "skincare set",
]

SHOPPING_ITEMS = [
    "mua áo",
    "mua quần",
    "mua tai nghe",
    "mua balo",
    "săn sale",
    "mua phụ kiện",
    "mua đồ tech",
    "mua giày",
    "mua ốp lưng",
]

EDU_ITEMS = [
    "đóng học phí",
    "mua sách",
    "đăng ký khóa học",
    "mua tài liệu",
    "lệ phí thi",
    "gia sư",
    "mua vở bút",
    "mua tài khoản học online",
    "workshop",
]

ENT_ITEMS = [
    "xem phim",
    "nạp game",
    "đi concert",
    "đi pub",
    "du lịch ngắn ngày",
    "karaoke",
    "vé lễ hội",
    "trip cuối tuần",
    "escape room",
]

SOCIAL_ITEMS = [
    "mừng cưới",
    "mua quà sinh nhật",
    "mua hoa",
    "góp quỹ lớp",
    "tiền mừng",
    "tiệc bạn bè",
    "gift card",
    "gửi phong bì",
    "quà handmade",
]

CHARITY_ITEMS = [
    "quyên góp",
    "ủng hộ",
    "mua nhu yếu phẩm tặng",
    "góp quỹ từ thiện",
    "mua chăn ấm",
    "ủng hộ y tế",
    "quỹ khuyến học",
    "ủng hộ bão lũ",
    "cho tặng bữa ăn",
]

INVEST_ITEMS = [
    "mua cổ phiếu",
    "mua vàng",
    "nạp crypto",
    "đầu tư quỹ mở",
    "mua trái phiếu",
    "hùn vốn",
    "mua ETF",
    "đầu tư đất nền",
    "nạp margin",
]

SAVINGS_ITEMS = [
    "gửi tiết kiệm",
    "bỏ heo",
    "quỹ dự phòng",
    "tiết kiệm online",
    "quỹ khẩn cấp",
    "tiết kiệm định kỳ",
    "để dành mua nhà",
    "quỹ du lịch",
    "tiết kiệm học phí",
]

DEBT_ITEMS = [
    "trả nợ",
    "đòi nợ",
    "thu hồi khoản vay",
    "trả góp",
    "hoàn tiền mượn",
    "thanh toán nợ thẻ",
    "trả nợ ngân hàng",
    "thu hồi tiền cọc",
    "vay tạm",
]

OTHER_ITEMS = [
    "lặt vặt",
    "chi phí linh tinh",
    "dịch vụ phát sinh",
    "phí nhỏ",
    "random charge",
    "support fee",
    "misc payment",
    "transaction fee",
    "khác",
]

LABEL_TO_ITEMS = {
    "Food": FOOD_ITEMS,
    "Transport": TRANSPORT_ITEMS,
    "Housing": HOUSING_ITEMS,
    "Essentials": ESSENTIAL_ITEMS,
    "Health": HEALTH_ITEMS,
    "Beauty": BEAUTY_ITEMS,
    "Shopping": SHOPPING_ITEMS,
    "Education": EDU_ITEMS,
    "Entertainment": ENT_ITEMS,
    "Social": SOCIAL_ITEMS,
    "Charity": CHARITY_ITEMS,
    "Investment": INVEST_ITEMS,
    "Savings": SAVINGS_ITEMS,
    "Debt": DEBT_ITEMS,
    "Others": OTHER_ITEMS,
}

NOISE_SENTENCES = [
    "hello ae, nay mood sao roi",
    "cho xin review nhe",
    "oke de toi check",
    "dang o van phong roi",
    "mai tinh tiep nha",
    "chot keo luc toi",
    "cho em xin slot voi",
    "di choi khong ne",
    "app nay lag qua troi",
    "toi ve nha day",
    "off 1 bua nha team",
    "nha co ai onl hem",
    "cmt giup tui cai",
    "xong viec ping toi",
    "hom nay troi oi la troi",
]


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in f)


def format_money_value() -> str:
    unit = random.choice(UNITS)
    if unit in {"k", "ngan", "nghin", "vnđ", "dong"}:
        value = random.choice(
            [
                random.randint(5, 999),
                random.randint(1000, 9999),
                random.choice([12, 15, 18, 20, 25, 33, 45, 69, 88, 99]),
            ]
        )
        if isinstance(value, int) and value >= 1000 and random.random() < 0.5:
            value_str = f"{value:,}".replace(",", ".")
        else:
            value_str = str(value)
        if unit == "vnđ":
            return f"{value_str}k vnđ"
        if unit == "dong":
            return f"{value_str}k đồng"
        return f"{value_str}{unit}"

    if unit in {"tr", "củ"}:
        integer_part = random.randint(1, 35)
        if random.random() < 0.5:
            decimal_part = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
            value_str = f"{integer_part}.{decimal_part}"
        else:
            value_str = str(integer_part)
        return f"{value_str}{unit}"

    return "0k"


def money_to_cell(text_money: str, is_noise: bool) -> str:
    if is_noise:
        return "0k"
    cleaned = text_money.replace(" đồng", "k").replace("vnđ", "k")
    cleaned = cleaned.replace("ngan", "k").replace("nghin", "k")
    cleaned = cleaned.replace("củ", "tr")
    return cleaned.strip()


def pick_context(label: str) -> str:
    if label in {"Food", "Transport", "Entertainment"}:
        return random.choice(STREET_PLACES)
    if label in {"Shopping", "Beauty", "Education"}:
        return random.choice(OFFICE_PLACES)
    if label in {"Housing", "Essentials", "Health"}:
        return random.choice(FAMILY_PLACES)
    return random.choice(STREET_PLACES + OFFICE_PLACES + FAMILY_PLACES)


def build_text(label: str, tx_type: str, money_text: str, idx: int) -> str:
    item = random.choice(LABEL_TO_ITEMS[label])
    context = pick_context(label)
    pay_way = random.choice(MONEY_ACTIONS)
    filler = random.choice(GENZ_FILLERS)

    patterns = [
        f"{item} ở {context}, {pay_way} {money_text}, {filler}",
        f"Nay {context} có vụ {item} hết {money_text}, thanh toán {pay_way}",
        f"{context} làm combo {item}; chốt {money_text} bằng {pay_way}",
        f"Đi {context} rồi {item}, bay {money_text}, cảm giác {filler}",
        f"Case văn phòng: {item} tại {context}, trả {money_text} qua {pay_way}",
        f"Gia đình ở {context}: {item} tổng {money_text}, xử lý bằng {pay_way}",
        f"Teencode mode on: {item} {money_text} ở {context}, {filler}",
        f"An sang khong dau: {item} {money_text} tai {context}, tra {pay_way}",
        f"English boi: {item} in {context}, pay {money_text} by {pay_way}",
        f"Story time {idx}: {context} {item} xong, mất {money_text}, {filler}",
    ]

    if tx_type == "Income":
        patterns = [
            f"Nhận tiền từ {item} ở {context}: +{money_text}, về ví qua {pay_way}",
            f"Thu nhập hôm nay: {item} {money_text}, bắn qua {pay_way}",
            f"Lương thưởng style genz: {item} +{money_text}, nhận tại {context}",
            f"Đòi được tiền: {item}, vào tài khoản {money_text}, {filler}",
            f"Cash-in from {item} at {context}: {money_text}, done by {pay_way}",
            f"Nay có khoản về từ {item}, {money_text} chuyển khoản cái rẹt",
            f"Boi tien roi ne: {item} +{money_text}, {context}",
        ]

    return random.choice(patterns)


def build_noise_text(label: str, idx: int) -> str:
    context = pick_context(label)
    base = random.choice(NOISE_SENTENCES)
    style = random.choice([
        f"{base}, gặp ở {context}",
        f"{context} ơi {base}",
        f"{base} #{idx}",
        f"{base}, {random.choice(GENZ_FILLERS)}",
    ])
    return style


def choose_label(tx_type: str, force_label: str | None = None) -> str:
    if force_label:
        return force_label
    if tx_type == "expense":
        return random.choice(EXPENSE_LABELS)
    if tx_type == "Income":
        return random.choice(INCOME_LABELS)
    return random.choice(NONE_LABELS)


def generate_rows(n: int) -> list[list[str]]:
    random.seed(RANDOM_SEED)
    rows: list[list[str]] = []
    seen_texts: set[str] = set()

    n_expense = int(round(n * 0.70))
    n_income = int(round(n * 0.25))
    n_none = n - n_expense - n_income

    schedule = (
        ["expense"] * n_expense
        + ["Income"] * n_income
        + ["None"] * n_none
    )
    random.shuffle(schedule)

    label_coverage = LABELS.copy()
    random.shuffle(label_coverage)

    i = 0
    while len(rows) < n:
        tx_type = schedule[len(rows)]
        force_label = None
        if label_coverage:
            force_label = label_coverage.pop()

        label = choose_label(tx_type, force_label)
        is_noise = tx_type == "None"

        if is_noise:
            text = build_noise_text(label, i)
            money_text = "0k"
        else:
            money_text = format_money_value()
            text = build_text(label, tx_type, money_text, i)

            # Inject some domain-specific noisy contexts and units.
            if label == "Transport" and random.random() < 0.25:
                liters = random.choice([1, 2, 3, 4, 5, 10])
                text += f", {liters} lít"

            if label in {"Shopping", "Housing"} and random.random() < 0.2:
                shop = random.choice(["Shopee", "Lazada", "Tiki"])
                text += f", order trên {shop}"

            if label in {"Education", "Housing"} and random.random() < 0.25:
                text += ", hóa đơn định kỳ"

        if text in seen_texts:
            i += 1
            continue

        seen_texts.add(text)
        money_cell = money_to_cell(money_text, is_noise)
        rows.append([text, label, tx_type, money_cell])
        i += 1

    return rows


def main() -> None:
    current_lines = count_lines(FILE_PATH)
    need_to_target = max(0, TARGET_TOTAL_LINES - current_lines)
    to_add = max(MIN_NEW_ROWS, need_to_target)

    new_rows = generate_rows(to_add)

    with FILE_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

    final_lines = count_lines(FILE_PATH)
    print(f"Current lines before: {current_lines}")
    print(f"Rows appended: {len(new_rows)}")
    print(f"Current lines after: {final_lines}")


if __name__ == "__main__":
    main()
