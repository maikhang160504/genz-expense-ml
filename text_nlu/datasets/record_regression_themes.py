"""Theme + quota cho sinh intent_record regression (~10k) — thay keyword runtime."""

from __future__ import annotations

RECORD_COLUMNS = ["text", "label", "type", "is_money"]

LABELS = frozenset({
    "Food", "Transport", "Housing", "Essentials", "Shopping", "Beauty",
    "Health", "Education", "Entertainment", "Social", "Investment", "Others",
    "Salary", "Bonus", "Business", "Debt", "Charity", "Savings",
})

# theme_id → mô tả prompt + label/type gợi ý
RECORD_THEMES: dict[str, dict] = {
    "cafe_social": {
        "hint": "Đi chơi/xã giao: đi cf với bạn, hẹn hò, date crush, uống cafe với nhóm. Entertainment expense.",
        "labels": "Entertainment",
        "type": "expense",
    },
    "cafe_buy_food": {
        "hint": "MUA mang về/order: mua cf, trà sữa takeaway, GrabFood. Food expense. KHÔNG 'với bạn/hẹn hò'.",
        "labels": "Food",
        "type": "expense",
    },
    "income_salary": {
        "hint": "Lương: lương tháng về, nhận lương, salary, lg về. Salary income.",
        "labels": "Salary",
        "type": "income",
    },
    "income_bonus": {
        "hint": "Thưởng, lì xì, red envelope. Bonus income.",
        "labels": "Bonus",
        "type": "income",
    },
    "income_business": {
        "hint": "Freelance, bán hàng online, hoa hồng, side hustle. Business income.",
        "labels": "Business",
        "type": "income",
    },
    "income_transfer": {
        "hint": "Nhận tiền từ người thân: mẹ/bố/chị chuyển khoản, gửi tiền về. Salary hoặc Others income. KHÔNG mua/chi.",
        "labels": "Salary,Others",
        "type": "income",
    },
    "income_refund": {
        "hint": "Hoàn tiền, refund Shopee/Lazada, trả lại tiền hàng. Others income.",
        "labels": "Others",
        "type": "income",
    },
    "record_not_action": {
        "hint": "MÔ TẢ đã chi (Record): mới tiêu, hôm nay hết, chi grab, mua đồ — KHÔNG phải lệnh đặt hạn mức/thống kê.",
        "labels": "Food,Transport,Shopping,Housing,Entertainment",
        "type": "expense",
    },
    "sub_entertainment": {
        "hint": "Subscription chi phí: Netflix, Spotify, YouTube Premium gia hạn. Entertainment EXPENSE (không phải thu nhập).",
        "labels": "Entertainment",
        "type": "expense",
    },
    "record_housing": {
        "hint": "Tiền nhà, KTX, điện nước, wifi. Housing expense.",
        "labels": "Housing",
        "type": "expense",
    },
    "record_transport": {
        "hint": "Grab, xăng, vé xe, gửi xe. Transport expense.",
        "labels": "Transport",
        "type": "expense",
    },
    "record_food_general": {
        "hint": "Ăn uống thường: cơm, phở, ăn trưa, ShopeeFood. Food expense.",
        "labels": "Food",
        "type": "expense",
    },
    "social_gift": {
        "hint": "Quà tặng, đám cưới, sinh nhật bạn. Social expense.",
        "labels": "Social",
        "type": "expense",
    },
    "entertainment_drink": {
        "hint": "Nhậu, bida, karaoke, bar. Entertainment expense.",
        "labels": "Entertainment",
        "type": "expense",
    },
    "health_education": {
        "hint": "Khám, thuốc, học phí, sách. Health/Education expense.",
        "labels": "Health,Education",
        "type": "expense",
    },
    "shopping_essentials": {
        "hint": "Shopee quần áo, đồ dùng KTX. Shopping/Essentials expense.",
        "labels": "Shopping,Essentials",
        "type": "expense",
    },
}

QUOTAS_10K: dict[str, int] = {
    "cafe_social": 1000,
    "cafe_buy_food": 700,
    "income_salary": 900,
    "income_bonus": 700,
    "income_business": 700,
    "income_transfer": 900,
    "income_refund": 700,
    "record_not_action": 900,
    "sub_entertainment": 700,
    "record_housing": 550,
    "record_transport": 700,
    "record_food_general": 700,
    "social_gift": 500,
    "entertainment_drink": 550,
    "health_education": 500,
    "shopping_essentials": 500,
}

# Mẫu vàng — case fail test + biên keyword cũ
REGRESSION_SEEDS: list[dict] = [
    {"text": "cf date với crush 80k", "label": "Entertainment", "type": "expense", "is_money": 1},
    {"text": "mẹ chuyển khoản 500k", "label": "Salary", "type": "income", "is_money": 1},
    {"text": "hoàn tiền shopee 89k", "label": "Others", "type": "income", "is_money": 1},
    {"text": "Netflix tháng 109k", "label": "Entertainment", "type": "expense", "is_money": 1},
    {"text": "Đi cà phê với bạn 19k", "label": "Entertainment", "type": "expense", "is_money": 1},
    {"text": "Mua cafe sữa đá 25k", "label": "Food", "type": "expense", "is_money": 1},
    {"text": "Mới tiêu 2tr", "label": "Others", "type": "expense", "is_money": 1},
    {"text": "nhận lương 15 củ", "label": "Salary", "type": "income", "is_money": 1},
    {"text": "bố gửi tiền ăn 2tr", "label": "Salary", "type": "income", "is_money": 1},
    {"text": "hoàn tiền đơn hàng 120k", "label": "Others", "type": "income", "is_money": 1},
    {"text": "gia hạn Netflix 109k", "label": "Entertainment", "type": "expense", "is_money": 1},
    {"text": "date quán cf view đẹp 150k", "label": "Entertainment", "type": "expense", "is_money": 1},
]
