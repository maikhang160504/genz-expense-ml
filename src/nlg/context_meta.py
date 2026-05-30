"""
Ngữ cảnh (context_meta) gắn kèm prompt NLG.

- ``build_context_metadata``: logic theo profile (dùng API / triển khai thật khi có dữ liệu).
- ``build_mock_context_metadata``: dữ liệu ngẫu nhiên cho demo / dev (không phụ thuộc DB).
"""
from __future__ import annotations

import random
from datetime import datetime
from typing import Any


def _get_time_of_day(hour: int) -> str:
    if 5 <= hour < 11:
        return "sáng_sớm"
    if 11 <= hour < 14:
        return "buổi_trưa"
    if 14 <= hour < 19:
        return "chiều_tối"
    return "đêm_muộn"


def _get_days_to_payday(day: int) -> int:
    """Tính số ngày đến kỳ lương gần nhất (ngày 15 hoặc cuối tháng 30)."""
    if day <= 15:
        return 15 - day
    return max(0, 30 - day)


def _get_wallet_health(budget_remain: float | None, budget_total: float | None) -> str:
    if budget_remain is None or not budget_total:
        return "không_rõ"
    ratio = budget_remain / max(budget_total, 1)
    if ratio >= 0.5:
        return "an_toan"
    if ratio >= 0.2:
        return "can_than"
    return "bao_dong"


def _build_environment_fields(profile: dict[str, Any]) -> dict[str, Any]:
    """Trả về các trường môi trường tính từ thời gian thực + profile."""
    now = datetime.now()
    hour = now.hour
    day = now.day
    return {
        "time_of_day": _get_time_of_day(hour),
        "day_of_month": day,
        "days_to_payday": _get_days_to_payday(day),
        "weather": profile.get("weather", "không_rõ"),
    }


def _pick_category_stats(
    category: str | None,
    profile: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    """Chọn stats phù hợp từ profile.category_stats.

    Ưu tiên:
    1. Category trùng với nhận dạng từ encoder.
    2. Ngẫu nhiên từ category khác trong gói (nếu category không có trong gói).
    3. Dict rỗng nếu không có category_stats.

    Trả về (picked_category, stats_dict).
    """
    cat_stats: dict[str, Any] = profile.get("category_stats") or {}
    if not cat_stats:
        return category, {}

    if category and category in cat_stats:
        return category, cat_stats[category]

    # Fallback: chọn ngẫu nhiên từ các category có trong gói
    keys = list(cat_stats.keys())
    picked = random.choice(keys)
    return picked, cat_stats[picked]


def build_context_metadata(nlu_result: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """So khớp số tiền / ngân sách / tần suất (luồng production khi có profile).

    Ưu tiên dữ liệu theo category đã nhận dạng (encoder → category_stats).
    Nếu category không có trong gói hoặc không có category_stats, chọn ngẫu nhiên.
    """
    env = _build_environment_fields(profile)
    intent = nlu_result.get("intent")
    spent_today = profile.get("spent_today", 0)
    spent_week = profile.get("spent_week", 0)
    spent_month = profile.get("spent_month", 0)

    if intent == "Record":
        amount = nlu_result.get("amount")
        budget_remain = profile.get("budget_remain")
        budget_total = profile.get("budget_total")
        threshold = profile.get("amount_threshold")

        # Lấy stats theo category đã nhận dạng (ưu tiên) hoặc random
        identified_category = nlu_result.get("category")
        picked_cat, cat = _pick_category_stats(identified_category, profile)

        frequency_week = cat.get("frequency_week") or profile.get("frequency_week") or 0
        avg_amount = cat.get("avg_amount") or profile.get("avg_amount") or 0
        category_pct = cat.get("pct") or 0

        is_triggered = False
        trigger_type: str | None = None

        if amount is not None and threshold is not None and amount > threshold:
            is_triggered = True
            trigger_type = "WARNING_AMOUNT"

        if amount is not None and budget_remain is not None and amount > budget_remain:
            is_triggered = True
            trigger_type = trigger_type or "OVER_BUDGET"

        if budget_remain is not None and budget_total:
            if budget_remain / max(budget_total, 1) < 0.2:
                is_triggered = True
                trigger_type = trigger_type or "WARNING_BUDGET"

        if frequency_week > 3:
            is_triggered = True
            trigger_type = trigger_type or "WARNING_FREQUENCY"

        wallet_health = _get_wallet_health(budget_remain, budget_total)

        cat_label = picked_cat or identified_category or "chi tiêu"
        
        # Build interesting, conditional facts dynamically
        interesting_facts = []

        # 1. Budget warnings
        if budget_remain is not None and budget_total and budget_total > 0:
            ratio = budget_remain / budget_total
            if ratio < 0.2:
                interesting_facts.append(f"Cảnh báo báo động: Ngân sách tháng này sắp cạn kiệt, chỉ còn lại {budget_remain:,.0f}đ ({ratio*100:.1f}%).")
            elif ratio < 0.5:
                interesting_facts.append(f"Cảnh báo nhẹ: Ngân sách đã tiêu quá nửa, còn lại {budget_remain:,.0f}đ.")

        # 2. Category specific high spending
        if category_pct > 30:
            interesting_facts.append(f"Danh mục {cat_label} đang chiếm tỷ trọng cực kỳ lớn: {category_pct}% tổng chi tiêu cả tháng.")
        elif category_pct > 15:
            interesting_facts.append(f"Danh mục {cat_label} chiếm tỷ trọng khá cao: {category_pct}% tổng chi tiêu cả tháng.")

        # 3. High frequency spending
        if frequency_week > 4:
            interesting_facts.append(f"Bạn đã chi cho {cat_label} tận {frequency_week} lần chỉ trong tuần này.")

        # 4. Payday countdown
        if env['days_to_payday'] == 0:
            interesting_facts.append("Hôm nay là ngày nhận lương 💸.")
        elif env['days_to_payday'] <= 3:
            interesting_facts.append(f"Chỉ còn {env['days_to_payday']} ngày nữa là tới ngày nhận lương.")

        # 5. Weather / Temperature / Extreme conditions
        weather_str = profile.get("weather", "không_rõ")
        if weather_str != "không_rõ" and any(k in weather_str.lower() for k in ["mưa", "bão", "ngập", "nóng", "lạnh", "tết", "lễ"]):
            interesting_facts.append(f"Thời tiết hôm nay có {weather_str}.")

        # 6. Special days (Birthday, Holiday, etc.)
        special_day = profile.get("special_day") or profile.get("special_event")
        if special_day:
            interesting_facts.append(f"Hôm nay là {special_day} 🎉.")

        # Assign historical_fact based on triggered conditions or fallback
        if not interesting_facts:
            if avg_amount > 0:
                historical_fact = f"Trung bình mỗi lần chi danh mục {cat_label}: {avg_amount:,.0f}đ."
            else:
                historical_fact = f"Hôm nay ghi nhận giao dịch thuộc danh mục {cat_label}."
        else:
            historical_fact = " ".join(interesting_facts[:2])

        return {
            "source": "profile",
            "is_triggered": is_triggered,
            "type": trigger_type,
            "spent_today": spent_today,
            "spent_week": spent_week,
            "spent_month": spent_month,
            **env,
            "wallet_health": wallet_health,
            "historical_fact": historical_fact,
            "message_data": {
                "remaining": budget_remain,
                "frequency_week": frequency_week,
                "avg_amount": avg_amount,
                "category": cat_label,
                "category_pct": category_pct,
            },
        }

    if intent == "Action":
        new_value = nlu_result.get("value")
        old_value = profile.get("old_value")
        is_triggered = False
        trigger_type = None
        if new_value is not None and old_value:
            diff_ratio = abs(new_value - old_value) / max(old_value, 1)
            if diff_ratio >= 0.5:
                is_triggered = True
                trigger_type = "WARNING_CHANGE"
        return {
            "source": "profile",
            "is_triggered": is_triggered,
            "type": trigger_type,
            "spent_today": spent_today,
            "spent_week": spent_week,
            "spent_month": spent_month,
            **env,
            "wallet_health": "không_rõ",
            "historical_fact": None,
            "message_data": {
                "old_value": old_value,
                "new_value": new_value,
            },
        }

    return {
        "source": "profile",
        "is_triggered": False,
        "type": None,
        "spent_today": spent_today,
        "spent_week": spent_week,
        "spent_month": spent_month,
        **env,
        "wallet_health": "không_rõ",
        "historical_fact": None,
        "message_data": None,
    }


def build_mock_context_metadata(
    nlu_result: dict[str, Any],
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Bối cảnh giả lập ngẫu nhiên — chỉ dùng demo; production nên dùng ``build_context_metadata``."""
    rng = random.Random(seed) if seed is not None else random.Random()

    now = datetime.now()
    time_of_day = _get_time_of_day(now.hour)
    day_of_month = now.day
    days_to_payday = _get_days_to_payday(day_of_month)

    weathers = [
        "Mưa bão ngập lụt ☔",
        "Nắng cháy da đầu ☀️",
        "Lạnh teo buzi 🥶",
        "Mát mẻ dễ chịu 🌤️",
        "Oi bức muốn xỉu 😮‍💨",
    ]
    wallet_healths = ["an_toan", "can_than", "bao_dong"]
    historical_facts = [
        "Tuần này chi nhiều hơn tuần trước khoảng 12%.",
        "Danh mục ăn uống đang chiếm >35% tổng chi tháng.",
        f"Còn {days_to_payday} ngày nữa mới tới kỳ lương.",
        "Tháng này đang over budget 20% danh mục mua sắm.",
        "Ví vừa tinh tinh sau kỳ lương hôm qua 💸",
        "Đây là lần chi thứ 4 danh mục này tuần này.",
        "Trung bình mỗi ngày bạn chi khoảng 150.000đ.",
    ]
    trigger_types = ["WARNING_AMOUNT", "WARNING_BUDGET", "TIP", None, None]

    wallet_health = rng.choice(wallet_healths)
    is_triggered = wallet_health == "bao_dong" or rng.choice([True, False, False])

    base: dict[str, Any] = {
        "source": "mock",
        "is_triggered": is_triggered,
        "type": rng.choice(trigger_types) if is_triggered else None,
        "time_of_day": time_of_day,
        "day_of_month": day_of_month,
        "days_to_payday": days_to_payday,
        "weather": rng.choice(weathers),
        "wallet_health": wallet_health,
        "historical_fact": rng.choice(historical_facts),
        "message_data": {
            "remaining": rng.randrange(20_000, 5_000_000, 10_000),
            "month_spend_hint": rng.randrange(500_000, 15_000_000, 50_000),
        },
    }

    if nlu_result.get("intent") == "Action":
        base["message_data"] = {
            "old_value": rng.randrange(100_000, 5_000_000, 25_000),
            "new_value": rng.randrange(100_000, 5_000_000, 25_000),
            "mock_note": "Giá trị cũ/mới chỉ để minh họa prompt.",
        }

    return base
