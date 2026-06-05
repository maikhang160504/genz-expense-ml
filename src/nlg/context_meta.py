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


def _is_income_record(nlu_result: dict[str, Any]) -> bool:
    if nlu_result.get("record_type") == "Income":
        return True
    return nlu_result.get("is_expense") is False


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

    return category, {}


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
    is_group = profile.get("wallet_type") == "group"

    def add_profile_fields(res: dict[str, Any]) -> dict[str, Any]:
        if "wallet_type" in profile:
            res["wallet_type"] = profile["wallet_type"]
        if "member_count" in profile:
            res["member_count"] = profile["member_count"]
        return res

    if intent == "Record":
        amount = nlu_result.get("amount")
        identified_category = nlu_result.get("category")
        picked_cat, cat = _pick_category_stats(identified_category, profile)
        cat_label = picked_cat or identified_category or "khác"

        if _is_income_record(nlu_result):
            income_type = nlu_result.get("income_type")
            interesting_facts: list[str] = []
            if env["days_to_payday"] == 0:
                interesting_facts.append("Hôm nay là ngày nhận lương 💸.")
            elif env["days_to_payday"] <= 3:
                interesting_facts.append(
                    f"Còn {env['days_to_payday']} ngày nữa tới kỳ lương — khoản thu này đúng hợp lý."
                )
            if amount is not None:
                interesting_facts.append(
                    f"Ghi nhận thu nhập {amount:,.0f}đ"
                    f"{f' ({income_type})' if income_type else ''} — {cat_label}."
                )
            historical_fact = " ".join(interesting_facts[:2]) if interesting_facts else None
            if is_group and historical_fact:
                historical_fact = historical_fact.replace("Bạn", "Nhóm").replace("bạn", "nhóm")

            return add_profile_fields({
                "record_type": "Income",
                "is_triggered": False,
                "type": None,
                **env,
                "wallet_health": "không_rõ",
                "historical_fact": historical_fact,
                "message_data": {
                    "category": cat_label,
                    **({"income_type": income_type} if income_type else {}),
                },
            })

        budget_remain = profile.get("budget_remain")
        budget_total = profile.get("budget_total")
        threshold = profile.get("amount_threshold")
        frequency_week = cat.get("frequency_week") or profile.get("frequency_week") or 0
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

        interesting_facts = []

        if budget_remain is not None and budget_total and budget_total > 0:
            ratio = budget_remain / budget_total
            if ratio < 0.2:
                interesting_facts.append(
                    f"Cảnh báo báo động: Ngân sách tháng này sắp cạn kiệt, chỉ còn lại {budget_remain:,.0f}đ ({ratio*100:.1f}%)."
                )
            elif ratio < 0.5:
                interesting_facts.append(
                    f"Cảnh báo nhẹ: Ngân sách đã tiêu quá nửa, còn lại {budget_remain:,.0f}đ."
                )

        if category_pct > 30:
            interesting_facts.append(
                f"Danh mục {cat_label} đang chiếm tỷ trọng cực kỳ lớn: {category_pct}% tổng chi tiêu cả tháng."
            )
        elif category_pct > 15:
            interesting_facts.append(
                f"Danh mục {cat_label} chiếm tỷ trọng khá cao: {category_pct}% tổng chi tiêu cả tháng."
            )

        if frequency_week > 4:
            interesting_facts.append(
                f"Bạn đã chi cho {cat_label} tận {frequency_week} lần chỉ trong tuần này."
            )

        if env["days_to_payday"] == 0:
            interesting_facts.append("Hôm nay là ngày nhận lương 💸.")
        elif env["days_to_payday"] <= 3:
            interesting_facts.append(f"Chỉ còn {env['days_to_payday']} ngày nữa là tới ngày nhận lương.")

        weather_str = profile.get("weather", "không_rõ")
        if weather_str != "không_rõ" and any(
            k in weather_str.lower() for k in ("mưa", "bão", "ngập", "nóng", "lạnh", "tết", "lễ")
        ):
            interesting_facts.append(f"Thời tiết hôm nay có {weather_str}.")

        special_day = profile.get("special_day") or profile.get("special_event")
        if special_day:
            interesting_facts.append(f"Hôm nay là {special_day} 🎉.")

        if interesting_facts:
            historical_fact = " ".join(interesting_facts[:2])
        elif is_triggered and amount is not None:
            historical_fact = f"Chi tiêu {amount:,.0f}đ — {cat_label}."
        else:
            historical_fact = None

        if is_group and historical_fact:
            historical_fact = historical_fact.replace("Bạn", "Nhóm").replace("bạn", "nhóm")

        message_data: dict[str, Any] = {"category": cat_label}
        if is_triggered or (budget_remain is not None and budget_total):
            if budget_remain is not None:
                message_data["remaining"] = budget_remain
            if frequency_week > 0:
                message_data["frequency_week"] = frequency_week
            if category_pct > 0:
                message_data["category_pct"] = category_pct

        return add_profile_fields({
            "record_type": "Expense",
            "is_triggered": is_triggered,
            "type": trigger_type,
            **env,
            "wallet_health": wallet_health,
            "historical_fact": historical_fact,
            "message_data": message_data,
        })

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
        action_facts: dict[str, Any] = profile.get("action_facts") or {}
        if old_value is not None and "old_value" not in action_facts:
            action_facts["old_value"] = old_value
        if new_value is not None and "new_value" not in action_facts:
            action_facts["new_value"] = new_value
        if spent_today or spent_week or spent_month:
            if "spent_today" not in action_facts:
                action_facts["spent_today"] = spent_today
            if "spent_week" not in action_facts:
                action_facts["spent_week"] = spent_week
            if "spent_month" not in action_facts:
                action_facts["spent_month"] = spent_month
        return add_profile_fields({
            "is_triggered": is_triggered,
            "type": trigger_type,
            **env,
            "action_facts": action_facts or None,
        })

    return add_profile_fields({**env})


def filter_context_metadata_for_prompt(
    ctx: dict[str, Any],
    nlu_result: dict[str, Any],
    user_text: str = "",
) -> dict[str, Any]:
    """Bỏ trường thừa / không liên quan câu user trước khi đưa vào NLG."""
    intent = nlu_result.get("intent")
    slim: dict[str, Any] = {}
    utter = (user_text or nlu_result.get("text") or "").strip().lower()

    # 1. Group wallet context handling
    is_group = (ctx.get("wallet_type") == "group" or ctx.get("is_shared_wallet") is True)
    if is_group:
        slim["is_shared_wallet"] = True

    # 2. Extract Priority 3 fields (lowest, only if mentioned by user)
    weather = ctx.get("weather")
    if weather and weather != "không_rõ":
        if any(k in utter for k in ("thời tiết", "mưa", "nắng", "trời", "weather")):
            slim["weather"] = weather
    
    days_to_payday = ctx.get("days_to_payday")
    if days_to_payday is not None:
        if any(k in utter for k in ("lương", "payday", "tiền về", "nhận lương")):
            slim["days_to_payday"] = days_to_payday

    time_of_day = ctx.get("time_of_day")
    if time_of_day is not None:
        if any(k in utter for k in ("giờ", "buổi", "sáng", "trưa", "chiều", "tối", "đêm", "ngày")):
            slim["time_of_day"] = time_of_day

    spent_last_month = ctx.get("spent_last_month")
    if spent_last_month is not None:
        if any(k in utter for k in ("tháng trước", "tháng ngoái", "last month", "so với tháng trước", "so sánh tháng trước")):
            slim["spent_last_month"] = spent_last_month

    # 3. Determine candidates for P1 and P2 fields
    p1_p2_candidates = {}

    health = ctx.get("wallet_health")
    if health and health != "không_rõ":
        if intent != "Record" or not _is_income_record(nlu_result):
            p1_p2_candidates["wallet_health"] = (health, 2)

    if intent == "Record":
        rt = nlu_result.get("record_type") or ctx.get("record_type")
        if rt:
            p1_p2_candidates["record_type"] = (rt, 2)
        
        if not _is_income_record(nlu_result) and ctx.get("is_triggered"):
            p1_p2_candidates["is_triggered"] = (True, 2)
            if ctx.get("type"):
                p1_p2_candidates["type"] = (ctx["type"], 2)
        
        hf = ctx.get("historical_fact")
        if hf:
            if is_group:
                hf = hf.replace("Bạn", "Nhóm").replace("bạn", "nhóm")
            p1_p2_candidates["historical_fact"] = (hf, 1)
            
        md = ctx.get("message_data") or {}
        if md:
            p1_p2_candidates["message_data"] = (md, 1)

    elif intent == "Action":
        if ctx.get("is_triggered"):
            p1_p2_candidates["is_triggered"] = (True, 2)
            if ctx.get("type"):
                p1_p2_candidates["type"] = (ctx["type"], 2)
                
        facts = ctx.get("action_facts")
        if facts:
            p1_p2_candidates["action_facts"] = (facts, 1)
            
        action_type = nlu_result.get("action_type")
        if action_type:
            p1_p2_candidates["action_type"] = (action_type, 1)

    # 4. Filter to keep at most 2 metadata field types from P1 and P2
    type_groups = []
    
    if "wallet_health" in p1_p2_candidates:
        type_groups.append({
            "keys": ["wallet_health"],
            "score": 2,
            "order": 1
        })
    if "is_triggered" in p1_p2_candidates or "type" in p1_p2_candidates:
        keys = []
        if "is_triggered" in p1_p2_candidates:
            keys.append("is_triggered")
        if "type" in p1_p2_candidates:
            keys.append("type")
        type_groups.append({
            "keys": keys,
            "score": 2,
            "order": 0
        })
    if "record_type" in p1_p2_candidates:
        type_groups.append({
            "keys": ["record_type"],
            "score": 2,
            "order": 2
        })
    if "historical_fact" in p1_p2_candidates:
        type_groups.append({
            "keys": ["historical_fact"],
            "score": 1,
            "order": 3
        })
    if "message_data" in p1_p2_candidates:
        type_groups.append({
            "keys": ["message_data"],
            "score": 1,
            "order": 4
        })
    if "action_facts" in p1_p2_candidates:
        type_groups.append({
            "keys": ["action_facts"],
            "score": 1,
            "order": 5
        })
    if "action_type" in p1_p2_candidates:
        type_groups.append({
            "keys": ["action_type"],
            "score": 1,
            "order": 6
        })

    type_groups.sort(key=lambda g: (-g["score"], g["order"]))
    kept_groups = type_groups[:2]
    
    for g in kept_groups:
        for k in g["keys"]:
            slim[k] = p1_p2_candidates[k][0]

    # 5. Essential NLU fields from nlu_result
    if intent == "Record":
        amount = nlu_result.get("amount")
        if amount is not None:
            slim["record_amount"] = amount
        item = (nlu_result.get("item") or "").strip()
        if item:
            slim["item"] = item
        cat = nlu_result.get("category")
        if cat:
            slim["category"] = cat

    content_keys = {
        "time_of_day", "days_to_payday", "weather", "wallet_health", "record_type",
        "is_triggered", "type", "historical_fact", "message_data", "action_facts", "action_type",
        "spent_last_month"
    }
    has_content = any(k in slim for k in content_keys)
    if not has_content:
        return {}

    return slim


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
