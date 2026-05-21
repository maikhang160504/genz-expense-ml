"""
Ngữ cảnh (context_meta) gắn kèm prompt NLG.

- ``build_context_metadata``: logic theo profile (dùng API / triển khai thật khi có dữ liệu).
- ``build_mock_context_metadata``: dữ liệu ngẫu nhiên cho demo / dev (không phụ thuộc DB).
"""
from __future__ import annotations

import random
from typing import Any


def build_context_metadata(nlu_result: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """So khớp số tiền / ngân sách / tần suất (luồng production khi có profile)."""
    intent = nlu_result.get("intent")
    if intent == "Record":
        amount = nlu_result.get("amount")
        budget_remain = profile.get("budget_remain")
        budget_total = profile.get("budget_total")
        frequency_week = profile.get("frequency_week")
        avg_amount = profile.get("avg_amount")
        threshold = profile.get("amount_threshold")
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

        if frequency_week is not None and frequency_week > 3:
            is_triggered = True
            trigger_type = trigger_type or "WARNING_FREQUENCY"

        return {
            "source": "profile",
            "is_triggered": is_triggered,
            "type": trigger_type,
            "message_data": {
                "remaining": budget_remain,
                "frequency_week": frequency_week,
                "avg_amount": avg_amount,
            },
        }

    if intent == "Action":
        new_value = nlu_result.get("value")
        old_value = profile.get("old_value")
        is_triggered = False
        trigger_type: str | None = None
        if new_value is not None and old_value:
            diff_ratio = abs(new_value - old_value) / max(old_value, 1)
            if diff_ratio >= 0.5:
                is_triggered = True
                trigger_type = "WARNING_CHANGE"
        return {
            "source": "profile",
            "is_triggered": is_triggered,
            "type": trigger_type,
            "message_data": {
                "old_value": old_value,
                "new_value": new_value,
            },
        }

    return {
        "source": "profile",
        "is_triggered": False,
        "type": None,
        "message_data": None,
    }


def build_mock_context_metadata(
    nlu_result: dict[str, Any],
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Bối cảnh giả lập ngẫu nhiên — chỉ dùng demo; production nên dùng ``build_context_metadata`` có điều kiện."""
    rng = random.Random(seed) if seed is not None else random.Random()

    tips = [
        "Tuần này chi nhiều hơn tuần trước khoảng 12%.",
        "Danh mục ăn uống đang chiếm >35% tổng chi.",
        "Còn 5 ngày cuối tháng, ngân sách còn dư khá ổn.",
        "Gợi ý: thử ghi thêm khoản cố định để dự báo chính xác hơn.",
    ]
    trigger_types = ["WARNING_AMOUNT", "WARNING_BUDGET", "TIP", None, None]

    base: dict[str, Any] = {
        "source": "mock",
        "is_triggered": rng.choice([True, False, False]),
        "type": rng.choice(trigger_types),
        "message_data": {
            "remaining": rng.randrange(20_000, 5_000_000, 10_000),
            "month_spend_hint": rng.randrange(500_000, 15_000_000, 50_000),
            "tip": rng.choice(tips),
        },
    }

    if nlu_result.get("intent") == "Action":
        base["message_data"] = {
            "old_value": rng.randrange(100_000, 5_000_000, 25_000),
            "new_value": rng.randrange(100_000, 5_000_000, 25_000),
            "mock_note": "Giá trị cũ/mới chỉ để minh họa prompt.",
        }

    return base
