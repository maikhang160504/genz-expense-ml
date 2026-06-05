"""Tests for Vietnamese time range parsing."""
from datetime import datetime, timezone, timedelta

import pytest

from src.nlu.time_parser import parse_time_range

VN_TZ = timezone(timedelta(hours=7))
# Sunday 31/05/2026
ANCHOR = datetime(2026, 5, 31, 15, 30, tzinfo=VN_TZ)


@pytest.mark.parametrize(
    "text,granularity",
    [
        ("Tổng chi tiêu tuần này", "week"),
        ("Tuần này tiêu hết bao nhiêu rồi", "week"),
        ("Tháng này tiêu bao nhiêu", "month"),
        ("Hôm nay chi bao nhiêu", "day"),
        ("Hôm qua tiêu gì", "yesterday"),
        ("Tuần trước sao rồi", "last_week"),
        ("Tháng trước tiêu nhiều không", "last_month"),
        ("Quý này tổng chi", "quarter"),
        ("7 ngày qua tiêu bao nhiêu", "rolling_7d"),
    ],
)
def test_parse_time_range_detects_granularity(text, granularity):
    result = parse_time_range(text, now=ANCHOR)
    assert result is not None
    assert result["granularity"] == granularity
    assert result["from"]
    assert result["to"]
    assert result["period_label"]


def test_week_range_monday_to_sunday_anchor():
    result = parse_time_range("tuần này", now=ANCHOR)
    assert result is not None
    assert result["from"].startswith("2026-05-25")
    assert "31/05" in result["period_label"] or "25/05" in result["period_label"]


def test_time_slots_from_ner():
    result = parse_time_range("tổng chi tiêu", time_slots=["tuần này"], now=ANCHOR)
    assert result is not None
    assert result["granularity"] == "week"


def test_unknown_returns_none():
    assert parse_time_range("xóa giao dịch vừa rồi", now=ANCHOR) is None
