"""Parse Vietnamese time phrases into a concrete date range."""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta, timezone

VN_TZ = timezone(timedelta(hours=7))

_GRANULARITY_LABELS = {
    "day": "Hôm nay",
    "yesterday": "Hôm qua",
    "week": "Tuần này",
    "last_week": "Tuần trước",
    "month": "Tháng này",
    "last_month": "Tháng trước",
    "quarter": "Quý này",
    "rolling_7d": "7 ngày qua",
    "rolling_30d": "30 ngày qua",
}


def _norm(s: str) -> str:
    nfd = unicodedata.normalize("NFD", (s or "").lower().strip())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def _as_vn(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=VN_TZ)
    return dt.astimezone(VN_TZ)


def _start_of_day(dt: datetime) -> datetime:
    dt = _as_vn(dt)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _end_of_day(dt: datetime) -> datetime:
    dt = _as_vn(dt)
    return dt.replace(hour=23, minute=59, second=59, microsecond=0)


def _monday_of(dt: datetime) -> datetime:
    dt = _start_of_day(dt)
    return dt - timedelta(days=dt.weekday())


def _fmt_short(dt: datetime) -> str:
    return dt.strftime("%d/%m")


def _fmt_iso(dt: datetime) -> str:
    return _as_vn(dt).isoformat()


def _build_result(
    *,
    granularity: str,
    start: datetime,
    end: datetime,
    label_prefix: str | None = None,
) -> dict:
    prefix = label_prefix or _GRANULARITY_LABELS.get(granularity, granularity)
    same_day = start.date() == end.date()
    if same_day:
        period_label = f"{prefix} ({_fmt_short(start)}/{start.year})"
    else:
        period_label = f"{prefix} ({_fmt_short(start)} - {_fmt_short(end)}/{end.year})"
    return {
        "period_label": period_label,
        "from": _fmt_iso(start),
        "to": _fmt_iso(end),
        "granularity": granularity,
    }


def _match_range(norm_text: str, now: datetime) -> dict | None:
    today = _start_of_day(now)

    if re.search(r"\bhom nay\b", norm_text):
        return _build_result(granularity="day", start=today, end=_end_of_day(now))

    if re.search(r"\bhom qua\b", norm_text):
        y = today - timedelta(days=1)
        return _build_result(granularity="yesterday", start=y, end=_end_of_day(y), label_prefix="Hôm qua")

    if re.search(r"\btuan truoc\b|\btuan vua roi\b", norm_text):
        last_mon = _monday_of(today) - timedelta(days=7)
        last_sun = _end_of_day(last_mon + timedelta(days=6))
        return _build_result(
            granularity="last_week",
            start=last_mon,
            end=last_sun,
            label_prefix="Tuần trước",
        )

    if re.search(r"\btuan nay\b|\btuan nay sao\b", norm_text):
        start = _monday_of(today)
        return _build_result(granularity="week", start=start, end=_end_of_day(now))

    if re.search(r"\bthang truoc\b", norm_text):
        first_this = today.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        start = last_prev.replace(day=1)
        return _build_result(
            granularity="last_month",
            start=_start_of_day(start),
            end=_end_of_day(last_prev),
            label_prefix="Tháng trước",
        )

    if re.search(r"\bthang nay\b", norm_text):
        start = today.replace(day=1)
        return _build_result(granularity="month", start=start, end=_end_of_day(now))

    if re.search(r"\bquy nay\b", norm_text):
        q = (today.month - 1) // 3
        start_month = q * 3 + 1
        start = today.replace(month=start_month, day=1)
        return _build_result(granularity="quarter", start=start, end=_end_of_day(now))

    m = re.search(r"\b(\d+)\s*ngay\s*qua\b", norm_text)
    if m:
        days = int(m.group(1))
        start = today - timedelta(days=max(days - 1, 0))
        gran = "rolling_7d" if days == 7 else "rolling_30d" if days == 30 else f"rolling_{days}d"
        return _build_result(
            granularity=gran,
            start=start,
            end=_end_of_day(now),
            label_prefix=f"{days} ngày qua",
        )

    if re.search(r"\b7 ngay\b|\b7 ngay qua\b", norm_text):
        start = today - timedelta(days=6)
        return _build_result(granularity="rolling_7d", start=start, end=_end_of_day(now), label_prefix="7 ngày qua")

    if re.search(r"\b30 ngay\b|\b30 ngay qua\b", norm_text):
        start = today - timedelta(days=29)
        return _build_result(granularity="rolling_30d", start=start, end=_end_of_day(now), label_prefix="30 ngày qua")

    return None


def parse_time_range(
    text: str,
    time_slots: list[str] | None = None,
    *,
    now: datetime | None = None,
) -> dict | None:
    """Return ``{ period_label, from, to, granularity }`` or ``None``."""
    anchor = _as_vn(now or datetime.now(VN_TZ))
    parts = [text or ""]
    if time_slots:
        parts.extend(str(s) for s in time_slots if s)
    combined = " ".join(parts)
    norm = _norm(combined)
    if not norm:
        return None
    return _match_range(norm, anchor)
