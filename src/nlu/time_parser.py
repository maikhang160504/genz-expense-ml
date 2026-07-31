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
    "year": "Năm nay",
    "weekend": "Cuối tuần",
    "weekday": "Ngày thường",
    "holiday": "Ngày lễ",
    "monday": "Thứ hai",
    "tuesday": "Thứ ba",
    "wednesday": "Thứ tư",
    "thursday": "Thứ năm",
    "friday": "Thứ sáu",
    "saturday": "Thứ bảy",
    "sunday": "Chủ nhật",
}


def _norm(s: str) -> str:
    s_clean = (s or "").lower().replace("đ", "d").replace("Đ", "d").strip()
    nfd = unicodedata.normalize("NFD", s_clean)
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

    # 0. Custom date ranges: "tu ngay 1 den ngay 10 thang nay", "tu 1 den 10 thang nay", "tu 01/07 den 10/07"
    m_range = re.search(
        r"(?:tu\s+)?(?:ngay\s+)?(\d{1,2})\s+(?:den|toi)\s+(?:ngay\s+)?(\d{1,2})(?:\s+(?:thang\s+(\d{1,2})|thang\s+(nay|truoc)))?",
        norm_text
    )
    if m_range and not re.search(r"\d{1,2}/\d{1,2}", norm_text):
        d1 = int(m_range.group(1))
        d2 = int(m_range.group(2))
        month_str = m_range.group(3)
        month_word = m_range.group(4)

        y = today.year
        if month_str:
            m = int(month_str)
        elif month_word == "truoc":
            m = today.month - 1
            if m == 0:
                m = 12
                y -= 1
        else:
            m = today.month

        try:
            start_dt = _start_of_day(today.replace(year=y, month=m, day=d1))
            end_dt = _end_of_day(today.replace(year=y, month=m, day=d2))
            label = f"Từ {d1:02d}/{m:02d} đến {d2:02d}/{m:02d}/{y}"
            return _build_result(
                granularity="custom",
                start=start_dt,
                end=end_dt,
                label_prefix=label
            )
        except ValueError:
            pass

    m_slash = re.search(
        r"(?:tu\s+)?(?:ngay\s+)?(\d{1,2})[\/\-](\d{1,2})(?:[\/\-](\d{4}))?\s+(?:den|toi)\s+(?:ngay\s+)?(\d{1,2})[\/\-](\d{1,2})(?:[\/\-](\d{4}))?",
        norm_text
    )
    if m_slash:
        d1, m1 = int(m_slash.group(1)), int(m_slash.group(2))
        y1 = int(m_slash.group(3)) if m_slash.group(3) else today.year
        d2, m2 = int(m_slash.group(4)), int(m_slash.group(5))
        y2 = int(m_slash.group(6)) if m_slash.group(6) else today.year
        try:
            start_dt = _start_of_day(today.replace(year=y1, month=m1, day=d1))
            end_dt = _end_of_day(today.replace(year=y2, month=m2, day=d2))
            label = f"Từ {d1:02d}/{m1:02d} đến {d2:02d}/{m2:02d}/{y2}"
            return _build_result(
                granularity="custom",
                start=start_dt,
                end=end_dt,
                label_prefix=label
            )
        except ValueError:
            pass

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

    # 1. Weekend / Weekday / Holiday
    if re.search(r"\bcuoi tuan\b", norm_text):
        mon = _monday_of(today)
        sat = mon + timedelta(days=5)
        sun = mon + timedelta(days=6)
        return _build_result(granularity="weekend", start=sat, end=_end_of_day(sun), label_prefix="Cuối tuần")

    if re.search(r"\bngay thuong\b", norm_text):
        mon = _monday_of(today)
        fri = mon + timedelta(days=4)
        return _build_result(granularity="weekday", start=mon, end=_end_of_day(fri), label_prefix="Ngày thường")

    if re.search(r"\bngay le\b", norm_text):
        return _build_result(granularity="holiday", start=today, end=_end_of_day(now), label_prefix="Ngày lễ")

    # 2. Days of week
    weekday_map = {
        r"\bthu (?:2|hai)\b": (0, "Thứ hai"),
        r"\bthu (?:3|ba)\b": (1, "Thứ ba"),
        r"\bthu (?:4|tu)\b": (2, "Thứ tư"),
        r"\bthu (?:5|nam)\b": (3, "Thứ năm"),
        r"\bthu (?:6|sau)\b": (4, "Thứ sáu"),
        r"\bthu (?:7|bay)\b": (5, "Thứ bảy"),
        r"\bchu nhat\b": (6, "Chủ nhật"),
    }
    for pat, (w_idx, label) in weekday_map.items():
        if re.search(pat, norm_text):
            day_dt = _monday_of(today) + timedelta(days=w_idx)
            gran = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][w_idx]
            return _build_result(granularity=gran, start=day_dt, end=_end_of_day(day_dt), label_prefix=label)

    # 3. Single base words fallback
    if re.search(r"\bthang\b", norm_text):
        start = today.replace(day=1)
        return _build_result(granularity="month", start=start, end=_end_of_day(now), label_prefix="Tháng này")

    if re.search(r"\btuan\b", norm_text):
        start = _monday_of(today)
        return _build_result(granularity="week", start=start, end=_end_of_day(now), label_prefix="Tuần này")

    if re.search(r"\bngay\b", norm_text):
        return _build_result(granularity="day", start=today, end=_end_of_day(now), label_prefix="Hôm nay")

    if re.search(r"\bnam\b", norm_text):
        start = today.replace(month=1, day=1)
        return _build_result(granularity="year", start=start, end=_end_of_day(now), label_prefix="Năm nay")

    return None


def parse_time_range(
    text: str,
    time_slots: list[str] | None = None,
    *,
    now: datetime | None = None,
) -> dict | None:
    """Chỉ parse từ slot time do model dự đoán; không quét keyword trên full câu."""
    anchor = _as_vn(now or datetime.now(VN_TZ))
    if not time_slots:
        return None
    for raw in time_slots:
        norm = _norm(str(raw))
        if not norm:
            continue
        hit = _match_range(norm, anchor)
        if hit:
            return hit
    return None
