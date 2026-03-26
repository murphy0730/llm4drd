from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any


def local_now() -> datetime:
    return datetime.now().astimezone()


def ensure_aware(value: datetime, fallback: datetime | None = None) -> datetime:
    if value.tzinfo is not None:
        return value
    tzinfo = fallback.tzinfo if fallback else local_now().tzinfo
    return value.replace(tzinfo=tzinfo)


def default_plan_start() -> datetime:
    now = local_now()
    base = datetime.combine(now.date(), time(hour=8), tzinfo=now.tzinfo)
    return base if now <= base else base


def parse_datetime_value(value: Any, fallback: datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        return ensure_aware(value, fallback)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return ensure_aware(parsed, fallback)
    raise TypeError(f"Unsupported datetime value: {value!r}")


def isoformat_or_none(value: datetime | None) -> str | None:
    if value is None:
        return None
    return ensure_aware(value).isoformat(timespec="seconds")


def offset_hours_to_datetime(plan_start: datetime, offset_hours: float | None) -> datetime | None:
    if offset_hours is None:
        return None
    return ensure_aware(plan_start) + timedelta(hours=float(offset_hours))


def datetime_to_offset_hours(plan_start: datetime, value: datetime | str | int | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    parsed = parse_datetime_value(value, plan_start)
    delta = parsed - ensure_aware(plan_start)
    return delta.total_seconds() / 3600.0


def round_hours(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)
