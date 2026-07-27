from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

from ..config import BaselineScheduleConfig


def next_daily_run(now: datetime, daily_at: str, timezone: str) -> datetime:
    """Return the next configured wall-clock time, never a catch-up run."""
    try:
        hour_text, minute_text = daily_at.split(":", 1)
        hour, minute = int(hour_text), int(minute_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("baseline_schedule.daily_at 必须为 HH:MM") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("baseline_schedule.daily_at 必须为有效的 HH:MM")

    zone = ZoneInfo(timezone)
    local_now = now.astimezone(zone)
    target = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= local_now:
        target += timedelta(days=1)
    return target


class DailyBaselineScheduler:
    """Single-process daily trigger; disabled schedules never create a thread."""

    def __init__(
        self,
        config: BaselineScheduleConfig,
        run_callback: Callable[[], None],
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.run_callback = run_callback
        self.logger = logger or logging.getLogger(__name__)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if not self.config.enabled:
            return False
        # Fail fast during application startup instead of silently missing nightly runs.
        next_daily_run(datetime.now(ZoneInfo(self.config.timezone)), self.config.daily_at, self.config.timezone)
        if self._thread and self._thread.is_alive():
            return True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="baseline-daily-scheduler",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run_loop(self) -> None:
        zone = ZoneInfo(self.config.timezone)
        while not self._stop_event.is_set():
            now = datetime.now(zone)
            target = next_daily_run(now, self.config.daily_at, self.config.timezone)
            self.logger.info("下一次夜间基准构建：%s", target.isoformat())
            if self._stop_event.wait(max(0.0, (target - now).total_seconds())):
                return
            try:
                self.run_callback()
            except Exception:  # noqa: BLE001
                self.logger.exception("夜间基准构建调度触发失败")
