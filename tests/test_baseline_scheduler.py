import logging
import json
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from llm4drd.config import BaselineScheduleConfig, load_config
from llm4drd.scheduling.baseline_scheduler import DailyBaselineScheduler, next_daily_run


class BaselineSchedulerTests(unittest.TestCase):
    def test_next_run_uses_configured_timezone_without_catchup(self):
        zone = ZoneInfo("Asia/Shanghai")
        before = datetime(2026, 7, 27, 1, 30, tzinfo=zone)
        after = datetime(2026, 7, 27, 2, 30, tzinfo=zone)

        self.assertEqual(
            next_daily_run(before, "02:00", "Asia/Shanghai"),
            datetime(2026, 7, 27, 2, 0, tzinfo=zone),
        )
        self.assertEqual(
            next_daily_run(after, "02:00", "Asia/Shanghai"),
            datetime(2026, 7, 28, 2, 0, tzinfo=zone),
        )

    def test_invalid_daily_time_is_rejected(self):
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        for value in ("2", "24:00", "12:60", "not-a-time"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                next_daily_run(now, value, "Asia/Shanghai")

    def test_disabled_schedule_does_not_create_thread(self):
        scheduler = DailyBaselineScheduler(
            BaselineScheduleConfig(enabled=False),
            lambda: self.fail("disabled scheduler must not run"),
        )
        self.assertFalse(scheduler.start())
        self.assertIsNone(scheduler._thread)

    def test_enabled_scheduler_can_be_stopped_while_waiting(self):
        scheduler = DailyBaselineScheduler(
            BaselineScheduleConfig(enabled=True, daily_at="23:59"),
            lambda: self.fail("future schedule must not run during this test"),
            logger=logging.getLogger("baseline-scheduler-test"),
        )
        self.assertTrue(scheduler.start())
        self.assertTrue(scheduler._thread and scheduler._thread.is_alive())
        scheduler.stop()
        self.assertFalse(scheduler._thread and scheduler._thread.is_alive())

    def test_due_schedule_invokes_callback_once(self):
        runs = []

        class _OneRunEvent:
            def __init__(self):
                self.wait_calls = 0
                self.stopped = False

            def is_set(self):
                return self.stopped

            def wait(self, _timeout):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    return False
                self.stopped = True
                return True

        scheduler = DailyBaselineScheduler(
            BaselineScheduleConfig(enabled=True),
            lambda: runs.append("ran"),
        )
        scheduler._stop_event = _OneRunEvent()
        scheduler._run_loop()
        self.assertEqual(runs, ["ran"])

    def test_config_file_exposes_disabled_schedule_entry(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({
                "baseline_schedule": {
                    "enabled": True,
                    "daily_at": "03:15",
                    "timezone": "Asia/Shanghai",
                    "time_limit_s": 7200,
                },
            }), encoding="utf-8")
            config = load_config(str(path)).baseline_schedule
        self.assertTrue(config.enabled)
        self.assertEqual(config.daily_at, "03:15")
        self.assertEqual(config.time_limit_s, 7200)
        self.assertEqual(config.generations, 1440)


if __name__ == "__main__":
    unittest.main()
