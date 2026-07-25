import unittest
from datetime import datetime, timedelta, timezone

from llm4drd.api import server
from llm4drd.core.models import Machine, MachineType, Shift, ShopFloor


PLAN_START = datetime(2026, 7, 13, 0, 0, tzinfo=timezone(timedelta(hours=8)))
CALENDAR_DAYS = 720
SHIFTS_PER_DAY = 3


def _shop_with_long_calendar() -> ShopFloor:
    """一台机器 720 天 × 每天 3 班：全量铺开就是 2160 行，正是撑爆 Excel 的量级来源。"""
    shop = ShopFloor(plan_start_at=PLAN_START)
    shop.machine_types["turning"] = MachineType("turning", "Turning", is_critical=False)
    shifts = [
        Shift(day=day, start_hour=start_hour, hours=8.0)
        for day in range(CALENDAR_DAYS)
        for start_hour in (0.0, 8.0, 16.0)
    ]
    shop.machines["M-1"] = Machine("M-1", "Turning-1", "turning", shifts=shifts)
    shop.build_indexes()
    return shop


class CalendarExportWindowTests(unittest.TestCase):
    def test_rows_limited_to_schedule_window(self):
        shop = _shop_with_long_calendar()
        schedule = [
            {"machine_id": "M-1", "op_id": "OP-1", "start": 0.0, "end": 6.0},
            {"machine_id": "M-1", "op_id": "OP-2", "start": 30.0, "end": 40.0},
        ]

        rows = server._calendar_export_rows(shop, schedule)

        # 窗口 [0, 40] 覆盖第 0~1 天：每天 3 班 + 第 2 天 00:00 起的班次不相交
        self.assertEqual(len(rows), 6)
        self.assertLess(len(rows), CALENDAR_DAYS * SHIFTS_PER_DAY)
        for row in rows:
            self.assertGreaterEqual(row["end"], 0.0)
            self.assertLessEqual(row["start"], 40.0)

    def test_boundary_touching_window_is_kept(self):
        shop = _shop_with_long_calendar()
        # 窗口右端点恰好落在第 2 天首个班次的起点上，相接也算有交集
        schedule = [{"machine_id": "M-1", "op_id": "OP-1", "start": 0.0, "end": 48.0}]

        rows = server._calendar_export_rows(shop, schedule)

        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[-1]["start"], 48.0)

    def test_missing_start_end_does_not_fall_back_to_full_calendar(self):
        shop = _shop_with_long_calendar()
        # 前端回传的精简条目可能只有 op_id / machine_id
        schedule = [{"machine_id": "M-1", "op_id": "OP-1"}]

        rows = server._calendar_export_rows(shop, schedule)

        self.assertLess(len(rows), CALENDAR_DAYS * SHIFTS_PER_DAY)
        self.assertEqual([row["start"] for row in rows], [0.0])

    def test_untouched_machine_contributes_no_rows(self):
        shop = _shop_with_long_calendar()
        shop.machines["M-2"] = Machine("M-2", "Turning-2", "turning", shifts=list(shop.machines["M-1"].shifts))
        shop.build_indexes()
        schedule = [{"machine_id": "M-1", "op_id": "OP-1", "start": 0.0, "end": 6.0}]

        rows = server._calendar_export_rows(shop, schedule)

        self.assertEqual({row["machine_id"] for row in rows}, {"M-1"})


class ScheduleTimeWindowTests(unittest.TestCase):
    def test_window_from_numeric_values_only(self):
        schedule = [
            {"start": 5.0, "end": 9.0},
            {"start": None, "end": "-"},
            {"start": 2.0, "end": 12.0},
        ]
        self.assertEqual(server._schedule_time_window(schedule), (2.0, 12.0))

    def test_empty_schedule_degrades_to_zero_window(self):
        self.assertEqual(server._schedule_time_window([]), (0.0, 0.0))

    def test_end_never_precedes_start(self):
        # 只有 start 的条目：窗口收敛到该点，而不是产出倒挂区间
        self.assertEqual(server._schedule_time_window([{"start": 7.0}]), (7.0, 7.0))


if __name__ == "__main__":
    unittest.main()
