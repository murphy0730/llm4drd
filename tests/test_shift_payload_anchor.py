import unittest
from datetime import datetime, timedelta, timezone

from llm4drd.api import server
from llm4drd.core.models import Machine, MachineType, Shift, ShopFloor


# plan_start_at 固定在非午夜钟点：anchor_hour 为 0 时本文件的断言会全部失真。
PLAN_START = datetime(2026, 7, 13, 16, 56, tzinfo=timezone(timedelta(hours=8)))


def _shop_with_day_shift() -> ShopFloor:
    shop = ShopFloor(plan_start_at=PLAN_START)
    shop.machine_types["turning"] = MachineType("turning", "Turning", is_critical=False)
    shop.machines["M-1"] = Machine(
        "M-1", "Turning-1", "turning",
        # 早班 08:00-18:00：start_hour 是墙上时钟小时。
        shifts=[Shift(day=0, start_hour=8.0, hours=10.0)],
    )
    shop.build_indexes()
    return shop


class ShiftPayloadAnchorTests(unittest.TestCase):
    def test_payload_windows_match_compiled_calendar(self):
        shop = _shop_with_day_shift()
        machine = shop.machines["M-1"]
        machine._ensure_calendar_cache()

        payload = server._resource_calendar_payload(shop, machine)
        windows = [(item["start"], item["end"]) for item in payload["shifts"]]
        compiled = [(round(start, 3), round(end, 3)) for start, end in machine._calendar_shift_windows]

        self.assertEqual(windows, compiled)

    def test_day_shift_keeps_wall_clock_label(self):
        shop = _shop_with_day_shift()
        machine = shop.machines["M-1"]

        shift = server._resource_calendar_payload(shop, machine)["shifts"][0]

        # 08:00 的早班必须仍标成 08:00，而不是被 anchor 推到 00:56。
        self.assertEqual(shift["start_at"][11:16], "08:00")
        self.assertEqual(shift["end_at"][11:16], "18:00")


if __name__ == "__main__":
    unittest.main()
