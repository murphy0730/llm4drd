from __future__ import annotations

import unittest

from llm4drd.api.delay_attribution import (
    analyze_delay_attribution,
    explain_order_delay,
)
from llm4drd.core.models import (
    Downtime, Machine, MachineType, Operation, Order, Shift, ShopFloor, Task,
)


DAY_SHIFTS = [Shift(day=day, start_hour=8.0, hours=10.0) for day in range(5)]
ROUND_THE_CLOCK = [Shift(day=day, start_hour=0.0, hours=24.0) for day in range(5)]


def _entry(op_id: str, machine_id: str, start: float, end: float) -> dict:
    return {
        "op_id": op_id,
        "machine_id": machine_id,
        "start": start,
        "end": end,
        "tooling_ids": [],
        "personnel_ids": [],
    }


def _add_order(shop: ShopFloor, order_id: str, due_date: float, operations: list[Operation]) -> None:
    """一个订单一个主任务，工序全挂在该任务下。"""
    task_id = f"T-{order_id}"
    shop.orders[order_id] = Order(order_id, f"Order {order_id}", release_time=0.0, due_date=due_date)
    task = Task(task_id, order_id, "Main", True, [], release_time=0.0, due_date=due_date)
    shop.tasks[task_id] = task
    shop.orders[order_id].task_ids.append(task_id)
    shop.orders[order_id].main_task_id = task_id
    for operation in operations:
        operation.task_id = task_id
        task.operations.append(operation)
        shop.operations[operation.id] = operation


def make_attribution_shop() -> ShopFloor:
    """三台机器覆盖四种等待成因：产能不足 / 班次外 / 停机 / 机器空闲。

    M-DAY 只上白班（08:00-18:00），M-FULL 与 M-DOWN 全天开机，M-DOWN 头 3 小时停机。
    偏移量以计划开始时刻（默认 08:00）为锚点，故白班窗口是 [0,10]、[24,34]……

    **每台机器各自独占一个工艺类型**：这样等待没有备选机器可去，成因才收敛到
    "这台机器不够用"。备选机器空闲的情形另有 make_alternative_shop 覆盖。
    """
    shop = ShopFloor()
    shop.machine_types["day"] = MachineType("day", "Day Mill", is_critical=True)
    shop.machine_types["full"] = MachineType("full", "Full Mill", is_critical=False)
    shop.machine_types["down"] = MachineType("down", "Down Mill", is_critical=False)
    shop.machines["M-DAY"] = Machine("M-DAY", "Day-only Mill", "day", shifts=list(DAY_SHIFTS))
    shop.machines["M-FULL"] = Machine("M-FULL", "Full Mill", "full", shifts=list(ROUND_THE_CLOCK))
    shop.machines["M-DOWN"] = Machine(
        "M-DOWN", "Down Mill", "down",
        shifts=list(ROUND_THE_CLOCK),
        downtimes=[Downtime("DT-1", "M-DOWN", "maintenance", 0.0, 3.0)],
    )

    # O-A 占满第一天白班，O-B 只能等到第二天开班 —— 等待里既有"机器被占"也有"没排班"。
    _add_order(shop, "O-A", 5.0, [Operation("OP-A", "", "Mill A", "day", 10.0)])
    _add_order(shop, "O-B", 5.0, [Operation("OP-B", "", "Mill B", "day", 3.0)])
    # O-C 卡在停机上；O-D 机器空着却没排（工装/人员/规则）。
    _add_order(shop, "O-C", 1.0, [Operation("OP-C", "", "Mill C", "down", 2.0)])
    _add_order(shop, "O-D", 1.0, [Operation("OP-D", "", "Mill D", "full", 2.0)])
    # O-E 准时完工，不该进入归因。
    _add_order(shop, "O-E", 99.0, [Operation("OP-E", "", "Mill E", "full", 1.0)])
    shop.build_indexes()
    return shop


def make_alternative_shop() -> ShopFloor:
    """同一工艺类型两台机器：M-1 被占，M-2 全程空闲。

    这时等待是选机/派工造成的，不是产能不足——加机台解决不了。
    """
    shop = ShopFloor()
    shop.machine_types["mill"] = MachineType("mill", "Milling", is_critical=True)
    shop.machines["M-1"] = Machine("M-1", "Mill 1", "mill", shifts=list(ROUND_THE_CLOCK))
    shop.machines["M-2"] = Machine("M-2", "Mill 2", "mill", shifts=list(ROUND_THE_CLOCK))
    _add_order(shop, "O-X", 1.0, [Operation("OP-X", "", "Blocking", "mill", 6.0)])
    _add_order(shop, "O-Y", 1.0, [Operation("OP-Y", "", "Waiting", "mill", 2.0)])
    shop.build_indexes()
    return shop


ALTERNATIVE_SCHEDULE = [
    _entry("OP-X", "M-1", 0.0, 6.0),   # 占住 M-1
    _entry("OP-Y", "M-1", 6.0, 8.0),   # ready=0，等 6h —— 但 M-2 全程空着
]


def make_cross_shift_shop() -> ShopFloor:
    """复刻审查发现的反例：工序墙钟跨班次，真实占机远小于墙钟跨度。

    班次 [0,10] 与 [24,34]。OP-B 墙钟占 [5,25]，其中只有 [5,10]+[24,25]=6h 是真加工。
    """
    shop = ShopFloor()
    shop.machine_types["m"] = MachineType("m", "Mill")
    shop.machines["M1"] = Machine("M1", "Mill 1", "m", shifts=list(DAY_SHIFTS))
    for order_id in ("O-A", "O-B", "O-Y"):
        _add_order(shop, order_id, 1.0, [Operation(f"OP-{order_id[-1]}", "", order_id, "m", 1.0)])
    shop.build_indexes()
    return shop


CROSS_SHIFT_SCHEDULE = [
    _entry("OP-A", "M1", 0.0, 3.0),     # 3h，全在班内
    _entry("OP-B", "M1", 5.0, 25.0),    # 墙钟 20h，真实占机 6h（[5,10]+[24,25]）
    _entry("OP-Y", "M1", 26.0, 27.0),   # ready=0，等 26h
]


SCHEDULE = [
    _entry("OP-A", "M-DAY", 0.0, 10.0),     # 占满第一天白班，自身不等待
    _entry("OP-B", "M-DAY", 24.0, 27.0),    # ready=0，等 24h = 机器被占 10h + 班次外 14h
    _entry("OP-C", "M-DOWN", 3.0, 5.0),     # ready=0，等 3h 停机
    _entry("OP-D", "M-FULL", 5.0, 7.0),     # ready=0，机器空着 5h → idle
    _entry("OP-E", "M-FULL", 20.0, 21.0),   # 不延误，不计入
]


class DelayAttributionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shop = make_attribution_shop()
        self.result = analyze_delay_attribution(self.shop, SCHEDULE, machine_limit=20)

    def _machine(self, machine_id: str) -> dict:
        return next(item for item in self.result["machines"] if item["machine_id"] == machine_id)

    def test_tardiness_is_reported_at_order_scope(self):
        """字段名必须点明是订单级——方案 KPI 的 total_tardiness 是任务级，两者对不上。"""
        self.assertIn("total_order_tardiness_hours", self.result)
        self.assertNotIn("total_tardiness_hours", self.result)
        # O-A 迟 5、O-B 迟 22、O-C 迟 4、O-D 迟 6
        self.assertAlmostEqual(self.result["total_order_tardiness_hours"], 37.0)

    def test_only_tardy_orders_are_attributed(self):
        self.assertEqual(self.result["tardy_order_count"], 4)
        affected = {
            order_id
            for machine in self.result["machines"]
            for order_id in machine["affected_order_ids"]
        }
        self.assertNotIn("O-E", affected)

    def test_off_shift_wait_is_not_blamed_on_the_machine(self):
        """只上白班导致的等待必须落在 off_shift，不能算成机器不够。"""
        day_machine = self._machine("M-DAY")
        self.assertAlmostEqual(day_machine["off_shift_wait_hours"], 14.0)
        self.assertAlmostEqual(day_machine["capacity_wait_hours"], 10.0)
        self.assertAlmostEqual(day_machine["dispatch_wait_hours"], 0.0)
        self.assertAlmostEqual(day_machine["downtime_wait_hours"], 0.0)
        self.assertAlmostEqual(day_machine["idle_wait_hours"], 0.0)

    def test_downtime_wait_is_separated_from_off_shift(self):
        down_machine = self._machine("M-DOWN")
        self.assertAlmostEqual(down_machine["downtime_wait_hours"], 3.0)
        self.assertAlmostEqual(down_machine["off_shift_wait_hours"], 0.0)
        self.assertAlmostEqual(down_machine["capacity_wait_hours"], 0.0)

    def test_idle_machine_wait_points_away_from_capacity(self):
        full_machine = self._machine("M-FULL")
        self.assertAlmostEqual(full_machine["idle_wait_hours"], 5.0)
        self.assertAlmostEqual(full_machine["capacity_wait_hours"], 0.0)

    def test_segments_sum_to_total_wait(self):
        for machine in self.result["machines"]:
            self.assertAlmostEqual(
                machine["capacity_wait_hours"]
                + machine["dispatch_wait_hours"]
                + machine["off_shift_wait_hours"]
                + machine["downtime_wait_hours"]
                + machine["idle_wait_hours"],
                machine["total_wait_hours"],
                places=6,
            )
        breakdown = self.result["wait_breakdown"]
        self.assertEqual(
            set(breakdown),
            {"capacity_bound", "dispatch_bound", "off_shift", "downtime", "idle"},
        )
        self.assertAlmostEqual(
            sum(breakdown.values()),
            sum(machine["total_wait_hours"] for machine in self.result["machines"]),
            places=6,
        )

    def test_machines_ranked_by_capacity_wait(self):
        ranked = [item["machine_id"] for item in self.result["machines"]]
        self.assertEqual(ranked[0], "M-DAY")
        self.assertEqual(len(ranked), 3)


    def test_machine_limit_truncates_ranking(self):
        limited = analyze_delay_attribution(self.shop, SCHEDULE, machine_limit=1)
        self.assertEqual(len(limited["machines"]), 1)
        # 全局分解仍统计全部机器，不受展示条数影响。
        self.assertAlmostEqual(
            limited["wait_breakdown"]["idle"],
            self.result["wait_breakdown"]["idle"],
        )

    def test_predecessor_turnover_pushes_ready_time(self):
        shop = make_attribution_shop()
        first = Operation("OP-P1", "T-O-A", "Pre", "full", 2.0, turnover_time=1.5)
        second = shop.operations["OP-A"]
        second.predecessor_ops = ["OP-P1"]
        shop.tasks["T-O-A"].operations.insert(0, first)
        shop.operations["OP-P1"] = first
        shop.build_indexes()
        schedule = [
            _entry("OP-P1", "M-FULL", 0.0, 2.0),
            _entry("OP-A", "M-FULL", 10.0, 15.0),
        ]
        result = analyze_delay_attribution(shop, schedule, machine_limit=20)
        machine = next(item for item in result["machines"] if item["machine_id"] == "M-FULL")
        # ready = 2.0 + 1.5 turnover = 3.5，等待 6.5 小时（机器全天开且空闲）。
        self.assertAlmostEqual(machine["idle_wait_hours"], 6.5)

    def test_no_tardy_orders_yields_empty_attribution(self):
        shop = make_attribution_shop()
        for order in shop.orders.values():
            order.due_date = 999.0
        result = analyze_delay_attribution(shop, SCHEDULE, machine_limit=20)
        self.assertEqual(result["tardy_order_count"], 0)
        self.assertEqual(result["machines"], [])
        self.assertAlmostEqual(result["wait_breakdown"]["capacity_bound"], 0.0)


class AlternativeMachineTests(unittest.TestCase):
    """审查发现 B：只看被分配的机器，会把选机问题误报成产能不足。"""

    def test_free_alternative_makes_the_wait_a_dispatch_problem(self):
        shop = make_alternative_shop()
        result = analyze_delay_attribution(shop, ALTERNATIVE_SCHEDULE, machine_limit=20)

        machine = next(m for m in result["machines"] if m["machine_id"] == "M-1")
        # M-1 忙了 6h，但同工艺的 M-2 全程空着 —— 加机台救不了，得改派工。
        self.assertAlmostEqual(machine["dispatch_wait_hours"], 6.0)
        self.assertAlmostEqual(machine["capacity_wait_hours"], 0.0)
        self.assertAlmostEqual(result["wait_breakdown"]["capacity_bound"], 0.0)

    def test_busy_alternative_still_counts_as_capacity(self):
        """备选机器同期也被占满时，才是真产能不足。"""
        shop = make_alternative_shop()
        _add_order(shop, "O-Z", 1.0, [Operation("OP-Z", "", "Filler", "mill", 6.0)])
        shop.build_indexes()
        schedule = ALTERNATIVE_SCHEDULE + [_entry("OP-Z", "M-2", 0.0, 6.0)]

        result = analyze_delay_attribution(shop, schedule, machine_limit=20)

        machine = next(m for m in result["machines"] if m["machine_id"] == "M-1")
        self.assertAlmostEqual(machine["capacity_wait_hours"], 6.0)
        self.assertAlmostEqual(machine["dispatch_wait_hours"], 0.0)

    def test_eligible_machine_ids_override_process_type(self):
        """工序显式限定了可用机器时，不能拿同类型的其他机器当备选。"""
        shop = make_alternative_shop()
        shop.operations["OP-Y"].eligible_machine_ids = ["M-1"]
        shop.build_indexes()

        result = analyze_delay_attribution(shop, ALTERNATIVE_SCHEDULE, machine_limit=20)

        machine = next(m for m in result["machines"] if m["machine_id"] == "M-1")
        self.assertAlmostEqual(machine["capacity_wait_hours"], 6.0)
        self.assertAlmostEqual(machine["dispatch_wait_hours"], 0.0)


class CrossShiftOccupancyTests(unittest.TestCase):
    """审查发现 A：墙钟重叠会把跨班休息算成占机，并把真实空闲吞掉。"""

    def setUp(self) -> None:
        self.shop = make_cross_shift_shop()
        self.result = analyze_delay_attribution(
            self.shop, CROSS_SHIFT_SCHEDULE, machine_limit=20
        )
        self.machine = next(
            m for m in self.result["machines"] if m["machine_id"] == "M1"
        )
        # 单订单归因隔离出 OP-Y 这一道等待（机器汇总里还含 OP-B 自身等待的 5h）。
        self.op_y = explain_order_delay(self.shop, CROSS_SHIFT_SCHEDULE, "O-Y")

    def test_occupancy_is_clipped_to_available_windows(self):
        # OP-Y 窗口 [0,26) 可用 12h；真实占机 = OP-A 3h + OP-B 的 [5,10]+[24,25] 6h = 9h。
        # 旧实现按墙钟累加得 3+20=23，再被 min() 钳到 12。
        self.assertAlmostEqual(self.op_y["attribution"]["capacity_bound"], 9.0)
        self.assertAlmostEqual(self.op_y["attribution"]["dispatch_bound"], 0.0)

    def test_genuine_idle_is_not_swallowed(self):
        # [3,5) 与 [25,26) 机器确实空着，共 3h —— 旧实现被钳位吞成 0
        self.assertAlmostEqual(self.op_y["attribution"]["idle"], 3.0)

    def test_saturation_excludes_cross_shift_rest(self):
        # 真实占机 3+6+1(OP-Y 自身)=10h，跨度 [0,27) 可用 13h。
        # 旧实现按墙钟得 3+20+1=24h，饱和度被钳成 1.0。
        self.assertAlmostEqual(self.machine["saturation"], round(10.0 / 13.0, 4))


class ScheduleCoverageTests(unittest.TestCase):
    """审查发现 C：没排进去的订单不能报成"未延误"——那比延误更严重。"""

    def setUp(self) -> None:
        self.shop = make_attribution_shop()
        # O-B 整个订单没排进来
        self.partial = [item for item in SCHEDULE if item["op_id"] != "OP-B"]

    def test_unscheduled_orders_are_listed_not_silently_dropped(self):
        result = analyze_delay_attribution(self.shop, self.partial, machine_limit=20)

        self.assertIn("O-B", result["unscheduled_order_ids"])
        self.assertEqual(result["unscheduled_order_count"], 1)

    def test_unscheduled_order_reports_unknown_tardiness(self):
        payload = explain_order_delay(self.shop, self.partial, "O-B")

        self.assertFalse(payload["planned"])
        self.assertIsNone(payload["tardiness_hours"])
        self.assertEqual(payload["scheduled_operation_count"], 0)
        self.assertEqual(payload["total_operation_count"], 1)

    def test_partially_scheduled_order_is_flagged(self):
        """只排了一半的订单，完工时刻被低估，延误必须标为未知。"""
        shop = make_attribution_shop()
        extra = Operation("OP-A2", "T-O-A", "Mill A2", "day", 2.0)
        shop.tasks["T-O-A"].operations.append(extra)
        shop.operations["OP-A2"] = extra
        shop.build_indexes()

        result = analyze_delay_attribution(shop, SCHEDULE, machine_limit=20)
        payload = explain_order_delay(shop, SCHEDULE, "O-A")

        self.assertIn("O-A", result["partially_scheduled_order_ids"])
        self.assertFalse(payload["planned"])
        self.assertIsNone(payload["tardiness_hours"])
        self.assertEqual(payload["scheduled_operation_count"], 1)
        self.assertEqual(payload["total_operation_count"], 2)

    def test_fully_scheduled_orders_stay_planned(self):
        payload = explain_order_delay(self.shop, SCHEDULE, "O-B")

        self.assertTrue(payload["planned"])
        self.assertAlmostEqual(payload["tardiness_hours"], 22.0)


class ExplainOrderDelayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shop = make_attribution_shop()

    def test_reports_tardiness_and_attribution(self):
        payload = explain_order_delay(self.shop, SCHEDULE, "O-B")
        self.assertEqual(payload["order_id"], "O-B")
        self.assertAlmostEqual(payload["tardiness_hours"], 22.0)  # 完工 27，交期 5
        self.assertAlmostEqual(payload["attribution"]["capacity_bound"], 10.0)
        self.assertAlmostEqual(payload["attribution"]["off_shift"], 14.0)

    def test_top_waits_carry_machine_identity(self):
        payload = explain_order_delay(self.shop, SCHEDULE, "O-B")
        top = payload["top_waits"][0]
        self.assertEqual(top["operation_id"], "OP-B")
        self.assertEqual(top["machine_name"], "Day-only Mill")
        self.assertAlmostEqual(top["wait_hours"], 24.0)

    def test_inevitable_tardiness_uses_critical_path(self):
        """工艺本身就来不及时，必须与资源竞争区分开。"""
        shop = make_attribution_shop()
        shop.orders["O-A"].due_date = 1.0  # 工序本身要 5 小时，交期 1 小时必然延误
        payload = explain_order_delay(shop, SCHEDULE, "O-A")
        self.assertGreater(payload["inevitable_tardiness_hours"], 0.0)

    def test_on_time_order_reports_zero_tardiness(self):
        payload = explain_order_delay(self.shop, SCHEDULE, "O-E")
        self.assertAlmostEqual(payload["tardiness_hours"], 0.0)
        self.assertEqual(payload["top_waits"], [])

    def test_unknown_order_raises(self):
        with self.assertRaises(KeyError):
            explain_order_delay(self.shop, SCHEDULE, "O-NOPE")


if __name__ == "__main__":
    unittest.main()
