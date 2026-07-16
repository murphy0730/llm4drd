"""仿真与优化算法健壮性回归测试。

覆盖以下修复：
1. 资源日历耗尽时工序不得以 end=inf 排出（资源会被永久锁死、指标被 inf 污染）
2. 初始在制工序完工时间非有限时的兜底
3. _compute_kpi 对未完成任务施加惩罚（部分排程不得优于完整排程）
4. build_schedule_analytics 对 None/inf 排程条目的容错
5. ExactSolver 对无交期任务（due_date=inf）不崩溃
6. 调度规则抛异常 / 返回非有限分数时的兜底
"""
import math
import unittest
from datetime import datetime, timezone

from llm4drd.core.models import (
    Machine,
    MachineType,
    Operation,
    Order,
    Shift,
    ShopFloor,
    Task,
)
from llm4drd.core.rules import BUILTIN_RULES
from llm4drd.core.simulator import SimResult, Simulator
from llm4drd.optimization.objectives import build_schedule_analytics


def _full_calendar(days: int = 30) -> list[Shift]:
    return [Shift(day=day, start_hour=0.0, hours=24.0) for day in range(days)]


def _build_shop(ops_spec, machines_spec, due_date: float = 100.0) -> ShopFloor:
    """ops_spec: [(op_id, process_type, hours, predecessor_ops)]
    或 [(op_id, process_type, hours, predecessor_ops, turnover_hours)]
    """
    operations = {}
    task = Task(id="T1", order_id="O1", name="T1", due_date=due_date, operations=[])
    for spec in ops_spec:
        op_id, process_type, hours, preds = spec[:4]
        turnover = spec[4] if len(spec) > 4 else 0.0
        op = Operation(
            id=op_id, task_id="T1", name=op_id, process_type=process_type,
            processing_time=hours, turnover_time=turnover, predecessor_ops=list(preds),
        )
        operations[op_id] = op
        task.operations.append(op)
    machines = {}
    machine_types = {}
    for machine_id, type_id, shifts in machines_spec:
        machines[machine_id] = Machine(id=machine_id, name=machine_id, type_id=type_id, shifts=shifts)
        machine_types.setdefault(type_id, MachineType(id=type_id, name=type_id))
    shop = ShopFloor(
        machine_types=machine_types, machines=machines,
        orders={"O1": Order(id="O1", name="O1", due_date=due_date, task_ids=["T1"], main_task_id="T1")},
        tasks={"T1": task}, operations=operations,
        # 班次 start_hour 是墙上时钟小时，build_indexes 会按 plan_start_at 的钟点锚定；
        # 固定为午夜，使 Shift(day=0, start_hour=0) 恰好对应偏移 0
        plan_start_at=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
    )
    shop.build_indexes()
    return shop


class CalendarExhaustionTests(unittest.TestCase):
    """资源日历耗尽：工序排不下时必须被挂起，而不是带着 inf 占用资源。"""

    def _run(self):
        # 机器只有第 0 天 8 小时班次；两道 6h 工序，第二道无法在日历内完成
        shop = _build_shop(
            ops_spec=[("op1", "P1", 6.0, []), ("op2", "P1", 6.0, [])],
            machines_spec=[("M1", "P1", [Shift(day=0, start_hour=0.0, hours=8.0)])],
        )
        return Simulator(shop, BUILTIN_RULES["ATC"]).run()

    def test_no_infinite_entries_in_schedule(self):
        result = self._run()
        for entry in result.schedule:
            self.assertTrue(math.isfinite(entry["start"]), f"start=inf in {entry}")
            self.assertTrue(math.isfinite(entry["end"]), f"end=inf in {entry}")

    def test_metrics_are_finite_and_infeasible_flagged(self):
        result = self._run()
        self.assertEqual(len(result.schedule), 1)
        self.assertFalse(result.feasible)
        self.assertEqual(result.scheduled_operations, 1)
        self.assertEqual(result.total_operations, 2)
        self.assertTrue(math.isfinite(result.makespan))
        self.assertTrue(math.isfinite(result.total_tardiness))
        analytics = build_schedule_analytics(
            _build_shop(
                ops_spec=[("op1", "P1", 6.0, []), ("op2", "P1", 6.0, [])],
                machines_spec=[("M1", "P1", [Shift(day=0, start_hour=0.0, hours=8.0)])],
            ),
            result,
        )
        self.assertFalse(analytics.feasible)
        self.assertTrue(math.isfinite(analytics.objective_values["makespan"]))
        self.assertTrue(math.isfinite(analytics.objective_values["total_tardiness"]))


class PartialSchedulePenaltyTests(unittest.TestCase):
    """部分排程必须被惩罚：排得少不能反而指标更优。"""

    def test_incomplete_schedule_penalized(self):
        # 完整排程：日历充足
        full_shop = _build_shop(
            ops_spec=[("op1", "P1", 6.0, []), ("op2", "P1", 6.0, [])],
            machines_spec=[("M1", "P1", _full_calendar())],
            due_date=1.0,  # 很紧的交期，保证有延误
        )
        full_result = Simulator(full_shop, BUILTIN_RULES["ATC"]).run()
        self.assertTrue(full_result.feasible)

        # 部分排程：只能排出第一道
        partial_shop = _build_shop(
            ops_spec=[("op1", "P1", 6.0, []), ("op2", "P1", 6.0, [])],
            machines_spec=[("M1", "P1", [Shift(day=0, start_hour=0.0, hours=8.0)])],
            due_date=1.0,
        )
        partial_result = Simulator(partial_shop, BUILTIN_RULES["ATC"]).run()
        self.assertFalse(partial_result.feasible)

        # 未完成任务按 惩罚完工时间 计延误：部分排程不得优于完整排程
        self.assertGreaterEqual(
            partial_result.total_tardiness + 1e-9, full_result.total_tardiness,
            "部分排程的总延误竟然比完整排程更小——惩罚未生效，会误导优化器",
        )
        self.assertGreater(partial_result.total_tardiness, 0.0)


class NormalScheduleRegressionTests(unittest.TestCase):
    """正常实例：所有内置规则都应排完并给出有限指标（回归保护）。"""

    def test_all_builtin_rules_complete(self):
        for rule_name, rule in BUILTIN_RULES.items():
            shop = _build_shop(
                ops_spec=[
                    ("op1", "P1", 2.0, []),
                    ("op2", "P1", 3.0, ["op1"]),
                    ("op3", "P2", 1.5, []),
                ],
                machines_spec=[("M1", "P1", _full_calendar()), ("M2", "P2", _full_calendar())],
            )
            result = Simulator(shop, rule).run()
            self.assertTrue(result.feasible, f"{rule_name} 未排完")
            self.assertEqual(len(result.schedule), 3, f"{rule_name} 排程条目数错误")
            self.assertTrue(math.isfinite(result.makespan))
            # op2 必须在 op1 之后
            entries = {entry["op_id"]: entry for entry in result.schedule}
            self.assertGreaterEqual(entries["op2"]["start"] + 1e-9, entries["op1"]["end"], rule_name)


class BrokenRuleTests(unittest.TestCase):
    """调度规则抛异常 / 返回 NaN 时仿真必须能继续。"""

    def test_raising_rule_falls_back(self):
        def broken_rule(op, machine, features, shop):
            raise RuntimeError("boom")

        shop = _build_shop(
            ops_spec=[("op1", "P1", 2.0, [])],
            machines_spec=[("M1", "P1", _full_calendar())],
        )
        result = Simulator(shop, broken_rule).run()
        self.assertTrue(result.feasible)

    def test_nan_rule_falls_back(self):
        def nan_rule(op, machine, features, shop):
            return float("nan")

        shop = _build_shop(
            ops_spec=[("op1", "P1", 2.0, []), ("op2", "P1", 2.0, [])],
            machines_spec=[("M1", "P1", _full_calendar())],
        )
        result = Simulator(shop, nan_rule).run()
        self.assertTrue(result.feasible)


class DirtyScheduleEntryTests(unittest.TestCase):
    """analytics 对 None / inf / 非数值条目不崩溃、不污染。"""

    def test_none_and_inf_entries(self):
        shop = _build_shop(
            ops_spec=[("op1", "P1", 2.0, [])],
            machines_spec=[("M1", "P1", _full_calendar())],
        )
        dirty = SimResult(schedule=[
            {"op_id": "op1", "task_id": "T1", "machine_id": "M1", "start": None, "end": None},
            {"op_id": "ghost", "task_id": "T1", "machine_id": "M1", "start": 0.0, "end": float("inf")},
            {"op_id": "junk", "task_id": "T1", "machine_id": "M1", "start": "abc", "end": "xyz"},
        ])
        analytics = build_schedule_analytics(shop, dirty)  # 不应抛异常
        self.assertTrue(math.isfinite(analytics.objective_values["makespan"]))
        self.assertTrue(math.isfinite(analytics.objective_values["total_tardiness"]))


class CrossTypeEligibleMachineTests(unittest.TestCase):
    """工序显式指定的机台类型 ≠ process_type 时不得被饿死。

    派工桶原按 op.process_type 建立，而机器只扫描自己类型的桶——
    指定机台跨类型的工序会永远无人问津（表现为与规则无关的
    "前驱已完成但抢不到资源"）。
    """

    def _shop_with_cross_type_op(self):
        op1 = Operation(
            id="op1", task_id="T1", name="op1",
            process_type="X",                 # 没有任何 X 类型机器
            processing_time=2.0,
            eligible_machine_ids=["M1"],      # 但显式指定了 A 类型的 M1
        )
        op2 = Operation(
            id="op2", task_id="T1", name="op2",
            process_type="A",                 # 类型匹配的机器存在(M1)
            processing_time=2.0,
            eligible_machine_ids=["M2"],      # 但只允许 B 类型的 M2
        )
        task = Task(id="T1", order_id="O1", name="T1", due_date=100.0, operations=[op1, op2])
        shop = ShopFloor(
            machine_types={"A": MachineType("A", "A"), "B": MachineType("B", "B")},
            machines={
                "M1": Machine(id="M1", name="M1", type_id="A", shifts=_full_calendar()),
                "M2": Machine(id="M2", name="M2", type_id="B", shifts=_full_calendar()),
            },
            orders={"O1": Order(id="O1", due_date=100.0, task_ids=["T1"], main_task_id="T1")},
            tasks={"T1": task},
            operations={"op1": op1, "op2": op2},
            plan_start_at=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        )
        shop.build_indexes()
        return shop

    def test_cross_type_eligible_ops_are_scheduled(self):
        result = Simulator(self._shop_with_cross_type_op(), BUILTIN_RULES["ATC"]).run()
        self.assertTrue(result.feasible, "跨类型指定机台的工序被饿死")
        self.assertEqual(len(result.schedule), 2)
        by_op = {entry["op_id"]: entry["machine_id"] for entry in result.schedule}
        self.assertEqual(by_op["op1"], "M1")
        self.assertEqual(by_op["op2"], "M2")

    def test_rule_independence(self):
        # 该场景对任何规则都必须可行（此前所有规则都饿死同一批工序）
        for rule_name, rule in BUILTIN_RULES.items():
            result = Simulator(self._shop_with_cross_type_op(), rule).run()
            self.assertTrue(result.feasible, f"{rule_name} 下跨类型工序被饿死")


class ExactSolverInfDueTests(unittest.TestCase):
    """无交期任务（due_date=inf）不得让 exact 求解器 OverflowError。"""

    def test_inf_due_date_does_not_crash(self):
        try:
            import ortools  # noqa: F401
        except ImportError:
            self.skipTest("ortools not installed")
        from llm4drd.optimization.exact import ExactSolver

        shop = _build_shop(
            ops_spec=[("op1", "P1", 2.0, [])],
            machines_spec=[("M1", "P1", _full_calendar())],
            due_date=float("inf"),
        )
        result = ExactSolver(shop, objectives=["total_tardiness"], time_limit_s=5).solve()
        self.assertIn(result.status, {"OPTIMAL", "FEASIBLE"})


class InitialWipFallbackTests(unittest.TestCase):
    """初始在制工序在日历排不下剩余工时时，完工时间退化为 start+duration。"""

    def test_wip_with_exhausted_calendar(self):
        shop = _build_shop(
            ops_spec=[("op1", "P1", 6.0, [])],
            machines_spec=[("M1", "P1", [Shift(day=0, start_hour=0.0, hours=2.0)])],
        )
        from llm4drd.core.models import OpStatus
        op = shop.operations["op1"]
        op.status = OpStatus.PROCESSING
        op.assigned_machine_id = "M1"
        op.start_time = 0.0
        op.end_time = None  # 触发 effective-end 计算 → 日历只有 2h，装不下 6h → inf → 兜底
        op.remaining_processing_time = 6.0

        result = Simulator(shop, BUILTIN_RULES["ATC"]).run()
        self.assertEqual(len(result.schedule), 1)
        entry = result.schedule[0]
        self.assertTrue(math.isfinite(entry["end"]))
        self.assertAlmostEqual(entry["end"], 6.0, places=3)
        self.assertTrue(result.feasible)
        self.assertTrue(math.isfinite(result.makespan))


if __name__ == "__main__":
    unittest.main()
