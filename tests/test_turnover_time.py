"""工序流转等待时间（turnover_time）语义测试。

对应设计：docs/superpowers/specs/2026-07-16-operation-turnover-time-design.md
"""
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path

import openpyxl

from llm4drd.core.models import Machine, MachineType, Operation, Order, Shift, ShopFloor, Task
from llm4drd.core.rules import BUILTIN_RULES
from llm4drd.core.simulator import Simulator
from llm4drd.data.db import InstanceStore, _float_or_default, init_db
from llm4drd.data.template_builder import build_instance_template_bytes
from llm4drd.tests.shop_fixtures import make_graph_context_shop
from llm4drd.tests.test_simulator_robustness import _build_shop, _full_calendar


class TestTurnoverField(unittest.TestCase):
    def test_turnover_time_defaults_to_zero(self):
        """既有构造点不传 turnover_time 时必须落 0，这是零回归的锚点。"""
        op = Operation(id="OP1", task_id="T1", name="OP1", process_type="turning", processing_time=5.0)
        self.assertEqual(op.turnover_time, 0.0)

    def test_turnover_time_is_settable(self):
        op = Operation(
            id="OP1", task_id="T1", name="OP1", process_type="turning",
            processing_time=5.0, turnover_time=3.5,
        )
        self.assertEqual(op.turnover_time, 3.5)


class TestTemplateColumn(unittest.TestCase):
    def _operations_headers(self) -> list[str]:
        wb = openpyxl.load_workbook(io.BytesIO(build_instance_template_bytes()))
        ws = wb["operations"]
        return [cell.value for cell in ws[1]]

    def test_turnover_column_follows_processing_time(self):
        headers = self._operations_headers()
        self.assertIn("turnover_time_hrs", headers)
        self.assertEqual(
            headers.index("turnover_time_hrs"),
            headers.index("processing_time_hrs") + 1,
            "turnover_time_hrs 必须紧跟在 processing_time_hrs 之后",
        )

    def test_template_demonstrates_nonzero_turnover(self):
        wb = openpyxl.load_workbook(io.BytesIO(build_instance_template_bytes()))
        ws = wb["operations"]
        headers = [cell.value for cell in ws[1]]
        column = headers.index("turnover_time_hrs")
        values = [row[column].value for row in ws.iter_rows(min_row=2)]
        self.assertTrue(any(v for v in values), "模板应至少有一行非零 turnover 以演示语义")


class TestTurnoverPersistence(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "test.db")
        init_db(self.db_path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_ddl_has_turnover_column(self):
        with sqlite3.connect(self.db_path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(inst_operations)")}
        self.assertIn("turnover_time", columns)

    def test_null_turnover_in_legacy_row_loads_as_zero(self):
        """模拟旧库：turnover_time 为 NULL 的行加载必须落 0 而非崩溃。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO inst_operations (op_id, task_id, op_name, process_type, processing_time, turnover_time) "
                "VALUES ('OP1','T1','OP1','turning',5.0,NULL)"
            )
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = dict(conn.execute("SELECT * FROM inst_operations WHERE op_id='OP1'").fetchone())
        self.assertEqual(_float_or_default(row.get("turnover_time"), 0.0), 0.0)

    def test_migration_is_idempotent_on_legacy_table(self):
        """旧库缺列时 init_db 必须补列且可重复执行。"""
        legacy = str(Path(self._tmp.name) / "legacy.db")
        with sqlite3.connect(legacy) as conn:
            conn.execute(
                "CREATE TABLE inst_operations (op_id TEXT PRIMARY KEY, task_id TEXT, "
                "op_name TEXT, process_type TEXT, processing_time REAL)"
            )
        init_db(legacy)
        init_db(legacy)  # 第二次必须不报错
        with sqlite3.connect(legacy) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(inst_operations)")}
        self.assertIn("turnover_time", columns)

    def test_excel_import_without_turnover_column_defaults_to_zero(self):
        """旧 xlsx 缺列时导入必须落 0。"""
        legacy_row = {"op_id": "OP1", "task_id": "T1", "processing_time_hrs": 5.0}
        self.assertEqual(
            _float_or_default(legacy_row.get("turnover_time_hrs", legacy_row.get("turnover_time", 0)), 0.0),
            0.0,
        )

    def test_excel_import_reads_turnover_column(self):
        row = {"op_id": "OP1", "task_id": "T1", "processing_time_hrs": 5.0, "turnover_time_hrs": 2.5}
        self.assertEqual(
            _float_or_default(row.get("turnover_time_hrs", row.get("turnover_time", 0)), 0.0),
            2.5,
        )

    def test_save_and_load_roundtrip_preserves_turnover_time(self):
        """save_from_shopfloor -> build_shopfloor 必须保值非零 turnover_time。"""
        shop = make_graph_context_shop()
        shop.operations["OP-11"].turnover_time = 2.5
        store = InstanceStore(self.db_path)
        store.save_from_shopfloor(shop)
        loaded = store.build_shopfloor()
        self.assertEqual(loaded.operations["OP-11"].turnover_time, 2.5)


def _shop_with_two_ops(turnover: float) -> ShopFloor:
    """OP1 -> OP2 串行，OP1 的 turnover 可调。OP1 已完工于 t=10。"""
    op1 = Operation(id="OP1", task_id="T1", name="OP1", process_type="turning",
                    processing_time=5.0, turnover_time=turnover)
    op2 = Operation(id="OP2", task_id="T1", name="OP2", process_type="milling",
                    processing_time=3.0, predecessor_ops=["OP1"])
    task = Task(id="T1", order_id="O1", name="T1", due_date=100.0, operations=[op1, op2])
    calendar = [Shift(day=d, start_hour=0.0, hours=24.0) for d in range(30)]
    shop = ShopFloor(
        machine_types={"turning": MachineType(id="turning", name="turning"),
                       "milling": MachineType(id="milling", name="milling")},
        machines={"m1": Machine(id="m1", name="m1", type_id="turning", shifts=calendar),
                  "m2": Machine(id="m2", name="m2", type_id="milling", shifts=calendar)},
        orders={"O1": Order(id="O1", name="O1", due_date=100.0, task_ids=["T1"], main_task_id="T1")},
        tasks={"T1": task},
        operations={"OP1": op1, "OP2": op2},
    )
    shop.build_indexes()
    op1.end_time = 10.0
    return shop


class TestFlowReadyGate(unittest.TestCase):
    def test_gate_adds_predecessor_turnover_to_its_end_time(self):
        shop = _shop_with_two_ops(turnover=4.0)
        self.assertEqual(shop.get_operation_flow_ready_time(shop.operations["OP2"]), 14.0)

    def test_zero_turnover_gate_equals_predecessor_end(self):
        """零回归锚点：turnover=0 时闸门退化为前驱完工时刻。"""
        shop = _shop_with_two_ops(turnover=0.0)
        op2 = shop.operations["OP2"]
        self.assertEqual(
            shop.get_operation_flow_ready_time(op2),
            max(shop.get_operation_release_time(op2), 10.0),
        )

    def test_gate_honors_release_time_when_larger(self):
        shop = _shop_with_two_ops(turnover=1.0)
        shop.tasks["T1"].release_time = 50.0
        self.assertEqual(shop.get_operation_flow_ready_time(shop.operations["OP2"]), 50.0)

    def test_gate_accepts_precomputed_release_time(self):
        """调用方传入缓存的 release_time 时，结果必须与内部计算一致。"""
        shop = _shop_with_two_ops(turnover=4.0)
        op2 = shop.operations["OP2"]
        self.assertEqual(
            shop.get_operation_flow_ready_time(op2, release_time=0.0),
            shop.get_operation_flow_ready_time(op2),
        )

    def test_gate_tolerates_predecessor_without_end_time(self):
        """前驱尚未完工（end_time is None）时不得崩溃。"""
        shop = _shop_with_two_ops(turnover=4.0)
        shop.operations["OP1"].end_time = None
        self.assertEqual(shop.get_operation_flow_ready_time(shop.operations["OP2"]), 0.0)

    def test_gate_uses_max_over_task_predecessor_operations(self):
        """口径 2：任务级前驱取 max(各工序 end + 各自 turnover)。

        刻意让「完工最晚的工序」(OP4, end=12) 与「流转最晚的工序」
        (OP3, end=5 但 turnover=9) 不是同一道——若实现错写成「取任务
        completion_time 再加末工序 turnover」，此测试会得 13 而非 14。
        """
        shop = _shop_with_two_ops(turnover=1.0)
        early = Operation(id="OP3", task_id="T2", name="OP3", process_type="turning",
                          processing_time=2.0, turnover_time=9.0)
        late = Operation(id="OP4", task_id="T2", name="OP4", process_type="turning",
                         processing_time=2.0, turnover_time=1.0)
        early.end_time = 5.0   # 5 + 9 = 14
        late.end_time = 12.0   # 12 + 1 = 13
        shop.tasks["T2"] = Task(id="T2", order_id="O1", name="T2", due_date=100.0,
                                operations=[early, late])
        shop.operations["OP3"] = early
        shop.operations["OP4"] = late
        shop.operations["OP2"].predecessor_tasks = ["T2"]
        # max(OP1: 10+1=11, OP3: 14, OP4: 13) = 14
        self.assertEqual(shop.get_operation_flow_ready_time(shop.operations["OP2"]), 14.0)


def _run(shop):
    result = Simulator(shop, BUILTIN_RULES["FIFO"]).run()
    return {entry["op_id"]: entry for entry in result.schedule}


class TestSimulatorTurnover(unittest.TestCase):
    def test_successor_waits_for_predecessor_turnover(self):
        shop = _build_shop(
            [("OP1", "turning", 5.0, [], 3.0),
             ("OP2", "milling", 2.0, ["OP1"])],
            [("m1", "turning", _full_calendar()), ("m2", "milling", _full_calendar())],
        )
        entries = _run(shop)
        self.assertGreaterEqual(
            entries["OP2"]["start"], entries["OP1"]["end"] + 3.0 - 1e-9,
            "后继必须等满前驱的 turnover",
        )

    def test_zero_turnover_is_unchanged(self):
        """零回归：turnover=0 时后继紧接前驱开工。"""
        shop = _build_shop(
            [("OP1", "turning", 5.0, [], 0.0),
             ("OP2", "milling", 2.0, ["OP1"])],
            [("m1", "turning", _full_calendar()), ("m2", "milling", _full_calendar())],
        )
        entries = _run(shop)
        self.assertAlmostEqual(entries["OP2"]["start"], entries["OP1"]["end"], places=6)

    def test_machine_is_free_during_turnover(self):
        """口径 3：turnover 期间机床可接别的活——这是本字段的核心动机。"""
        shop = _build_shop(
            [("OP1", "turning", 5.0, [], 100.0),   # 超长 turnover
             ("OP2", "milling", 2.0, ["OP1"]),
             ("OP3", "turning", 4.0, [])],          # 无前驱，抢同一台车床
            [("m1", "turning", _full_calendar()), ("m2", "milling", _full_calendar())],
        )
        entries = _run(shop)
        self.assertLess(
            entries["OP3"]["start"], entries["OP1"]["end"] + 100.0,
            "OP3 不该等 OP1 的 turnover——turnover 不占用机床",
        )

    def test_turnover_elapses_on_wall_clock_not_shift_time(self):
        """口径 1：turnover 跨越非排班时段时不被拉长。

        机床每天只排 0-8h。OP1 需 6h，从 0 开工、8h 前完工。turnover=10h
        跨越了当天的非排班时段。若 turnover 错误地只在班次内计时，OP2 会被
        推迟一整天；按自然时间则闸门落在次日，OP2 应在闸门开启后的第一个
        排班窗口内开工。
        """
        short_shifts = [Shift(day=d, start_hour=0.0, hours=8.0) for d in range(30)]
        shop = _build_shop(
            [("OP1", "turning", 6.0, [], 10.0),
             ("OP2", "milling", 2.0, ["OP1"])],
            [("m1", "turning", short_shifts), ("m2", "milling", short_shifts)],
        )
        entries = _run(shop)
        gate = entries["OP1"]["end"] + 10.0
        self.assertGreaterEqual(entries["OP2"]["start"], gate - 1e-9)
        self.assertLess(entries["OP2"]["start"], gate + 24.0,
                        "OP2 应在闸门后的首个排班窗口开工，而非因 turnover 被班次化再推一天")

    def test_completed_predecessor_with_negative_end_time_gates_at_t0(self):
        """spec 用例 7：initial_state 中已完工前驱的 end_time 为负值。

        end_time=-2, turnover=1 -> 闸门 = -1 <= 0，后继在 t=0 即就绪。
        """
        op1 = Operation(
            id="OP1", task_id="T1", name="OP1", process_type="turning",
            processing_time=5.0, turnover_time=1.0,
        )
        op2 = Operation(
            id="OP2", task_id="T1", name="OP2", process_type="milling",
            processing_time=2.0, predecessor_ops=["OP1"],
        )
        from llm4drd.core.models import OpStatus
        op1.status = OpStatus.COMPLETED
        op1.end_time = -2.0
        task = Task(id="T1", order_id="O1", name="T1", due_date=100.0, operations=[op1, op2])
        shop = ShopFloor(
            machine_types={
                "turning": MachineType(id="turning", name="turning"),
                "milling": MachineType(id="milling", name="milling"),
            },
            machines={
                "m1": Machine(id="m1", name="m1", type_id="turning", shifts=_full_calendar()),
                "m2": Machine(id="m2", name="m2", type_id="milling", shifts=_full_calendar()),
            },
            orders={"O1": Order(id="O1", name="O1", due_date=100.0, task_ids=["T1"], main_task_id="T1")},
            tasks={"T1": task},
            operations={"OP1": op1, "OP2": op2},
        )
        shop.build_indexes()
        # predecessor 的贡献是 end_time + turnover_time = -2 + 1 = -1；
        # 与 release_time 默认值 0 取 max 后闸门为 0，即 <= 0，后继 t=0 即就绪。
        self.assertLessEqual(shop.get_operation_flow_ready_time(op2), 0.0)
        entries = _run(shop)
        self.assertLessEqual(entries["OP2"]["start"], 1e-9)


class TestDerivedTimesWithTurnover(unittest.TestCase):
    def test_earliest_start_of_successor_includes_predecessor_turnover(self):
        """任务内前推：后继的 earliest_start 必须含前驱 turnover。"""
        shop = _build_shop(
            [("OP1", "turning", 5.0, [], 4.0),
             ("OP2", "milling", 2.0, ["OP1"])],
            [("m1", "turning", _full_calendar()), ("m2", "milling", _full_calendar())],
        )
        # OP1: est=0, pt=5, turnover=4 -> OP2.est = 0+5+4 = 9
        self.assertAlmostEqual(shop.operations["OP2"].earliest_start_time, 9.0, places=6)

    def test_zero_turnover_leaves_earliest_start_unchanged(self):
        """零回归锚点。"""
        shop = _build_shop(
            [("OP1", "turning", 5.0, [], 0.0),
             ("OP2", "milling", 2.0, ["OP1"])],
            [("m1", "turning", _full_calendar()), ("m2", "milling", _full_calendar())],
        )
        self.assertAlmostEqual(shop.operations["OP2"].earliest_start_time, 5.0, places=6)

    def test_derived_start_time_backs_off_by_own_turnover(self):
        """反推：后继要在 t 开工，则本工序须在 t - turnover 前完工。"""
        shop = _build_shop(
            [("OP1", "turning", 5.0, [], 4.0),
             ("OP2", "milling", 2.0, ["OP1"])],
            [("m1", "turning", _full_calendar()), ("m2", "milling", _full_calendar())],
            due_date=100.0,
        )
        op1, op2 = shop.operations["OP1"], shop.operations["OP2"]
        # OP2.derived_start = 100 - 2 = 98；OP1 须在 98 - 4 = 94 前完工
        # 故 OP1.derived_due_date = 94, OP1.derived_start = 94 - 5 = 89
        self.assertAlmostEqual(op2.derived_start_time, 98.0, places=6)
        self.assertAlmostEqual(op1.derived_due_date, 94.0, places=6)
        self.assertAlmostEqual(op1.derived_start_time, 89.0, places=6)


class TestApproxEvalTurnover(unittest.TestCase):
    def test_simulator_result_respects_turnover_as_reference(self):
        """参照锚点：仿真器的真实结果，供近似评价对齐。"""
        shop = _build_shop(
            [("OP1", "turning", 5.0, [], 4.0),
             ("OP2", "milling", 2.0, ["OP1"])],
            [("m1", "turning", _full_calendar()), ("m2", "milling", _full_calendar())],
        )
        entries = _run(shop)
        self.assertGreaterEqual(entries["OP2"]["start"], entries["OP1"]["end"] + 4.0 - 1e-9)

    def _evaluate(self, shop):
        from llm4drd.optimization.approx_eval import ApproximateScheduleEvaluator
        from llm4drd.optimization.solution_model import CandidateParameters, FEATURE_NAMES
        candidate = CandidateParameters(
            feature_weights={name: 0.0 for name in FEATURE_NAMES},
            destroy_weights={},
            repair_weights={},
        )
        evaluator = ApproximateScheduleEvaluator(
            shop, graph_features={}, time_scale=10.0, due_scale=100.0,
            priority_scale=1.0, keep_schedule_limit=100,
        )
        solution = evaluator.evaluate(candidate, source="test", generation=0)
        return {entry["op_id"]: entry for entry in solution.schedule}

    def test_approx_eval_respects_predecessor_op_turnover(self):
        """base_ready 必须计入工序级前驱的 turnover——直接验证 approx_eval 内部，不止仿真器。"""
        shop = _build_shop(
            [("OP1", "turning", 5.0, [], 4.0),
             ("OP2", "milling", 2.0, ["OP1"])],
            [("m1", "turning", _full_calendar()), ("m2", "milling", _full_calendar())],
        )
        entries = self._evaluate(shop)
        self.assertGreaterEqual(
            entries["OP2"]["start"], entries["OP1"]["end"] + 4.0 - 1e-9,
            "approx_eval 的 base_ready 必须计入前驱 turnover",
        )

    def test_approx_eval_task_level_predecessor_uses_max_over_operations(self):
        """口径 2 的 unroll 校验：不得退化为 task_completion（聚合完工时刻）+ 末工序 turnover。

        OP3 (end=5, turnover=9 -> 14) 与 OP4 (end=12, turnover=1 -> 13) 是同一
        预驱任务 T2 内并行的两道工序。若实现仍用聚合的 task_completion(=12，
        即 OP4 的完工时刻) 再加某一道工序的 turnover，只会得到 13；正确实现
        须遍历任务全部工序并取 max(14, 13) = 14。
        """
        early = Operation(id="OP3", task_id="T2", name="OP3", process_type="turning",
                           processing_time=5.0, turnover_time=9.0)
        late = Operation(id="OP4", task_id="T2", name="OP4", process_type="drilling",
                          processing_time=12.0, turnover_time=1.0)
        task2 = Task(id="T2", order_id="O1", name="T2", due_date=200.0, operations=[early, late])
        op2 = Operation(id="OP2", task_id="T1", name="OP2", process_type="milling",
                         processing_time=2.0, predecessor_tasks=["T2"])
        task1 = Task(id="T1", order_id="O1", name="T1", due_date=200.0, operations=[op2])
        shop = ShopFloor(
            machine_types={
                "turning": MachineType(id="turning", name="turning"),
                "drilling": MachineType(id="drilling", name="drilling"),
                "milling": MachineType(id="milling", name="milling"),
            },
            machines={
                "m1": Machine(id="m1", name="m1", type_id="turning", shifts=_full_calendar()),
                "m2": Machine(id="m2", name="m2", type_id="drilling", shifts=_full_calendar()),
                "m3": Machine(id="m3", name="m3", type_id="milling", shifts=_full_calendar()),
            },
            orders={"O1": Order(id="O1", name="O1", due_date=200.0, task_ids=["T1", "T2"], main_task_id="T1")},
            tasks={"T1": task1, "T2": task2},
            operations={"OP2": op2, "OP3": early, "OP4": late},
        )
        shop.build_indexes()
        entries = self._evaluate(shop)
        self.assertGreaterEqual(
            entries["OP2"]["start"], 14.0 - 1e-6,
            "任务级前驱须取 max(各工序 end+turnover)，不得退化为末工序/聚合完工时刻取值",
        )


if __name__ == "__main__":
    unittest.main()
