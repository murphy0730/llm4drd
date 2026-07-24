"""按机器类型聚合的资源利用率对比：纯函数口径 + 端点结构与未知方案跳过。

利用率口径：单机 util = Σ(end-start) / (max(end)-min(start))，类型取有排产机器均值，
同时报告"有排产台数 / 总台数"。缺 machine_id / 未知机台的条目被跳过。
"""
import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import HTTPException

from llm4drd.api import server
from llm4drd.core.models import Machine, Shift
from llm4drd.data.db import InstanceStore, WorkflowProgressStore, init_db
from llm4drd.tests.shop_fixtures import make_graph_context_shop


def run(coro):
    return asyncio.run(coro)


def _entry(machine_id, start, end):
    return {"machine_id": machine_id, "start": start, "end": end}


class MachineTypeUtilizationFunctionTests(unittest.TestCase):
    def test_type_mean_and_used_machines(self):
        shop = make_graph_context_shop()
        schedule = [
            _entry("M-C1", 0.0, 10.0),                       # 满窗：busy10 / 窗口10 = 1.0
            _entry("M-C2", 0.0, 5.0), _entry("M-C2", 15.0, 20.0),  # 半窗：busy10 / 窗口20 = 0.5
            _entry("M-A1", 0.0, 4.0),                        # asm：busy4 / 窗口4 = 1.0
            _entry(None, 0.0, 3.0),                          # 缺 machine_id → 跳过
            _entry("M-UNKNOWN", 0.0, 3.0),                   # 未知机台 → 跳过
        ]
        result = server._machine_type_utilization(shop, schedule)
        self.assertEqual(result["cut"], {"utilization": 0.75, "used_machines": 2})
        self.assertEqual(result["asm"], {"utilization": 1.0, "used_machines": 1})

    def test_zero_length_window_is_zero(self):
        shop = make_graph_context_shop()
        result = server._machine_type_utilization(shop, [_entry("M-C1", 5.0, 5.0)])
        self.assertEqual(result["cut"], {"utilization": 0.0, "used_machines": 1})

    def test_none_start_or_end_skipped(self):
        shop = make_graph_context_shop()
        result = server._machine_type_utilization(shop, [_entry("M-C1", None, 5.0)])
        self.assertEqual(result, {})


class MachineTypeDailyUtilizationTests(unittest.TestCase):
    """日利用率分母 = 当日「已排产机器」的可用工时之和（班次扣除停机）。"""

    def _cut_row(self, shop, schedule):
        payload = server._machine_type_daily_utilization(shop, schedule)
        return next(row for row in payload["types"] if row["type_id"] == "cut"), payload

    def test_denominator_excludes_unused_machines(self):
        shop = make_graph_context_shop()  # cut 类型 2 台，均 24h 班次
        row, payload = self._cut_row(shop, [_entry("M-C1", 0.0, 24.0)])
        self.assertEqual(payload["days"], [1])
        self.assertEqual(row["per_day"], [1.0])  # 旧口径：24 /(2 台 × 24h) = 0.5
        self.assertEqual((row["machines_used"], row["machines_total"]), (1, 2))

    def test_denominator_follows_shift_calendar(self):
        shop = make_graph_context_shop()
        shop.machines["M-C1"] = Machine("M-C1", "Cutter 1", "cut", shifts=[Shift(day=0, start_hour=8.0, hours=8.0)])
        shop.build_indexes()
        row, _ = self._cut_row(shop, [_entry("M-C1", 8.0, 16.0)])
        self.assertEqual(row["per_day"], [1.0])  # 旧口径：8 /(1 台 × 24h) = 0.333

    def test_day_without_schedule_is_none(self):
        shop = make_graph_context_shop()
        row, payload = self._cut_row(shop, [_entry("M-C1", 0.0, 6.0), _entry("M-C1", 48.0, 54.0)])
        self.assertEqual(payload["days"], [1, 2, 3])
        self.assertEqual(row["per_day"], [0.25, None, 0.25])


class MachineTypeUtilizationApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = str(Path(self.tmp.name) / "api.db")
        init_db(self.db_path)

        originals = (
            server.inst_store,
            server.workflow_store,
            server.shop,
            server._active_shop_cache,
            server._sim_runtime_cache,
            server.last_sim_payload,
            server._hybrid_tasks,
            server._latest_hybrid_task_id,
        )

        def _restore():
            (
                server.inst_store,
                server.workflow_store,
                server.shop,
                server._active_shop_cache,
                server._sim_runtime_cache,
                server.last_sim_payload,
                server._hybrid_tasks,
                server._latest_hybrid_task_id,
            ) = originals

        self.addCleanup(_restore)

        server.inst_store = InstanceStore(self.db_path)
        server.workflow_store = WorkflowProgressStore(self.db_path)
        server.shop = None
        server._active_shop_cache = None
        server._sim_runtime_cache = None
        server.last_sim_payload = None
        server._hybrid_tasks = {}
        server._latest_hybrid_task_id = None
        server.inst_store.save_from_shopfloor(make_graph_context_shop())

        server._hybrid_tasks["t1"] = {
            "status": "done",
            "phase": "done",
            "result": {"solutions": [{"solution_id": "S-1"}, {"solution_id": "S-2"}]},
            "export_result": {
                "solutions": [
                    {"solution_id": "S-1", "schedule": [
                        _entry("M-C1", 0.0, 10.0),
                        _entry("M-C2", 0.0, 5.0), _entry("M-C2", 15.0, 20.0),
                        _entry("M-A1", 0.0, 4.0),
                    ]},
                    {"solution_id": "S-2", "schedule": [_entry("M-C1", 0.0, 20.0)]},
                ],
            },
            "reference_solutions": [],
        }
        server._latest_hybrid_task_id = "t1"

    def _call(self, solution_ids):
        return run(server.optimize_machine_type_utilization("t1", solution_ids))

    def test_response_structure_and_values(self):
        payload = self._call("S-1,S-2")
        self.assertEqual(payload["solutions"], ["S-1", "S-2"])
        types = {item["type_id"]: item for item in payload["types"]}
        self.assertEqual([item["type_id"] for item in payload["types"]], ["asm", "cut"])

        cut = types["cut"]
        self.assertEqual(cut["machines_total"], 2)
        self.assertTrue(cut["is_critical"])
        self.assertEqual(cut["per_solution"]["S-1"], {"utilization": 0.75, "used_machines": 2})
        self.assertEqual(cut["per_solution"]["S-2"], {"utilization": 1.0, "used_machines": 1})

        asm = types["asm"]
        self.assertEqual(asm["machines_total"], 1)
        self.assertFalse(asm["is_critical"])
        self.assertEqual(asm["per_solution"]["S-1"], {"utilization": 1.0, "used_machines": 1})
        self.assertNotIn("S-2", asm["per_solution"])  # S-2 无 asm 排产 → 该列缺省

    def test_unknown_solution_id_skipped(self):
        payload = self._call("S-1,S-missing")
        self.assertEqual(payload["solutions"], ["S-1"])

    def test_no_instance_raises_400(self):
        server.inst_store.clear_all()
        server._active_shop_cache = None
        with self.assertRaises(HTTPException) as ctx:
            self._call("S-1")
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
