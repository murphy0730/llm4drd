"""按订单筛选返回某方案排程条目的端点。

评审页方案详情甘特图不再全量下发排程：服务端返回订单 facet + 单个订单的 entries。
覆盖默认订单、指定订单、基线按 id 命中、未知方案报错、缺 order_id 归桶五件事。
"""
import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import HTTPException

from llm4drd.api import server
from llm4drd.data.db import InstanceStore, WorkflowProgressStore, init_db
from llm4drd.tests.shop_fixtures import make_graph_context_shop


def run(coro):
    return asyncio.run(coro)


def _entry(op_id, order_id, order_name, machine_id="M-1", start=0.0, end=1.0):
    return {
        "op_id": op_id,
        "task_id": f"T-{op_id}",
        "machine_id": machine_id,
        "start": start,
        "end": end,
        "order_id": order_id,
        "order_name": order_name,
    }


class OptimizeSolutionScheduleApiTests(unittest.TestCase):
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

        self.schedule = [
            _entry("op-a1", "O-0001", "订单0001"),
            _entry("op-a2", "O-0001", "订单0001"),
            _entry("op-b1", "O-0002", "订单0002"),
        ]
        server._hybrid_tasks["t1"] = {
            "status": "done",
            "phase": "done",
            "result": {"solutions": [{"solution_id": "S-1"}]},
            "export_result": {
                "baseline": {"solution_id": "S-base", "schedule": [_entry("op-c1", "O-0003", "订单0003")]},
                "solutions": [{"solution_id": "S-1", "schedule": self.schedule}],
            },
            "reference_solutions": [],
        }
        server._latest_hybrid_task_id = "t1"

    def _call(self, solution_id="S-1", order_id=None):
        return run(server.optimize_solution_schedule("t1", solution_id, order_id))

    def test_default_order_is_first_facet(self):
        payload = self._call()
        self.assertEqual(payload["total_operations"], 3)
        facet = {item["order_id"]: item["op_count"] for item in payload["orders"]}
        self.assertEqual(facet, {"O-0001": 2, "O-0002": 1})
        self.assertEqual(payload["order_id"], "O-0001")
        self.assertTrue(all(e["order_id"] == "O-0001" for e in payload["entries"]))
        self.assertEqual(len(payload["entries"]), 2)

    def test_explicit_order_filters_entries_only(self):
        payload = self._call(order_id="O-0002")
        self.assertEqual(payload["order_id"], "O-0002")
        self.assertEqual([e["op_id"] for e in payload["entries"]], ["op-b1"])
        self.assertEqual({item["order_id"] for item in payload["orders"]}, {"O-0001", "O-0002"})

    def test_baseline_resolved_by_its_solution_id(self):
        payload = self._call(solution_id="S-base")
        self.assertEqual(payload["solution_id"], "S-base")
        self.assertEqual(payload["total_operations"], 1)
        self.assertEqual(payload["order_id"], "O-0003")

    def test_unknown_solution_raises_404(self):
        with self.assertRaises(HTTPException) as ctx:
            self._call(solution_id="S-missing")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_entries_without_order_id_bucketed(self):
        server._hybrid_tasks["t1"]["export_result"]["solutions"][0]["schedule"] = [
            _entry("op-x", None, ""),
            _entry("op-y", "O-0001", "订单0001"),
        ]
        payload = self._call()
        facet = {item["order_id"] for item in payload["orders"]}
        self.assertEqual(facet, {"-", "O-0001"})
        # 默认订单为排序后首个（"-" 排在 "O-0001" 之前），其 entries 为缺 order_id 的条目
        self.assertEqual(payload["order_id"], "-")
        self.assertEqual([e["op_id"] for e in payload["entries"]], ["op-x"])


if __name__ == "__main__":
    unittest.main()
