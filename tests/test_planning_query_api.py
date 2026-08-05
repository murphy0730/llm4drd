from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import HTTPException

from llm4drd.api import server
from llm4drd.data.db import InstanceStore, WorkflowProgressStore, init_db
from llm4drd.tests.shop_fixtures import make_graph_context_shop


def _solution(solution_id: str, entries: list[dict]) -> dict:
    return {
        "solution_id": solution_id,
        "source": "hybrid",
        "feasible": True,
        "objectives": {"total_tardiness": 2.5, "makespan": 20.0},
        "metrics": {"main_order_tardy_total_time": 1.0},
        "schedule": entries,
    }


class PlanningQueryApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        db_path = str(Path(self.tmp.name) / "planning-api.db")
        init_db(db_path)
        originals = (
            server.inst_store,
            server.workflow_store,
            server.shop,
            server._active_shop_cache,
            server._hybrid_tasks,
            server._latest_hybrid_task_id,
        )
        self.addCleanup(self._restore, originals)
        server.inst_store = InstanceStore(db_path)
        server.workflow_store = WorkflowProgressStore(db_path)
        server.inst_store.save_from_shopfloor(make_graph_context_shop())
        server.shop = None
        server._active_shop_cache = None

        entries = [
            {"op_id": "OP-11", "machine_id": "M-C1", "start": 0.0, "end": 4.0},
            {"op_id": "OP-12", "machine_id": "M-C1", "start": 4.0, "end": 6.0},
            {"op_id": "OP-13", "machine_id": "M-A1", "start": 7.0, "end": 10.0},
        ]
        full = _solution("S-1", entries)
        server._hybrid_tasks = {
            "task-1": {
                "status": "done",
                "result": {
                    "archive_size": 3,
                    "solutions": [{**full, "schedule": entries[:1]}],
                    "baseline": {},
                },
                "export_result": {
                    "archive_size": 3,
                    "solutions": [full],
                    "baseline": {},
                    "reference_solutions": [],
                },
            }
        }
        server._latest_hybrid_task_id = "task-1"

    @staticmethod
    def _restore(originals) -> None:
        (
            server.inst_store,
            server.workflow_store,
            server.shop,
            server._active_shop_cache,
            server._hybrid_tasks,
            server._latest_hybrid_task_id,
        ) = originals

    def test_planning_routes_are_registered(self) -> None:
        paths = {route.path for route in server.app.routes}

        self.assertTrue({
            "/api/query/planning/overview",
            "/api/query/planning/solutions",
            "/api/query/planning/orders/search",
            "/api/query/planning/order/{order_id}",
            "/api/query/planning/operations/search",
            "/api/query/planning/operation/{operation_id}",
            "/api/query/planning/rules",
            "/api/command/planning/run-rule",
            "/api/query/planning/bottleneck",
            "/api/query/planning/order/{order_id}/delay",
        }.issubset(paths))

    def test_bottleneck_requires_a_solution_and_suggests_the_available_ones(self) -> None:
        """瓶颈随方案而变，不给默认值；但错误里要把可选方案递到手上。"""
        with self.assertRaises(HTTPException) as caught:
            server.planning_bottleneck()

        error = caught.exception.detail["error"]
        self.assertEqual(error["code"], "SOLUTION_REQUIRED")
        self.assertEqual(
            [item["solution_id"] for item in error["suggestions"]], ["S-1"]
        )
        self.assertTrue(error["suggestions"][0]["solution_name"])
        self.assertTrue(error["suggestions"][0]["category_label"])

    def test_bottleneck_attributes_waiting_to_causes(self) -> None:
        payload = server.planning_bottleneck(solution_id="S-1")

        self.assertTrue(payload["ok"])
        data = payload["data"]
        self.assertEqual(data["solution_id"], "S-1")
        self.assertEqual(
            set(data["wait_breakdown"]),
            {"capacity_bound", "dispatch_bound", "off_shift", "downtime", "idle"},
        )
        # 口径说明必须随 payload 一起走，别让 Agent 又拿利用率榜当瓶颈。
        self.assertIn("machine_utilization_ranking）不能用来回答瓶颈问题", data["method_note"])

    def test_bottleneck_rejects_unknown_solution(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            server.planning_bottleneck(solution_id="S-NOPE")

        self.assertEqual(caught.exception.detail["error"]["code"], "SOLUTION_NOT_FOUND")

    def test_order_delay_explains_a_single_order(self) -> None:
        payload = server.planning_order_delay("O-1", solution_id="S-1")

        self.assertTrue(payload["ok"])
        data = payload["data"]
        self.assertEqual(data["order_id"], "O-1")
        self.assertIn("inevitable_tardiness_hours", data)
        self.assertEqual(
            set(data["attribution"]),
            {"capacity_bound", "dispatch_bound", "off_shift", "downtime", "idle"},
        )

    def test_order_delay_rejects_unknown_order(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            server.planning_order_delay("O-NOPE", solution_id="S-1")

        self.assertEqual(caught.exception.detail["error"]["code"], "ORDER_NOT_FOUND")

    def test_truncated_schedule_is_rejected_not_analysed(self) -> None:
        """剪过的 payload 上算归因是垃圾——大批工序会被误当成"没排入"。"""
        for payload in server._hybrid_tasks["task-1"].values():
            if isinstance(payload, dict) and payload.get("solutions"):
                payload["solutions"][0]["schedule_truncated"] = True

        for call in (
            lambda: server.planning_bottleneck(solution_id="S-1"),
            lambda: server.planning_order_delay("O-1", solution_id="S-1"),
        ):
            with self.assertRaises(HTTPException) as caught:
                call()
            self.assertEqual(
                caught.exception.detail["error"]["code"], "SCHEDULE_TRUNCATED"
            )

    def test_rules_endpoint_lists_builtin_rules(self) -> None:
        payload = server.planning_rules()

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["rule_count"], len(server.BUILTIN_RULES))
        self.assertEqual(
            {item["rule_name"] for item in payload["data"]["rules"]},
            set(server.BUILTIN_RULES),
        )
        atc = next(item for item in payload["data"]["rules"] if item["rule_name"] == "ATC")
        self.assertTrue(atc["is_default"])
        self.assertTrue(atc["description"])

    def test_run_rule_planning_rejects_unknown_rule(self) -> None:
        original = server._simulate_locked
        calls = []
        server._simulate_locked = lambda req: calls.append(req)
        self.addCleanup(setattr, server, "_simulate_locked", original)

        with self.assertRaises(HTTPException) as raised:
            server.run_rule_planning(server.RunRulePlanningReq(rule_name="unknown"))

        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(raised.exception.detail["error"]["code"], "PLANNING_RULE_NOT_FOUND")
        self.assertEqual(calls, [])

    def test_run_rule_planning_returns_compact_result(self) -> None:
        original = server._simulate_locked
        captured = []

        def fake_simulate(req):
            captured.append(req.rule_name)
            return {
                "rule": req.rule_name,
                "metrics": {"total_tardiness": 3.5, "makespan": 20.0},
                "gantt": [{"op_id": f"OP-{index}"} for index in range(25)],
                "diagnosis": "排产完成",
                "diagnosis_detail": {"level": "info"},
            }

        server._simulate_locked = fake_simulate
        self.addCleanup(setattr, server, "_simulate_locked", original)

        payload = server.run_rule_planning(server.RunRulePlanningReq(rule_name=" atc "))

        self.assertEqual(captured, ["ATC"])
        self.assertEqual(payload["data"]["rule_name"], "ATC")
        self.assertEqual(payload["data"]["operation_count"], 25)
        self.assertEqual(len(payload["data"]["planning_preview"]), 20)
        self.assertTrue(payload["data"]["planning_truncated"])

    def test_overview_wraps_data_and_metadata(self) -> None:
        payload = server.planning_overview()

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["candidate_count"], 1)
        self.assertEqual(payload["data"]["archive_size"], 3)
        self.assertEqual(payload["data"]["solutions"][0]["solution_name"], "方案一")
        self.assertEqual(payload["meta"]["time_unit"], "hour")
        self.assertTrue(payload["meta"]["task_defaulted"])

    def test_order_endpoint_returns_full_export_planning(self) -> None:
        payload = server.planning_order("O-1", solution_ids="S-1")

        self.assertEqual(payload["data"]["solutions"][0]["operation_count"], 3)
        self.assertEqual(payload["data"]["solutions"][0]["solution_name"], "方案一")

    def test_operation_endpoint_returns_placement(self) -> None:
        payload = server.planning_operation("OP-13", solution_ids="S-1")

        self.assertTrue(payload["data"]["placements"][0]["planned"])
        self.assertEqual(payload["data"]["placements"][0]["solution_name"], "方案一")
        self.assertEqual(payload["data"]["placements"][0]["machine_id"], "M-A1")

    def test_domain_error_becomes_structured_http_error(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            server.planning_order("missing")

        self.assertEqual(raised.exception.status_code, 404)
        self.assertEqual(raised.exception.detail["error"]["code"], "ORDER_NOT_FOUND")


if __name__ == "__main__":
    unittest.main()
