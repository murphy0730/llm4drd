import inspect
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from llm4drd.api import server
from llm4drd.api.review_read import ReviewReadCache
from llm4drd.data.db import (
    InstanceStore,
    RuleReferenceCacheStore,
    WorkflowProgressStore,
    init_db,
)
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class ReviewApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        db_path = str(Path(self.tmp.name) / "review-api.db")
        init_db(db_path)
        originals = (
            server.inst_store,
            server.workflow_store,
            server.shop,
            server._active_shop_cache,
            server._hybrid_tasks,
            server._latest_hybrid_task_id,
            getattr(server, "review_read_cache", None),
            server.rule_reference_cache_store,
        )
        self.addCleanup(self._restore, originals)
        server.inst_store = InstanceStore(db_path)
        server.workflow_store = WorkflowProgressStore(db_path)
        server.rule_reference_cache_store = RuleReferenceCacheStore(db_path)
        server.inst_store.save_from_shopfloor(make_graph_context_shop())
        server.shop = None
        server._active_shop_cache = None
        if hasattr(server, "review_read_cache"):
            server.review_read_cache = ReviewReadCache(8)

        def entry(solution_id, order_id, machine_id, start):
            return {
                "op_id": f"{solution_id}-{order_id}",
                "order_id": order_id,
                "order_name": f"订单{order_id}",
                "machine_id": machine_id,
                "start": start,
                "end": start + 1,
            }

        solutions = [
            {
                "solution_id": f"S-{index}",
                "schedule": [entry(f"S{index}", "O-001", "M-C1", index)],
            }
            for index in range(1, 6)
        ]
        server._hybrid_tasks = {
            "t1": {
                "status": "done",
                "result": {"solutions": [{"solution_id": item["solution_id"]} for item in solutions]},
                "export_result": {"solutions": solutions},
                "reference_solutions": [],
            }
        }
        server._latest_hybrid_task_id = "t1"

    def _restore(self, originals):
        (
            server.inst_store,
            server.workflow_store,
            server.shop,
            server._active_shop_cache,
            server._hybrid_tasks,
            server._latest_hybrid_task_id,
            review_cache,
            server.rule_reference_cache_store,
        ) = originals
        if hasattr(server, "review_read_cache"):
            server.review_read_cache = review_cache

    def test_batch_returns_two_schemes_and_utilization(self):
        response = server.optimize_review_data("t1", "S-1,S-2", "O-001", True)
        payload = json.loads(response.body)
        self.assertEqual(payload["solutions"], ["S-1", "S-2"])
        self.assertEqual(set(payload["schemes"]), {"S-1", "S-2"})
        self.assertIsNotNone(payload["type_utilization"])

    def test_order_only_request_omits_utilization(self):
        response = server.optimize_review_data("t1", "S-1,S-2", "O-001", False)
        self.assertIsNone(json.loads(response.body)["type_utilization"])

    def test_search_returns_ranked_union(self):
        response = server.optimize_review_orders("t1", "S-1,S-2", "001", 50)
        self.assertEqual(json.loads(response.body)["orders"][0]["order_id"], "O-001")

    def test_solution_ids_are_deduplicated_and_capped_at_four(self):
        response = server.optimize_review_data(
            "t1", "S-1,S-1,S-2,S-3,S-4,S-5", "O-001", False
        )
        payload = json.loads(response.body)
        self.assertEqual(payload["solutions"], ["S-1", "S-2", "S-3", "S-4"])

    def test_uncached_rule_is_reported_without_losing_valid_scheme(self):
        response = server.optimize_review_data(
            "t1", "S-1,RULE:NEVER-CACHED", "O-001", True
        )
        payload = json.loads(response.body)
        self.assertEqual(payload["solutions"], ["S-1"])
        self.assertEqual(payload["failed_solution_ids"], ["RULE:NEVER-CACHED"])
        self.assertIn("尚未计算完整排程", payload["failure_messages"]["RULE:NEVER-CACHED"])

    def test_repeated_reads_reuse_built_solution_index(self):
        with patch.object(
            server,
            "build_review_solution_index",
            wraps=server.build_review_solution_index,
        ) as build:
            server.optimize_review_data("t1", "S-1,S-2", "O-001", True)
            server.optimize_review_orders("t1", "S-1,S-2", "001", 50)
        self.assertEqual(build.call_count, 2)

    def test_endpoints_are_synchronous(self):
        self.assertFalse(inspect.iscoroutinefunction(server.optimize_review_data))
        self.assertFalse(inspect.iscoroutinefunction(server.optimize_review_orders))


if __name__ == "__main__":
    unittest.main()
