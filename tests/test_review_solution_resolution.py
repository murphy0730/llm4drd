import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException

from llm4drd.api import server
from llm4drd.data.db import InstanceStore, RuleReferenceCacheStore, init_db
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class ReviewSolutionResolutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        db_path = str(Path(self.tmp.name) / "review.db")
        init_db(db_path)
        InstanceStore(db_path).save_from_shopfloor(make_graph_context_shop())
        self.cache = RuleReferenceCacheStore(db_path)

    def test_export_reference_wins_over_truncated_task_view(self):
        full = {"solution_id": "R-1", "schedule": [{"op_id": "full"}]}
        truncated = {
            "solution_id": "R-1",
            "schedule": [],
            "summary": {"total_operations": 1},
        }
        task = {
            "export_result": {"reference_solutions": [full], "solutions": []},
            "reference_solutions": [truncated],
        }
        self.assertIs(server._resolve_export_solution(None, task, "R-1"), full)

    def test_rule_cache_hit_never_runs_simulation(self):
        cached = {"solution_id": "RULE:EDD", "schedule": [{"op_id": "cached"}]}
        self.cache.put("EDD", cached)
        with (
            patch.object(server, "rule_reference_cache_store", self.cache),
            patch.object(
                server,
                "_rule_reference_solution",
                side_effect=AssertionError("must not simulate"),
            ),
        ):
            resolved = server._resolve_export_solution(
                make_graph_context_shop(),
                {"export_result": {"solutions": [], "reference_solutions": []}},
                "RULE:EDD",
            )
        self.assertEqual(resolved["schedule"], [{"op_id": "cached"}])

    def test_rule_cache_miss_returns_409_without_simulation(self):
        with (
            patch.object(server, "rule_reference_cache_store", self.cache),
            patch.object(
                server,
                "_rule_reference_solution",
                side_effect=AssertionError("must not simulate"),
            ),
            self.assertRaises(HTTPException) as ctx,
        ):
            server._resolve_export_solution(
                make_graph_context_shop(),
                {"export_result": {"solutions": [], "reference_solutions": []}},
                "RULE:SPT",
            )
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("尚未计算", str(ctx.exception.detail))


if __name__ == "__main__":
    unittest.main()
