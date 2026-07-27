import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import HTTPException

from llm4drd.api import server
from llm4drd.data.db import DispatchStrategyStore, InstanceStore, init_db
from llm4drd.optimization.solution_model import CandidateParameters, FEATURE_NAMES
from llm4drd.tests.shop_fixtures import make_graph_context_shop


def run(coro):
    return asyncio.run(coro)


class DispatchStrategyApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = str(Path(self.tmp.name) / "api.db")
        init_db(self.db_path)
        self.originals = (
            server.dispatch_strategy_store,
            server.inst_store,
            server.shop,
            server._active_shop_cache,
            server._sim_runtime_cache,
            server._hybrid_tasks,
        )
        self.addCleanup(self._restore)
        server.dispatch_strategy_store = DispatchStrategyStore(self.db_path)
        server.inst_store = InstanceStore(self.db_path)
        server.inst_store.save_from_shopfloor(make_graph_context_shop())
        server.shop = None
        server._active_shop_cache = None
        server._sim_runtime_cache = None
        server._hybrid_tasks = {}

        candidate = CandidateParameters(
            feature_weights={name: (1.0 if name == "processing_time" else 0.0) for name in FEATURE_NAMES},
            destroy_weights={},
            repair_weights={},
            op_bias={"OP-11": 0.2},
            seed_rule_name="SPT",
            graph_profile="balanced",
        )
        solution = {
            "solution_id": "S-1",
            "candidate_parameters": candidate.to_dict(),
            "objectives": {"total_tardiness": 1.0},
            "metrics": {"makespan": 14.0},
        }
        server._hybrid_tasks["task-1"] = {
            "status": "done",
            "instance_version": 2,
            "instance_fingerprint": "fp",
            "graph_context_mode": "legacy",
            "config": {"cold_start": True, "objective_keys": ["total_tardiness"]},
            "strategy_context": {"feature_names": list(FEATURE_NAMES), "scale_json": {}},
            "result": {"objective_keys": ["total_tardiness"], "solutions": [solution]},
            "export_result": {"solutions": [solution]},
        }

    def _restore(self):
        (
            server.dispatch_strategy_store,
            server.inst_store,
            server.shop,
            server._active_shop_cache,
            server._sim_runtime_cache,
            server._hybrid_tasks,
        ) = self.originals

    def _save(self):
        return run(server.create_dispatch_strategy_set(server.DispatchStrategySetCreateReq(
            task_id="task-1",
            name="冷启动十方案",
            solutions=[server.DispatchStrategyNameReq(solution_id="S-1", name="方案一｜均衡型")],
        )))

    def test_save_list_and_simulate_named_strategy(self):
        saved = self._save()
        listed = run(server.dispatch_strategy_sets())
        self.assertEqual(listed["sets"][0]["name"], "冷启动十方案")
        strategy_id = saved["strategies"][0]["id"]
        result = server._simulate_locked(server.SimReq(strategy_id=strategy_id))
        self.assertEqual(result["rule"], "方案一｜均衡型")
        self.assertEqual(result["strategy_id"], strategy_id)
        self.assertGreater(result["metrics"]["completed_operations"], 0)

    def test_delete_dispatch_strategy_set_then_404_on_repeat(self):
        saved = self._save()
        deleted = run(server.delete_dispatch_strategy_set(saved["id"]))
        self.assertEqual(deleted["status"], "ok")
        listed = run(server.dispatch_strategy_sets())
        self.assertEqual(listed["count"], 0)
        with self.assertRaises(HTTPException) as ctx:
            run(server.delete_dispatch_strategy_set(saved["id"]))
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
