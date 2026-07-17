import asyncio
import json
import subprocess
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import BackgroundTasks

from llm4drd.api import server
from llm4drd.optimization.hybrid_nsga3_alns import (
    HybridConfig,
    HybridNSGA3ALNSOptimizer,
)
from llm4drd.optimization.exact import ExactSolver
from llm4drd.optimization.pareto import NSGA2Optimizer
from llm4drd.tests.shop_fixtures import make_graph_context_shop


ROOT = Path(__file__).resolve().parents[1]


class HybridOptimizerRealProgressTests(unittest.TestCase):
    def test_progress_is_monotonic_and_covers_internal_stage_work(self):
        optimizer = HybridNSGA3ALNSOptimizer(
            make_graph_context_shop(),
            HybridConfig(
                objective_keys=["total_tardiness", "makespan"],
                target_solution_count=2,
                population_size=4,
                generations=1,
                alns_iterations_per_candidate=1,
                refine_rounds=1,
                time_limit_s=60,
                parallel_workers=1,
                seed=23,
            ),
        )
        snapshots = []

        result = optimizer.run(progress_callback=lambda item: snapshots.append(dict(item)))

        self.assertGreaterEqual(result.found_solution_count, 1)
        self.assertTrue(result.solutions[0]["schedule"])
        self.assertGreater(len(snapshots), 6)
        scores = [snapshot["real_progress"] for snapshot in snapshots]
        self.assertEqual(scores, sorted(scores))
        self.assertEqual(scores[-1], 1.0)
        for snapshot in snapshots:
            self.assertGreaterEqual(snapshot["phase_progress"], 0.0)
            self.assertLessEqual(snapshot["phase_progress"], 1.0)
            self.assertGreaterEqual(snapshot["phase_total"], snapshot["phase_completed"])

        exact_updates = [
            snapshot
            for snapshot in snapshots
            if snapshot["phase"] == "exact_promotion"
        ]
        self.assertGreaterEqual(len(exact_updates), 2)
        self.assertTrue(
            any(
                0 < snapshot["phase_completed"] < snapshot["phase_total"]
                for snapshot in exact_updates
            )
        )
        self.assertTrue(
            any(snapshot["phase"] == "finalize" for snapshot in snapshots)
        )


class RelatedOptimizerProgressTests(unittest.TestCase):
    def test_nsga2_reports_each_completed_evaluation(self):
        updates = []
        optimizer = NSGA2Optimizer(
            make_graph_context_shop(),
            ["total_tardiness", "makespan"],
            pop_size=4,
            generations=1,
            seed=31,
        )

        solutions = optimizer.run(
            callback=lambda done, total, generation: updates.append(
                (done, total, generation)
            )
        )

        self.assertTrue(solutions)
        self.assertEqual(updates[-1][:2], (8, 8))
        self.assertEqual(
            sorted({done for done, _total, _generation in updates}),
            list(range(1, 9)),
        )

    def test_exact_solver_reports_model_search_and_completion(self):
        updates = []

        result = ExactSolver(
            make_graph_context_shop(),
            ["makespan"],
            time_limit_s=5,
        ).solve(progress_callback=lambda snapshot: updates.append(dict(snapshot)))

        self.assertIn(result.status, {"OPTIMAL", "FEASIBLE"})
        self.assertTrue(result.schedule)
        self.assertGreater(len(updates), 4)
        scores = [snapshot["real_progress"] for snapshot in updates]
        self.assertEqual(scores, sorted(scores))
        self.assertEqual(scores[-1], 1.0)
        self.assertIn("model_build", {item["phase"] for item in updates})
        self.assertIn("search", {item["phase"] for item in updates})
        self.assertEqual(updates[-1]["phase"], "done")


class HybridStatusActivityTests(unittest.TestCase):
    def setUp(self):
        self.original_state = (
            server.shop,
            server._hybrid_tasks,
            server._latest_hybrid_task_id,
        )
        server.shop = make_graph_context_shop()
        server._hybrid_tasks = {}
        server._latest_hybrid_task_id = None
        self.addCleanup(self._restore_state)

    def _restore_state(self):
        (
            server.shop,
            server._hybrid_tasks,
            server._latest_hybrid_task_id,
        ) = self.original_state

    def test_heartbeat_marks_stall_without_claiming_real_progress(self):
        entered_run = threading.Event()
        release_run = threading.Event()

        class BlockingOptimizer:
            graph_context_diff = None

            def __init__(self, *_args, **_kwargs):
                pass

            def run(self, progress_callback=None):
                entered_run.set()
                release_run.wait(timeout=2.0)
                raise RuntimeError("injected stop")

        async def submit():
            background = BackgroundTasks()
            response = await server.optimize_hybrid(
                server.HybridOptimizeReq(time_limit_s=30),
                background,
            )
            return response, background

        with (
            patch.object(
                server,
                "resolve_graph_context_mode",
                return_value=server.GraphContextMode.LEGACY,
            ),
            patch.object(server, "HybridNSGA3ALNSOptimizer", BlockingOptimizer),
            patch.object(server, "_active_shop", return_value=server.shop),
            patch.object(server, "OPTIMIZE_HEARTBEAT_INTERVAL_S", 0.01),
            patch.object(server, "OPTIMIZE_STALL_THRESHOLD_S", 0.05),
        ):
            response, background = asyncio.run(submit())
            task_id = response["task_id"]
            worker = threading.Thread(target=lambda: asyncio.run(background()))
            worker.start()
            self.assertTrue(entered_run.wait(timeout=1.0))
            initial_real_at = server._hybrid_tasks[task_id][
                "last_real_progress_at"
            ]
            time.sleep(0.09)

            status = asyncio.run(server.optimize_hybrid_status(task_id))
            self.assertTrue(status["stalled"])
            self.assertGreaterEqual(status["seconds_since_real_progress"], 0.05)
            self.assertEqual(status["last_real_progress_at"], initial_real_at)
            self.assertGreater(status["updated_at"], status["last_real_progress_at"])
            self.assertEqual(status["real_progress"], 0.0)

            release_run.set()
            worker.join(timeout=1.0)
            self.assertFalse(worker.is_alive())


class FrontendOptimizeProgressTests(unittest.TestCase):
    def _node_eval(self, expression: str):
        script = (
            "const p=require('./frontend/optimize_progress.js');"
            f"console.log(JSON.stringify({expression}));"
        )
        result = subprocess.run(
            ["node", "-e", script],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    def test_wall_clock_does_not_move_progress(self):
        values = self._node_eval(
            "[p.optimizeProgress({status:'running',phase:'coarse',"
            "real_progress:0.27,elapsed_s:10}),"
            "p.optimizeProgress({status:'running',phase:'coarse',"
            "real_progress:0.27,elapsed_s:1200})]"
        )
        self.assertEqual(values, [27, 27])

    def test_stage_subprogress_and_stall_are_exposed(self):
        payload = self._node_eval(
            "({progress:p.optimizeProgress({status:'running',"
            "phase:'exact_promotion',real_progress:0.73,"
            "phase_completed:2,phase_total:5}),"
            "activity:p.optimizeActivity({status:'running',"
            "last_real_progress_at:100,stalled:false},165,60)})"
        )
        self.assertEqual(payload["progress"], 73)
        self.assertTrue(payload["activity"]["stalled"])
        self.assertEqual(payload["activity"]["secondsSinceRealProgress"], 65)


if __name__ == "__main__":
    unittest.main()
