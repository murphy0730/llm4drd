import asyncio
import threading
import time
import unittest
from unittest.mock import patch

from fastapi import BackgroundTasks

from llm4drd.api import server
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class OptimizeProgressHeartbeatTests(unittest.TestCase):
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

    def test_initial_population_work_keeps_task_alive_and_moves_past_initializing(self):
        entered_run = threading.Event()
        release_run = threading.Event()

        class BlockingOptimizer:
            graph_context_diff = None

            def __init__(self, *_args, **_kwargs):
                pass

            def run(self, progress_callback=None):
                entered_run.set()
                release_run.wait(timeout=2.0)
                raise RuntimeError("test stops after observing heartbeat")

        async def submit():
            background = BackgroundTasks()
            response = await server.optimize_hybrid(
                server.HybridOptimizeReq(time_limit_s=30),
                background,
            )
            return response, background

        with (
            patch.object(server, "resolve_graph_context_mode", return_value=server.GraphContextMode.LEGACY),
            patch.object(server, "HybridNSGA3ALNSOptimizer", BlockingOptimizer),
            patch.object(server, "_active_shop", return_value=server.shop),
            patch.object(server, "OPTIMIZE_HEARTBEAT_INTERVAL_S", 0.05, create=True),
        ):
            response, background = asyncio.run(submit())
            task_id = response["task_id"]
            initial_updated_at = server._hybrid_tasks[task_id]["updated_at"]
            worker = threading.Thread(target=lambda: asyncio.run(background()))
            worker.start()
            self.assertTrue(entered_run.wait(timeout=1.0))

            deadline = time.time() + 1.0
            while (
                (
                    server._hybrid_tasks[task_id]["updated_at"] <= initial_updated_at
                    or server._hybrid_tasks[task_id]["elapsed_s"] <= 0.0
                )
                and time.time() < deadline
            ):
                time.sleep(0.01)

            task = server._hybrid_tasks[task_id]
            self.assertEqual(task["phase"], "coarse")
            self.assertGreater(task["updated_at"], initial_updated_at)
            self.assertGreater(task["elapsed_s"], 0.0)

            release_run.set()
            worker.join(timeout=1.0)
            self.assertFalse(worker.is_alive())

            stopped_at = task["updated_at"]
            time.sleep(0.12)
            self.assertEqual(task["updated_at"], stopped_at)
            self.assertEqual(task["status"], "error")


if __name__ == "__main__":
    unittest.main()
