"""重启后从库里恢复流程进度，而不是把整条流程重跑一遍。

覆盖三件事：跑完的步骤会落库；进程重启（_restore_workflow_progress）后导出和
评审接口仍能用；实例一改动，落库的进度自动失效。
"""
import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from llm4drd.api import server
from llm4drd.data.db import InstanceStore, WorkflowProgressStore, init_db
from llm4drd.tests.shop_fixtures import make_graph_context_shop


def run(coro):
    return asyncio.run(coro)


class WorkflowProgressApiTests(unittest.TestCase):
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

    def _simulate_restart(self):
        """丢掉全部内存态，只留库——等价于重启进程。"""
        server._active_shop_cache = None
        server._sim_runtime_cache = None
        server.last_sim_payload = None
        server._hybrid_tasks = {}
        server._latest_hybrid_task_id = None
        server._restore_workflow_progress()

    # --- 强校验 ---

    def test_validation_result_is_persisted_and_reused(self):
        first = run(server.validate_instance())
        self.assertEqual(server.workflow_store.load("validation"), first)

        with patch.object(server, "_validate_instance") as recompute:
            second = run(server.validate_instance())
        recompute.assert_not_called()
        self.assertEqual(second, first, "库里已有匹配当前实例的校验结论，不该重算")

    def test_validation_force_recomputes_and_overwrites(self):
        run(server.validate_instance())
        server.workflow_store.save("validation", {"status": "stale-marker"})

        refreshed = run(server.validate_instance(force=True))
        self.assertNotEqual(refreshed.get("status"), "stale-marker")
        self.assertEqual(
            server.workflow_store.load("validation"), refreshed, "重新执行必须覆盖旧数据",
        )

    def test_instance_edit_forces_validation_recompute(self):
        run(server.validate_instance())
        server.inst_store.update_order(
            "O-1", {"order_name": "Renamed", "release_time": 0.0, "due_date": 40.0, "priority": 3})

        with patch.object(
            server, "_validate_instance", wraps=server._validate_instance,
        ) as recompute:
            run(server.validate_instance())
        recompute.assert_called_once()

    # --- 仿真 ---

    def test_simulation_survives_restart_and_keeps_export_working(self):
        payload = server.simulate(server.SimReq(rule_name="ATC"))
        self.assertTrue(payload["gantt"])

        self._simulate_restart()
        self.assertEqual(server.last_sim_payload, payload, "重启后导出端点依赖的仿真结果必须恢复")

        progress = run(server.workflow_progress())
        self.assertEqual(progress["simulation"], payload)

    def test_rerunning_simulation_overwrites_stored_result(self):
        server.simulate(server.SimReq(rule_name="ATC"))
        server.simulate(server.SimReq(rule_name="EDD"))
        self.assertEqual(server.workflow_store.load("simulation")["rule"], "EDD")

    def test_instance_edit_drops_stored_simulation(self):
        server.simulate(server.SimReq(rule_name="ATC"))
        server.inst_store.update_order(
            "O-1", {"order_name": "Renamed", "release_time": 0.0, "due_date": 40.0, "priority": 3})

        self._simulate_restart()
        self.assertIsNone(server.last_sim_payload, "实例已变，旧排程不能当作当前仿真结果")
        self.assertIsNone(run(server.workflow_progress())["simulation"])

    # --- 优化 ---

    def _store_finished_optimization(self, task_id="opt-1"):
        task = {
            "status": "done",
            "phase": "done",
            "result": {
                "objective_keys": ["makespan"],
                "solutions": [{"solution_id": "S-1", "objectives": {"makespan": 10.0}}],
                "archive_size": 1,
            },
            "export_result": {"solutions": [{"solution_id": "S-1", "schedule": []}]},
            "reference_solutions": [],
        }
        server._hybrid_tasks[task_id] = task
        server._latest_hybrid_task_id = task_id
        server._save_workflow_progress("optimization", {"task_id": task_id, "task": task})
        return task_id, task

    def test_optimization_result_survives_restart(self):
        task_id, task = self._store_finished_optimization()
        self._simulate_restart()

        self.assertEqual(server._latest_hybrid_task_id, task_id)
        self.assertEqual(server._hybrid_tasks[task_id]["result"], task["result"])
        restored = run(server.optimize_hybrid_result(task_id))
        self.assertEqual(restored["solutions"], task["result"]["solutions"])

    def test_restored_optimization_keeps_export_payload(self):
        task_id, task = self._store_finished_optimization()
        self._simulate_restart()
        self.assertEqual(
            server._hybrid_tasks[task_id]["export_result"],
            task["export_result"],
            "导出用的完整排程也要恢复，否则重启后方案无法导出",
        )

    def test_workflow_progress_exposes_optimization_result(self):
        task_id, task = self._store_finished_optimization()
        self._simulate_restart()

        optimization = run(server.workflow_progress())["optimization"]
        self.assertEqual(optimization["task_id"], task_id)
        self.assertEqual(optimization["status"], "done")
        self.assertEqual(optimization["solutions"], task["result"]["solutions"])

    def test_instance_edit_drops_stored_optimization(self):
        self._store_finished_optimization()
        server.inst_store.update_order(
            "O-1", {"order_name": "Renamed", "release_time": 0.0, "due_date": 40.0, "priority": 3})

        self._simulate_restart()
        self.assertEqual(server._hybrid_tasks, {})
        self.assertIsNone(run(server.workflow_progress())["optimization"])

    # --- 评审 ---

    def test_review_selection_roundtrips(self):
        run(server.save_review_progress(
            server.ReviewProgressReq(selection=["S-1", "S-2"], detail_id="S-2")))

        review = run(server.workflow_progress())["review"]
        self.assertEqual(review["selection"], ["S-1", "S-2"])
        self.assertEqual(review["detail_id"], "S-2")

    def test_instance_edit_drops_review_selection(self):
        run(server.save_review_progress(server.ReviewProgressReq(selection=["S-1"])))
        server.inst_store.clear_all()
        self.assertIsNone(run(server.workflow_progress())["review"])

    # --- 持久化失败不影响计算 ---

    def test_progress_save_failure_does_not_break_simulation(self):
        with patch.object(
            server.workflow_store, "save", side_effect=RuntimeError("disk full"),
        ):
            payload = server.simulate(server.SimReq(rule_name="ATC"))
        self.assertTrue(payload["gantt"], "存进度失败不该把跑完的仿真判成失败")


if __name__ == "__main__":
    unittest.main()
