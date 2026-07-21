"""GET /api/workflow/overview 五步进度条聚合状态 + hybrid 运行日志事件。

- 空状态：import 为 current，其余 todo，接口始终返回 5 步；
- 快照齐备时各步 state/tone/detail 正确，current 唯一；
- hybrid status 响应带 events（最近 60 条正序），优化过程记录关键事件。
"""
import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import BackgroundTasks

from llm4drd.api import server
from llm4drd.data.db import GraphStore, InstanceStore, WorkflowProgressStore, init_db
from llm4drd.tests.shop_fixtures import make_graph_context_shop


def run(coro):
    return asyncio.run(coro)


class WorkflowOverviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = str(Path(self.tmp.name) / "overview.db")
        init_db(self.db_path)

        originals = (
            server.inst_store,
            server.workflow_store,
            server.graph_store,
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
                server.graph_store,
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
        server.graph_store = GraphStore(self.db_path)
        server.shop = None
        server._active_shop_cache = None
        server._sim_runtime_cache = None
        server.last_sim_payload = None
        server._hybrid_tasks = {}
        server._latest_hybrid_task_id = None

    # --- 工具 ---

    def _overview(self):
        payload = run(server.workflow_overview())
        return {step["key"]: step for step in payload["steps"]}, payload

    def _load_instance(self):
        server.inst_store.save_from_shopfloor(make_graph_context_shop())

    def _save_validation(self, status, errors=0, warnings=0):
        server.workflow_store.save(
            "validation",
            {"status": status, "error_count": errors, "warning_count": warnings},
        )

    def _graph_meta_ready(self, nodes=428, edges=1204):
        return patch.object(
            server,
            "_graph_meta_payload",
            return_value={
                "total_nodes": nodes,
                "total_edges": edges,
                "cache_ready": True,
                "invalid_reason": "",
            },
        )

    def _store_done_optimization(self, solution_count=2, task_id="opt-1", persist=True):
        task = {
            "status": "done",
            "phase": "done",
            "result": {
                "solutions": [
                    {"solution_id": f"S-{index}"} for index in range(solution_count)
                ],
                "archive_size": solution_count,
            },
            "export_result": {"solutions": []},
            "reference_solutions": [],
        }
        server._hybrid_tasks[task_id] = task
        server._latest_hybrid_task_id = task_id
        if persist:
            server._save_workflow_progress(
                "optimization", {"task_id": task_id, "task": task},
            )
        return task_id, task

    @staticmethod
    def _current_count(payload):
        return sum(1 for step in payload["steps"] if step["state"] == "current")

    # --- 空状态 ---

    def test_empty_state_marks_import_current_and_rest_todo(self):
        steps, payload = self._overview()
        self.assertEqual(
            [step["key"] for step in payload["steps"]],
            ["import", "graph", "simulate", "optimize", "review"],
        )
        self.assertEqual(steps["import"]["state"], "current")
        self.assertEqual(steps["import"]["tone"], "none")
        self.assertEqual(steps["import"]["detail"], "未导入实例")
        for key in ("graph", "simulate", "optimize", "review"):
            self.assertEqual(steps[key]["state"], "todo", key)
            self.assertEqual(steps[key]["tone"], "none", key)
        self.assertEqual(steps["review"]["badge"], "0/4")
        self.assertEqual(self._current_count(payload), 1)

    def test_overview_survives_snapshot_load_failure(self):
        with patch.object(
            server.workflow_store, "load_all", side_effect=RuntimeError("db gone"),
        ):
            payload = run(server.workflow_overview())
        self.assertEqual(
            [step["key"] for step in payload["steps"]],
            ["import", "graph", "simulate", "optimize", "review"],
        )

    # --- import 步 ---

    def test_import_without_validation_snapshot_warns(self):
        self._load_instance()
        steps, payload = self._overview()
        self.assertEqual(steps["import"]["state"], "done")
        self.assertEqual(steps["import"]["tone"], "warn")
        self.assertEqual(steps["import"]["detail"], "已导入 · 未校验")
        # 导入完成但图谱未构建 → graph 被提为 current
        self.assertEqual(steps["graph"]["state"], "current")
        self.assertEqual(steps["graph"]["detail"], "未构建")
        self.assertEqual(self._current_count(payload), 1)

    def test_import_validation_warning_and_failed_tones(self):
        self._load_instance()
        self._save_validation("warning", warnings=3)
        steps, _ = self._overview()
        self.assertEqual(steps["import"]["tone"], "warn")
        self.assertEqual(steps["import"]["detail"], "校验通过 · 3 警告")

        self._save_validation("failed", errors=2, warnings=1)
        steps, _ = self._overview()
        self.assertEqual(steps["import"]["tone"], "err")
        self.assertEqual(steps["import"]["detail"], "校验失败 · 2 错误")

    # --- graph 步 ---

    def test_graph_invalid_marks_blocked(self):
        self._load_instance()
        with patch.object(
            server,
            "_graph_meta_payload",
            return_value={
                "total_nodes": 5,
                "total_edges": 6,
                "cache_ready": False,
                "invalid_reason": "实例已变更",
            },
        ):
            steps, _ = self._overview()
        self.assertEqual(steps["graph"]["state"], "blocked")
        self.assertEqual(steps["graph"]["tone"], "err")
        self.assertEqual(steps["graph"]["detail"], "图谱失效 · 需重建")
        # graph 被 blocked 跳过，第一个非 done 非 blocked 的 simulate 提为 current
        self.assertEqual(steps["simulate"]["state"], "current")
        self.assertEqual(steps["optimize"]["state"], "todo")
        self.assertEqual(steps["review"]["state"], "todo")

    # --- 全链路 ---

    def test_full_flow_all_done_has_no_current(self):
        self._load_instance()
        self._save_validation("passed")
        server.workflow_store.save(
            "simulation", {"rule": "ATC", "metrics": {"feasible": True}},
        )
        self._store_done_optimization(solution_count=2)
        server.workflow_store.save(
            "review", {"selection": ["S-0", "S-1"], "detail_id": "S-0"},
        )
        with self._graph_meta_ready():
            steps, payload = self._overview()
        self.assertEqual(steps["import"]["tone"], "ok")
        self.assertEqual(steps["import"]["detail"], "校验通过")
        self.assertEqual(steps["graph"]["state"], "done")
        self.assertEqual(steps["graph"]["tone"], "ok")
        self.assertEqual(steps["graph"]["detail"], "428 节点 · 1,204 边")
        self.assertEqual(steps["simulate"]["state"], "done")
        self.assertEqual(steps["simulate"]["tone"], "ok")
        self.assertEqual(steps["simulate"]["detail"], "ATC · 可行")
        self.assertEqual(steps["optimize"]["state"], "done")
        self.assertEqual(steps["optimize"]["tone"], "ok")
        self.assertEqual(steps["optimize"]["detail"], "2 候选方案")
        self.assertEqual(steps["review"]["state"], "done")
        self.assertEqual(steps["review"]["tone"], "ok")
        self.assertEqual(steps["review"]["detail"], "已选 2 个方案")
        self.assertEqual(steps["review"]["badge"], "2/4")
        self.assertEqual(self._current_count(payload), 0)

    def test_simulation_incomplete_marks_warn_and_review_current(self):
        self._load_instance()
        self._save_validation("passed")
        server.workflow_store.save(
            "simulation", {"rule": "EDD", "metrics": {"feasible": False}},
        )
        self._store_done_optimization(solution_count=3)
        with self._graph_meta_ready():
            steps, payload = self._overview()
        self.assertEqual(steps["simulate"]["state"], "done")
        self.assertEqual(steps["simulate"]["tone"], "warn")
        self.assertEqual(steps["simulate"]["detail"], "EDD · 不完整")
        # 有候选但未选 → review 是当前步
        self.assertEqual(steps["review"]["state"], "current")
        self.assertEqual(steps["review"]["tone"], "none")
        self.assertEqual(steps["review"]["detail"], "已有 3 候选")
        self.assertEqual(steps["review"]["badge"], "0/4")
        self.assertEqual(self._current_count(payload), 1)

    # --- optimize 步 ---

    def test_running_optimize_is_current(self):
        self._load_instance()
        self._save_validation("passed")
        server.workflow_store.save(
            "simulation", {"rule": "ATC", "metrics": {"feasible": True}},
        )
        server._hybrid_tasks["run-1"] = {
            "status": "running",
            "phase": "coarse",
            "real_progress": 0.63,
        }
        server._latest_hybrid_task_id = "run-1"
        with self._graph_meta_ready():
            steps, payload = self._overview()
        self.assertEqual(steps["optimize"]["state"], "current")
        self.assertEqual(steps["optimize"]["tone"], "run")
        self.assertEqual(steps["optimize"]["detail"], "运行中 · 63%")
        self.assertEqual(steps["review"]["state"], "todo")
        self.assertEqual(self._current_count(payload), 1)

    def test_review_shows_snapshot_candidates_while_new_run_in_progress(self):
        self._load_instance()
        self._save_validation("passed")
        server.workflow_store.save(
            "simulation", {"rule": "ATC", "metrics": {"feasible": True}},
        )
        old_task = {
            "status": "done",
            "phase": "done",
            "result": {
                "solutions": [{"solution_id": f"S-{index}"} for index in range(4)],
            },
        }
        server._save_workflow_progress(
            "optimization", {"task_id": "old-1", "task": old_task},
        )
        server._hybrid_tasks["run-1"] = {
            "status": "running",
            "phase": "coarse",
            "real_progress": 0.1,
        }
        server._latest_hybrid_task_id = "run-1"
        with self._graph_meta_ready():
            steps, _ = self._overview()
        self.assertEqual(steps["optimize"]["state"], "current")
        self.assertEqual(steps["optimize"]["tone"], "run")
        self.assertEqual(steps["review"]["state"], "todo")
        self.assertEqual(steps["review"]["detail"], "已有 4 候选")

    def test_optimize_error_marks_blocked(self):
        self._load_instance()
        server._hybrid_tasks["bad-1"] = {"status": "error", "error": "boom"}
        server._latest_hybrid_task_id = "bad-1"
        steps, _ = self._overview()
        self.assertEqual(steps["optimize"]["state"], "blocked")
        self.assertEqual(steps["optimize"]["tone"], "err")
        self.assertEqual(steps["optimize"]["detail"], "优化失败")

    def test_done_optimization_from_snapshot_only(self):
        """重启后内存任务丢失，只有落库快照时 optimize 仍显示 done。"""
        self._load_instance()
        _, task = self._store_done_optimization(solution_count=5)
        server._hybrid_tasks = {}
        server._latest_hybrid_task_id = None
        steps, _ = self._overview()
        self.assertEqual(steps["optimize"]["state"], "done")
        self.assertEqual(steps["optimize"]["detail"], "5 候选方案")
        self.assertEqual(steps["review"]["detail"], "已有 5 候选")


class HybridEventLogTests(unittest.TestCase):
    """hybrid 任务 events：记录关键节点，status 响应返回最近 60 条。"""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = str(Path(self.tmp.name) / "events.db")
        init_db(self.db_path)

        originals = (
            server.shop,
            server.workflow_store,
            server._hybrid_tasks,
            server._latest_hybrid_task_id,
        )

        def _restore():
            (
                server.shop,
                server.workflow_store,
                server._hybrid_tasks,
                server._latest_hybrid_task_id,
            ) = originals

        self.addCleanup(_restore)

        server.shop = make_graph_context_shop()
        server.workflow_store = WorkflowProgressStore(self.db_path)
        server._hybrid_tasks = {}
        server._latest_hybrid_task_id = None

    def test_log_hybrid_event_keeps_last_120(self):
        task = {"phase": "coarse"}
        for index in range(130):
            server._log_hybrid_event(task, f"事件 {index}")
        self.assertEqual(len(task["events"]), 120)
        self.assertEqual(task["events"][0]["text"], "事件 10")
        latest = task["events"][-1]
        self.assertEqual(latest["text"], "事件 129")
        self.assertEqual(latest["phase"], "coarse")
        self.assertIn("ts", latest)
        server._log_hybrid_event(task, "显式阶段", phase="finalize")
        self.assertEqual(task["events"][-1]["phase"], "finalize")

    def test_status_response_includes_last_60_events(self):
        server._hybrid_tasks["t-1"] = {
            "status": "running",
            "phase": "coarse",
            "events": [
                {"ts": float(index), "phase": "coarse", "text": f"e{index}"}
                for index in range(80)
            ],
        }
        payload = run(server.optimize_hybrid_status("t-1"))
        self.assertEqual(len(payload["events"]), 60)
        self.assertEqual(payload["events"][0]["text"], "e20")
        self.assertEqual(payload["events"][-1]["text"], "e79")

        # 旧任务（重启前创建，没有 events 字段）也不报错
        server._hybrid_tasks["t-2"] = {"status": "running", "phase": "coarse"}
        payload = run(server.optimize_hybrid_status("t-2"))
        self.assertEqual(payload["events"], [])

    def _run_with_optimizer(self, optimizer_cls):
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
            patch.object(server, "HybridNSGA3ALNSOptimizer", optimizer_cls),
            patch.object(server, "_active_shop", return_value=server.shop),
        ):
            response, background = asyncio.run(submit())
            asyncio.run(background())
        return response["task_id"], server._hybrid_tasks[response["task_id"]]

    def test_run_records_key_events(self):
        class _FakeResult:
            def to_dict(self):
                return {
                    "solutions": [{"solution_id": "S-1"}],
                    "archive_size": 3,
                    "generations_completed": 2,
                    "elapsed_s": 1.5,
                    "total_evaluations": 42,
                }

            def to_export_dict(self):
                return {"solutions": [{"solution_id": "S-1", "schedule": []}]}

        class _FakeOptimizer:
            graph_context_diff = None

            def __init__(self, *_args, **_kwargs):
                pass

            def run(self, progress_callback=None):
                progress_callback(
                    {"phase": "coarse", "archive_size": 1, "real_progress": 0.2},
                )
                progress_callback(
                    {"phase": "exact_promotion", "archive_size": 3, "real_progress": 0.6},
                )
                return _FakeResult()

        task_id, task = self._run_with_optimizer(_FakeOptimizer)
        texts = [event["text"] for event in task["events"]]
        self.assertIn("初始化完成 · 种群 24 · 基线 ATC 注入", texts)
        self.assertIn("阶段切换 · 精确评估", texts)
        self.assertIn("发现新非支配解 · 前沿规模 1", texts)
        self.assertIn("发现新非支配解 · 前沿规模 3", texts)
        self.assertIn("优化完成 · 候选方案 1 个 · 总评估 42 次", texts)
        # phase 未变化的回调不重复记阶段切换（任务进入 run 前已是 coarse）
        self.assertNotIn("阶段切换 · 近似广搜", texts)
        # status 端点透出同样的 events（不足 60 条全量返回，正序）
        status = run(server.optimize_hybrid_status(task_id))
        self.assertEqual(status["status"], "done")
        self.assertEqual([event["text"] for event in status["events"]], texts)

    def test_failed_run_records_error_event(self):
        class _FailingOptimizer:
            graph_context_diff = None

            def __init__(self, *_args, **_kwargs):
                pass

            def run(self, progress_callback=None):
                raise RuntimeError("boom")

        _, task = self._run_with_optimizer(_FailingOptimizer)
        texts = [event["text"] for event in task["events"]]
        self.assertIn("优化失败 · boom", texts)
        # 失败前的初始化事件仍然保留
        self.assertIn("初始化完成 · 种群 24 · 基线 ATC 注入", texts)


if __name__ == "__main__":
    unittest.main()
