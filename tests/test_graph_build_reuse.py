"""/api/graph/build 在图谱已构建且指纹未变时应复用持久化产物，而不是每次重建。"""
import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import BackgroundTasks

from llm4drd.api import server
from llm4drd.data.db import GraphStore, InstanceStore, init_db
from llm4drd.data.graph_artifact_store import GraphArtifactStore
from llm4drd.knowledge.context_service import GraphContextService
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class GraphBuildReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        db_path = str(Path(self.tmp.name) / "graph.db")
        init_db(db_path)

        originals = (
            server.graph_store, server.graph_artifact_store, server.graph_context_service,
            server.inst_store, server.shop, server._graph_tasks, server._active_shop_cache,
        )

        def restore():
            (
                server.graph_store, server.graph_artifact_store, server.graph_context_service,
                server.inst_store, server.shop, server._graph_tasks, server._active_shop_cache,
            ) = originals

        self.addCleanup(restore)

        server.graph_store = GraphStore(db_path)
        server.graph_artifact_store = GraphArtifactStore(db_path)
        server.graph_context_service = GraphContextService(server.graph_artifact_store)
        server.inst_store = InstanceStore(db_path)
        server.shop = None
        server._graph_tasks = {}
        server._active_shop_cache = None
        server.inst_store.save_from_shopfloor(make_graph_context_shop())

    def _run_build(self, force=False):
        """跑一次构建端点并同步执行它排的后台任务，返回任务状态。"""
        bg = BackgroundTasks()
        submitted = asyncio.run(server.build_graph(bg, force=force))
        asyncio.run(bg())
        return server._graph_tasks[submitted["task_id"]]

    def test_first_build_builds_from_scratch(self):
        task = self._run_build()
        self.assertEqual(task["status"], "done", task.get("error"))
        self.assertEqual(task["graph_context"]["cache_level"], "built")
        self.assertFalse(task["graph_context"]["cache_hit"])

    def test_second_build_reuses_cached_artifact(self):
        self._run_build()
        task = self._run_build()
        self.assertEqual(task["status"], "done", task.get("error"))
        self.assertTrue(task["graph_context"]["cache_hit"], "指纹未变时应复用已构建的图谱，而不是重建")
        self.assertIn(task["graph_context"]["cache_level"], {"l1", "sqlite"})

    def test_build_reuses_sqlite_artifact_after_process_restart(self):
        """进程重启（L1 内存缓存丢失）后，仍应从 SQLite 复用而不是重建。"""
        self._run_build()
        server.graph_context_service = GraphContextService(server.graph_artifact_store)

        task = self._run_build()
        self.assertEqual(task["status"], "done", task.get("error"))
        self.assertEqual(task["graph_context"]["cache_level"], "sqlite", "重启后应命中 SQLite 持久化产物")
        self.assertTrue(task["graph_context"]["cache_hit"])

    def test_force_rebuilds_even_when_cached(self):
        self._run_build()
        task = self._run_build(force=True)
        self.assertEqual(task["status"], "done", task.get("error"))
        self.assertEqual(task["graph_context"]["cache_level"], "built", "force=True 必须真正重建")

    def test_explicitly_invalidated_graph_is_never_reused(self):
        """服务端标记失效（status != ready）后，即使指纹相同也必须重建。"""
        self._run_build()
        server._invalidate_graph_context("operation_updated")

        task = self._run_build()
        self.assertEqual(task["status"], "done", task.get("error"))
        self.assertEqual(task["graph_context"]["cache_level"], "built", "已标记失效的图谱不能被复用")

    def test_instance_change_invalidates_reuse_and_rebuilds(self):
        self._run_build()
        server.inst_store.update_operation("OP-11", {
            "task_id": "T-11", "op_name": "Cut", "process_type": "cut",
            "processing_time": 99.0, "turnover_time": 0.0,
            "predecessor_ops": "", "predecessor_tasks": "",
            "eligible_machine_ids": "M-C1",
            "required_tooling_types": "", "required_personnel_skills": "",
            "initial_status": "",
        })

        task = self._run_build()
        self.assertEqual(task["status"], "done", task.get("error"))
        self.assertEqual(task["graph_context"]["cache_level"], "built", "实例变更后指纹不同，必须重建而不是复用旧图")


if __name__ == "__main__":
    unittest.main()
