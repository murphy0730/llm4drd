import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from fastapi import BackgroundTasks
from fastapi import HTTPException

from llm4drd.api import server
from llm4drd.data.db import GraphStore, InstanceStore, init_db
from llm4drd.data.graph_artifact_store import GraphArtifactStore
from llm4drd.knowledge.context_service import GraphContextService
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class GraphApiContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        db_path = str(Path(self.tmp.name) / "graph.db")
        init_db(db_path)
        self.originals = (
            server.graph_store,
            server.graph_artifact_store,
            server.graph_context_service,
            server.inst_store,
            server.shop,
            server._graph_tasks,
        )
        server.graph_store = GraphStore(db_path)
        server.graph_artifact_store = GraphArtifactStore(db_path)
        server.graph_context_service = GraphContextService(
            server.graph_artifact_store
        )
        server.inst_store = InstanceStore(db_path)
        server.shop = None
        server._graph_tasks = {}

    def tearDown(self):
        (
            server.graph_store,
            server.graph_artifact_store,
            server.graph_context_service,
            server.inst_store,
            server.shop,
            server._graph_tasks,
        ) = self.originals
        self.tmp.cleanup()

    def test_graph_build_writes_display_and_compute_meta(self):
        context, diagnostics = server._build_graph_artifacts(
            make_graph_context_shop(), force_rebuild=True
        )
        display_meta = server.graph_store.load_meta()
        compute_meta = server.graph_artifact_store.load_context_meta()

        self.assertEqual(
            display_meta["instance_hash"], compute_meta["instance_hash"]
        )
        self.assertEqual(context.fingerprint.instance_hash, display_meta["instance_hash"])
        self.assertEqual(diagnostics.cache_level, "built")
        self.assertGreater(diagnostics.build_time_ms, 0.0)
        self.assertEqual(
            display_meta["build_time_ms"], diagnostics.build_time_ms
        )
        self.assertEqual(
            compute_meta["build_time_ms"], diagnostics.build_time_ms
        )

    def test_instance_operation_update_invalidates_one_service(self):
        calls = []
        original = server.graph_context_service
        server.graph_context_service = SimpleNamespace(invalidate=calls.append)
        try:
            server._invalidate_graph_context("operation_updated")
        finally:
            server.graph_context_service = original
        self.assertEqual(calls, ["operation_updated"])

    def test_graph_meta_keeps_legacy_fields_and_adds_cache_diagnostics(self):
        server._build_graph_artifacts(
            make_graph_context_shop(), force_rebuild=True
        )

        payload = asyncio.run(server.graph_meta())

        self.assertTrue(
            {
                "total_nodes",
                "total_edges",
                "node_type_counts",
                "edge_type_counts",
                "instance_hash_prefix",
                "topology_hash_prefix",
                "feature_hash_prefix",
                "schema_version",
                "builder_version",
                "cache_ready",
                "build_time_ms",
                "invalid_reason",
            }.issubset(payload)
        )
        self.assertTrue(payload["cache_ready"])
        self.assertEqual(len(payload["instance_hash_prefix"]), 12)

    def test_generated_instance_builds_graph_with_default_shift_values(self):
        async def run_flow():
            await server.gen_instance(
                server.GenReq(
                    num_orders=1,
                    tasks_per_order_min=1,
                    tasks_per_order_max=1,
                    ops_per_task_min=1,
                    ops_per_task_max=1,
                    machines_per_type=1,
                    schedule_days=2,
                    maintenance_prob=0.0,
                    seed=42,
                )
            )
            background = BackgroundTasks()
            response = await server.build_graph(background)
            await background()
            return server._graph_tasks[response["task_id"]]

        task = asyncio.run(run_flow())

        self.assertEqual(task["status"], "done", task.get("error"))

    def test_invalidated_graph_rows_are_not_served(self):
        server._build_graph_artifacts(
            make_graph_context_shop(), force_rebuild=True
        )
        server._invalidate_graph_context("operation_updated")

        meta = asyncio.run(server.graph_meta())
        self.assertFalse(meta["cache_ready"])
        self.assertEqual(meta["invalid_reason"], "operation_updated")

        requests = (
            server.graph_nodes(),
            server.graph_edges(),
            server.graph_order("O-1"),
            server.graph_order_search("O-1"),
            server.node_neighbors("O:O-1"),
        )
        for request in requests:
            with self.subTest(request=request):
                with self.assertRaises(HTTPException) as raised:
                    asyncio.run(request)
                self.assertEqual(raised.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
