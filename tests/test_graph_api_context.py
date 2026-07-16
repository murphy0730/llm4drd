import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from llm4drd.api import server
from llm4drd.data.db import GraphStore, init_db
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
        )
        server.graph_store = GraphStore(db_path)
        server.graph_artifact_store = GraphArtifactStore(db_path)
        server.graph_context_service = GraphContextService(
            server.graph_artifact_store
        )

    def tearDown(self):
        (
            server.graph_store,
            server.graph_artifact_store,
            server.graph_context_service,
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


if __name__ == "__main__":
    unittest.main()
