import os
import sqlite3
import threading
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from llm4drd.data.db import init_db
from llm4drd.data.graph_artifact_store import GraphArtifactStore
from llm4drd.knowledge.canonical import CanonicalGraphBuilder, GraphFingerprint
from llm4drd.knowledge.context import (
    GraphContextBuildError,
    GraphContextCorruptError,
    GraphContextStaleError,
)
from llm4drd.knowledge.context_service import (
    GraphContextMode,
    GraphContextService,
    resolve_graph_context_mode,
)
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class GraphContextServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "graph.db")
        init_db(self.db_path)
        self.store = GraphArtifactStore(self.db_path)
        self.service = GraphContextService(self.store)
        self.shop = make_graph_context_shop()

    def tearDown(self):
        self.tmp.cleanup()

    def test_miss_then_l1_hit_then_sqlite_hit(self):
        with self.assertLogs(
            "llm4drd.knowledge.context_service", level="INFO"
        ) as captured:
            first, a = self.service.get_or_build(self.shop)
            second, b = self.service.get_or_build(self.shop)
            self.service.clear_memory_cache()
            third, c = self.service.get_or_build(self.shop)
        self.assertEqual(a.cache_level, "built")
        self.assertIs(second, first)
        self.assertEqual(b.cache_level, "l1")
        self.assertEqual(third, first)
        self.assertEqual(c.cache_level, "sqlite")
        self.assertTrue(c.cache_hit)
        output = "\n".join(captured.output)
        for event in (
            "graph_context.miss",
            "graph_context.build_started",
            "graph_context.build_completed",
            "graph_context.l1_hit",
            "graph_context.sqlite_hit",
        ):
            self.assertIn(event, output)
        self.assertIn("operation_count=4", output)
        self.assertIn("elapsed_ms=", output)

    def test_invalidation_forces_rebuild(self):
        _, first = self.service.get_or_build(self.shop)
        with self.assertLogs(
            "llm4drd.knowledge.context_service", level="INFO"
        ) as captured:
            self.service.invalidate("operation_updated")
        _, second = self.service.get_or_build(self.shop)
        self.assertEqual(first.cache_level, "built")
        self.assertEqual(second.cache_level, "built")
        self.assertEqual(second.invalid_reason, "operation_updated")
        self.assertIn("graph_context.invalidated", "\n".join(captured.output))

    def test_force_rebuild_bypasses_l1_and_sqlite(self):
        self.service.get_or_build(self.shop)

        _, diagnostics = self.service.get_or_build(self.shop, force_rebuild=True)

        self.assertEqual(diagnostics.cache_level, "built")

    def test_same_fingerprint_builds_once_under_concurrency(self):
        barrier = threading.Barrier(3)
        results = []
        errors = []

        def run():
            try:
                barrier.wait()
                results.append(self.service.get_or_build(self.shop)[1].cache_level)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=5)

        self.assertFalse(errors)
        self.assertEqual(results.count("built"), 1)
        self.assertEqual(len(results), 2)

    def test_corrupt_sqlite_cache_is_cleared_and_rebuilt_once(self):
        expected, _ = self.service.get_or_build(self.shop)
        self.service.clear_memory_cache()
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("DELETE FROM graph_operation_features WHERE op_ordinal=0")
            conn.commit()

        with self.assertLogs(
            "llm4drd.knowledge.context_service", level="INFO"
        ) as captured:
            rebuilt, diagnostics = self.service.get_or_build(self.shop)

        self.assertEqual(rebuilt, expected)
        self.assertEqual(diagnostics.cache_level, "built")
        self.assertIn("feature", diagnostics.invalid_reason)
        self.assertIn("graph_context.corrupt", "\n".join(captured.output))

    def test_second_corrupt_build_failure_is_not_retried(self):
        class CorruptBuilder:
            def __init__(self):
                self.calls = 0

            def build(self, *args, **kwargs):
                self.calls += 1
                raise GraphContextCorruptError("bad build")

        builder = CorruptBuilder()
        service = GraphContextService(self.store, builder=builder)

        with self.assertLogs(
            "llm4drd.knowledge.context_service", level="INFO"
        ) as captured:
            with self.assertRaises(GraphContextBuildError):
                service.get_or_build(self.shop)

        self.assertEqual(builder.calls, 2)
        self.assertIn("graph_context.rebuild_failed", "\n".join(captured.output))

    def test_instance_change_before_save_aborts_persistence(self):
        changed = GraphFingerprint("changed", "changed", "changed")

        with self.assertRaises(GraphContextStaleError):
            self.service.get_or_build(
                self.shop,
                current_fingerprint_provider=lambda: changed,
            )

        self.assertIsNone(self.store.load_context_meta())


class GraphContextModeTests(unittest.TestCase):
    def test_mode_resolution_accepts_known_values_and_rejects_unknown(self):
        for value, expected in (
            (None, GraphContextMode.ACTIVE),
            ("legacy", GraphContextMode.LEGACY),
            ("shadow", GraphContextMode.SHADOW),
            ("active", GraphContextMode.ACTIVE),
        ):
            with self.subTest(value=value):
                environment = {} if value is None else {"LLM4DRD_GRAPH_CONTEXT_MODE": value}
                with patch.dict(os.environ, environment, clear=True):
                    self.assertEqual(resolve_graph_context_mode(), expected)

        with patch.dict(
            os.environ, {"LLM4DRD_GRAPH_CONTEXT_MODE": "invalid"}, clear=True
        ):
            with self.assertLogs(
                "llm4drd.knowledge.context_service", level="WARNING"
            ):
                self.assertEqual(
                    resolve_graph_context_mode(), GraphContextMode.LEGACY
                )


if __name__ == "__main__":
    unittest.main()
