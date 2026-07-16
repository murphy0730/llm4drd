import copy
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from llm4drd.data.db import GraphStore, init_db
from llm4drd.data.graph_artifact_store import GraphArtifactStore
from llm4drd.knowledge.canonical import CanonicalGraphBuilder
from llm4drd.knowledge.context import (
    ComputeGraphProjection,
    DisplayGraphProjection,
    GraphContextCorruptError,
)
from llm4drd.knowledge.graph import HeterogeneousGraph
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class GraphArtifactStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "graph.db")
        init_db(self.db_path)
        self.shop = make_graph_context_shop()
        canonical = CanonicalGraphBuilder().build(self.shop)
        self.display = DisplayGraphProjection.from_canonical(canonical)
        self.context = ComputeGraphProjection().build(self.shop, canonical)
        self.store = GraphArtifactStore(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def _artifacts_for(self, shop):
        canonical = CanonicalGraphBuilder().build(shop)
        return (
            DisplayGraphProjection.from_canonical(canonical),
            ComputeGraphProjection().build(shop, canonical),
        )

    def test_schema_is_idempotent_and_context_round_trips(self):
        init_db(self.db_path)
        self.store.save_artifacts(self.display, self.context)

        loaded = self.store.load_context(self.context.fingerprint)

        self.assertEqual(loaded, self.context)
        with closing(sqlite3.connect(self.db_path)) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            graph_meta_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(graph_meta)")
            }
        self.assertTrue(
            {
                "graph_context_meta",
                "graph_entity_index",
                "graph_context_relations",
                "graph_operation_features",
                "graph_operation_groups",
            }.issubset(tables)
        )
        self.assertTrue(
            {
                "instance_hash",
                "topology_hash",
                "feature_hash",
                "schema_version",
                "builder_version",
                "build_time_ms",
                "invalid_reason",
            }.issubset(graph_meta_columns)
        )

    def test_display_rows_preserve_canonical_node_order(self):
        self.store.save_artifacts(self.display, self.context)

        _, nodes = GraphStore(self.db_path).load_nodes(limit=1000)
        self.assertEqual(
            tuple(node["node_id"] for node in nodes),
            self.display._node_order,
        )

    def test_display_edges_preserve_legacy_networkx_iteration_order(self):
        legacy = HeterogeneousGraph()
        legacy.build_from_shopfloor(self.shop)
        expected = tuple(
            (source, target, attrs["edge_type"])
            for source, target, attrs in legacy.graph.edges(data=True)
        )

        self.store.save_artifacts(self.display, self.context)
        _, edges = GraphStore(self.db_path).load_edges(limit=1000)

        self.assertEqual(
            tuple(
                (edge["source"], edge["target"], edge["edge_type"])
                for edge in edges
            ),
            expected,
        )

    def test_mismatched_fingerprint_is_a_cache_miss(self):
        self.store.save_artifacts(self.display, self.context)
        stale = self.context.fingerprint.__class__(
            "x", "y", "z", 1, "canonical-v1"
        )
        self.assertIsNone(self.store.load_context(stale))

    def test_precommit_failure_rolls_back_every_projection(self):
        self.store.save_artifacts(self.display, self.context)
        before = self.store.load_context_meta()

        with self.assertRaisesRegex(RuntimeError, "instance changed"):
            self.store.save_artifacts(
                self.display,
                self.context,
                precommit_check=lambda: (_ for _ in ()).throw(
                    RuntimeError("instance changed")
                ),
            )

        self.assertEqual(self.store.load_context_meta(), before)

    def test_missing_feature_row_is_reported_as_corruption(self):
        self.store.save_artifacts(self.display, self.context)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("DELETE FROM graph_operation_features WHERE op_ordinal=0")
            conn.commit()

        with self.assertRaises(GraphContextCorruptError):
            self.store.load_context(self.context.fingerprint)

    def test_non_finite_feature_is_reported_as_corruption(self):
        self.store.save_artifacts(self.display, self.context)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "UPDATE graph_operation_features SET predecessor_depth=? WHERE op_ordinal=0",
                (float("inf"),),
            )
            conn.commit()

        with self.assertRaises(GraphContextCorruptError):
            self.store.load_context(self.context.fingerprint)

    def test_display_and_compute_fingerprint_mismatch_is_corruption(self):
        self.store.save_artifacts(self.display, self.context)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("UPDATE graph_meta SET feature_hash='corrupt'")
            conn.commit()

        with self.assertRaises(GraphContextCorruptError):
            self.store.load_context(self.context.fingerprint)

    def test_legacy_graph_store_still_saves_after_schema_migration(self):
        self.store.save_artifacts(self.display, self.context)
        graph = HeterogeneousGraph()
        graph.build_from_shopfloor(self.shop)
        legacy_store = GraphStore(self.db_path)

        legacy_store.save_graph(graph)

        self.assertEqual(
            legacy_store.load_meta()["total_nodes"], graph.graph.number_of_nodes()
        )
        compute_meta = self.store.load_context_meta()
        self.assertEqual(compute_meta["status"], "invalid")
        self.assertEqual(compute_meta["invalid_reason"], "legacy_graph_saved")
        self.assertIsNone(self.store.load_context(self.context.fingerprint))

    def test_failure_after_relation_insert_keeps_old_artifacts(self):
        self.store.save_artifacts(self.display, self.context)
        before_context_meta = self.store.load_context_meta()
        before_display_meta = GraphStore(self.db_path).load_meta()
        changed_shop = copy.deepcopy(self.shop)
        changed_shop.operations["OP-11"].processing_time += 1.0
        display, context = self._artifacts_for(changed_shop)

        with patch.object(
            self.store,
            "_after_relation_insert",
            side_effect=RuntimeError("injected failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                self.store.save_artifacts(display, context)

        self.assertEqual(self.store.load_context_meta(), before_context_meta)
        self.assertEqual(GraphStore(self.db_path).load_meta(), before_display_meta)
        self.assertEqual(
            self.store.load_context(self.context.fingerprint), self.context
        )
        self.assertIsNone(self.store.load_context(context.fingerprint))

    def test_mark_invalid_and_clear_all_cover_both_projections(self):
        self.store.save_artifacts(self.display, self.context)

        self.store.mark_invalid("manual invalidation")

        self.assertIsNone(self.store.load_context(self.context.fingerprint))
        self.assertEqual(
            self.store.load_context_meta()["invalid_reason"],
            "manual invalidation",
        )
        self.assertEqual(
            GraphStore(self.db_path).load_meta()["invalid_reason"],
            "manual invalidation",
        )

        self.store.clear_all()
        self.assertIsNone(self.store.load_context_meta())
        self.assertFalse(GraphStore(self.db_path).has_data())


if __name__ == "__main__":
    unittest.main()
