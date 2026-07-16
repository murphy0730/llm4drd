import math
import unittest
from dataclasses import replace

import llm4drd.knowledge as knowledge
from llm4drd.knowledge.canonical import CanonicalGraphBuilder
from llm4drd.knowledge.context import (
    ComputeGraphProjection,
    DisplayGraphProjection,
    GraphContextCorruptError,
    compare_legacy_context,
    validate_graph_context,
)
from llm4drd.knowledge.graph import HeterogeneousGraph
from llm4drd.optimization.hybrid_nsga3_alns import build_legacy_graph_features
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class GraphContextTests(unittest.TestCase):
    def setUp(self):
        self.shop = make_graph_context_shop()
        self.canonical = CanonicalGraphBuilder().build(self.shop)
        self.context = ComputeGraphProjection().build(self.shop, self.canonical)

    def test_operation_index_is_stable_and_complete(self):
        self.assertEqual(self.context.operation_ids, tuple(sorted(self.shop.operations)))
        self.assertEqual(set(self.context.operation_index), set(self.shop.operations))

    def test_predecessor_successor_accessors_are_inverse(self):
        self.assertEqual(self.context.predecessors("OP-12"), ("OP-11",))
        self.assertIn("OP-12", self.context.successors("OP-11"))
        self.assertEqual(set(self.context.predecessors("OP-13")), {"OP-11", "OP-12"})

    def test_explicit_and_derived_machine_edges_are_available(self):
        self.assertEqual(self.context.eligible_machines("OP-11"), ("M-C1",))
        self.assertEqual(set(self.context.eligible_machines("OP-12")), {"M-C1", "M-C2"})

    def test_feature_view_contains_legacy_feature_names(self):
        values = self.context.operation_features("OP-13")
        self.assertEqual(
            set(values),
            {
                "predecessor_depth",
                "assembly_criticality",
                "shared_resource_degree",
                "bottleneck_adjacency",
                "graph_out_degree",
            },
        )

    def test_groups_support_profile_expansion(self):
        self.assertEqual(
            set(self.context.operations_in_group("process_type", "cut")),
            {"OP-11", "OP-12", "OP-21"},
        )
        self.assertEqual(
            self.context.operations_in_group("personnel_skill", "SK-ASM"),
            ("OP-13",),
        )
        self.assertEqual(
            self.context.operations_in_group("tooling_type", "TL-CUT"),
            ("OP-13",),
        )

    def test_context_validates_against_shop(self):
        validate_graph_context(self.shop, self.context)

    def test_context_maps_are_read_only(self):
        with self.assertRaises(TypeError):
            self.context.operation_index["OP-X"] = 4
        with self.assertRaises(TypeError):
            self.context.operation_groups[("process_type", "new")] = ()

    def test_context_types_are_exported_from_knowledge_package(self):
        self.assertIs(knowledge.GraphContext, type(self.context))
        self.assertIs(knowledge.ComputeGraphProjection, ComputeGraphProjection)
        self.assertIs(knowledge.DisplayGraphProjection, DisplayGraphProjection)

    def test_features_match_legacy_formula(self):
        graph = HeterogeneousGraph()
        graph.build_from_shopfloor(self.shop)
        expected = build_legacy_graph_features(self.shop, graph.graph)

        for operation_id in self.context.operation_ids:
            for feature_name, expected_value in expected[operation_id].items():
                self.assertAlmostEqual(
                    self.context.operation_features(operation_id)[feature_name],
                    expected_value,
                    delta=1e-12,
                )

    def test_display_projection_preserves_canonical_order_metadata(self):
        display = DisplayGraphProjection.from_canonical(self.canonical)

        self.assertEqual(display.nodes, self.canonical.nodes)
        self.assertEqual(display.edges, self.canonical.edges)
        self.assertEqual(display._node_order, self.canonical._node_order)
        with self.assertRaises(TypeError):
            display.stats["total_nodes"] = 0

    def test_validation_rejects_corrupt_relations_features_and_metadata(self):
        corrupt_contexts = (
            replace(
                self.context,
                predecessor_offsets=(0,) * len(self.context.predecessor_offsets),
            ),
            replace(
                self.context,
                predecessor_indices=(len(self.context.operation_ids),),
                predecessor_offsets=(0, 1, 1, 1, 1),
            ),
            replace(
                self.context,
                feature_matrix=(
                    (math.nan,) + self.context.feature_matrix[0][1:],
                    *self.context.feature_matrix[1:],
                ),
            ),
            replace(
                self.context,
                feature_names=("renamed", *self.context.feature_names[1:]),
            ),
            replace(self.context, operation_index={"OP-11": 0}),
        )

        for context in corrupt_contexts:
            with self.subTest(context=context):
                with self.assertRaises(GraphContextCorruptError):
                    validate_graph_context(self.shop, context)

    def test_shadow_comparison_reports_no_difference(self):
        diff = compare_legacy_context(self.shop, self.context)
        self.assertEqual(diff.total_differences, 0)

    def test_shadow_comparison_detects_successor_difference(self):
        first = self.context.successor_indices[0]
        changed = ((first + 1) % len(self.context.operation_ids),)
        changed += self.context.successor_indices[1:]
        context = replace(self.context, successor_indices=changed)

        self.assertGreater(compare_legacy_context(self.shop, context).total_differences, 0)


if __name__ == "__main__":
    unittest.main()
