import copy
import math
import unittest
from unittest.mock import patch

from llm4drd.core.models import Order, Shift, ShopFloor
from llm4drd.knowledge.canonical import CanonicalGraphBuilder, compute_graph_fingerprint
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class GraphFingerprintTests(unittest.TestCase):
    def test_fingerprint_is_independent_of_dictionary_order(self):
        left = make_graph_context_shop()
        right = make_graph_context_shop()
        right.operations = dict(reversed(list(right.operations.items())))
        right.tasks = dict(reversed(list(right.tasks.items())))
        self.assertEqual(compute_graph_fingerprint(left), compute_graph_fingerprint(right))

    def test_processing_time_changes_feature_not_topology(self):
        left = make_graph_context_shop()
        right = copy.deepcopy(left)
        right.operations["OP-11"].processing_time += 1.0
        a = compute_graph_fingerprint(left)
        b = compute_graph_fingerprint(right)
        self.assertNotEqual(a.instance_hash, b.instance_hash)
        self.assertEqual(a.topology_hash, b.topology_hash)
        self.assertNotEqual(a.feature_hash, b.feature_hash)

    def test_turnover_time_changes_feature_not_topology(self):
        """turnover_time 进入 instance_hash/feature_hash 输入清单，但不改变 topology_hash。

        对应 docs/superpowers/specs/2026-07-16-unified-graph-context-design.md §7.3：
        流转等待时间与加工时间并列为 feature_hash 输入，不改变任何边的存在性。
        """
        left = make_graph_context_shop()
        right = copy.deepcopy(left)
        right.operations["OP-11"].turnover_time = 2.0
        a = compute_graph_fingerprint(left)
        b = compute_graph_fingerprint(right)
        self.assertNotEqual(a.instance_hash, b.instance_hash)
        self.assertEqual(a.topology_hash, b.topology_hash)
        self.assertNotEqual(a.feature_hash, b.feature_hash)

    def test_predecessor_change_changes_topology(self):
        left = make_graph_context_shop()
        right = copy.deepcopy(left)
        right.operations["OP-21"].predecessor_ops.append("OP-12")
        self.assertNotEqual(
            compute_graph_fingerprint(left).topology_hash,
            compute_graph_fingerprint(right).topology_hash,
        )

    def test_main_task_change_is_a_feature_not_topology_change(self):
        left = make_graph_context_shop()
        right = copy.deepcopy(left)
        right.orders["O-1"].main_task_id = "T-11"
        a = compute_graph_fingerprint(left)
        b = compute_graph_fingerprint(right)
        self.assertEqual(a.topology_hash, b.topology_hash)
        self.assertNotEqual(a.feature_hash, b.feature_hash)

    def test_task_operation_membership_changes_feature_not_topology(self):
        left = make_graph_context_shop()
        right = copy.deepcopy(left)
        right.tasks["T-11"].operations.clear()
        a = compute_graph_fingerprint(left)
        b = compute_graph_fingerprint(right)
        self.assertNotEqual(a.instance_hash, b.instance_hash)
        self.assertEqual(a.topology_hash, b.topology_hash)
        self.assertNotEqual(a.feature_hash, b.feature_hash)

    def test_non_finite_graph_input_is_rejected(self):
        shop = make_graph_context_shop()
        shop.operations["OP-11"].processing_time = math.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compute_graph_fingerprint(shop)

    def test_numeric_representation_does_not_change_fingerprint(self):
        left = make_graph_context_shop()
        for resource in [
            *left.machines.values(),
            *left.toolings.values(),
            *left.personnel.values(),
        ]:
            resource.shifts = [
                Shift(day=shift.day, start_hour=int(shift.start_hour), hours=int(shift.hours))
                for shift in resource.shifts
            ]
        right = copy.deepcopy(left)
        for resource in [
            *right.machines.values(),
            *right.toolings.values(),
            *right.personnel.values(),
        ]:
            resource.shifts = [
                Shift(day=shift.day, start_hour=float(shift.start_hour), hours=float(shift.hours))
                for shift in resource.shifts
            ]

        self.assertEqual(
            compute_graph_fingerprint(left),
            compute_graph_fingerprint(right),
        )

    def test_mapping_keys_and_model_ids_are_both_fingerprinted(self):
        entity_examples = (
            ("machine_types", "cut"),
            ("tooling_types", "TL-CUT"),
            ("machines", "M-C1"),
            ("toolings", "TL-1"),
            ("personnel", "P-1"),
            ("orders", "O-1"),
            ("tasks", "T-11"),
            ("operations", "OP-11"),
        )
        for collection_name, entity_key in entity_examples:
            with self.subTest(collection=collection_name):
                left = make_graph_context_shop()
                right = copy.deepcopy(left)
                entity = getattr(right, collection_name)[entity_key]
                entity.id = f"{entity.id}-changed"
                self.assertNotEqual(
                    compute_graph_fingerprint(left).instance_hash,
                    compute_graph_fingerprint(right).instance_hash,
                )

    def test_tooling_model_id_changes_topology_and_canonical_edge(self):
        left = make_graph_context_shop()
        right = copy.deepcopy(left)
        right.toolings["TL-1"].id = "TL-X"

        left_edges = {
            (edge.source, edge.target, edge.edge_type)
            for edge in CanonicalGraphBuilder().build(left).edges
        }
        right_edges = {
            (edge.source, edge.target, edge.edge_type)
            for edge in CanonicalGraphBuilder().build(right).edges
        }

        self.assertNotEqual(left_edges, right_edges)
        self.assertNotEqual(
            compute_graph_fingerprint(left).topology_hash,
            compute_graph_fingerprint(right).topology_hash,
        )

    def test_task_model_id_changes_legacy_feature_fingerprint(self):
        from llm4drd.optimization.hybrid_nsga3_alns import (
            HybridConfig,
            HybridNSGA3ALNSOptimizer,
        )

        left = make_graph_context_shop()
        right = copy.deepcopy(left)
        right.tasks["T-11"].id = "T-X"
        config = HybridConfig(
            objective_keys=["total_tardiness", "makespan"],
            parallel_workers=1,
        )

        left_value = HybridNSGA3ALNSOptimizer(left, config).graph_features["OP-11"][
            "assembly_criticality"
        ]
        right_value = HybridNSGA3ALNSOptimizer(right, config).graph_features["OP-11"][
            "assembly_criticality"
        ]

        self.assertNotEqual(left_value, right_value)
        self.assertNotEqual(
            compute_graph_fingerprint(left).feature_hash,
            compute_graph_fingerprint(right).feature_hash,
        )


class CanonicalGraphBuilderTests(unittest.TestCase):
    def test_canonical_types_are_exported_from_knowledge_package(self):
        from llm4drd import knowledge

        self.assertIs(knowledge.CanonicalGraphBuilder, CanonicalGraphBuilder)

    def test_canonical_rows_match_legacy_construction_order(self):
        shop = make_graph_context_shop()
        canonical = CanonicalGraphBuilder().build(shop)

        self.assertEqual(
            tuple(node.node_id for node in canonical.nodes),
            (
                "M:M-C1", "M:M-C2", "M:M-A1", "TL:TL-1", "P:P-1",
                "O:O-1", "O:O-2", "T:T-11", "T:T-12", "T:T-21",
                "OP:OP-11", "OP:OP-12", "OP:OP-13", "OP:OP-21",
            ),
        )
        self.assertEqual(
            tuple((edge.source, edge.target, edge.edge_type) for edge in canonical.edges),
            (
                ("O:O-1", "T:T-11", "order_has_task"),
                ("O:O-1", "T:T-12", "order_has_task"),
                ("T:T-11", "T:T-12", "task_predecessor"),
                ("O:O-2", "T:T-21", "order_has_task"),
                ("T:T-11", "OP:OP-11", "task_has_operation"),
                ("OP:OP-11", "M:M-C1", "machine_eligible"),
                ("T:T-11", "OP:OP-12", "task_has_operation"),
                ("OP:OP-11", "OP:OP-12", "operation_sequence"),
                ("OP:OP-12", "M:M-C1", "machine_eligible"),
                ("OP:OP-12", "M:M-C2", "machine_eligible"),
                ("T:T-12", "OP:OP-13", "task_has_operation"),
                ("T:T-11", "OP:OP-13", "op_depends_task"),
                ("OP:OP-13", "M:M-A1", "machine_eligible"),
                ("OP:OP-13", "TL:TL-1", "tooling_eligible"),
                ("OP:OP-13", "P:P-1", "personnel_eligible"),
                ("T:T-21", "OP:OP-21", "task_has_operation"),
                ("OP:OP-21", "M:M-C1", "machine_eligible"),
                ("OP:OP-21", "M:M-C2", "machine_eligible"),
            ),
        )

    def test_canonical_edges_match_legacy_graph(self):
        from llm4drd.knowledge.graph import HeterogeneousGraph

        shop = make_graph_context_shop()
        canonical = CanonicalGraphBuilder().build(shop)
        legacy = HeterogeneousGraph()
        legacy.build_from_shopfloor(shop)
        expected = {(a, b, d["edge_type"]) for a, b, d in legacy.graph.edges(data=True)}
        actual = {(edge.source, edge.target, edge.edge_type) for edge in canonical.edges}
        self.assertEqual(actual, expected)

    def test_operation_predecessor_tasks_create_task_level_edges(self):
        shop = make_graph_context_shop()
        shop.tasks["T-12"].predecessor_task_ids.clear()

        canonical = CanonicalGraphBuilder().build(shop)
        edges = {(edge.source, edge.target, edge.edge_type) for edge in canonical.edges}

        self.assertIn(("T:T-11", "T:T-12", "task_predecessor"), edges)
        self.assertIn(("T:T-11", "OP:OP-13", "op_depends_task"), edges)

    def test_progress_reports_legacy_digraph_counts(self):
        shop = make_graph_context_shop()
        shop.operations["OP-11"].eligible_machine_ids.append("M-C1")
        calls = []

        CanonicalGraphBuilder().build(shop, lambda *values: calls.append(values))

        self.assertEqual(calls, [(0, 9, 5, 0), (9, 9, 14, 18)])

    def test_deadline_is_checked_at_order_checkpoint(self):
        shop = ShopFloor()
        for index in range(500):
            order_id = f"O-{index}"
            shop.orders[order_id] = Order(id=order_id, due_date=24.0)
        calls = []

        with patch(
            "llm4drd.knowledge.canonical.time.monotonic", side_effect=[0.0, 2.0]
        ):
            with self.assertRaisesRegex(TimeoutError, "超过时间限制"):
                CanonicalGraphBuilder().build(
                    shop,
                    lambda *values: calls.append(values),
                    deadline=1.0,
                )

        self.assertEqual(calls, [(0, 500, 0, 0)])


if __name__ == "__main__":
    unittest.main()
