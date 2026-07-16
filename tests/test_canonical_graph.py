import copy
import math
import unittest

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


if __name__ == "__main__":
    unittest.main()
