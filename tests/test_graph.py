import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from llm4drd.core.models import Operation, Order, ShopFloor, Task
from llm4drd.data.db import GraphStore, init_db
from llm4drd.knowledge.canonical import (
    CanonicalEdge,
    CanonicalGraph,
    CanonicalNode,
    GraphFingerprint,
)
from llm4drd.knowledge.graph import HeterogeneousGraph


class HeterogeneousGraphTests(unittest.TestCase):
    def test_forward_predecessor_preserves_legacy_networkx_iteration_order(self):
        from llm4drd.tests.shop_fixtures import make_graph_context_shop

        shop = make_graph_context_shop()
        shop.operations = {
            operation_id: shop.operations[operation_id]
            for operation_id in ("OP-12", "OP-21", "OP-11", "OP-13")
        }

        graph = HeterogeneousGraph()
        graph.build_from_shopfloor(shop)

        expected_node_order = (
            "M:M-C1", "M:M-C2", "M:M-A1", "TL:TL-1", "P:P-1",
            "O:O-1", "O:O-2", "T:T-11", "T:T-12", "T:T-21",
            "OP:OP-12", "OP:OP-11", "OP:OP-21", "OP:OP-13",
        )
        self.assertEqual(
            tuple(graph.graph.nodes),
            expected_node_order,
        )
        expected_node_attrs = {
            "M:M-C1": {
                "node_type": "machine", "entity_id": "M-C1", "label": "Cutter 1",
                "type_id": "cut", "type_name": "Cut", "is_critical": True,
            },
            "M:M-C2": {
                "node_type": "machine", "entity_id": "M-C2", "label": "Cutter 2",
                "type_id": "cut", "type_name": "Cut", "is_critical": True,
            },
            "M:M-A1": {
                "node_type": "machine", "entity_id": "M-A1", "label": "Assembly 1",
                "type_id": "asm", "type_name": "Assembly", "is_critical": False,
            },
            "TL:TL-1": {
                "node_type": "tooling", "entity_id": "TL-1", "label": "Fixture 1",
                "type_id": "TL-CUT", "type_name": "Cut fixture",
            },
            "P:P-1": {
                "node_type": "personnel", "entity_id": "P-1", "label": "Assembler",
                "skills": "SK-ASM",
            },
            "O:O-1": {
                "node_type": "order", "entity_id": "O-1", "label": "Order 1",
                "due_date": 40.0, "due_at": shop.time_label(40.0), "priority": 3,
                "release_time": 0.0, "release_at": shop.time_label(0.0),
            },
            "O:O-2": {
                "node_type": "order", "entity_id": "O-2", "label": "Order 2",
                "due_date": 48.0, "due_at": shop.time_label(48.0), "priority": 1,
                "release_time": 1.0, "release_at": shop.time_label(1.0),
            },
            "T:T-11": {
                "node_type": "task", "entity_id": "T-11", "label": "Part",
                "order_id": "O-1", "is_main": False, "due_date": 24.0,
                "release_time": 0.0, "due_at": shop.time_label(24.0),
                "release_at": shop.time_label(0.0), "derived_due_date": 24.0,
                "derived_due_at": shop.time_label(24.0), "derived_start_time": 18.0,
                "derived_start_at": shop.time_label(18.0), "critical_path_time": 6.0,
                "critical_slack": 18.0,
            },
            "T:T-12": {
                "node_type": "task", "entity_id": "T-12", "label": "Main",
                "order_id": "O-1", "is_main": True, "due_date": 40.0,
                "release_time": 0.0, "due_at": shop.time_label(40.0),
                "release_at": shop.time_label(0.0), "derived_due_date": 40.0,
                "derived_due_at": shop.time_label(40.0), "derived_start_time": 37.0,
                "derived_start_at": shop.time_label(37.0), "critical_path_time": 3.0,
                "critical_slack": 31.0,
            },
            "T:T-21": {
                "node_type": "task", "entity_id": "T-21", "label": "Second",
                "order_id": "O-2", "is_main": True, "due_date": 48.0,
                "release_time": 1.0, "due_at": shop.time_label(48.0),
                "release_at": shop.time_label(1.0), "derived_due_date": 48.0,
                "derived_due_at": shop.time_label(48.0), "derived_start_time": 43.0,
                "derived_start_at": shop.time_label(43.0), "critical_path_time": 5.0,
                "critical_slack": 42.0,
            },
            "OP:OP-11": {
                "node_type": "operation", "entity_id": "OP-11", "label": "Cut",
                "task_id": "T-11", "process_type": "cut", "processing_time": 4.0,
                "turnover_time": 0.0,
                "required_tooling_types": "", "required_personnel_skills": "",
                "status": "pending", "derived_due_date": 22.0,
                "derived_due_at": shop.time_label(22.0), "derived_start_time": 18.0,
                "derived_start_at": shop.time_label(18.0), "critical_slack": 18.0,
            },
            "OP:OP-12": {
                "node_type": "operation", "entity_id": "OP-12", "label": "Finish",
                "task_id": "T-11", "process_type": "cut", "processing_time": 2.0,
                "turnover_time": 0.0,
                "required_tooling_types": "", "required_personnel_skills": "",
                "status": "pending", "derived_due_date": 24.0,
                "derived_due_at": shop.time_label(24.0), "derived_start_time": 22.0,
                "derived_start_at": shop.time_label(22.0), "critical_slack": 18.0,
            },
            "OP:OP-13": {
                "node_type": "operation", "entity_id": "OP-13", "label": "Assemble",
                "task_id": "T-12", "process_type": "asm", "processing_time": 3.0,
                "turnover_time": 0.0,
                "required_tooling_types": "TL-CUT",
                "required_personnel_skills": "SK-ASM", "status": "pending",
                "derived_due_date": 40.0, "derived_due_at": shop.time_label(40.0),
                "derived_start_time": 37.0, "derived_start_at": shop.time_label(37.0),
                "critical_slack": 31.0,
            },
            "OP:OP-21": {
                "node_type": "operation", "entity_id": "OP-21", "label": "Other cut",
                "task_id": "T-21", "process_type": "cut", "processing_time": 5.0,
                "turnover_time": 0.0,
                "required_tooling_types": "", "required_personnel_skills": "",
                "status": "pending", "derived_due_date": 48.0,
                "derived_due_at": shop.time_label(48.0), "derived_start_time": 43.0,
                "derived_start_at": shop.time_label(43.0), "critical_slack": 42.0,
            },
        }
        self.assertEqual(
            tuple((node_id, dict(attrs)) for node_id, attrs in graph.graph.nodes(data=True)),
            tuple((node_id, expected_node_attrs[node_id]) for node_id in expected_node_order),
        )
        self.assertEqual(
            tuple(
                (source, target, attrs["edge_type"])
                for source, target, attrs in graph.graph.edges(data=True)
            ),
            (
                ("O:O-1", "T:T-11", "order_has_task"),
                ("O:O-1", "T:T-12", "order_has_task"),
                ("O:O-2", "T:T-21", "order_has_task"),
                ("T:T-11", "T:T-12", "task_predecessor"),
                ("T:T-11", "OP:OP-12", "task_has_operation"),
                ("T:T-11", "OP:OP-11", "task_has_operation"),
                ("T:T-11", "OP:OP-13", "op_depends_task"),
                ("T:T-12", "OP:OP-13", "task_has_operation"),
                ("T:T-21", "OP:OP-21", "task_has_operation"),
                ("OP:OP-12", "M:M-C1", "machine_eligible"),
                ("OP:OP-12", "M:M-C2", "machine_eligible"),
                ("OP:OP-11", "OP:OP-12", "operation_sequence"),
                ("OP:OP-11", "M:M-C1", "machine_eligible"),
                ("OP:OP-21", "M:M-C1", "machine_eligible"),
                ("OP:OP-21", "M:M-C2", "machine_eligible"),
                ("OP:OP-13", "M:M-A1", "machine_eligible"),
                ("OP:OP-13", "TL:TL-1", "tooling_eligible"),
                ("OP:OP-13", "P:P-1", "personnel_eligible"),
            ),
        )

    def test_build_translates_rows_from_canonical_builder(self):
        canonical = CanonicalGraph(
            nodes=(CanonicalNode("M:M-1", "machine", "M-1", {"label": "Machine"}),),
            edges=(CanonicalEdge("OP:OP-1", "M:M-1", "machine_eligible", {}),),
            fingerprint=GraphFingerprint("instance", "topology", "feature"),
        )
        shop = ShopFloor()
        progress_callback = object()

        with patch("llm4drd.knowledge.graph.CanonicalGraphBuilder") as builder_type:
            builder_type.return_value.build.return_value = canonical
            graph = HeterogeneousGraph()
            graph.build_from_shopfloor(shop, progress_callback, 123.0)

        builder_type.return_value.build.assert_called_once_with(
            shop, progress_callback, 123.0
        )
        self.assertEqual(
            graph.graph.nodes["M:M-1"],
            {"node_type": "machine", "entity_id": "M-1", "label": "Machine"},
        )
        self.assertEqual(
            graph.graph.edges["OP:OP-1", "M:M-1"],
            {"edge_type": "machine_eligible"},
        )

    def test_order_subgraph_excludes_other_orders_and_keeps_resources(self):
        graph = HeterogeneousGraph()
        for node_id, node_type, entity_id in [
            ("O:O-1", "order", "O-1"),
            ("T:T-1", "task", "T-1"),
            ("OP:OP-1", "operation", "OP-1"),
            ("O:O-2", "order", "O-2"),
            ("T:T-2", "task", "T-2"),
            ("OP:OP-2", "operation", "OP-2"),
            ("M:M-1", "machine", "M-1"),
        ]:
            graph.graph.add_node(
                node_id, node_type=node_type, entity_id=entity_id, label=entity_id
            )
        graph.graph.add_edge("O:O-1", "T:T-1", edge_type="order_has_task")
        graph.graph.add_edge("T:T-1", "OP:OP-1", edge_type="task_has_operation")
        graph.graph.add_edge("OP:OP-1", "M:M-1", edge_type="machine_eligible")
        graph.graph.add_edge("O:O-2", "T:T-2", edge_type="order_has_task")
        graph.graph.add_edge("T:T-2", "OP:OP-2", edge_type="task_has_operation")
        graph.graph.add_edge("OP:OP-2", "M:M-1", edge_type="machine_eligible")

        with TemporaryDirectory() as directory:
            db_path = str(Path(directory) / "graph.db")
            init_db(db_path)
            store = GraphStore(db_path)
            store.save_graph(graph)
            result = store.load_order_subgraph("O-1")

        node_ids = {node["node_id"] for node in result["nodes"]}
        self.assertEqual(node_ids, {"O:O-1", "T:T-1", "OP:OP-1", "M:M-1"})
        self.assertEqual(len(result["edges"]), 3)

    def _build_order_with_machine(self, machine_node_id, machine_entity_id):
        graph = HeterogeneousGraph()
        for node_id, node_type, entity_id in [
            ("O:O-1", "order", "O-1"),
            ("T:T-1", "task", "T-1"),
            ("OP:OP-1", "operation", "OP-1"),
            (machine_node_id, "machine", machine_entity_id),
        ]:
            graph.graph.add_node(
                node_id, node_type=node_type, entity_id=entity_id, label=entity_id
            )
        graph.graph.add_edge("O:O-1", "T:T-1", edge_type="order_has_task")
        graph.graph.add_edge("T:T-1", "OP:OP-1", edge_type="task_has_operation")
        graph.graph.add_edge("OP:OP-1", machine_node_id, edge_type="machine_eligible")
        return graph

    def test_order_subgraph_filters_os_machines(self):
        # 机器 entity_id 以 OS_ 开头，子图应在 SQL 层过滤掉该机器及其关联边。
        graph = self._build_order_with_machine("M:OS_stub", "OS_stub")
        with TemporaryDirectory() as directory:
            db_path = str(Path(directory) / "graph.db")
            init_db(db_path)
            store = GraphStore(db_path)
            store.save_graph(graph)
            result = store.load_order_subgraph("O-1")

        node_ids = {node["node_id"] for node in result["nodes"]}
        self.assertEqual(node_ids, {"O:O-1", "T:T-1", "OP:OP-1"})
        self.assertTrue(all(edge["target"] != "M:OS_stub" for edge in result["edges"]))

    def test_search_order_subgraph_resolves_and_filters_os_machines(self):
        graph = self._build_order_with_machine("M:OS_stub", "OS_stub")
        with TemporaryDirectory() as directory:
            db_path = str(Path(directory) / "graph.db")
            init_db(db_path)
            store = GraphStore(db_path)
            store.save_graph(graph)
            # 精确解析 + 模糊解析都应命中，且 OS_ 机器被过滤。
            exact = store.search_order_subgraph("O-1")
            fuzzy = store.search_order_subgraph("O")
            missing = store.search_order_subgraph("not-exist")

        self.assertEqual(exact["order_id"], "O:O-1")
        self.assertEqual(fuzzy["order_id"], "O:O-1")
        for result in (exact, fuzzy):
            node_ids = {node["node_id"] for node in result["nodes"]}
            self.assertEqual(node_ids, {"O:O-1", "T:T-1", "OP:OP-1"})
        self.assertIsNone(missing["order_id"])
        self.assertEqual(missing["nodes"], [])

    def test_operation_predecessor_tasks_create_all_task_level_edges(self):
        predecessor_ids = ["T-P1", "T-P2", "T-P3"]
        target_task = Task(
            id="T-NEXT",
            order_id="O-1",
            name="Next",
            due_date=24.0,
            derived_due_date=24.0,
            derived_start_time=0.0,
        )
        target_op = Operation(
            id="OP-NEXT",
            task_id=target_task.id,
            name="Next operation",
            process_type="assembly",
            processing_time=1.0,
            predecessor_tasks=predecessor_ids,
            derived_due_date=24.0,
            derived_start_time=0.0,
        )
        target_task.operations.append(target_op)

        shop = ShopFloor()
        shop.orders["O-1"] = Order(
            id="O-1",
            due_date=24.0,
            task_ids=[*predecessor_ids, target_task.id],
        )
        for task_id in predecessor_ids:
            shop.tasks[task_id] = Task(
                id=task_id,
                order_id="O-1",
                name=task_id,
                due_date=24.0,
                derived_due_date=24.0,
                derived_start_time=0.0,
            )
        shop.tasks[target_task.id] = target_task
        shop.operations[target_op.id] = target_op

        graph = HeterogeneousGraph()
        graph.build_from_shopfloor(shop)

        for predecessor_id in predecessor_ids:
            task_edge = graph.graph.get_edge_data(
                f"T:{predecessor_id}", f"T:{target_task.id}"
            )
            self.assertEqual(
                task_edge["edge_type"], HeterogeneousGraph.EDGE_TASK_PREDECESSOR
            )
            operation_edge = graph.graph.get_edge_data(
                f"T:{predecessor_id}", f"OP:{target_op.id}"
            )
            self.assertEqual(
                operation_edge["edge_type"], HeterogeneousGraph.EDGE_OP_PRED_TASK
            )

        self.assertEqual(
            graph.get_graph_stats()["edge_types"][HeterogeneousGraph.EDGE_TASK_PREDECESSOR],
            len(predecessor_ids),
        )


if __name__ == "__main__":
    unittest.main()
