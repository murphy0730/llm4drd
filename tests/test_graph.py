import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm4drd.core.models import Operation, Order, ShopFloor, Task
from llm4drd.data.db import GraphStore, init_db
from llm4drd.knowledge.graph import HeterogeneousGraph


class HeterogeneousGraphTests(unittest.TestCase):
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
