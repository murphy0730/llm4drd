import unittest

from llm4drd.core.models import Operation, Order, ShopFloor, Task
from llm4drd.knowledge.graph import HeterogeneousGraph


class HeterogeneousGraphTests(unittest.TestCase):
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
