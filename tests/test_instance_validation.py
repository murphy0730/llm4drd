import unittest

from llm4drd.api.server import _validate_instance
from llm4drd.core.models import Operation, Order, Task
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class InstanceValidationWarningTests(unittest.TestCase):
    @staticmethod
    def _set_order_due(shop, due_date):
        shop.orders["O-1"].due_date = due_date
        shop.tasks["T-12"].due_date = due_date

    @staticmethod
    def _add_parallel_child(shop):
        child = Task("T-13", "O-1", "Parallel child", False, [], release_time=0.0, due_date=40.0)
        operation = Operation("OP-14", "T-13", "Parallel work", "cut", 7.0, turnover_time=1.0)
        child.operations.append(operation)
        shop.tasks[child.id] = child
        shop.operations[operation.id] = operation
        shop.orders["O-1"].task_ids.append(child.id)
        shop.tasks["T-12"].predecessor_task_ids.append(child.id)
        shop.operations["OP-13"].predecessor_tasks.append(child.id)
        shop.build_indexes()

    def test_due_date_at_plan_start_warns_once_and_skips_critical_path(self):
        shop = make_graph_context_shop()
        self._set_order_due(shop, 0.0)

        result = _validate_instance(shop)

        matches = [warning for warning in result["warnings"] if warning["entity"] == "T-12"]
        self.assertEqual(len(matches), 1)
        self.assertIn("主订单已经超期", matches[0]["message"])
        self.assertNotIn("关键路径", matches[0]["message"])

    def test_parallel_children_are_not_summed_when_critical_path_meets_due_date(self):
        shop = make_graph_context_shop()
        shop.operations["OP-11"].turnover_time = 2.0
        self._add_parallel_child(shop)
        self._set_order_due(shop, 12.0)

        result = _validate_instance(shop)

        matches = [warning for warning in result["warnings"] if warning["entity"] == "T-12" and "关键路径" in warning["message"]]
        self.assertEqual(matches, [])

    def test_critical_path_processing_and_turnover_beyond_due_date_warns(self):
        shop = make_graph_context_shop()
        shop.operations["OP-11"].turnover_time = 2.0
        self._add_parallel_child(shop)
        self._set_order_due(shop, 10.0)

        result = _validate_instance(shop)

        matches = [warning for warning in result["warnings"] if warning["entity"] == "T-12" and "关键路径理想最早完工（11.0h）" in warning["message"]]
        self.assertEqual(len(matches), 1)
        self.assertIn("最少延误 1.0h", matches[0]["message"])

    def test_critical_path_equal_to_due_date_does_not_warn(self):
        shop = make_graph_context_shop()
        shop.operations["OP-11"].turnover_time = 2.0
        shop.operations["OP-13"].turnover_time = 100.0
        self._set_order_due(shop, 11.0)

        result = _validate_instance(shop)

        matches = [warning for warning in result["warnings"] if "关键路径" in warning["message"]]
        self.assertEqual(matches, [])

    def test_validation_uses_order_due_date_like_review_kpi(self):
        shop = make_graph_context_shop()
        shop.tasks["T-12"].due_date = 0.0
        shop.orders["O-1"].due_date = 40.0

        result = _validate_instance(shop)

        matches = [warning for warning in result["warnings"] if warning["entity"] == "T-12"]
        self.assertEqual(matches, [])

    def test_all_warnings_only_reference_main_tasks(self):
        shop = make_graph_context_shop()
        shop.orders["O-2"].due_date = 0.0
        shop.orders["O-empty"] = Order("O-empty", "Empty")

        result = _validate_instance(shop)

        main_task_ids = {task.id for task in shop.tasks.values() if task.is_main}
        self.assertTrue(result["warnings"])
        self.assertTrue(all(warning["entity"] in main_task_ids for warning in result["warnings"]))
        self.assertNotIn("O-empty", {warning["entity"] for warning in result["warnings"]})


if __name__ == "__main__":
    unittest.main()
