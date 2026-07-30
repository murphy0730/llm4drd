import unittest

from llm4drd.api.server import _validate_instance
from llm4drd.core.models import Order
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class InstanceValidationWarningTests(unittest.TestCase):
    def test_zero_due_date_warns_for_main_task(self):
        shop = make_graph_context_shop()
        shop.tasks["T-12"].due_date = 0.0

        result = _validate_instance(shop)

        matches = [
            warning for warning in result["warnings"]
            if warning["entity"] == "T-12" and "交期为 0h" in warning["message"]
        ]
        self.assertEqual(len(matches), 1)

    def test_total_processing_and_turnover_of_main_and_child_tasks_is_compared_with_due_date(self):
        shop = make_graph_context_shop()
        shop.operations["OP-11"].turnover_time = 2.0
        shop.tasks["T-12"].due_date = 10.0

        result = _validate_instance(shop)

        matches = [
            warning for warning in result["warnings"]
            if warning["entity"] == "T-12" and "加工与转运总时长（11.0h）大于交期（10.0h）" in warning["message"]
        ]
        self.assertEqual(len(matches), 1)

    def test_total_duration_equal_to_due_date_does_not_warn(self):
        shop = make_graph_context_shop()
        shop.operations["OP-11"].turnover_time = 2.0
        shop.tasks["T-12"].due_date = 11.0

        result = _validate_instance(shop)

        matches = [warning for warning in result["warnings"] if "加工与转运总时长" in warning["message"]]
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
