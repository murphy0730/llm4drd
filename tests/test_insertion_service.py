import unittest
from io import BytesIO

import openpyxl

from llm4drd.api.insertion_service import (
    InsertionError,
    InsertionRunStore,
    build_insertion_export,
    build_insertion_template,
    evaluate_insertion,
    parse_insertion_file,
)
from llm4drd.core.models import (
    Machine,
    MachineType,
    Personnel,
    ShopFloor,
    Tooling,
    ToolingType,
)


def insertion_shop():
    shop = ShopFloor()
    shop.machine_types["cut"] = MachineType("cut", "Cut", is_critical=True)
    shop.machines["M1"] = Machine("M1", "Machine 1", "cut")
    shop.machines["M2"] = Machine("M2", "Machine 2", "cut")
    shop.tooling_types["fixture"] = ToolingType("fixture", "Fixture")
    shop.toolings["TL1"] = Tooling("TL1", "Fixture 1", "fixture")
    shop.personnel["P1"] = Personnel("P1", "Operator 1", ["operator"])
    shop.build_indexes()
    return shop


class InsertionServiceTests(unittest.TestCase):
    def test_frozen_mode_uses_joint_resource_gap_then_tail(self):
        shop = insertion_shop()
        base = [
            {"op_id": "OLD-1", "machine_id": "M1", "tooling_ids": ["TL1"], "personnel_ids": ["P1"], "start": 0, "end": 4},
            {"op_id": "OLD-2", "machine_id": "M1", "tooling_ids": ["TL1"], "personnel_ids": ["P1"], "start": 8, "end": 12},
        ]
        orders = [{"order_id": "URG-1", "release_time": 0, "expected_due_date": 20, "main_task_id": "T1"}]
        operations = [
            {
                "order_id": "URG-1", "op_id": "NEW-1", "task_id": "T1", "process_type": "cut",
                "processing_time_hrs": 3, "turnover_time_hrs": 1, "eligible_machine_ids": "M1",
                "required_tooling_types": "fixture", "required_personnel_skills": "operator",
            },
            {
                "order_id": "URG-1", "op_id": "NEW-2", "task_id": "T1", "process_type": "cut",
                "processing_time_hrs": 2, "predecessor_ops": "NEW-1", "eligible_machine_ids": "M1",
                "required_tooling_types": "fixture", "required_personnel_skills": "operator",
            },
        ]

        run = evaluate_insertion(shop, base, orders, operations, policy="frozen", instance_version=1)

        by_id = {entry["op_id"]: entry for entry in run.payload["inserted_schedule"]}
        self.assertEqual((by_id["NEW-1"]["start"], by_id["NEW-1"]["end"]), (4.0, 7.0))
        self.assertEqual(by_id["NEW-1"]["placement"], "gap")
        self.assertEqual((by_id["NEW-2"]["start"], by_id["NEW-2"]["end"]), (12.0, 14.0))
        self.assertEqual(by_id["NEW-2"]["placement"], "tail")
        self.assertEqual(run.payload["order_results"][0]["conclusion"], "met")
        self.assertTrue(run.payload["existing_orders_protected"])

    def test_multiple_orders_and_optional_due_date(self):
        shop = insertion_shop()
        orders = [
            {"order_id": "A", "release_time": 0, "expected_due_date": 4, "main_task_id": "TA"},
            {"order_id": "B", "release_time": 0, "expected_due_date": None, "main_task_id": "TB"},
        ]
        operations = [
            {"order_id": "A", "op_id": "A1", "task_id": "TA", "process_type": "cut", "processing_time_hrs": 2},
            {"order_id": "B", "op_id": "B1", "task_id": "TB", "process_type": "cut", "processing_time_hrs": 2},
        ]

        run = evaluate_insertion(shop, [], orders, operations, policy="frozen", instance_version=1)

        conclusions = {item["order_id"]: item["conclusion"] for item in run.payload["order_results"]}
        self.assertEqual(conclusions, {"A": "met", "B": "not_set"})
        self.assertEqual(len(run.payload["inserted_schedule"]), 2)

    def test_cross_order_predecessor_is_rejected(self):
        shop = insertion_shop()
        orders = [
            {"order_id": "A", "release_time": 0, "main_task_id": "TA"},
            {"order_id": "B", "release_time": 0, "main_task_id": "TB"},
        ]
        operations = [
            {"order_id": "A", "op_id": "A1", "task_id": "TA", "process_type": "cut", "processing_time_hrs": 1},
            {"order_id": "B", "op_id": "B1", "task_id": "TB", "process_type": "cut", "processing_time_hrs": 1, "predecessor_ops": "A1"},
        ]
        with self.assertRaisesRegex(InsertionError, "跨订单"):
            evaluate_insertion(shop, [], orders, operations, policy="frozen", instance_version=1)

    def test_template_round_trip_has_two_sheets(self):
        payload = build_insertion_template()
        parsed = parse_insertion_file(payload, "insertion.xlsx")
        self.assertTrue(parsed["orders"])
        self.assertTrue(parsed["operations"])
        self.assertIn("order_id", parsed["operations"][0])

    def test_due_protected_mode_keeps_frozen_candidate_as_safe_fallback(self):
        shop = insertion_shop()
        orders = [{"order_id": "A", "release_time": 0, "expected_due_date": 4, "main_task_id": "TA"}]
        operations = [
            {"order_id": "A", "op_id": "A1", "task_id": "TA", "process_type": "cut", "processing_time_hrs": 2},
        ]

        run = evaluate_insertion(shop, [], orders, operations, policy="due_protected", instance_version=7)

        self.assertTrue(run.payload["existing_orders_protected"])
        self.assertEqual(run.payload["search_status"], "best_found")
        self.assertEqual(len(run.payload["inserted_schedule"]), 1)

    def test_export_contains_required_sheets_and_run_store_invalidates(self):
        shop = insertion_shop()
        run = evaluate_insertion(
            shop,
            [],
            [{"order_id": "A", "release_time": 0, "main_task_id": "TA"}],
            [{"order_id": "A", "op_id": "A1", "task_id": "TA", "process_type": "cut", "processing_time_hrs": 1}],
            policy="frozen",
            instance_version=3,
        )
        workbook = openpyxl.load_workbook(BytesIO(build_insertion_export(run)), read_only=True)
        try:
            self.assertEqual(
                workbook.sheetnames,
                ["交期结论", "新工序排程", "KPI对比", "现有订单影响", "合并排程"],
            )
        finally:
            workbook.close()

        store = InsertionRunStore(max_runs=1)
        store.put(run)
        self.assertIs(store.get(run.run_id, 3), run)
        with self.assertRaisesRegex(InsertionError, "实例已发生变化"):
            store.get(run.run_id, 4)


if __name__ == "__main__":
    unittest.main()
