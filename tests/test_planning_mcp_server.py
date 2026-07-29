from __future__ import annotations

import unittest

from llm4drd.mcp_server.errors import PlanningAPIError
from llm4drd.mcp_server.server import TOOL_DEFINITIONS, handle_tool_call


class FakePlanningClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.order_matches = [{
            "order_id": "O-1",
            "order_name": "Order 1",
            "match_type": "exact_id",
        }]
        self.operation_matches = [{
            "operation_id": "OP-13",
            "operation_name": "Assemble",
            "match_type": "exact_id",
        }]

    def get_overview(self, task_id=None):
        self.calls.append(("overview", task_id))
        return {"ok": True, "data": {"candidate_count": 2}}

    def compare_solutions(self, task_id=None, solution_ids=None, metric_keys=None):
        self.calls.append(("compare", task_id, solution_ids, metric_keys))
        return {"ok": True, "data": {"solution_count": 2}}

    def search_orders(self, query, limit=20):
        self.calls.append(("search_orders", query, limit))
        return {"ok": True, "data": list(self.order_matches)}

    def search_operations(self, query, order_id=None, limit=20):
        self.calls.append(("search_operations", query, order_id, limit))
        return {"ok": True, "data": list(self.operation_matches)}

    def get_order_planning(self, order_id, task_id=None, solution_ids=None):
        self.calls.append(("order", order_id, task_id, solution_ids))
        return {"ok": True, "data": {"order": {"order_id": order_id}}}

    def get_operation_planning(self, operation_id, task_id=None, solution_ids=None):
        self.calls.append(("operation", operation_id, task_id, solution_ids))
        return {"ok": True, "data": {"operation": {"operation_id": operation_id}}}

    def list_rules(self):
        self.calls.append(("rules",))
        return {
            "ok": True,
            "data": {
                "rule_count": 2,
                "rules": [
                    {"rule_name": "ATC", "description": "综合交期与加工时间"},
                    {"rule_name": "EDD", "description": "最早交期优先"},
                ],
            },
        }

    def run_rule_planning(self, rule_name):
        self.calls.append(("run_rule", rule_name))
        return {
            "ok": True,
            "data": {
                "rule_name": rule_name,
                "metrics": {"total_tardiness": 3.5},
                "operation_count": 4,
            },
        }


class PlanningMCPServerTests(unittest.TestCase):
    def test_publishes_the_planning_and_whatif_tools(self) -> None:
        self.assertEqual(
            {item["name"] for item in TOOL_DEFINITIONS},
            {
                # 只读排产查询 + 执行
                "get_planning_overview",
                "compare_planning_solutions",
                "search_planning_entities",
                "get_order_planning",
                "get_operation_planning",
                "list_planning_rules",
                "run_rule_planning",
                # what-if 场景推演
                "create_whatif_scenario",
                "apply_whatif_patch",
                "describe_whatif_scenario",
                "revert_whatif_patch",
                "run_whatif_planning",
                "get_whatif_run",
                "compare_whatif_runs",
                "apply_whatif_to_instance",
            },
        )
        self.assertTrue(all(item["inputSchema"]["type"] == "object" for item in TOOL_DEFINITIONS))

    def test_lists_builtin_planning_rules(self) -> None:
        client = FakePlanningClient()

        result = handle_tool_call("list_planning_rules", {}, client)

        self.assertFalse(result["isError"])
        self.assertEqual(result["structuredContent"]["data"]["rule_count"], 2)
        self.assertIn(("rules",), client.calls)

    def test_runs_planning_with_normalized_rule_name(self) -> None:
        client = FakePlanningClient()

        result = handle_tool_call("run_rule_planning", {"rule_name": " atc "}, client)

        self.assertFalse(result["isError"])
        self.assertEqual(result["structuredContent"]["data"]["rule_name"], "ATC")
        self.assertIn(("run_rule", "ATC"), client.calls)

    def test_run_rule_requires_rule_name(self) -> None:
        result = handle_tool_call("run_rule_planning", {}, FakePlanningClient())

        self.assertFalse(result["isError"])
        self.assertEqual(result["structuredContent"]["error"]["code"], "INVALID_ARGUMENT")

    def test_order_tool_resolves_a_unique_business_query(self) -> None:
        client = FakePlanningClient()

        result = handle_tool_call(
            "get_order_planning",
            {"order_query": "O-1", "solution_ids": ["S-1"]},
            client,
        )

        self.assertFalse(result["isError"])
        self.assertEqual(result["structuredContent"]["data"]["order"]["order_id"], "O-1")
        self.assertIn(("order", "O-1", None, ["S-1"]), client.calls)

    def test_order_tool_returns_ambiguity_as_business_data(self) -> None:
        client = FakePlanningClient()
        client.order_matches = [
            {"order_id": "O-1", "order_name": "Alpha"},
            {"order_id": "O-2", "order_name": "Alpha 2"},
        ]

        result = handle_tool_call(
            "get_order_planning",
            {"order_query": "Alpha"},
            client,
        )

        self.assertFalse(result["isError"])
        self.assertFalse(result["structuredContent"]["ok"])
        self.assertEqual(result["structuredContent"]["error"]["code"], "AMBIGUOUS_ORDER")

    def test_infrastructure_failure_sets_mcp_is_error(self) -> None:
        class BrokenClient(FakePlanningClient):
            def get_overview(self, task_id=None):
                raise PlanningAPIError("PLANNING_API_UNAVAILABLE", "服务不可用")

        result = handle_tool_call("get_planning_overview", {}, BrokenClient())

        self.assertTrue(result["isError"])
        self.assertEqual(
            result["structuredContent"]["error"]["code"],
            "PLANNING_API_UNAVAILABLE",
        )

    def test_rejects_more_than_four_solutions_before_http_call(self) -> None:
        result = handle_tool_call(
            "compare_planning_solutions",
            {"solution_ids": ["1", "2", "3", "4", "5"]},
            FakePlanningClient(),
        )

        self.assertFalse(result["isError"])
        self.assertEqual(result["structuredContent"]["error"]["code"], "INVALID_ARGUMENT")


if __name__ == "__main__":
    unittest.main()
