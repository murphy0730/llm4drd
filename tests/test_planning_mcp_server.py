from __future__ import annotations

import unittest

from llm4drd.mcp_server.errors import PlanningAPIError
from llm4drd.mcp_server.server import TOOL_DEFINITIONS, handle_tool_call


class FakePlanningClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.order_planned = True
        self.bottleneck_payload_extra: dict = {}
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

    def compare_solutions(self, task_id=None, solution_ids=None, metric_keys=None, machine_limit=None):
        self.calls.append(("compare", task_id, solution_ids, metric_keys, machine_limit))
        return {
            "ok": True,
            "data": {
                "solution_count": 1,
                "metric_keys": ["total_tardiness"],
                "machine_limit": machine_limit or 20,
                "ranking_source": "utilization_ranking",
                "solutions": [{
                    "solution_id": "S-1",
                    "machine_utilization_ranking": [
                        {"machine_id": "M-C1", "machine_name": "车床1", "full_horizon_utilization": 0.92},
                        {"machine_id": "M-C2", "machine_name": "车床2", "full_horizon_utilization": 0.71},
                    ],
                }],
            },
        }

    def diagnose_bottleneck(self, solution_id, task_id=None, machine_limit=None):
        self.calls.append(("bottleneck", solution_id, task_id, machine_limit))
        return {
            "ok": True,
            "data": {
                "solution_name": "方案二",
                "tardy_order_count": 3,
                "total_order_tardiness_hours": 42.0,
                "unscheduled_order_count": 0,
                "partially_scheduled_order_count": 0,
                "wait_breakdown": {
                    "capacity_bound": 5.0, "dispatch_bound": 1.0,
                    "off_shift": 88.0, "downtime": 0.0, "idle": 2.0,
                },
                "machines": [{
                    "machine_id": "M-C1", "machine_name": "车床1",
                    "capacity_wait_hours": 5.0, "saturation": 0.93,
                }],
                **self.bottleneck_payload_extra,
            },
        }

    def explain_order_delay(self, order_id, solution_id, task_id=None):
        self.calls.append(("order_delay", order_id, solution_id, task_id))
        return {
            "ok": True,
            "data": {
                "order_name": "订单甲",
                "planned": self.order_planned,
                "scheduled_operation_count": 3 if self.order_planned else 0,
                "total_operation_count": 3,
                "tardiness_hours": 30.0 if self.order_planned else None,
                "inevitable_tardiness_hours": 12.0,
                "attribution": {
                    "capacity_bound": 4.0, "dispatch_bound": 0.0,
                    "off_shift": 14.0, "downtime": 0.0, "idle": 0.0,
                },
            },
        }

    def get_whatif_run(self, run_id, machine_limit=None):
        self.calls.append(("whatif_run", run_id, machine_limit))
        return {"ok": True, "data": {"run_id": run_id, "status": "running"}}

    def compare_whatif_runs(self, run_ids, metric_keys=None, machine_limit=None):
        self.calls.append(("whatif_compare", run_ids, metric_keys, machine_limit))
        return {
            "ok": True,
            "data": {
                "baseline": {"scenario_name": "现状（基线）", "rule_name": "ATC"},
                "entries": [{
                    "rule_name": "ATC",
                    "machine_utilization_ranking": [
                        {"machine_id": "M-C1", "machine_name": "车床1", "full_horizon_utilization": 0.88},
                    ],
                }],
            },
        }

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
                # 延误归因
                "diagnose_bottleneck",
                "explain_order_delay",
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

    def test_passes_machine_limit_through_to_the_planning_api(self) -> None:
        client = FakePlanningClient()

        handle_tool_call("compare_planning_solutions", {"machine_limit": 3}, client)
        handle_tool_call("get_whatif_run", {"run_id": "R-1", "machine_limit": 3}, client)
        handle_tool_call("compare_whatif_runs", {"run_ids": ["R-1"], "machine_limit": 3}, client)

        self.assertEqual([call[-1] for call in client.calls], [3, 3, 3])

    def test_omitting_machine_limit_leaves_the_default_to_the_backend(self) -> None:
        client = FakePlanningClient()

        handle_tool_call("compare_planning_solutions", {}, client)

        # 传 None 而不是 20：默认值只在后端定义一处，避免两边各写一份后漂移。
        self.assertIsNone(client.calls[0][-1])

    def test_rejects_out_of_range_machine_limit(self) -> None:
        result = handle_tool_call(
            "compare_planning_solutions",
            {"machine_limit": 99},
            FakePlanningClient(),
        )

        self.assertEqual(result["structuredContent"]["error"]["code"], "INVALID_ARGUMENT")

    def test_summarises_the_top_loaded_machine(self) -> None:
        result = handle_tool_call("compare_planning_solutions", {}, FakePlanningClient())

        self.assertIn("车床1", result["content"][0]["text"])
        self.assertIn("92%", result["content"][0]["text"])
        # 摘要必须说"负荷"而不是"瓶颈"，否则 Agent 又会把排行榜当归因。
        self.assertIn("非瓶颈判定", result["content"][0]["text"])

    def test_missing_solution_id_reaches_the_backend_for_suggestions(self) -> None:
        """本端不抢先拦：后端的 SOLUTION_REQUIRED 会把可选方案一并递给 Agent，
        比一句"solution_id 不能为空"更容易自我修正。"""
        client = FakePlanningClient()
        handle_tool_call("diagnose_bottleneck", {}, client)

        self.assertEqual(client.calls, [("bottleneck", None, None, None)])

    def test_diagnose_bottleneck_summary_names_the_dominant_cause(self) -> None:
        """摘要要直接点名主因——白班场景的锅不能扣在机器头上。"""
        client = FakePlanningClient()
        result = handle_tool_call("diagnose_bottleneck", {"solution_id": "S-1"}, client)

        text = result["content"][0]["text"]
        self.assertIn("该时段没排班", text)
        self.assertNotIn("产能不足（同类机器", text)
        self.assertEqual(client.calls, [("bottleneck", "S-1", None, None)])

    def test_unscheduled_orders_are_warned_before_the_attribution(self) -> None:
        """没排进去的订单必须先说，否则会被当成"这方案挺好"。"""
        client = FakePlanningClient()
        client.bottleneck_payload_extra = {"unscheduled_order_count": 2}
        result = handle_tool_call("diagnose_bottleneck", {"solution_id": "S-1"}, client)

        text = result["content"][0]["text"]
        self.assertTrue(text.startswith("⚠"))
        self.assertIn("2 个订单完全未排入该方案", text)

    def test_unplanned_order_is_never_summarised_as_on_time(self) -> None:
        client = FakePlanningClient()
        client.order_planned = False
        result = handle_tool_call(
            "explain_order_delay", {"order_id": "O-1", "solution_id": "S-1"}, client
        )

        text = result["content"][0]["text"]
        self.assertIn("未完整排入该方案", text)
        self.assertNotIn("未延误", text)

    def test_explain_order_delay_separates_inevitable_tardiness(self) -> None:
        client = FakePlanningClient()
        result = handle_tool_call(
            "explain_order_delay", {"order_id": "O-1", "solution_id": "S-1"}, client
        )

        text = result["content"][0]["text"]
        self.assertIn("工艺链本身就来不及", text)
        self.assertIn("12.0h", text)
        self.assertEqual(client.calls, [("order_delay", "O-1", "S-1", None)])

    def test_scenario_id_is_rejected_for_non_resource_entities(self) -> None:
        client = FakePlanningClient()
        result = handle_tool_call(
            "search_planning_entities",
            {"entity_type": "order", "query": "O-1", "scenario_id": "SC-1"},
            client,
        )

        self.assertFalse(result["structuredContent"]["ok"])
        self.assertEqual(client.calls, [])

    def test_summarises_the_top_loaded_machine_for_whatif_compare(self) -> None:
        result = handle_tool_call(
            "compare_whatif_runs", {"run_ids": ["R-1"]}, FakePlanningClient()
        )

        self.assertIn("车床1", result["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
