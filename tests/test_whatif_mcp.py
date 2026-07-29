"""MCP 层 what-if 工具的分派、参数映射与摘要生成。

与 test_planning_mcp_server.py 同样用假 client——MCP 层是薄代理，
这里只验证"参数怎么翻译成 HTTP 调用"和"payload 怎么翻译成人话摘要"。
"""
from __future__ import annotations

import unittest

from llm4drd.mcp_server.errors import PlanningAPIError
from llm4drd.mcp_server.server import handle_tool_call

NEW_MACHINE = {
    "op": "add", "entity": "machine",
    "values": {"machine_id": "M20", "machine_name": "车床20", "type_id": "turning", "shifts": "0/8/10"},
}


class FakeWhatIfClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def search_resources(self, entity_type, query="", limit=20):
        self.calls.append(("search_resources", entity_type, query, limit))
        return {"ok": True, "data": {"entity_type": entity_type, "total": 3, "items": []}}

    def search_orders(self, query, limit=20):
        self.calls.append(("search_orders", query, limit))
        return {"ok": True, "data": []}

    def search_operations(self, query, order_id=None, limit=20):
        self.calls.append(("search_operations", query, order_id, limit))
        return {"ok": True, "data": []}

    def create_whatif_scenario(self, name=None):
        self.calls.append(("create", name))
        return {"ok": True, "data": {"scenario_id": "wf-1", "name": name, "patch_count": 0,
                                     "validation": {"errors": [], "warnings": []}}}

    def list_whatif_scenarios(self):
        self.calls.append(("list_scenarios",))
        return {"ok": True, "data": {"scenario_count": 2, "scenarios": []}}

    def describe_whatif_scenario(self, scenario_id):
        self.calls.append(("describe", scenario_id))
        return {"ok": True, "data": {
            "scenario_id": scenario_id, "name": "加一台车床", "patch_count": 1,
            "scale_delta": {"machines": 1}, "apply_token": "4-abc",
            "validation": {"errors": [], "warnings": []},
        }}

    def apply_whatif_patch(self, scenario_id, patches):
        self.calls.append(("patch", scenario_id, patches))
        return {"ok": True, "data": {
            "applied": [{"message": "已新增机器 M20", "matched": 1}],
            "scale": {"machines": 20},
        }}

    def revert_whatif_patch(self, scenario_id, count=1):
        self.calls.append(("revert", scenario_id, count))
        return {"ok": True, "data": {"scenario_id": scenario_id, "name": "x", "patch_count": 0,
                                     "validation": {"errors": [], "warnings": []}}}

    def run_whatif_planning(self, scenario_id, rule_names=None, include_baseline=False, wait_seconds=None):
        self.calls.append(("run", scenario_id, rule_names, include_baseline))
        return {"ok": True, "data": {
            "run_id": "wr-1", "scenario_name": "加一台车床", "rule_names": rule_names or ["ATC"],
            "status": "done",
            "results": [{"rule_name": "ATC", "metrics": {"total_tardiness": 8.1, "makespan": 40.0}}],
        }}

    def get_whatif_run(self, run_id):
        self.calls.append(("get_run", run_id))
        return {"ok": True, "data": {"run_id": run_id, "scenario_name": "x", "rule_names": ["ATC"],
                                     "status": "running", "results": []}}

    def compare_whatif_runs(self, run_ids, metric_keys=None):
        self.calls.append(("compare_runs", run_ids, metric_keys))
        return {"ok": True, "data": {
            "baseline": {"run_id": run_ids[0], "scenario_name": "现状", "rule_name": "ATC"},
            "metrics": [], "entries": [{}, {}],
        }}

    def apply_whatif_to_instance(self, scenario_id, confirm_token):
        self.calls.append(("apply", scenario_id, confirm_token))
        return {"ok": True, "data": {"applied": True, "instance_version": 9,
                                     "backup_path": "/tmp/backup.db"}}


class WhatIfDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeWhatIfClient()

    def call(self, name, arguments):
        return handle_tool_call(name, arguments, self.client)

    def test_create_scenario(self):
        result = self.call("create_whatif_scenario", {"name": "加一台车床"})
        self.assertFalse(result["isError"])
        self.assertEqual(self.client.calls[0], ("create", "加一台车床"))

    def test_apply_patch_passes_patches_through(self):
        result = self.call("apply_whatif_patch", {"scenario_id": "wf-1", "patches": [NEW_MACHINE]})
        self.assertFalse(result["isError"])
        self.assertEqual(self.client.calls[0], ("patch", "wf-1", [NEW_MACHINE]))
        self.assertIn("已新增机器 M20", result["content"][0]["text"])

    def test_apply_patch_rejects_empty_patches(self):
        result = self.call("apply_whatif_patch", {"scenario_id": "wf-1", "patches": []})
        self.assertFalse(result["isError"])  # 业务错误，让模型自己改
        self.assertEqual(result["structuredContent"]["error"]["code"], "INVALID_ARGUMENT")
        self.assertEqual(self.client.calls, [])

    def test_describe_without_id_lists_scenarios(self):
        result = self.call("describe_whatif_scenario", {})
        self.assertEqual(self.client.calls[0], ("list_scenarios",))
        self.assertIn("2 个 what-if 场景", result["content"][0]["text"])

    def test_describe_with_id_summarizes_changes(self):
        result = self.call("describe_whatif_scenario", {"scenario_id": "wf-1"})
        self.assertEqual(self.client.calls[0], ("describe", "wf-1"))
        text = result["content"][0]["text"]
        self.assertIn("1 处改动", text)
        self.assertIn("machines", text)

    def test_describe_surfaces_validation_errors_in_summary(self):
        self.client.describe_whatif_scenario = lambda scenario_id: {"ok": True, "data": {
            "scenario_id": scenario_id, "name": "删机", "patch_count": 1,
            "validation": {"errors": [{"message": "工序无可用机器"}], "warnings": []},
        }}
        result = self.call("describe_whatif_scenario", {"scenario_id": "wf-1"})
        self.assertIn("不能直接跑", result["content"][0]["text"])

    def test_revert_normalizes_count(self):
        self.call("revert_whatif_patch", {"scenario_id": "wf-1", "count": 3})
        self.assertEqual(self.client.calls[0], ("revert", "wf-1", 3))

    def test_revert_rejects_out_of_range_count(self):
        result = self.call("revert_whatif_patch", {"scenario_id": "wf-1", "count": 0})
        self.assertEqual(result["structuredContent"]["error"]["code"], "INVALID_ARGUMENT")

    def test_run_uppercases_and_dedups_rules(self):
        self.call("run_whatif_planning", {"scenario_id": "wf-1", "rule_names": ["atc", "ATC", "edd"]})
        # include_baseline 默认开启：一次调用同时拿到改动后与现状，省下三步
        self.assertEqual(self.client.calls[0], ("run", "wf-1", ["ATC", "EDD"], True))

    def test_run_summary_reports_best_rule(self):
        result = self.call("run_whatif_planning", {"scenario_id": "wf-1"})
        self.assertIn("ATC", result["content"][0]["text"])
        self.assertIn("总延迟 8.1", result["content"][0]["text"])

    def test_running_status_tells_model_to_poll(self):
        result = self.call("get_whatif_run", {"run_id": "wr-1"})
        self.assertIn("get_whatif_run 轮询", result["content"][0]["text"])

    def test_compare_requires_run_ids(self):
        result = self.call("compare_whatif_runs", {"run_ids": []})
        self.assertEqual(result["structuredContent"]["error"]["code"], "INVALID_ARGUMENT")

    def test_compare_dedups_run_ids(self):
        self.call("compare_whatif_runs", {"run_ids": ["wr-1", "wr-1", "wr-2"], "metric_keys": ["makespan"]})
        self.assertEqual(self.client.calls[0], ("compare_runs", ["wr-1", "wr-2"], ["makespan"]))

    def test_apply_requires_confirm_token(self):
        result = self.call("apply_whatif_to_instance", {"scenario_id": "wf-1"})
        self.assertEqual(result["structuredContent"]["error"]["code"], "INVALID_ARGUMENT")
        self.assertEqual(self.client.calls, [])

    def test_apply_summary_warns_about_side_effects(self):
        result = self.call("apply_whatif_to_instance", {"scenario_id": "wf-1", "confirm_token": "4-abc"})
        self.assertEqual(self.client.calls[0], ("apply", "wf-1", "4-abc"))
        self.assertIn("四步流程", result["content"][0]["text"])

    def test_infrastructure_failure_is_flagged_as_error(self):
        def boom(name=None):
            raise PlanningAPIError("PLANNING_API_UNAVAILABLE", "连不上")

        self.client.create_whatif_scenario = boom
        result = self.call("create_whatif_scenario", {"name": "x"})
        self.assertTrue(result["isError"])


class WaitBudgetTests(unittest.TestCase):
    """内联等待预算的双重上限。

    这条规则曾经漏掉服务端上限：客户端超时配到 50s 时算出 wait_seconds=48，
    被 /api/whatif/runs 的请求体校验（le=45）直接打回 422，推演一次都跑不起来。
    """

    def _sent_wait(self, timeout_seconds: float, wait_seconds=None) -> float:
        from llm4drd.mcp_server.planning_client import PlanningAPIClient

        client = PlanningAPIClient("http://127.0.0.1:8888", timeout_seconds)
        captured = {}

        def fake_post(path, payload):
            captured.update(payload)
            return {"ok": True, "data": {}}

        client._post = fake_post
        client.run_whatif_planning("wf-1", ["ATC"], wait_seconds)
        return captured["wait_seconds"]

    def test_never_exceeds_server_limit(self):
        from llm4drd.mcp_server.planning_client import MAX_SERVER_WAIT_SECONDS

        self.assertLessEqual(self._sent_wait(60.0), MAX_SERVER_WAIT_SECONDS)
        self.assertLessEqual(self._sent_wait(50.0), MAX_SERVER_WAIT_SECONDS)

    def test_leaves_headroom_under_client_timeout(self):
        self.assertEqual(self._sent_wait(10.0), 8.0)

    def test_explicit_wait_is_still_clamped(self):
        self.assertEqual(self._sent_wait(10.0, wait_seconds=30.0), 8.0)

    def test_never_negative(self):
        self.assertGreaterEqual(self._sent_wait(1.0), 0.0)


class ResourceSearchDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeWhatIfClient()

    def test_machine_search_allows_empty_query(self):
        result = handle_tool_call("search_planning_entities", {"entity_type": "machine"}, self.client)
        self.assertFalse(result["isError"])
        self.assertEqual(self.client.calls[0], ("search_resources", "machine", "", 20))

    def test_resource_types_route_to_resource_endpoint(self):
        for entity_type in ("machine", "machine_type", "tooling", "personnel"):
            with self.subTest(entity_type=entity_type):
                client = FakeWhatIfClient()
                handle_tool_call("search_planning_entities",
                                 {"entity_type": entity_type, "query": "车"}, client)
                self.assertEqual(client.calls[0][0], "search_resources")

    def test_order_search_still_requires_query(self):
        result = handle_tool_call("search_planning_entities", {"entity_type": "order"}, self.client)
        self.assertEqual(result["structuredContent"]["error"]["code"], "INVALID_ARGUMENT")

    def test_unknown_entity_type_lists_valid_options(self):
        result = handle_tool_call("search_planning_entities",
                                  {"entity_type": "robot", "query": "x"}, self.client)
        self.assertIn("machine", result["structuredContent"]["error"]["message"])


if __name__ == "__main__":
    unittest.main()
