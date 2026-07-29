"""What-if REST 端点：信封、错误码、隔离性。

隔离性是这组测试的重点——推演跑完之后，主流程的 inst_version、实例内容、
四步流程快照都必须原封不动。
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from llm4drd.api import server
from llm4drd.api.whatif_service import WhatIfService
from llm4drd.data.db import (
    DowntimeStore,
    InstanceStore,
    WorkflowProgressStore,
    get_instance_version,
    init_db,
)
from llm4drd.tests.shop_fixtures import make_graph_context_shop

NEW_MACHINE = {
    "op": "add", "entity": "machine",
    "values": {"machine_id": "M-C3", "machine_name": "Cutter 3", "type_id": "cut", "shifts": "0/0/24"},
}


class WhatIfApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = str(Path(self.tmp.name) / "whatif-api.db")
        init_db(self.db_path)
        originals = (
            server.inst_store,
            server.downtime_store,
            server.workflow_store,
            server.whatif_service,
            server.shop,
            server._active_shop_cache,
        )
        self.addCleanup(self._restore, originals)

        server.inst_store = InstanceStore(self.db_path)
        server.downtime_store = DowntimeStore(self.db_path)
        server.workflow_store = WorkflowProgressStore(self.db_path)
        server.inst_store.save_from_shopfloor(make_graph_context_shop())
        server.shop = None
        server._active_shop_cache = None
        server.whatif_service = WhatIfService(server.inst_store, server.downtime_store)
        self.client = TestClient(server.app)

    def _restore(self, originals) -> None:
        (
            server.inst_store,
            server.downtime_store,
            server.workflow_store,
            server.whatif_service,
            server.shop,
            server._active_shop_cache,
        ) = originals

    # --- helpers ---

    def create_scenario(self, name: str = "测试场景") -> str:
        response = self.client.post("/api/whatif/scenarios", json={"name": name})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["data"]["scenario_id"]

    def add_patches(self, scenario_id: str, patches: list[dict]):
        return self.client.post(f"/api/whatif/scenarios/{scenario_id}/patches", json={"patches": patches})

    def run_rules(self, scenario_id: str, rules: list[str]):
        return self.client.post("/api/whatif/runs", json={
            "scenario_id": scenario_id, "rule_names": rules, "wait_seconds": 45,
        })


class ScenarioEndpointTests(WhatIfApiTests):
    def test_create_and_describe(self):
        scenario_id = self.create_scenario("加一台车床")
        response = self.client.get(f"/api/whatif/scenarios/{scenario_id}")
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["name"], "加一台车床")
        self.assertEqual(payload["data"]["patch_count"], 0)
        self.assertFalse(payload["data"]["base_stale"])

    def test_add_patch_reports_scale_delta(self):
        scenario_id = self.create_scenario()
        payload = self.add_patches(scenario_id, [NEW_MACHINE]).json()["data"]
        self.assertEqual(payload["applied"][0]["matched"], 1)
        self.assertEqual(payload["scale"]["machines"], 4)
        self.assertEqual(payload["detail"]["scale_delta"], {"machines": 1})

    def test_where_selector_batch_update(self):
        scenario_id = self.create_scenario()
        payload = self.add_patches(scenario_id, [
            {"op": "update", "entity": "machine", "where": {"type_id": "cut"},
             "values": {"shifts": "0/8/10"}},
        ]).json()["data"]
        self.assertEqual(payload["applied"][0]["matched"], 2)

    def test_revert_patch(self):
        scenario_id = self.create_scenario()
        self.add_patches(scenario_id, [NEW_MACHINE])
        response = self.client.delete(f"/api/whatif/scenarios/{scenario_id}/patches?count=1")
        self.assertEqual(response.json()["data"]["patch_count"], 0)

    def test_unknown_scenario_returns_404_with_envelope(self):
        response = self.client.get("/api/whatif/scenarios/wf-nope")
        self.assertEqual(response.status_code, 404)
        detail = response.json()["detail"]
        self.assertFalse(detail["ok"])
        self.assertEqual(detail["error"]["code"], "WHATIF_SCENARIO_NOT_FOUND")

    def test_bad_patch_returns_422_with_suggestions(self):
        scenario_id = self.create_scenario()
        response = self.add_patches(scenario_id, [{"op": "add", "entity": "robot", "values": {"id": "R1"}}])
        self.assertEqual(response.status_code, 422)
        error = response.json()["detail"]["error"]
        self.assertEqual(error["code"], "UNKNOWN_ENTITY")
        self.assertTrue(error["suggestions"])

    def test_stale_base_returns_409(self):
        scenario_id = self.create_scenario()
        server.inst_store.update_machine("M-C1", {"machine_name": "改名", "type_id": "cut", "shifts": ""})
        response = self.add_patches(scenario_id, [NEW_MACHINE])
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["error"]["code"], "WHATIF_BASE_STALE")

    def test_delete_scenario(self):
        scenario_id = self.create_scenario()
        self.assertTrue(self.client.delete(f"/api/whatif/scenarios/{scenario_id}").json()["data"]["deleted"])
        self.assertEqual(self.client.get(f"/api/whatif/scenarios/{scenario_id}").status_code, 404)


class RunEndpointTests(WhatIfApiTests):
    def test_run_returns_metrics_inline_for_small_instance(self):
        scenario_id = self.create_scenario()
        payload = self.run_rules(scenario_id, ["ATC", "EDD"]).json()["data"]
        self.assertEqual(payload["status"], "done", payload.get("error"))
        self.assertEqual(len(payload["results"]), 2)
        self.assertIn("total_tardiness", payload["results"][0]["metrics"])

    def test_run_then_poll(self):
        scenario_id = self.create_scenario()
        run_id = self.run_rules(scenario_id, ["ATC"]).json()["data"]["run_id"]
        payload = self.client.get(f"/api/whatif/runs/{run_id}").json()["data"]
        self.assertEqual(payload["status"], "done")

    def test_unknown_rule_returns_422(self):
        scenario_id = self.create_scenario()
        response = self.run_rules(scenario_id, ["NOPE"])
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["error"]["code"], "PLANNING_RULE_NOT_FOUND")

    def test_compare_runs_across_scenarios(self):
        baseline_id = self.create_scenario("现状")
        baseline_run = self.run_rules(baseline_id, ["ATC"]).json()["data"]["run_id"]

        slower_id = self.create_scenario("工时翻倍")
        self.add_patches(slower_id, [
            {"op": "update", "entity": "operation", "id": "OP-11", "values": {"processing_time": 40.0}},
        ])
        slower_run = self.run_rules(slower_id, ["ATC"]).json()["data"]["run_id"]

        response = self.client.get(
            "/api/whatif/runs/compare",
            params={"run_ids": f"{baseline_run},{slower_run}", "metric_keys": "makespan"},
        )
        payload = response.json()["data"]
        self.assertEqual(payload["baseline"]["run_id"], baseline_run)
        self.assertEqual(len(payload["entries"]), 2)
        self.assertFalse(payload["entries"][1]["deltas"]["makespan"]["better"])

    def test_compare_path_is_not_swallowed_by_run_id_route(self):
        """/runs/compare 必须先于 /runs/{run_id} 匹配，否则跨场景对比永远 404。"""
        response = self.client.get("/api/whatif/runs/compare", params={"run_ids": ""})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["error"]["code"], "INVALID_ARGUMENT")

    def test_validation_error_blocks_the_run(self):
        scenario_id = self.create_scenario("删掉唯一可用机")
        self.add_patches(scenario_id, [{"op": "remove", "entity": "machine", "id": "M-C1"}])
        payload = self.run_rules(scenario_id, ["ATC"]).json()["data"]
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error"]["code"], "WHATIF_VALIDATION_FAILED")


class IsolationEndpointTests(WhatIfApiTests):
    def test_whatif_leaves_instance_and_workflow_untouched(self):
        version_before = get_instance_version(self.db_path)
        validate_before = self.client.get("/api/instance/validate").json()
        workflow_before = self.client.get("/api/workflow/progress").json()

        scenario_id = self.create_scenario("隔离验证")
        self.add_patches(scenario_id, [
            NEW_MACHINE,
            {"op": "update", "entity": "operation", "id": "OP-11", "values": {"processing_time": 12.0}},
        ])
        self.assertEqual(self.run_rules(scenario_id, ["ATC", "EDD"]).json()["data"]["status"], "done")

        self.assertEqual(get_instance_version(self.db_path), version_before)
        self.assertEqual(self.client.get("/api/instance/validate").json(), validate_before)
        self.assertEqual(self.client.get("/api/workflow/progress").json(), workflow_before)
        self.assertNotIn("M-C3", server.inst_store.build_shopfloor().machines)

    def test_whatif_does_not_disturb_main_simulation_result(self):
        main_before = self.client.post("/api/simulate", json={"rule_name": "ATC"}).json()

        scenario_id = self.create_scenario("扰动")
        self.add_patches(scenario_id, [
            {"op": "update", "entity": "operation", "id": "OP-11", "values": {"processing_time": 40.0}},
        ])
        self.run_rules(scenario_id, ["ATC"])

        main_after = self.client.post("/api/simulate", json={"rule_name": "ATC"}).json()
        self.assertEqual(main_after["metrics"]["makespan"], main_before["metrics"]["makespan"])
        self.assertEqual(main_after["metrics"]["total_tardiness"], main_before["metrics"]["total_tardiness"])


class ApplyEndpointTests(WhatIfApiTests):
    def test_apply_requires_confirm_token(self):
        scenario_id = self.create_scenario("落库")
        self.add_patches(scenario_id, [NEW_MACHINE])
        response = self.client.post(
            f"/api/whatif/scenarios/{scenario_id}/apply", json={"confirm_token": "bogus"})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["error"]["code"], "WHATIF_CONFIRM_REQUIRED")

    def test_apply_writes_instance_and_warns_about_side_effects(self):
        scenario_id = self.create_scenario("落库")
        self.add_patches(scenario_id, [NEW_MACHINE])
        token = self.client.get(f"/api/whatif/scenarios/{scenario_id}").json()["data"]["apply_token"]
        version_before = get_instance_version(self.db_path)

        payload = self.client.post(
            f"/api/whatif/scenarios/{scenario_id}/apply", json={"confirm_token": token}).json()["data"]

        self.assertTrue(payload["applied"])
        self.assertGreater(payload["instance_version"], version_before)
        self.assertIn("四步流程", payload["side_effect_notice"])
        self.assertTrue(Path(payload["backup_path"]).exists())
        self.assertIn("M-C3", server.inst_store.build_shopfloor().machines)


class ResourceSearchTests(WhatIfApiTests):
    def test_machine_search_exposes_shift_template(self):
        payload = self.client.get(
            "/api/query/planning/resources/search", params={"entity_type": "machine"}).json()["data"]
        self.assertEqual(payload["total"], 3)
        machine = next(item for item in payload["items"] if item["machine_id"] == "M-C1")
        self.assertEqual(machine["type_id"], "cut")
        # fixture 的每台机器都是 14 天 × 每天一班 0:00 起 24h
        self.assertEqual(machine["shift_pattern"], "0/0.0/24.0")
        self.assertEqual(machine["shifts_per_day"], 1)
        self.assertTrue(machine["calendar_uniform"])
        self.assertGreaterEqual(machine["shift_days"], 14)

    def test_shift_pattern_collapses_the_expanded_calendar(self):
        """日历被 ensure_calendar_capacity 铺开到几百天；回给 Agent 的必须是一天的模式，
        否则响应会大到被 MCP 宿主转成 artifact，多花一步 read_artifact 才能读。"""
        payload = self.client.get(
            "/api/query/planning/resources/search", params={"entity_type": "machine"}).json()["data"]
        machine = payload["items"][0]
        self.assertNotIn("shifts", machine)
        self.assertNotIn("shifts_detail", machine)
        self.assertLess(len(machine["shift_pattern"]), 40)
        self.assertGreater(machine["shift_days"], machine["shifts_per_day"])

    def test_non_uniform_calendar_is_flagged(self):
        shop = server.inst_store.build_shopfloor()
        from llm4drd.core.models import Shift
        shop.machines["M-C1"].shifts = [Shift(day=0, start_hour=0.0, hours=24.0),
                                        Shift(day=1, start_hour=8.0, hours=8.0)]
        server._active_shop_cache = (server.get_instance_version(self.db_path), shop)
        self.addCleanup(setattr, server, "_active_shop_cache", None)
        payload = self.client.get(
            "/api/query/planning/resources/search", params={"entity_type": "machine", "q": "M-C1"}).json()["data"]
        self.assertFalse(payload["items"][0]["calendar_uniform"])

    def test_machine_type_search_reports_counts(self):
        payload = self.client.get(
            "/api/query/planning/resources/search", params={"entity_type": "machine_type"}).json()["data"]
        cut = next(item for item in payload["items"] if item["type_id"] == "cut")
        self.assertEqual(cut["machine_count"], 2)
        self.assertTrue(cut["is_critical"])

    def test_keyword_filter(self):
        payload = self.client.get(
            "/api/query/planning/resources/search",
            params={"entity_type": "machine", "q": "M-C1"}).json()["data"]
        self.assertEqual([item["machine_id"] for item in payload["items"]], ["M-C1"])

    def test_unknown_entity_type_rejected(self):
        response = self.client.get(
            "/api/query/planning/resources/search", params={"entity_type": "robot"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["error"]["code"], "INVALID_ARGUMENT")


if __name__ == "__main__":
    unittest.main()
