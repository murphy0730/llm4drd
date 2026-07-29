"""What-if 编排层：物化正确性、跑规则、对比、隔离性。

最关键的一条是 test_empty_scenario_matches_build_shopfloor——它钉死
build_shopfloor_from_rows 的抽取是零行为变化的；一旦有人改坏取数路径，
what-if 的所有结论都会悄悄偏离主流程。
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm4drd.api import whatif_service as whatif_service_module
from llm4drd.api.whatif_service import WhatIfError, WhatIfService, rows_to_save_args
from llm4drd.data.db import DowntimeStore, InstanceStore, get_instance_version, init_db
from llm4drd.tests.shop_fixtures import make_graph_context_shop


def shop_signature(shop) -> dict:
    """对 ShopFloor 取可比较的结构化指纹（覆盖会被 patch 影响的全部字段）。"""
    return {
        "plan_start_at": shop.plan_start_at.isoformat(),
        "machine_types": sorted(
            (t.id, t.name, t.is_critical) for t in shop.machine_types.values()
        ),
        "machines": sorted(
            (m.id, m.name, m.type_id,
             tuple((s.day, s.start_hour, s.hours) for s in m.shifts),
             tuple(sorted((d.machine_id, d.start_time, d.end_time) for d in m.downtimes)))
            for m in shop.machines.values()
        ),
        "toolings": sorted((t.id, t.name, t.type_id) for t in shop.toolings.values()),
        "personnel": sorted(
            (p.id, p.name, tuple(p.skills)) for p in shop.personnel.values()
        ),
        "orders": sorted(
            (o.id, o.name, o.release_time, o.due_date, o.priority,
             tuple(sorted(o.task_ids)), o.main_task_id)
            for o in shop.orders.values()
        ),
        "tasks": sorted(
            (t.id, t.order_id, t.name, t.is_main, tuple(t.predecessor_task_ids),
             t.release_time, t.due_date)
            for t in shop.tasks.values()
        ),
        "operations": sorted(
            (o.id, o.task_id, o.name, o.process_type, o.processing_time, o.turnover_time,
             tuple(o.predecessor_ops), tuple(o.predecessor_tasks),
             tuple(o.eligible_machine_ids), tuple(o.required_tooling_types),
             tuple(o.required_personnel_skills), str(o.status))
            for o in shop.operations.values()
        ),
    }


class WhatIfServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = str(Path(self.tmp.name) / "inst.db")
        init_db(self.db_path)
        self.store = InstanceStore(self.db_path)
        self.downtimes = DowntimeStore(self.db_path)
        self.store.save_from_shopfloor(make_graph_context_shop())
        self.service = WhatIfService(self.store, self.downtimes)

    def run_scenario(self, scenario, rules):
        run = self.service.submit_run(scenario, rules)
        self.service.wait_for(run.run_id, 60)
        return self.service.get_run(run.run_id)


class MaterializeTests(WhatIfServiceTestCase):
    def test_empty_scenario_matches_build_shopfloor(self):
        """空 patch 场景物化出的实例，必须与主流程 build_shopfloor() 逐字段一致。"""
        scenario = self.service.create_scenario("空场景")
        shop = self.service._materialize_shop(scenario)
        self.assertEqual(shop_signature(shop), shop_signature(self.store.build_shopfloor()))

    def test_describe_does_not_build_a_simulation_runtime(self):
        """describe 只做校验，绝不能顺带深拷贝整个实例——大实例上那是十几秒，会撑爆 MCP 超时。"""
        built = []
        original = whatif_service_module.SimulationRuntime

        def spy(shop, *args, **kwargs):
            built.append(shop)
            return original(shop, *args, **kwargs)

        whatif_service_module.SimulationRuntime = spy
        self.addCleanup(setattr, whatif_service_module, "SimulationRuntime", original)

        scenario = self.service.create_scenario("只描述")
        self.service.describe(scenario)
        self.assertEqual(built, [], "describe 不应构建 SimulationRuntime")

        self.run_scenario(scenario, ["ATC"])
        self.assertEqual(len(built), 1, "跑仿真时才应构建一次 SimulationRuntime")

    def test_runtime_is_reused_across_runs_of_the_same_scenario(self):
        scenario = self.service.create_scenario("复用")
        self.run_scenario(scenario, ["ATC"])
        _, first = self.service._materialize(scenario)
        self.run_scenario(scenario, ["EDD"])
        _, second = self.service._materialize(scenario)
        self.assertIs(first, second)

    def test_shop_and_runtime_stay_same_source(self):
        """runtime 建好后，缓存里必须还是同一份 shop——校验看的和仿真跑的不能是两个实例。"""
        scenario = self.service.create_scenario("同源")
        shop_before = self.service._materialize_shop(scenario)
        shop_after, _ = self.service._materialize(scenario)
        self.assertIs(shop_after, shop_before)
        self.assertIs(self.service._materialize_shop(scenario), shop_before)

    def test_materialize_reuses_single_slot_cache(self):
        scenario = self.service.create_scenario("缓存")
        first_shop, first_runtime = self.service._materialize(scenario)
        second_shop, second_runtime = self.service._materialize(scenario)
        self.assertIs(first_shop, second_shop)
        self.assertIs(first_runtime, second_runtime)

    def test_patch_invalidates_materialized_cache(self):
        scenario = self.service.create_scenario("失效")
        first_shop, _ = self.service._materialize(scenario)
        self.service.add_patches(scenario, [
            {"op": "add", "entity": "machine", "values": {
                "machine_id": "M-C3", "machine_name": "Cutter 3", "type_id": "cut",
                "shifts": "0/0/24"}},
        ])
        second_shop, _ = self.service._materialize(self.service.require_scenario(scenario.scenario_id))
        self.assertIsNot(first_shop, second_shop)
        self.assertIn("M-C3", second_shop.machines)

    def test_added_machine_lands_in_type_index(self):
        """新增机器必须进 _machine_by_type 反向索引，否则派工永远看不到它。"""
        scenario = self.service.create_scenario("加机")
        self.service.add_patches(scenario, [
            {"op": "add", "entity": "machine", "values": {
                "machine_id": "M-C3", "machine_name": "Cutter 3", "type_id": "cut",
                "shifts": "0/0/24"}},
        ])
        shop, _ = self.service._materialize(self.service.require_scenario(scenario.scenario_id))
        self.assertIn("M-C3", [m.id for m in shop.get_machines_for_type("cut")])

    def test_downtime_patch_reaches_machine(self):
        scenario = self.service.create_scenario("停机")
        self.service.add_patches(scenario, [
            {"op": "add", "entity": "downtime", "values": {
                "machine_id": "M-C1", "downtime_type": "maintenance",
                "start_time": 5.0, "end_time": 9.0}},
        ])
        shop, _ = self.service._materialize(self.service.require_scenario(scenario.scenario_id))
        self.assertEqual(len(shop.machines["M-C1"].downtimes), 1)
        self.assertFalse(shop.machines["M-C1"].is_available_at(6.0))


class IsolationTests(WhatIfServiceTestCase):
    def test_whatif_never_touches_the_database(self):
        version_before = get_instance_version(self.db_path)
        scenario = self.service.create_scenario("只读")
        self.service.add_patches(scenario, [
            {"op": "remove", "entity": "machine", "id": "M-C2"},
            {"op": "update", "entity": "operation", "id": "OP-11", "values": {"processing_time": 9.0}},
        ])
        self.run_scenario(self.service.require_scenario(scenario.scenario_id), ["ATC"])

        self.assertEqual(get_instance_version(self.db_path), version_before)
        reloaded = self.store.build_shopfloor()
        self.assertIn("M-C2", reloaded.machines)
        self.assertEqual(reloaded.operations["OP-11"].processing_time, 4.0)

    def test_stale_base_is_detected(self):
        scenario = self.service.create_scenario("过期")
        self.store.update_machine("M-C1", {"machine_name": "改名", "type_id": "cut", "shifts": ""})
        with self.assertRaises(WhatIfError) as ctx:
            self.service.add_patches(scenario, [{"op": "remove", "entity": "machine", "id": "M-C2"}])
        self.assertEqual(ctx.exception.code, "WHATIF_BASE_STALE")


class RunTests(WhatIfServiceTestCase):
    def test_run_multiple_rules_returns_metrics(self):
        scenario = self.service.create_scenario("基线")
        run = self.run_scenario(scenario, ["ATC", "EDD", "SPT"])
        self.assertEqual(run.status, "done", run.error)
        self.assertEqual(len(run.results), 3)
        for item in run.results:
            self.assertIn("total_tardiness", item["metrics"])
            self.assertIn("makespan", item["metrics"])
            self.assertTrue(item["metrics"]["feasible"])

    def test_run_matches_main_flow_kpi(self):
        """同一份数据、同一条规则，what-if 的 KPI 必须与主流程仿真逐位一致。"""
        from llm4drd.core.rules import BUILTIN_RULES
        from llm4drd.core.simulator import Simulator
        from llm4drd.optimization.objectives import build_schedule_analytics

        reference_shop = self.store.build_shopfloor()
        reference = Simulator(reference_shop, BUILTIN_RULES["ATC"]).run()
        reference_kpi = build_schedule_analytics(reference_shop, reference).objective_values

        run = self.run_scenario(self.service.create_scenario("对齐"), ["ATC"])
        metrics = run.results[0]["metrics"]
        for key, value in reference_kpi.items():
            self.assertAlmostEqual(metrics[key], round(float(value), 4), places=4, msg=key)

    def test_removing_the_only_eligible_machine_is_rejected(self):
        """OP-11 只能上 M-C1，删掉它会让工序永远排不出——必须在跑之前拦住。"""
        scenario = self.service.create_scenario("删瓶颈机")
        self.service.add_patches(scenario, [{"op": "remove", "entity": "machine", "id": "M-C1"}])
        run = self.run_scenario(self.service.require_scenario(scenario.scenario_id), ["ATC"])
        self.assertEqual(run.status, "failed")
        self.assertEqual(run.error["code"], "WHATIF_VALIDATION_FAILED")

    def test_unknown_rule_rejected(self):
        with self.assertRaises(WhatIfError) as ctx:
            self.service.submit_run(self.service.create_scenario("坏规则"), ["NOPE"])
        self.assertEqual(ctx.exception.code, "PLANNING_RULE_NOT_FOUND")

    def test_removing_a_parallel_machine_worsens_makespan(self):
        """OP-12 / OP-21 按 process_type 匹配到 cut 类的两台机；砍掉一台就得排队。"""
        baseline = self.run_scenario(self.service.create_scenario("现状"), ["SPT"])
        scenario = self.service.create_scenario("砍一台车床")
        self.service.add_patches(scenario, [{"op": "remove", "entity": "machine", "id": "M-C2"}])
        reduced = self.run_scenario(self.service.require_scenario(scenario.scenario_id), ["SPT"])

        self.assertEqual(reduced.status, "done", reduced.error)
        self.assertGreater(
            reduced.results[0]["metrics"]["makespan"],
            baseline.results[0]["metrics"]["makespan"],
        )

    def test_slower_operations_increase_tardiness(self):
        baseline = self.run_scenario(self.service.create_scenario("基线"), ["ATC"])
        scenario = self.service.create_scenario("工时翻倍")
        self.service.add_patches(scenario, [
            {"op": "update", "entity": "operation", "id": "OP-11", "values": {"processing_time": 40.0}},
        ])
        slower = self.run_scenario(self.service.require_scenario(scenario.scenario_id), ["ATC"])
        self.assertGreater(
            slower.results[0]["metrics"]["makespan"],
            baseline.results[0]["metrics"]["makespan"],
        )


class IncludeBaselineTests(WhatIfServiceTestCase):
    """一次调用同时跑「改动后」与「现状」——宿主 Agent 的单次 Run 步数预算很紧，
    省下「建基线场景 + 再跑一次 + 对比」三步是能不能跑完整个推演的关键。"""

    def slower_scenario(self):
        scenario = self.service.create_scenario("工时翻倍")
        self.service.add_patches(scenario, [
            {"op": "update", "entity": "operation", "id": "OP-11", "values": {"processing_time": 40.0}},
        ])
        return self.service.require_scenario(scenario.scenario_id)

    def test_run_includes_baseline_variant(self):
        run = self.service.submit_run(self.slower_scenario(), ["ATC"], include_baseline=True)
        self.service.wait_for(run.run_id, 60)
        run = self.service.get_run(run.run_id)
        self.assertEqual(run.status, "done", run.error)
        variants = {item["variant"] for item in run.results}
        self.assertEqual(variants, {"scenario", "baseline"})

    def test_baseline_matches_a_plain_unpatched_run(self):
        run = self.service.submit_run(self.slower_scenario(), ["ATC"], include_baseline=True)
        self.service.wait_for(run.run_id, 60)
        baseline = next(i for i in self.service.get_run(run.run_id).results if i["variant"] == "baseline")

        plain = self.run_scenario(self.service.create_scenario("现状"), ["ATC"])
        self.assertEqual(baseline["metrics"]["makespan"], plain.results[0]["metrics"]["makespan"])
        self.assertEqual(
            baseline["metrics"]["total_tardiness"], plain.results[0]["metrics"]["total_tardiness"])

    def test_compare_anchors_on_the_baseline_variant(self):
        """基线必须当锚点：否则「变好/变坏」会以改动后的场景为准，方向全反。"""
        run = self.service.submit_run(self.slower_scenario(), ["ATC"], include_baseline=True)
        self.service.wait_for(run.run_id, 60)
        payload = self.service.compare_runs([run.run_id], ["makespan"])

        self.assertEqual(payload["baseline"]["variant"], "baseline")
        changed = next(e for e in payload["entries"] if e["variant"] == "scenario")
        self.assertGreater(changed["deltas"]["makespan"]["abs"], 0)
        self.assertFalse(changed["deltas"]["makespan"]["better"])

    def test_baseline_skipped_when_scenario_has_no_changes(self):
        run = self.service.submit_run(self.service.create_scenario("现状"), ["ATC"], include_baseline=True)
        self.assertFalse(run.include_baseline)
        self.service.wait_for(run.run_id, 60)
        self.assertEqual(len(self.service.get_run(run.run_id).results), 1)

    def test_baseline_does_not_evict_the_scenario_cache(self):
        """基线不该走单槽缓存——否则每轮带基线的推演都把场景那份顶掉，下轮又要重建。"""
        scenario = self.slower_scenario()
        shop_before, _ = self.service._materialize(scenario)
        run = self.service.submit_run(scenario, ["ATC"], include_baseline=True)
        self.service.wait_for(run.run_id, 60)
        self.assertIs(self.service._materialize_shop(scenario), shop_before)


class CompareTests(WhatIfServiceTestCase):
    def test_compare_across_scenarios(self):
        baseline = self.run_scenario(self.service.create_scenario("现状"), ["ATC"])
        scenario = self.service.create_scenario("工时翻倍")
        self.service.add_patches(scenario, [
            {"op": "update", "entity": "operation", "id": "OP-11", "values": {"processing_time": 40.0}},
        ])
        slower = self.run_scenario(self.service.require_scenario(scenario.scenario_id), ["ATC"])

        payload = self.service.compare_runs([baseline.run_id, slower.run_id], ["makespan"])
        self.assertEqual(len(payload["entries"]), 2)
        self.assertEqual(payload["baseline"]["run_id"], baseline.run_id)
        # makespan 是 min 方向，变大即变差
        self.assertFalse(payload["entries"][1]["deltas"]["makespan"]["better"])
        self.assertGreater(payload["entries"][1]["deltas"]["makespan"]["abs"], 0)

    def test_compare_rejects_unfinished_run(self):
        # 先等它真跑完再改状态：否则后台线程会在 TemporaryDirectory 清理后继续写库。
        run = self.run_scenario(self.service.create_scenario("未完成"), ["ATC"])
        run.status = "running"
        with self.assertRaises(WhatIfError) as ctx:
            self.service.compare_runs([run.run_id])
        self.assertEqual(ctx.exception.code, "WHATIF_RUN_NOT_READY")

    def test_compare_rejects_unknown_metrics(self):
        run = self.run_scenario(self.service.create_scenario("指标"), ["ATC"])
        with self.assertRaises(WhatIfError) as ctx:
            self.service.compare_runs([run.run_id], ["not_a_metric"])
        self.assertEqual(ctx.exception.code, "INVALID_ARGUMENT")


class DescribeTests(WhatIfServiceTestCase):
    def test_describe_reports_scale_delta_and_token(self):
        scenario = self.service.create_scenario("描述")
        self.service.add_patches(scenario, [
            {"op": "add", "entity": "machine", "values": {
                "machine_id": "M-C3", "machine_name": "Cutter 3", "type_id": "cut",
                "shifts": "0/0/24"}},
        ])
        payload = self.service.describe(self.service.require_scenario(scenario.scenario_id))
        self.assertEqual(payload["scale_delta"], {"machines": 1})
        self.assertEqual(payload["base_scale"]["machines"], 3)
        self.assertEqual(payload["scale"]["machines"], 4)
        self.assertTrue(payload["apply_token"])
        self.assertFalse(payload["validation"]["errors"])

    def test_revert_drops_last_patch(self):
        scenario = self.service.create_scenario("撤销")
        self.service.add_patches(scenario, [
            {"op": "update", "entity": "operation", "id": "OP-11", "values": {"processing_time": 9.0}},
        ])
        payload = self.service.revert_patches(self.service.require_scenario(scenario.scenario_id), 1)
        self.assertEqual(payload["patch_count"], 0)


class ApplyToInstanceTests(WhatIfServiceTestCase):
    def test_apply_requires_matching_token(self):
        scenario = self.service.create_scenario("落库")
        self.service.add_patches(scenario, [
            {"op": "add", "entity": "machine", "values": {
                "machine_id": "M-C3", "machine_name": "Cutter 3", "type_id": "cut",
                "shifts": "0/0/24"}},
        ])
        scenario = self.service.require_scenario(scenario.scenario_id)
        with self.assertRaises(WhatIfError) as ctx:
            self.service.apply_to_instance(scenario, "wrong-token")
        self.assertEqual(ctx.exception.code, "WHATIF_CONFIRM_REQUIRED")

    def test_apply_writes_instance_and_bumps_version(self):
        scenario = self.service.create_scenario("落库")
        self.service.add_patches(scenario, [
            {"op": "add", "entity": "machine", "values": {
                "machine_id": "M-C3", "machine_name": "Cutter 3", "type_id": "cut",
                "shifts": "0/0/24"}},
            {"op": "add", "entity": "downtime", "values": {
                "machine_id": "M-C1", "downtime_type": "maintenance",
                "start_time": 5.0, "end_time": 9.0}},
        ])
        scenario = self.service.require_scenario(scenario.scenario_id)
        token = self.service.describe(scenario)["apply_token"]
        version_before = get_instance_version(self.db_path)

        result = self.service.apply_to_instance(scenario, token)

        self.assertTrue(result["applied"])
        self.assertGreater(get_instance_version(self.db_path), version_before)
        self.assertTrue(Path(result["backup_path"]).exists())
        reloaded = self.store.build_shopfloor()
        self.assertIn("M-C3", reloaded.machines)
        self.assertEqual(len(reloaded.machines["M-C1"].downtimes), 1)

    def test_apply_preserves_untouched_data(self):
        """落库走的是 clear_all + 全量重写，必须证明没被改的实体逐字段还原。"""
        scenario = self.service.create_scenario("落库")
        self.service.add_patches(scenario, [
            {"op": "update", "entity": "operation", "id": "OP-11", "values": {"processing_time": 7.5}},
        ])
        scenario = self.service.require_scenario(scenario.scenario_id)
        expected, _ = self.service._materialize(scenario)
        expected_signature = shop_signature(expected)

        self.service.apply_to_instance(scenario, self.service.describe(scenario)["apply_token"])
        self.assertEqual(shop_signature(self.store.build_shopfloor()), expected_signature)

    def test_apply_rejects_scenario_without_changes(self):
        scenario = self.service.create_scenario("空")
        token = self.service.describe(scenario)["apply_token"]
        with self.assertRaises(WhatIfError) as ctx:
            self.service.apply_to_instance(scenario, token)
        self.assertEqual(ctx.exception.code, "INVALID_ARGUMENT")


class RowsAdapterTests(WhatIfServiceTestCase):
    def test_initial_state_rows_are_split_out(self):
        rows = self.service.load_base_rows()
        rows["operations"][0]["initial_status"] = "COMPLETED"
        rows["operations"][0]["initial_end_time"] = 3.0
        args = rows_to_save_args(rows)
        self.assertEqual(len(args["initial_state_rows"]), 1)
        self.assertEqual(args["initial_state_rows"][0]["op_id"], rows["operations"][0]["op_id"])
        self.assertEqual(args["initial_state_rows"][0]["initial_status"], "COMPLETED")

    def test_no_initial_state_yields_empty_rows(self):
        args = rows_to_save_args(self.service.load_base_rows())
        self.assertEqual(args["initial_state_rows"], [])
        self.assertIsNotNone(args["plan_start_at"])


if __name__ == "__main__":
    unittest.main()
