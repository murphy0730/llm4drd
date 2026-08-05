from __future__ import annotations

import unittest

from llm4drd.api.planning_query_service import (
    PlanningQueryError,
    PlanningQueryService,
)
from llm4drd.tests.shop_fixtures import make_graph_context_shop


def _entry(
    op_id: str,
    *,
    start: float,
    end: float,
    machine_id: str = "M-C1",
) -> dict:
    return {
        "op_id": op_id,
        "machine_id": machine_id,
        "start": start,
        "end": end,
        "tooling_ids": [],
        "personnel_ids": [],
    }


def _solution(solution_id: str, entries: list[dict], *, tardiness: float, makespan: float) -> dict:
    return {
        "solution_id": solution_id,
        "source": "hybrid",
        "feasible": True,
        "objectives": {
            "total_tardiness": tardiness,
            "makespan": makespan,
        },
        "metrics": {
            "total_tardiness": tardiness,
            "main_order_tardy_total_time": tardiness / 2,
            "makespan": makespan,
        },
        "schedule": entries,
    }


class PlanningQueryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shop = make_graph_context_shop()
        first_full = [
            _entry("OP-11", start=2.0, end=6.0),
            _entry("OP-12", start=7.0, end=9.0),
            _entry("OP-13", start=42.0, end=45.0, machine_id="M-A1"),
            _entry("OP-21", start=1.0, end=6.0, machine_id="M-C2"),
        ]
        second_full = [
            _entry("OP-11", start=1.0, end=5.0),
            _entry("OP-12", start=5.0, end=7.0),
            _entry("OP-21", start=2.0, end=7.0, machine_id="M-C2"),
        ]
        first = _solution("S-1", first_full, tardiness=12.5, makespan=45.0)
        second = _solution("S-2", second_full, tardiness=3.0, makespan=41.0)
        self.task = {
            "status": "done",
            "result": {
                "archive_size": 7,
                "solutions": [
                    {**first, "schedule": first_full[:1], "schedule_truncated": True},
                    {**second, "schedule": second_full[:1], "schedule_truncated": True},
                ],
                "baseline": {"solution_id": "BASE", "schedule": []},
            },
            "export_result": {
                "archive_size": 7,
                "solutions": [first, second],
                "baseline": {"solution_id": "BASE", "schedule": []},
                "reference_solutions": [
                    {"solution_id": "REF-1", "schedule": [], "metrics": {}}
                ],
            },
        }
        self.service = PlanningQueryService(
            shop=self.shop,
            tasks={"task-1": self.task},
            latest_task_id="task-1",
            instance_version=4,
        )

    def test_overview_keeps_candidate_and_archive_counts_distinct(self) -> None:
        result = self.service.get_overview()

        self.assertEqual(result["task_id"], "task-1")
        self.assertEqual(result["instance_version"], 4)
        self.assertEqual(result["solution_count"], 4)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["archive_size"], 7)
        self.assertEqual(result["baseline_count"], 1)
        self.assertEqual(result["reference_count"], 1)
        self.assertEqual(result["solutions"][1]["solution_name"], "方案二")
        self.assertEqual(result["solutions"][1]["total_tardiness_hours"], 12.5)
        self.assertEqual(result["solutions"][1]["main_order_tardiness_hours"], 6.25)

    def test_overview_lists_baseline_and_reference_next_to_candidates(self) -> None:
        result = self.service.get_overview()

        self.assertEqual(
            [(item["solution_id"], item["solution_name"], item["category"])
             for item in result["solutions"]],
            [
                ("BASE", "方案一", "baseline"),
                ("S-1", "方案二", "pareto"),
                ("S-2", "方案三", "pareto"),
                ("REF-1", "方案四", "heuristic"),
            ],
        )
        self.assertEqual(result["solutions"][0]["category_label"], "基线方案")

    def test_baseline_feasibility_falls_back_to_metrics(self) -> None:
        self.task["export_result"]["baseline"]["metrics"] = {"feasible": True}

        result = self.service.get_overview()

        self.assertTrue(result["solutions"][0]["feasible"])

    def test_compare_without_ids_starts_from_the_baseline(self) -> None:
        result = self.service.compare_solutions()

        self.assertEqual(
            [item["solution_name"] for item in result["solutions"]],
            ["方案一", "方案二", "方案三", "方案四"],
        )
        self.assertFalse(result["truncated"])

    def test_solution_names_match_review_candidate_order(self) -> None:
        result = self.service.compare_solutions(
            solution_ids=["BASE", "S-1", "S-2", "REF-1"]
        )

        self.assertEqual(
            [item["solution_name"] for item in result["solutions"]],
            ["方案一", "方案二", "方案三", "方案四"],
        )

    def test_order_planning_uses_full_export_and_computes_order_tardiness(self) -> None:
        result = self.service.get_order_planning("O-1", solution_ids=["S-1"])

        self.assertEqual(result["order"]["order_id"], "O-1")
        planning = result["solutions"][0]
        self.assertEqual(planning["solution_name"], "方案二")
        self.assertEqual(planning["operation_count"], 3)
        self.assertEqual(planning["order_completion_hours"], 45.0)
        self.assertEqual(planning["order_tardiness_hours"], 5.0)
        self.assertEqual(planning["total_tardiness_hours"], 12.5)
        self.assertEqual(
            [entry["operation_id"] for entry in planning["operations"]],
            ["OP-11", "OP-12", "OP-13"],
        )
        self.assertTrue(all(entry["start_at"] for entry in planning["operations"]))

    def test_operation_planning_returns_one_placement_per_solution(self) -> None:
        result = self.service.get_operation_planning(
            "OP-13", solution_ids=["S-1", "S-2"]
        )

        self.assertEqual(result["operation"]["order_id"], "O-1")
        self.assertEqual(result["placements"][0]["planned"], True)
        self.assertEqual(result["placements"][0]["solution_name"], "方案二")
        self.assertEqual(result["placements"][0]["machine_id"], "M-A1")
        self.assertEqual(result["placements"][1], {
            "solution_id": "S-2",
            "solution_name": "方案三",
            "planned": False,
            "reason": "operation_not_present_in_solution",
        })

    def test_searches_orders_and_operations_by_business_fields(self) -> None:
        orders = self.service.search_orders("order 1")
        operations = self.service.search_operations("assemble")

        self.assertEqual([item["order_id"] for item in orders], ["O-1"])
        self.assertEqual([item["operation_id"] for item in operations], ["OP-13"])
        self.assertEqual(operations[0]["order_id"], "O-1")

    def test_not_ready_task_raises_stable_domain_error(self) -> None:
        service = PlanningQueryService(
            shop=self.shop,
            tasks={"running": {"status": "running", "result": None}},
            latest_task_id="running",
            instance_version=4,
        )

        with self.assertRaises(PlanningQueryError) as raised:
            service.get_overview()

        self.assertEqual(raised.exception.code, "PLANNING_TASK_NOT_READY")

    def test_unknown_solution_raises_stable_domain_error(self) -> None:
        with self.assertRaises(PlanningQueryError) as raised:
            self.service.compare_solutions(solution_ids=["missing"])

        self.assertEqual(raised.exception.code, "SOLUTION_NOT_FOUND")

    def test_task_from_an_old_instance_is_rejected(self) -> None:
        stale_task = {**self.task, "instance_version": 3}
        service = PlanningQueryService(
            shop=self.shop,
            tasks={"stale": stale_task},
            latest_task_id="stale",
            instance_version=4,
        )

        with self.assertRaises(PlanningQueryError) as raised:
            service.get_overview()

        self.assertEqual(raised.exception.code, "PLANNING_SNAPSHOT_STALE")

    def test_non_finite_operation_times_are_returned_as_null(self) -> None:
        broken = _solution(
            "S-BROKEN",
            [
                _entry("OP-11", start=float("inf"), end=float("inf")),
                _entry("OP-12", start=4.0, end=6.0),
            ],
            tardiness=float("inf"),
            makespan=float("inf"),
        )
        task = {
            "status": "done",
            "result": {"archive_size": 1, "solutions": [broken]},
            "export_result": {"archive_size": 1, "solutions": [broken]},
        }
        service = PlanningQueryService(
            shop=self.shop,
            tasks={"broken": task},
            latest_task_id="broken",
            instance_version=4,
        )

        result = service.get_order_planning("O-1")

        planning = result["solutions"][0]
        self.assertEqual(planning["order_completion_hours"], 6.0)
        self.assertEqual(planning["order_tardiness_hours"], 0.0)
        invalid = next(
            item for item in planning["operations"] if item["operation_id"] == "OP-11"
        )
        self.assertIsNone(invalid["start_hours"])
        self.assertIsNone(planning["total_tardiness_hours"])


class BottleneckMachineTests(unittest.TestCase):
    """compare_solutions 里的瓶颈机器排名。"""

    def setUp(self) -> None:
        self.shop = make_graph_context_shop()
        self.solution = {
            "solution_id": "S-1",
            "source": "hybrid",
            "feasible": True,
            "objectives": {"total_tardiness": 3.0, "makespan": 40.0},
            "metrics": {},
            "schedule": [],
            "summary": {
                "machine_utilization": {"M-A1": 0.42, "M-C1": 0.91, "M-C2": 0.77},
            },
        }
        self.task = {
            "status": "done",
            "result": {"solutions": [self.solution]},
            "export_result": {"solutions": [self.solution]},
        }

    def _service(self) -> PlanningQueryService:
        return PlanningQueryService(
            shop=self.shop,
            tasks={"task-1": self.task},
            latest_task_id="task-1",
            instance_version=4,
        )

    def test_ranks_machines_by_utilization_with_names_and_critical_flag(self) -> None:
        result = self._service().compare_solutions()

        machines = result["solutions"][0]["machine_utilization_ranking"]
        self.assertEqual(
            [(item["machine_id"], item["full_horizon_utilization"]) for item in machines],
            [("M-C1", 0.91), ("M-C2", 0.77), ("M-A1", 0.42)],
        )
        self.assertEqual(machines[0]["machine_name"], "Cutter 1")
        self.assertTrue(machines[0]["is_critical"])
        self.assertFalse(machines[2]["is_critical"])
        self.assertEqual(result["ranking_source"], "utilization_ranking")

    def test_defaults_to_twenty_machines(self) -> None:
        result = self._service().compare_solutions()

        # 车间只有 3 台机器，默认 20 条时应当全量返回而不是报错或补空位。
        self.assertEqual(result["machine_limit"], 20)
        self.assertEqual(len(result["solutions"][0]["machine_utilization_ranking"]), 3)

    def test_honours_an_explicit_limit(self) -> None:
        result = self._service().compare_solutions(machine_limit=2)

        self.assertEqual(result["machine_limit"], 2)
        self.assertEqual(
            [item["machine_id"] for item in result["solutions"][0]["machine_utilization_ranking"]],
            ["M-C1", "M-C2"],
        )

    def test_clamps_an_out_of_range_limit(self) -> None:
        self.assertEqual(self._service().compare_solutions(machine_limit=0)["machine_limit"], 1)
        self.assertEqual(self._service().compare_solutions(machine_limit=999)["machine_limit"], 50)

    def test_falls_back_to_the_legacy_top5_ids_on_old_payloads(self) -> None:
        # 改这次功能之前跑的任务：payload 里只有 id，没有逐机利用率。
        self.solution["summary"] = {"bottleneck_machine_ids": ["M-C1", "M-A1"]}

        result = self._service().compare_solutions()

        machines = result["solutions"][0]["machine_utilization_ranking"]
        self.assertEqual(result["ranking_source"], "legacy_top5")
        self.assertEqual([item["machine_id"] for item in machines], ["M-C1", "M-A1"])
        self.assertEqual(machines[0]["machine_name"], "Cutter 1")
        self.assertIsNone(machines[0]["full_horizon_utilization"])

    def test_survives_a_payload_without_any_bottleneck_data(self) -> None:
        self.solution["summary"] = {}

        result = self._service().compare_solutions()

        self.assertEqual(result["solutions"][0]["machine_utilization_ranking"], [])

    def test_keeps_unknown_machine_ids_instead_of_dropping_them(self) -> None:
        # 实例改过之后，历史方案里可能留着已被删掉的机器；宁可给个空名字也别静默丢行。
        self.solution["summary"]["machine_utilization"]["M-GONE"] = 0.99

        machines = self._service().compare_solutions()["solutions"][0]["machine_utilization_ranking"]

        self.assertEqual(machines[0]["machine_id"], "M-GONE")
        self.assertEqual(machines[0]["machine_name"], "")
        self.assertFalse(machines[0]["is_critical"])


if __name__ == "__main__":
    unittest.main()
