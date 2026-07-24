"""基准方案抽取与质量门禁。

抽取要覆盖极端解 + 均衡膝点 + 簇代表并按签名去重；门禁按 emphasis 分档：均衡解要
多数目标不劣于 ATC，极端解只要强调目标显著更优、其余不灾难退化即可。
"""
import unittest

from llm4drd.optimization.baseline_extract import (
    extract_baseline_solutions,
    passes_quality_gate,
)
from llm4drd.optimization.objectives import get_objective_specs
from llm4drd.optimization.solution_model import CandidateParameters, OptimizationSolution

OBJ_KEYS = ["main_order_tardy_ratio", "critical_active_window_utilization", "avg_flowtime"]
SPECS = get_objective_specs(OBJ_KEYS)  # min, max, min


def _sol(sid: str, tardy: float, util: float, flow: float, weight: float = 0.0) -> OptimizationSolution:
    candidate = CandidateParameters(
        feature_weights={"urgency": weight, "due_date": weight * 0.5},
        destroy_weights={},
        repair_weights={},
    )
    return OptimizationSolution(
        solution_id=sid,
        source="test",
        generation=0,
        candidate=candidate,
        objectives={
            "main_order_tardy_ratio": tardy,
            "critical_active_window_utilization": util,
            "avg_flowtime": flow,
        },
        metrics={},
        schedule=[],
        feasible=True,
        schedule_signature=sid,
    )


class ExtractTests(unittest.TestCase):
    def test_extracts_extremes_knee_and_dedups(self):
        # 三个各自极端 + 一个均衡；权重各异保证签名不同。
        solutions = [
            _sol("best_tardy", 0.05, 0.60, 400, weight=1.0),   # min tardy 最优
            _sol("best_util", 0.30, 0.95, 500, weight=2.0),    # max util 最优
            _sol("best_flow", 0.25, 0.55, 250, weight=3.0),    # min flow 最优
            _sol("balanced", 0.15, 0.80, 320, weight=4.0),     # 均衡
        ]
        extracted = extract_baseline_solutions(solutions, SPECS, cluster_target=8)
        emphases = {e for e, _ in extracted}
        self.assertIn("min_main_order_tardy_ratio", emphases)
        self.assertIn("max_critical_active_window_utilization", emphases)
        self.assertIn("min_avg_flowtime", emphases)
        self.assertIn("balanced", emphases)
        # 去重：同一 solution 不会以两个标签重复出现（按 candidate.signature 唯一）。
        sigs = [sol.candidate.signature() for _, sol in extracted]
        self.assertEqual(len(sigs), len(set(sigs)))

    def test_empty_and_infeasible(self):
        self.assertEqual(extract_baseline_solutions([], SPECS), [])
        infeasible = _sol("x", 0.1, 0.9, 300)
        infeasible.feasible = False
        self.assertEqual(extract_baseline_solutions([infeasible], SPECS), [])

    def test_extreme_solution_labeled_correctly(self):
        extracted = dict(
            (e, s) for e, s in extract_baseline_solutions(
                [
                    _sol("a", 0.05, 0.60, 400, weight=1.0),
                    _sol("b", 0.30, 0.95, 500, weight=2.0),
                    _sol("c", 0.25, 0.55, 250, weight=3.0),
                ],
                SPECS,
            )
        )
        self.assertEqual(extracted["min_main_order_tardy_ratio"].solution_id, "a")
        self.assertEqual(extracted["max_critical_active_window_utilization"].solution_id, "b")
        self.assertEqual(extracted["min_avg_flowtime"].solution_id, "c")


class QualityGateTests(unittest.TestCase):
    def setUp(self):
        # ATC 基线：延误 0.30 / 利用率 0.70 / 流程 500
        self.atc = _sol("atc", 0.30, 0.70, 500)

    def test_balanced_passes_when_dominates(self):
        cand = _sol("cand", 0.20, 0.80, 400)  # 三项都优于 ATC
        ok, evidence = passes_quality_gate(cand, self.atc, SPECS, "balanced")
        self.assertTrue(ok)
        self.assertEqual(evidence["verdict"]["tier"], "balanced")

    def test_balanced_rejected_when_mostly_worse(self):
        cand = _sol("cand", 0.40, 0.60, 550)  # 三项都劣于 ATC
        ok, _ = passes_quality_gate(cand, self.atc, SPECS, "cluster_0")
        self.assertFalse(ok)

    def test_extreme_passes_on_emphasized_gain_without_catastrophe(self):
        # 延误从 0.30 → 0.15（改进 50%），利用率/流程略差但未灾难退化。
        cand = _sol("cand", 0.15, 0.66, 560)
        ok, evidence = passes_quality_gate(cand, self.atc, SPECS, "min_main_order_tardy_ratio")
        self.assertTrue(ok)
        self.assertEqual(evidence["verdict"]["tier"], "extreme")

    def test_extreme_rejected_on_catastrophic_other_objective(self):
        # 延误大幅改进，但流程时间从 500 涨到 900（>500*1.5=750）灾难退化。
        cand = _sol("cand", 0.15, 0.70, 900)
        ok, _ = passes_quality_gate(cand, self.atc, SPECS, "min_main_order_tardy_ratio")
        self.assertFalse(ok)

    def test_extreme_rejected_when_emphasized_gain_too_small(self):
        # 延误只从 0.30 → 0.29（改进 <10% 阈值）。
        cand = _sol("cand", 0.29, 0.70, 500)
        ok, _ = passes_quality_gate(cand, self.atc, SPECS, "min_main_order_tardy_ratio")
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
