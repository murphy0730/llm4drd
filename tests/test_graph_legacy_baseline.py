import unittest

from llm4drd.knowledge.graph import HeterogeneousGraph
from llm4drd.optimization.hybrid_nsga3_alns import HybridConfig, HybridNSGA3ALNSOptimizer
from llm4drd.tests.shop_fixtures import (
    canonical_graph_signature, hybrid_result_signature, make_graph_context_shop,
)


class LegacyGraphBaselineTests(unittest.TestCase):
    def test_hybrid_signature_recursively_removes_only_wall_time(self):
        payload = {
            "baseline": {
                "metrics": {
                    "wall_time_ms": 0.4,
                    "makespan": 9.0,
                    "diagnostics": {"wall_time_ms": 0.2, "event_count": 13},
                },
            },
            "solutions": [{
                "metrics": {
                    "wall_time_ms": 0.3,
                    "total_evaluations": 7,
                    "phases": [{"wall_time_ms": 0.1, "name": "exact"}],
                },
            }],
            "archive_size": 1,
            "found_solution_count": 1,
            "generations_completed": 1,
            "total_evaluations": 7,
            "approximate_evaluations": 5,
            "exact_evaluations": 2,
        }

        class Result:
            def to_export_dict(self):
                return payload

        signature = hybrid_result_signature(Result())

        self.assertEqual(
            signature["baseline"]["metrics"],
            {"makespan": 9.0, "diagnostics": {"event_count": 13}},
        )
        self.assertEqual(
            signature["solutions"][0]["metrics"],
            {"total_evaluations": 7, "phases": [{"name": "exact"}]},
        )
        self.assertEqual(payload["baseline"]["metrics"]["wall_time_ms"], 0.4)

    def test_legacy_graph_is_deterministic(self):
        left = HeterogeneousGraph(); left.build_from_shopfloor(make_graph_context_shop())
        right = HeterogeneousGraph(); right.build_from_shopfloor(make_graph_context_shop())
        self.assertEqual(canonical_graph_signature(left.graph), canonical_graph_signature(right.graph))

    def test_legacy_hybrid_is_deterministic(self):
        config = HybridConfig(
            objective_keys=["total_tardiness", "makespan"],
            target_solution_count=2, population_size=4, generations=1,
            alns_iterations_per_candidate=0, time_limit_s=60,
            parallel_workers=1, seed=17,
        )
        left = HybridNSGA3ALNSOptimizer(make_graph_context_shop(), config).run()
        right = HybridNSGA3ALNSOptimizer(make_graph_context_shop(), config).run()
        self.assertEqual(hybrid_result_signature(left), hybrid_result_signature(right))


if __name__ == "__main__":
    unittest.main()
