import unittest

from llm4drd.knowledge.graph import HeterogeneousGraph
from llm4drd.optimization.hybrid_nsga3_alns import HybridConfig, HybridNSGA3ALNSOptimizer
from llm4drd.tests.shop_fixtures import (
    canonical_graph_signature, hybrid_result_signature, make_graph_context_shop,
)


class LegacyGraphBaselineTests(unittest.TestCase):
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
