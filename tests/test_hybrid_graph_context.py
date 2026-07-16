import unittest
from unittest.mock import patch

from llm4drd.knowledge.canonical import CanonicalGraphBuilder
from llm4drd.knowledge.context import ComputeGraphProjection
from llm4drd.optimization.hybrid_nsga3_alns import (
    HybridConfig,
    HybridNSGA3ALNSOptimizer,
)
from llm4drd.tests.shop_fixtures import (
    hybrid_result_signature,
    make_graph_context_shop,
)


class HybridGraphContextTests(unittest.TestCase):
    def setUp(self):
        self.config = HybridConfig(
            objective_keys=["total_tardiness", "makespan"],
            target_solution_count=2,
            population_size=4,
            generations=1,
            alns_iterations_per_candidate=0,
            time_limit_s=60,
            parallel_workers=1,
            seed=17,
        )

    def context(self, shop):
        canonical = CanonicalGraphBuilder().build(shop)
        return ComputeGraphProjection().build(shop, canonical)

    def test_active_matches_legacy_without_constructing_networkx(self):
        legacy_shop = make_graph_context_shop()
        active_shop = make_graph_context_shop()
        legacy = HybridNSGA3ALNSOptimizer(
            legacy_shop, self.config, graph_context_mode="legacy"
        ).run()

        with patch(
            "llm4drd.optimization.hybrid_nsga3_alns.HeterogeneousGraph",
            side_effect=AssertionError("active mode constructed NetworkX"),
        ):
            active = HybridNSGA3ALNSOptimizer(
                active_shop,
                self.config,
                self.context(active_shop),
                "active",
            ).run()

        self.assertEqual(
            hybrid_result_signature(active), hybrid_result_signature(legacy)
        )

    def test_shadow_uses_legacy_solver_and_reports_zero_diff(self):
        shop = make_graph_context_shop()
        optimizer = HybridNSGA3ALNSOptimizer(
            shop, self.config, self.context(shop), "shadow"
        )
        self.assertEqual(optimizer.graph_context_diff.total_differences, 0)
        self.assertEqual(
            hybrid_result_signature(optimizer.run()),
            hybrid_result_signature(
                HybridNSGA3ALNSOptimizer(
                    make_graph_context_shop(), self.config
                ).run()
            ),
        )

    def test_active_and_shadow_require_context(self):
        for mode in ("active", "shadow"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "GraphContext is required"):
                    HybridNSGA3ALNSOptimizer(
                        make_graph_context_shop(), self.config, None, mode
                    )

    def test_active_reuses_current_shop_indexes(self):
        shop = make_graph_context_shop()
        context = self.context(shop)

        with patch.object(shop, "build_indexes", wraps=shop.build_indexes) as build:
            HybridNSGA3ALNSOptimizer(shop, self.config, context, "active")

        build.assert_not_called()

    def test_active_rebuilds_stale_shop_indexes(self):
        shop = make_graph_context_shop()
        context = self.context(shop)
        shop._ops_by_task.clear()

        with patch.object(shop, "build_indexes", wraps=shop.build_indexes) as build:
            HybridNSGA3ALNSOptimizer(shop, self.config, context, "active")

        build.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
