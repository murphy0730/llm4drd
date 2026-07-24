"""白天热启动：_seed_population 从基准方案库追加种子的护栏与零回归契约。

库空 → 与改前逐字节一致；库非空 → active 方案进种群且 population_size 上调防截断；
feature_names / scale 护栏不过的方案被跳过。
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm4drd.data.db import BaselineSolutionStore, init_db
from llm4drd.optimization.hybrid_nsga3_alns import HybridConfig, HybridNSGA3ALNSOptimizer
from llm4drd.optimization.solution_model import FEATURE_NAMES
from llm4drd.tests.shop_fixtures import make_graph_context_shop


def _seed_signatures(population) -> list[str]:
    return [candidate.signature() for candidate in population]


class WarmStartTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = str(Path(self.tmp.name) / "baseline.db")
        init_db(self.db_path)
        self.store = BaselineSolutionStore(self.db_path)

    def _config(self, **overrides) -> HybridConfig:
        base = dict(
            objective_keys=["total_tardiness", "makespan"],
            target_solution_count=2,
            population_size=24,
            generations=1,
            alns_iterations_per_candidate=0,
            time_limit_s=60,
            parallel_workers=1,
            seed=17,
            baseline_db_path=self.db_path,
        )
        base.update(overrides)
        return HybridConfig(**base)

    def _optimizer(self, **overrides) -> HybridNSGA3ALNSOptimizer:
        return HybridNSGA3ALNSOptimizer(
            make_graph_context_shop(), self._config(**overrides), graph_context_mode="legacy"
        )

    def _baseline_row(self, rid: str, scales: dict, feature_names=None) -> dict:
        return {
            "id": rid,
            "emphasis": "balanced",
            "objective_keys": ["main_order_tardy_ratio"],
            "feature_names": feature_names if feature_names is not None else list(FEATURE_NAMES),
            # 特意用一组显著的权重，保证投影后仍与内置规则种子签名不同。
            "feature_weights": {name: 0.7 for name in FEATURE_NAMES},
            "scale_json": scales,
            "op_bias": {},
            "destroy_weights": {},
            "repair_weights": {},
            "destroy_fraction": 0.25,
            "reached_objectives": {},
            "baseline_compare": {},
            "snapshot_id": "s",
            "snapshot_version": 1,
            "created_at": "2026-07-24T02:00:00",
        }

    def _current_scales(self, optimizer) -> dict:
        return {
            "time_scale": optimizer.time_scale,
            "due_scale": optimizer.due_scale,
            "priority_scale": optimizer.priority_scale,
        }

    def test_empty_library_is_zero_regression(self):
        # 库空(enabled=True) 的种群签名必须与显式关闭(enabled=False) 逐字节一致。
        disabled = self._optimizer(baseline_seeds_enabled=False)._seed_population()
        enabled_empty = self._optimizer(baseline_seeds_enabled=True)._seed_population()
        self.assertEqual(_seed_signatures(disabled), _seed_signatures(enabled_empty))

    def test_active_baseline_injected_into_population(self):
        probe = self._optimizer()
        self.store.save_batch("b1", [self._baseline_row("s1", self._current_scales(probe))])

        opt = self._optimizer()
        seeds = opt._load_baseline_seeds()
        self.assertEqual(len(seeds), 1)
        # 复现投影签名（_seed_population 内部对基线以 blend=0.22 投影到 balanced 图空间）。
        projected_sig = opt._project_candidate_to_graph_space(
            opt._load_baseline_seeds()[0], "balanced", blend=0.22
        ).signature()

        with_sigs = {c.signature() for c in self._optimizer()._seed_population()}
        without_sigs = {c.signature() for c in self._optimizer(baseline_seeds_enabled=False)._seed_population()}
        self.assertIn(projected_sig, with_sigs)
        self.assertNotIn(projected_sig, without_sigs)

    def test_many_baselines_bump_population_size_to_avoid_truncation(self):
        probe = self._optimizer()
        scales = self._current_scales(probe)
        rows = []
        for i in range(30):
            row = self._baseline_row(f"s{i}", scales)
            # 每条权重各异，保证 30 条基线签名互不相同、不被去重吞掉。
            row["feature_weights"] = {name: 0.3 + 0.02 * i for name in FEATURE_NAMES}
            rows.append(row)
        self.store.save_batch("b1", rows)

        opt = self._optimizer()
        self.assertEqual(len(opt._load_baseline_seeds()), 30)
        population = opt._seed_population()
        # 30 条基线 + 现有种子 > 24，population_size 被上调，全部种子都进种群未被截断。
        self.assertGreater(opt.config.population_size, 24)
        self.assertEqual(len(population), opt.config.population_size)

    def test_feature_names_mismatch_skipped(self):
        probe = self._optimizer()
        self.store.save_batch(
            "b1",
            [self._baseline_row("s1", self._current_scales(probe), feature_names=["urgency", "slack"])],
        )
        self.assertEqual(self._optimizer()._load_baseline_seeds(), [])

    def test_scale_out_of_tolerance_skipped(self):
        probe = self._optimizer()
        off = {k: v * 10 for k, v in self._current_scales(probe).items()}
        self.store.save_batch("b1", [self._baseline_row("s1", off)])
        self.assertEqual(self._optimizer()._load_baseline_seeds(), [])

    def test_full_run_with_active_library_completes(self):
        # 仅测 _seed_population 不够：投影后的基线候选还要流经评估/ALNS 全程。这里跑一次
        # 完整 run()，确认带库热启动不会在优化循环里崩溃、仍产出可行解。
        probe = self._optimizer()
        self.store.save_batch("b1", [self._baseline_row("s1", self._current_scales(probe))])
        optimizer = HybridNSGA3ALNSOptimizer(
            make_graph_context_shop(),
            self._config(generations=2, alns_iterations_per_candidate=2),
            graph_context_mode="legacy",
        )
        result = optimizer.run()
        self.assertGreaterEqual(result.found_solution_count, 1)
        self.assertGreaterEqual(len(optimizer.archive.solutions()), 1)


if __name__ == "__main__":
    unittest.main()
