import pickle
import unittest
from concurrent.futures.process import BrokenProcessPool

from llm4drd.optimization import parallel_eval
from llm4drd.optimization.hybrid_nsga3_alns import HybridConfig, HybridNSGA3ALNSOptimizer
from llm4drd.tests.shop_fixtures import make_graph_context_shop


def _small_config(**overrides) -> HybridConfig:
    params = dict(
        objective_keys=["total_tardiness", "makespan"],
        target_solution_count=2, population_size=4, generations=1,
        alns_iterations_per_candidate=0, time_limit_s=60,
        parallel_workers=1, seed=17,
    )
    params.update(overrides)
    return HybridConfig(**params)


class ParallelEvalTests(unittest.TestCase):
    def test_worker_matches_inline_simulation(self):
        # 在当前进程直接调用 worker 的初始化与任务函数（不真正 spawn），
        # 验证 worker 路径与主进程 _simulate_candidate 产出一致。
        shop = make_graph_context_shop()
        optimizer = HybridNSGA3ALNSOptimizer(shop, _small_config())
        candidate = optimizer._default_candidate(optimizer.config.baseline_rule_name)

        parallel_eval.init_worker(optimizer._worker_payload_bytes())
        worker_sim = parallel_eval.run_exact_simulation(candidate)
        worker_solution = optimizer._solution_from_sim_result(
            candidate, worker_sim, "test", 0,
        )
        inline_solution = optimizer._simulate_candidate(candidate, "test", 0)

        def comparable(solution):
            metrics = dict(solution.metrics)
            metrics["wall_time_ms"] = 0.0
            return metrics, solution.schedule_signature

        self.assertEqual(comparable(worker_solution), comparable(inline_solution))

    def test_worker_payload_is_picklable(self):
        shop = make_graph_context_shop()
        optimizer = HybridNSGA3ALNSOptimizer(shop, _small_config())
        payload = pickle.loads(optimizer._worker_payload_bytes())
        self.assertIn("shop", payload)
        self.assertIn("graph_features", payload)
        self.assertEqual(
            sorted(payload["scales"]),
            ["busy_scale", "due_scale", "priority_scale", "time_scale"],
        )

    def test_process_backend_runs_end_to_end(self):
        shop = make_graph_context_shop()
        config = _small_config(parallel_workers=2, parallel_backend="process")
        optimizer = HybridNSGA3ALNSOptimizer(shop, config)
        result = optimizer.run()
        self.assertGreaterEqual(result.found_solution_count, 1)
        for solution in result.solutions:
            self.assertTrue(solution["metrics"]["feasible"])
        # 回退到线程池也会让上面的断言通过——必须显式确认进程后端没被放弃，
        # 否则 spawn 一旦坏掉这个测试就成了摆设。
        self.assertFalse(
            optimizer._process_backend_failed,
            "process backend silently fell back to threads",
        )

    def test_process_backend_falls_back_when_pool_breaks(self):
        shop = make_graph_context_shop()
        config = _small_config(parallel_workers=2, parallel_backend="process")
        optimizer = HybridNSGA3ALNSOptimizer(shop, config)

        class _BrokenExecutor:
            def submit(self, *args, **kwargs):
                raise BrokenProcessPool("injected failure")

        optimizer._ensure_process_pool = lambda worker_count: (
            _BrokenExecutor() if worker_count > 1 else None
        )
        result = optimizer.run()
        # 进程池全程抛错，但结果必须完整——回退路径要把候选补齐。
        self.assertGreaterEqual(result.found_solution_count, 1)
        self.assertTrue(optimizer._process_backend_failed)

    def test_process_and_thread_backends_agree(self):
        left = HybridNSGA3ALNSOptimizer(
            make_graph_context_shop(),
            _small_config(parallel_workers=2, parallel_backend="thread"),
        ).run()
        right = HybridNSGA3ALNSOptimizer(
            make_graph_context_shop(),
            _small_config(parallel_workers=2, parallel_backend="process"),
        ).run()
        self.assertEqual(
            sorted(s["metrics"]["makespan"] for s in left.solutions),
            sorted(s["metrics"]["makespan"] for s in right.solutions),
        )


if __name__ == "__main__":
    unittest.main()
