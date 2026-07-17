import os
import pickle
import unittest
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool

from llm4drd.core.models import OpStatus
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
    def _init_worker(self, optimizer) -> None:
        """在本进程里初始化 worker；payload 文件由测试自己回收。

        正常路径下文件随 _shutdown_process_pool 删除，直接调 init_worker 的测试
        绕开了池的生命周期，不清理就会在临时目录里堆下大实例 payload。
        """
        path = optimizer._write_worker_payload()
        self.addCleanup(os.unlink, path)
        parallel_eval.init_worker(path)

    def test_worker_matches_inline_simulation(self):
        # 在当前进程直接调用 worker 的初始化与任务函数（不真正 spawn），
        # 验证 worker 路径与主进程 _simulate_candidate 产出一致。
        shop = make_graph_context_shop()
        optimizer = HybridNSGA3ALNSOptimizer(shop, _small_config())
        candidate = optimizer._default_candidate(optimizer.config.baseline_rule_name)

        self._init_worker(optimizer)
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

    def test_worker_keeps_a_single_shop_copy(self):
        # 每份 shop 在生产实例上达数百 MB：worker 里 runtime / approx / _STATE
        # 必须是同一个对象，多一份深拷贝就会把并行 worker 推进 MemoryError。
        optimizer = HybridNSGA3ALNSOptimizer(make_graph_context_shop(), _small_config())
        self._init_worker(optimizer)
        runtime_shop = parallel_eval._STATE["runtime"].shop
        self.assertIs(parallel_eval._STATE["shop"], runtime_shop)
        self.assertIs(parallel_eval._STATE["approx"].shop, runtime_shop)

    def test_approx_never_sees_the_exact_runs_terminal_state(self):
        # 共用 shop 的代价：精确仿真结束后 shop 停在"全部完成"的终态，而
        # approx 的 build_schedule_analytics 会读 op.status/end_time。不复位就会
        # 把候选算成 feasible 并混入上一轮的完工时间——数值上未必立刻看得出，
        # 所以这里直接钉住"approx 评估时看到的必须是初始状态"。
        optimizer = HybridNSGA3ALNSOptimizer(make_graph_context_shop(), _small_config())
        candidate = optimizer._default_candidate(optimizer.config.baseline_rule_name)
        self._init_worker(optimizer)
        before = parallel_eval.run_approx_evaluation(candidate, "test", 0)

        parallel_eval.run_exact_simulation(candidate)

        approx = parallel_eval._STATE["approx"]
        seen: dict = {}
        original_evaluate = approx.evaluate

        def spy(*args, **kwargs):
            seen["statuses"] = {op.status for op in approx.shop.operations.values()}
            seen["ended"] = [op.end_time for op in approx.shop.operations.values()]
            return original_evaluate(*args, **kwargs)

        approx.evaluate = spy
        after = parallel_eval.run_approx_evaluation(candidate, "test", 0)

        self.assertEqual(seen["statuses"], {OpStatus.PENDING})
        self.assertEqual(seen["ended"], [None] * len(approx.shop.operations))
        self.assertEqual(before.metrics, after.metrics)

    def test_pool_capacity_is_not_pinned_by_the_first_batch(self):
        # 近似阶段先以小 worker 数建池，精确阶段再要求更大并发时，
        # 不能被首批的容量永久卡住。
        optimizer = HybridNSGA3ALNSOptimizer(
            make_graph_context_shop(),
            _small_config(parallel_workers=8, parallel_backend="process"),
        )
        try:
            small = optimizer._ensure_process_pool(2)
            self.assertIsNotNone(small)
            large = optimizer._ensure_process_pool(8)
            self.assertIsNotNone(large)
            self.assertGreaterEqual(large._max_workers, 8)
        finally:
            optimizer._shutdown_process_pool()

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

    def test_process_backend_falls_back_on_startup_failures(self):
        # 进程通常在首次 submit() 才真正启动，届时抛的是 RuntimeError/OSError
        # 而非 BrokenProcessPool——例如 daemon 进程内不允许再起子进程（本项目
        # 跑在 FastAPI worker 里，正是该场景）。这些必须同样回退，而不是炸穿。
        for exc in (
            RuntimeError("cannot start new process (daemon)"),
            OSError("[Errno 24] Too many open files"),
            AssertionError("daemonic processes are not allowed to have children"),
        ):
            with self.subTest(exception=type(exc).__name__):
                optimizer = HybridNSGA3ALNSOptimizer(
                    make_graph_context_shop(),
                    _small_config(parallel_workers=2, parallel_backend="process"),
                )

                class _FailingExecutor:
                    def submit(self, *args, _exc=exc, **kwargs):
                        raise _exc

                optimizer._ensure_process_pool = lambda worker_count: (
                    _FailingExecutor() if worker_count > 1 else None
                )
                result = optimizer.run()
                self.assertGreaterEqual(result.found_solution_count, 1)
                self.assertTrue(optimizer._process_backend_failed)

    def test_task_level_exceptions_still_propagate(self):
        # 回退只该吃"基础设施"异常；worker 内部的业务异常必须照常上抛，
        # 否则真实 bug 会被静默降级成"慢一点但结果照出"。
        optimizer = HybridNSGA3ALNSOptimizer(
            make_graph_context_shop(),
            _small_config(parallel_workers=2, parallel_backend="process"),
        )

        class _TaskFailingExecutor:
            def submit(self, *args, **kwargs):
                future = Future()
                future.set_exception(ValueError("bug inside worker task"))
                return future

        optimizer._ensure_process_pool = lambda worker_count: (
            _TaskFailingExecutor() if worker_count > 1 else None
        )
        with self.assertRaises(ValueError):
            optimizer.run()

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
