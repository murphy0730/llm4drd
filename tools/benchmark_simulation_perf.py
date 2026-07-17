#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
仿真与优化器性能基准。

测两件事：单次仿真耗时、混合优化器端到端耗时。用生成器造一个有实际规模的
实例——tests 里的 make_graph_context_shop() 只有 4 道工序，跑一次 0.05 秒，
测不出任何差异。

与 tools/benchmark_graph_context.py 的区别：那个是图上下文 legacy-vs-active
的对比验收工具；这里只测评估吞吐（deepcopy 复用、并行后端、克隆开销）。

用法:
    python -m llm4drd.tools.benchmark_simulation_perf
    python -m llm4drd.tools.benchmark_simulation_perf --orders 30 --workers 4
"""
from __future__ import annotations

import argparse
import statistics
import time

from ..core.rules import BUILTIN_RULES
from ..core.simulator import Simulator
from ..data.generator import InstanceGenerator
from ..optimization.hybrid_nsga3_alns import HybridConfig, HybridNSGA3ALNSOptimizer


def build_shop(orders: int, seed: int):
    return InstanceGenerator(seed=seed).generate(
        num_orders=orders,
        tasks_per_order=(2, 5),
        ops_per_task=(2, 5),
        machines_per_type=3,
    )


def bench_simulation(shop, repeats: int):
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = Simulator(shop, BUILTIN_RULES["ATC"]).run()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), result


def bench_optimizer(shop, workers: int, seed: int, backend: str):
    config = HybridConfig(
        objective_keys=["total_tardiness", "makespan"],
        population_size=12, generations=4, alns_iterations_per_candidate=2,
        time_limit_s=120, parallel_workers=workers, seed=seed,
        parallel_backend=backend,
    )
    optimizer = HybridNSGA3ALNSOptimizer(shop, config)
    started = time.perf_counter()
    result = optimizer.run()
    return time.perf_counter() - started, optimizer, result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--orders", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sim-repeats", type=int, default=5)
    parser.add_argument("--backend", choices=("process", "thread"), default="process")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    shop = build_shop(args.orders, args.seed)
    print(f"instance: orders={len(shop.orders)} tasks={len(shop.tasks)} "
          f"ops={len(shop.operations)} machines={len(shop.machines)}")

    sim_median, sim_result = bench_simulation(shop, args.sim_repeats)
    print(f"single_simulation_median_s: {sim_median:.4f} "
          f"(feasible={sim_result.feasible} events={sim_result.event_count})")

    elapsed, optimizer, result = bench_optimizer(shop, args.workers, args.seed, args.backend)
    # 各阶段实际并发取决于工序规模（见 _phase_parallel_workers），打印出来避免
    # 凭想当然下结论。
    print(f"workers: approx={optimizer.approx_parallel_workers} "
          f"exact={optimizer.exact_parallel_workers} "
          f"refine={optimizer.refine_parallel_workers} "
          f"pool_capacity={optimizer._max_pool_workers}")
    print(f"optimizer_elapsed_s: {elapsed:.2f}")
    print(f"exact_evaluations: {optimizer.exact_evaluations} "
          f"approx_evaluations: {optimizer.approximate_evaluations} "
          f"exact_eval_time_total: {optimizer.exact_eval_time_total:.2f}")
    print(f"process_backend_failed: {optimizer._process_backend_failed}")
    print(f"found_solution_count: {result.found_solution_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
