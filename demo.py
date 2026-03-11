#!/usr/bin/env python3
"""
LLM4DRD 智能调度平台 — 完整演示
==================================
覆盖主要流程:
  Step 1  问题实例生成
  Step 2  仿真 + 内置调度规则对比
  Step 3  异构图建模
  Step 4  帕累托前沿 (内置规则枚举)
  Step 5  NSGA-II 真实帕累托前沿
  Step 6  精确求解 (OR-Tools CP-SAT, 需安装 ortools)
  Step 7  在线调度 (事件驱动推进)
  Step 8  数据库：规则存取
  Step 9  LLM 进化引擎 (无 API Key 时使用模板回退)

运行:
  python -m llm4drd_platform                           # 无需 API Key
  LLM_API_KEY=sk-xxx python -m llm4drd_platform        # 使用真实 LLM
"""

import os
import sys
import time
import logging

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# 当直接 python demo.py 运行时确保包可导入
_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from llm4drd_platform.config import get_config
from llm4drd_platform.data.generator import InstanceGenerator
from llm4drd_platform.core.simulator import Simulator
from llm4drd_platform.core.rules import BUILTIN_RULES, compile_rule_from_code
from llm4drd_platform.knowledge.graph import HeterogeneousGraph
from llm4drd_platform.optimization.pareto import ParetoOptimizer, NSGA2Optimizer
from llm4drd_platform.optimization.exact import ExactSolver
from llm4drd_platform.scheduling.online import OnlineSchedulerV3
from llm4drd_platform.data.db import init_db, RuleStore
from llm4drd_platform.ai.evolution import EvolutionEngine, EvolutionConfig, LLMInterface

DIVIDER = "─" * 60


def section(title: str):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


def ok(msg: str):
    print(f"  ✓  {msg}")


def info(msg: str):
    print(f"     {msg}")


# ── Step 1: 问题实例生成 ────────────────────────────────────────────────
def step1_generate():
    section("Step 1 / 9  问题实例生成")
    gen = InstanceGenerator(seed=42)
    shop = gen.generate(
        num_orders=8,
        tasks_per_order=(2, 4),
        ops_per_task=(2, 4),
        machines_per_type=2,
        processing_time_range=(1, 8),
        due_date_factor=1.4,
    )
    s = shop.summary()
    ok(f"车间: {s['orders']} 订单 / {s['tasks']} 任务 / "
       f"{s['operations']} 工序 / {s['machines']} 机器 ({s['machine_types']} 类)")
    return shop


# ── Step 2: 仿真 + 内置规则对比 ─────────────────────────────────────────
def step2_simulate(shop):
    section("Step 2 / 9  仿真 + 内置调度规则对比")
    rules_to_compare = ["ATC", "EDD", "SPT", "COMPOSITE", "BOTTLENECK"]
    results = {}
    for name in rules_to_compare:
        r = Simulator(shop, BUILTIN_RULES[name]).run()
        results[name] = r
        ok(f"{name:12s}  makespan={r.makespan:6.1f}h  "
           f"tardiness={r.total_tardiness:7.2f}  "
           f"tardy={r.tardy_job_count}  wall={r.wall_time_ms:.1f}ms")

    best = min(results, key=lambda k: results[k].total_tardiness)
    info(f"→ 最低总延迟: {best} ({results[best].total_tardiness:.2f})")

    # 自定义规则编译
    fn_custom = compile_rule_from_code(
        "def my_rule(op, m, f, s): return f['priority']*3 - f['processing_time']*0.1"
    )
    r_c = Simulator(shop, fn_custom).run()
    ok(f"自定义规则      makespan={r_c.makespan:6.1f}h  tardiness={r_c.total_tardiness:7.2f}")
    return results


# ── Step 3: 异构图建模 ──────────────────────────────────────────────────
def step3_graph(shop):
    section("Step 3 / 9  异构图建模")
    hg = HeterogeneousGraph()
    hg.build_from_shopfloor(shop)
    stats = hg.get_graph_stats()
    ok(f"节点: {stats['total_nodes']}  边: {stats['total_edges']}")
    info(f"节点类型: {stats['node_types']}")
    info(f"边类型:   {stats['edge_types']}")
    return hg


# ── Step 4: 帕累托前沿 (内置规则枚举) ──────────────────────────────────
def step4_pareto(shop):
    section("Step 4 / 9  帕累托前沿 (内置 11 条规则)")
    opt = ParetoOptimizer(shop, ["total_tardiness", "makespan"])
    solutions = opt.evaluate()
    pareto = [s for s in solutions if s.rank == 0]
    ok(f"{len(solutions)} 个解, 帕累托前沿 {len(pareto)} 点")
    for s in sorted(pareto, key=lambda x: x.objectives["total_tardiness"]):
        info(f"  {s.rule_name:12s}  tard={s.objectives['total_tardiness']:7.2f}  "
             f"mksp={s.objectives['makespan']:6.2f}")
    return solutions


# ── Step 5: NSGA-II 真实帕累托前沿 ──────────────────────────────────────
def step5_nsga2(shop):
    section("Step 5 / 9  NSGA-II 真实帕累托前沿")
    t0 = time.time()
    nsga2 = NSGA2Optimizer(
        shop, objective_keys=["total_tardiness", "makespan"],
        pop_size=12, generations=5, seed=42,
    )

    def callback(done, total, gen):
        pct = int(done / total * 30)
        print(f"\r     进化中 [{'#'*pct}{' '*(30-pct)}] {done}/{total}", end="", flush=True)

    solutions = nsga2.run(callback=callback)
    print()
    elapsed = time.time() - t0
    pareto = [s for s in solutions if s.rank == 0]
    ok(f"NSGA-II 完成 ({elapsed:.1f}s): {len(solutions)} 个解, 前沿 {len(pareto)} 点")
    info(f"前沿最优延迟: {min(s.objectives['total_tardiness'] for s in pareto):.2f}  "
         f"前沿最优 Makespan: {min(s.objectives['makespan'] for s in pareto):.2f}")
    return solutions


# ── Step 6: 精确求解 ────────────────────────────────────────────────────
def step6_exact(shop):
    section("Step 6 / 9  精确求解 (OR-Tools CP-SAT)")
    solver = ExactSolver(shop, objectives=["makespan"], time_limit_s=10)
    t0 = time.time()
    result = solver.solve()
    elapsed = time.time() - t0
    if result.status == "ERROR":
        info("⚠  ortools 未安装, 跳过 (pip install ortools>=9.7)")
    elif result.status in ("OPTIMAL", "FEASIBLE"):
        ok(f"求解 {result.status} ({elapsed:.1f}s): {result.objectives}")
        if result.bounds:
            info(f"下界: {result.bounds}")
    else:
        info(f"求解状态: {result.status} ({elapsed:.1f}s)")
    return result


# ── Step 7: 在线调度 ────────────────────────────────────────────────────
def step7_online(shop):
    section("Step 7 / 9  在线调度 (事件驱动)")
    scheduler = OnlineSchedulerV3(shop, rule_name="ATC")
    ok(f"初始化完成, 机器数={len(shop.machines)}")

    status = scheduler.advance(20.0)
    ok(f"推进 20h → 完成 {status['ops_completed']}/{status['ops_total']}, "
       f"进行中 {status['ops_in_progress']}, 就绪 {status['ops_ready']}")

    first_mid = list(shop.machines.keys())[0]
    scheduler.on_breakdown(first_mid, repair_at=30.0)
    info(f"→ 机器 {first_mid} 故障 (修复时间=30h)")

    status = scheduler.advance(15.0)
    ok(f"推进 15h → 完成 {status['ops_completed']}/{status['ops_total']}")

    scheduler.on_repair(first_mid)
    info(f"→ 机器 {first_mid} 恢复")

    status = scheduler.advance(200.0)
    ok(f"全部推进完毕 → 完成 {status['ops_completed']}/{status['ops_total']} 工序")
    gantt = status.get("gantt", [])
    if gantt:
        last = gantt[-1]
        info(f"甘特最后一条: {last['op_name']} on {last['machine_name']} "
             f"[{last['start']:.1f}h ~ {last['end']:.1f}h]")
    return status


# ── Step 8: 数据库 ──────────────────────────────────────────────────────
def step8_database():
    section("Step 8 / 9  数据库：规则存取")
    db_path = "/tmp/llm4drd_demo.db"
    init_db(db_path)
    ok(f"数据库初始化: {db_path}")

    store = RuleStore(db_path)
    import uuid
    store.save_rule(
        rule_id=str(uuid.uuid4())[:8],
        name="demo_rule",
        code="def demo_rule(op,m,f,s): return f['priority']",
        objective="total_tardiness",
        fitness=42.0,
    )
    ok("规则已保存")

    rules = store.get_all_rules(active_only=True)
    ok(f"活跃规则数: {len(rules)}, 首条: {rules[0]['name'] if rules else '(空)'}")
    return db_path


# ── Step 9: LLM 进化引擎 ────────────────────────────────────────────────
def step9_evolution(shop, db_path):
    section("Step 9 / 9  LLM 进化引擎")
    cfg = get_config()
    use_real = bool(cfg.llm.api_key)
    info(f"API Key: {'已配置 → 真实 LLM' if use_real else '未配置 → 模板回退模式'}")

    llm = LLMInterface()
    engine = EvolutionEngine(
        config=EvolutionConfig(population_size=4, max_generations=2),
        llm=llm,
    )

    call_count = [0]
    def on_log(entry):
        call_count[0] += 1

    llm.set_callback(on_log)
    t0 = time.time()
    best_rule = engine.evolve(train_instances=[shop])
    elapsed = time.time() - t0

    gen_count = len(engine.history)
    best_fit = engine.best.fitness if engine.best else float('inf')
    ok(f"进化完成 ({elapsed:.1f}s): {gen_count} 代, "
       f"最优 fitness={best_fit:.2f}, LLM 调用 {call_count[0]} 次")
    if best_rule:
        info(f"最优规则: {best_rule.name}")
    return best_rule


# ── 主程序 ──────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  LLM4DRD 智能调度平台  完整演示")
    print("=" * 60)

    shop    = step1_generate()
    step2_simulate(shop)
    step3_graph(shop)
    step4_pareto(shop)
    step5_nsga2(shop)
    step6_exact(shop)
    step7_online(shop)
    db_path = step8_database()
    step9_evolution(shop, db_path)

    print(f"\n{'=' * 60}")
    print("  演示完成！所有核心模块运行正常。")
    print("  启动 Web 服务:")
    print("    uvicorn llm4drd_platform.api.server:app --reload --port 8000")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
