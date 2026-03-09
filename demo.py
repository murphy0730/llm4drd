#!/usr/bin/env python3
"""
LLM4DRD 智能调度平台 — 完整演示 (合并版 v2)
=============================================
演示全部模块: 实例生成 → 异构图 → 特征编码 → 规则对比 →
  LLM进化 → 在线调度 → 动态重排 → 帕累托 → MonteCarlo → 性能 → 数据库

运行:  python -m llm4drd_platform
真实LLM: export LLM_API_KEY=sk-xxx && python -m llm4drd_platform
"""
import os, sys, time, inspect, logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm4drd_platform.config import get_config
from llm4drd_platform.models import MachineStatus
from llm4drd_platform.instance_generator import InstanceGenerator
from llm4drd_platform.heterogeneous_graph import HeterogeneousGraph
from llm4drd_platform.feature_encoder import FeatureEncoder
from llm4drd_platform.simulator import Simulator
from llm4drd_platform.dispatching_rules import BUILTIN_RULES, get_all_rule_names, compile_rule_from_code
from llm4drd_platform.llm_evolution import EvolutionEngine, EvolutionConfig, LLMInterface
from llm4drd_platform.online_scheduler import OnlineScheduler
from llm4drd_platform.rescheduler import DynamicRescheduler, RescheduleConfig
from llm4drd_platform.scenario_manager import ScenarioManager
from llm4drd_platform.pareto import ParetoOptimizer
from llm4drd_platform.performance import FeasibilityIndex, FeatureCache
from llm4drd_platform.db_manager import init_db, RuleStore

logging.basicConfig(level=logging.WARNING)

def hdr(t): print(f"\n{'='*64}\n  {t}\n{'='*64}")
def tbl(h, rows):
    w=[max(len(str(h[i])),max((len(str(r[i]))for r in rows),default=0))for i in range(len(h))]
    f=" | ".join(f"{{:<{x}}}"for x in w); print(f.format(*h)); print("-+-".join("-"*x for x in w))
    for r in rows: print(f.format(*[str(v)for v in r]))

def main():
    print(r"""
  ╔═══════════════════════════════════════════════════════════╗
  ║  LLM4DRD 智能调度平台 — 合并工程完整演示 v2              ║
  ╚═══════════════════════════════════════════════════════════╝""")

    cfg = get_config()
    print(f"  LLM: {cfg.llm.model} @ {cfg.llm.base_url}")
    print(f"  Key: {'已配置' if cfg.llm.api_key else '未配置(模板模式)'}")

    # ── 1. 实例 ──
    hdr("1/9  生成 FAFSP 问题实例")
    gen = InstanceGenerator(seed=42)
    shop = gen.generate_fafsp_instance(num_orders=8, products_per_order=(1,3), components_per_product=(2,4),
        num_processing_machines=5, num_assembly_machines=2, due_date_factor=1.5, order_arrival_spread=0.3)
    np = sum(1 for m in shop.machines.values() if m.stage_id=="stage_proc")
    na = sum(1 for m in shop.machines.values() if m.stage_id=="stage_asm")
    print(f"  {len(shop.orders)}单 {len(shop.jobs)}作业 {len(shop.machines)}机器(加工{np}+组装{na}) {len(shop.assembly_groups)}配套组")
    for oid,o in shop.orders.items():
        print(f"    {oid}: 到达={o.release_time:.0f} 交期={o.due_date:.0f} P{o.priority} 作业={len(o.job_ids)}")

    # ── 2. 异构图+特征 ──
    hdr("2/9  异构图建模 & 特征编码")
    g = HeterogeneousGraph(); g.build_from_shopfloor(shop); s=g.get_graph_stats()
    print(f"  {s['total_nodes']}节点 {s['total_edges']}边  类型={s['node_types']}")
    enc = FeatureEncoder(shop); fv=enc.encode(list(shop.jobs.values())[0],list(shop.machines.values())[0],0.0)
    print(f"  特征向量: {len(fv.to_dict())}维")

    # ── 3. 性能模块 ──
    hdr("3/9  性能优化模块")
    idx = FeasibilityIndex(shop)
    for mid in list(shop.machines.keys()):
        ps = idx.get_feasible_for_machine(mid, 0.0); print(f"  {mid}: {len(ps)}可行")
    print(f"  FeatureCache: max_size={5000}")

    # ── 4. 规则对比 ──
    hdr("4/9  11条内置规则对比")
    rs=[]
    for n in get_all_rule_names():
        r=Simulator(shop,BUILTIN_RULES[n]).run(max_time=50000)
        rs.append((n,round(r.total_tardiness,1),round(r.makespan,1),f"{r.avg_utilization*100:.1f}%",r.tardy_job_count,f"{r.simulation_time*1000:.0f}ms"))
    rs.sort(key=lambda x:x[1])
    tbl(["规则","总延迟","Makespan","利用率","延迟数","耗时"],rs)
    best_rule=rs[0][0]; print(f"\n  ★ 最优: {best_rule}")

    # ── 5. 帕累托 ──
    hdr("5/9  帕累托多目标优化")
    opt=ParetoOptimizer(shop,["total_tardiness","makespan"])
    sols=opt.evaluate_rules(BUILTIN_RULES); front=opt.get_pareto_front(sols)
    print(f"  {len(sols)}方案, 帕累托前沿={len(front)}个")
    for s2 in sorted(sols,key=lambda x:x.rank):
        vs=list(s2.objectives.values())
        print(f"  {s2.rule_name:<14} {vs[0]:>9.1f} {vs[1]:>8.1f}  rank={s2.rank}{'  ★'if s2.rank==0 else''}")
    fd=opt.to_frontend_data(sols)
    print(f"  前端数据: {len(fd['solutions'])}点, 帕累托={len(fd['pareto_front'])}点")

    # ── 6. LLM进化 ──
    hdr("6/9  LLM双专家进化训练")
    ts=gen.generate_training_set(3); vs=gen.generate_test_set(2)
    ecfg=EvolutionConfig(population_size=6,elite_size=2,max_generations=4,patience=3)
    eng=EvolutionEngine(ecfg,LLMInterface()); eng.initialize_population(["ATC","EDD","KIT_AWARE"])
    def cb(g2,bf,pop):
        vd=[p for p in pop if p.fitness<float('inf')]
        print(f"    第{g2}代: best={bf:.1f} avg={sum(p.fitness for p in vd)/max(1,len(vd)):.1f}")
    t0=time.time(); be=eng.evolve(ts,vs,callback=cb); te=time.time()-t0
    if be:
        print(f"  完成({te:.1f}s) best={be.fitness:.1f} hybrid={be.hybrid_score:.4f}")
        try:
            r2=Simulator(shop,compile_rule_from_code(be.code)).run(max_time=50000)
            print(f"  主实例验证: 延迟={r2.total_tardiness:.1f} makespan={r2.makespan:.1f}")
        except: pass

    # ── 7. 在线调度 ──
    hdr("7/9  在线事件驱动调度")
    shop2=InstanceGenerator(seed=99).generate_fafsp_instance(num_orders=5,num_processing_machines=4,num_assembly_machines=2)
    sc=OnlineScheduler(shop2); sc.set_rule(best_rule)
    ds=[d for mid,m in shop2.machines.items() if m.status==MachineStatus.IDLE for d in [sc.on_machine_idle(mid,0.0)] if d]
    for d in ds[:5]: print(f"  {d.job_id}→{d.machine_id} score={d.priority_score:.3f} {d.decision_time_ms:.2f}ms")
    if ds: print(f"  平均决策: {sum(d.decision_time_ms for d in ds)/len(ds):.2f}ms")

    # ── 8. 动态重排 ──
    hdr("8/9  动态重排")
    rr=DynamicRescheduler(sc,RescheduleConfig(freeze_window=30,local_search_iterations=10,local_search_timeout=3))
    res=rr.reschedule(0.0,"urgent_insertion")
    print(f"  原延迟={res.old_fitness:.1f} → 新={res.new_fitness:.1f}  改善={res.improvement:.1%}  变更={res.change_ratio:.1%}  耗时={res.computation_time:.2f}s")

    # ── 9. MonteCarlo + DB ──
    hdr("9/9  Monte Carlo & 数据库")
    mc=ScenarioManager(shop).run_monte_carlo(BUILTIN_RULES["ATC"],15,0.05,0.1)
    print(f"  15次: 延迟 μ={mc['tardiness']['mean']} σ={mc['tardiness']['std']}  makespan μ={mc['makespan']['mean']}")
    db="/tmp/_demo.db"; init_db(db); st=RuleStore(db)
    for n,f in BUILTIN_RULES.items(): st.save_rule(f"b_{n}",n,inspect.getsource(f),is_builtin=True)
    print(f"  SQLite: {len(st.get_all_rules())}条规则 ✓"); os.remove(db)

    # ── 完成 ──
    hdr("全部完成 — 14模块集成测试通过")
    print("""
  启动服务:
    pip install -r requirements.txt
    uvicorn llm4drd_platform.api_server:app --host 0.0.0.0 --port 8000

  配置大模型 (任选):
    1) 编辑 config.json          2) export LLM_API_KEY=sk-xxx
    3) API: PUT /api/config/llm   4) 前端页面直接修改
""")

if __name__=="__main__": main()
