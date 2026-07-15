#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析仿真中"未排产(unscheduled)"的工序，定位根因。

为什么需要它
------------
前端/后端的仿真诊断只把未排工序粗略分成"被上游级联阻塞(非根因)"和若干
根因。当一批工序"全是级联受阻、却没有任何根因"时(这正是"机器班次很长却
还是排不下"的典型现象)，说明真正的根因没被现有诊断暴露出来。常见真因有四类:

  1) 依赖环(cycle): 若干工序/任务互相把对方列为前驱，永远无法就绪。
  2) 断链(dangling): 某工序的前驱指向了不存在的工序/任务。
  3) 空任务前驱: 作为前驱的任务没有任何工序，永远无法"完成"，其下游被卡死。
  4) 资源日历耗尽(calendar-exhausted): 工序已就绪，但机器/工装/人员三类中
     某一类的可用日历覆盖不到所需工时，被仿真器挂起(排在 _unschedulable_ops)。
     —— 很多时候你只把"机器"班次设长了，却忘了工装/人员的日历。

用法
----
  # ① 结构性分析(快, 不需要跑仿真): 检测依赖环 / 断链 / 空任务前驱
  python tools/analyze_unscheduled.py --db llm4drd.db

  # ② 完整分析(会实跑 ATC 仿真, 较慢): 额外捕获"已就绪但排不下"的工序,
  #    并逐资源定位是哪类资源的日历不够长
  python tools/analyze_unscheduled.py --db llm4drd.db --simulate

  # ③ 把未排工序清单导出成 CSV(便于在 Excel 里逐道核对)
  python tools/analyze_unscheduled.py --db llm4drd.db --simulate --out unscheduled.csv

说明
----
  - 默认数据库为当前目录的 llm4drd.db，可用 --db 指定你本机/服务器上的实例库。
  - 必须在仓库根目录运行(脚本会自动注册 llm4drd_platform 包)。
  - --simulate 模式会完整跑一遍仿真，实例很大时可能要几分钟。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import pathlib
import sys
import time
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent  # 仓库根目录


def _bootstrap_package() -> None:
    """复用 run_server.py 的方式注册 llm4drd_platform 包，使相对导入可用。"""
    spec = importlib.util.spec_from_file_location(
        "llm4drd_platform", ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("无法创建 llm4drd_platform 包 spec，请确认在仓库根目录运行。")
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["llm4drd_platform"] = pkg
    spec.loader.exec_module(pkg)


def _iterative_tarjan_scc(nodes, adj):
    """迭代式 Tarjan 强连通分量，避免大模型上的递归爆栈。

    adj: dict[node -> list[前驱节点]]。返回 list[list[node]]，其中长度>1 的
    分量即为依赖环(成员互相依赖，永远无法就绪)。
    """
    index_counter = [0]
    index: dict = {}
    lowlink: dict = {}
    on_stack: dict = {}
    stack: list = []
    result: list = []

    for start in nodes:
        if start in index:
            continue
        work = [(start, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                index[v] = index_counter[0]
                lowlink[v] = index_counter[0]
                index_counter[0] += 1
                stack.append(v)
                on_stack[v] = True
            recurse = False
            neighbors = list(adj[v])
            i = pi
            while i < len(neighbors):
                w = neighbors[i]
                if w not in index:
                    work[-1] = (v, i + 1)
                    work.append((w, 0))
                    recurse = True
                    break
                elif on_stack.get(w):
                    lowlink[v] = min(lowlink[v], index[w])
                i += 1
            if recurse:
                continue
            if lowlink[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == v:
                        break
                result.append(comp)
            work.pop()
            if work:
                u = work[-1][0]
                lowlink[u] = min(lowlink[u], lowlink[v])
    return result


# 足够大的上界，覆盖绝大多数日历(单位与模型时间单位一致)
_HORIZON = 1e7


def _available_hours(resource) -> float | None:
    """某资源(机器/工装/人员)在 [0, _HORIZON] 内的总可用工时。

    单个资源对象没有 calendar_days 属性(那是 ShopFloor 级别的方法)，
    用 CalendarResourceMixin.available_time_between 求和可用窗口即可。
    """
    try:
        return resource.available_time_between(0.0, _HORIZON)
    except Exception:
        return None


def analyze_structural(shop) -> dict:
    """纯结构性检查：断链、空任务前驱、依赖环。不依赖仿真结果。"""
    print("\n=== [结构性分析] 依赖关系检查(不需要跑仿真) ===", flush=True)

    # 1) 断链: 前驱指向不存在的工序/任务
    dangling_ops, dangling_tasks = [], []
    for op in shop.operations.values():
        for p in op.predecessor_ops:
            if p not in shop.operations:
                dangling_ops.append((op.id, p))
        for t in op.predecessor_tasks:
            if t not in shop.tasks:
                dangling_tasks.append((op.id, t))
    print(f"  · 断链(前驱不存在): 工序级 {len(dangling_ops)} 处, 任务级 {len(dangling_tasks)} 处")
    for oid, p in dangling_ops[:10]:
        print(f"      op {oid} -> 缺失工序 {p}")
    for oid, t in dangling_tasks[:10]:
        print(f"      op {oid} -> 缺失任务 {t}")

    # 2) 空任务前驱: 作为前驱的任务自身没有任何工序，永远无法"完成"
    task_op_ids: dict[str, list[str]] = defaultdict(list)
    for op in shop.operations.values():
        task_op_ids[op.task_id].append(op.id)
    empty_task_preds = set()
    for op in shop.operations.values():
        for t in op.predecessor_tasks:
            if t in shop.tasks and not task_op_ids.get(t):
                empty_task_preds.add(t)
    print(f"  · 空任务前驱(任务无工序、永远不完成): {len(empty_task_preds)} 个")
    for t in list(empty_task_preds)[:10]:
        print(f"      任务 {t}")

    # 3) 依赖环: 在 工序前驱 + 任务前驱(展开为该任务的所有工序) 上做 SCC
    adj = {op.id: set() for op in shop.operations.values()}
    for op in shop.operations.values():
        for p in op.predecessor_ops:
            if p in adj and p != op.id:
                adj[op.id].add(p)
        for t in op.predecessor_tasks:
            for po in task_op_ids.get(t, []):
                if po in adj and po != op.id:
                    adj[op.id].add(po)
    sccs = _iterative_tarjan_scc(list(adj.keys()), adj)
    cyclic = [c for c in sccs if len(c) > 1]
    self_loops = [c[0] for c in sccs if len(c) == 1 and c[0] in adj[c[0]]]
    print(f"  · 依赖环: 自环 {len(self_loops)} 个, 长度>1 的环 {len(cyclic)} 个")
    if self_loops[:10]:
        print(f"      自环节点示例: {self_loops[:10]}")
    if cyclic:
        sizes = sorted((len(c) for c in cyclic), reverse=True)
        print(f"      环大小(降序): {sizes[:20]}")
        c0 = max(cyclic, key=len)
        print(f"      最大环含 {len(c0)} 道工序，示例:")
        for oid in c0[:15]:
            op = shop.operations[oid]
            in_comp = [p for p in op.predecessor_ops if p in set(c0)]
            print(f"        {oid}  指向环内前驱: {in_comp[:6]}")
    return {
        "dangling_ops": len(dangling_ops),
        "dangling_tasks": len(dangling_tasks),
        "empty_task_preds": len(empty_task_preds),
        "cyclic_count": len(cyclic),
        "self_loop_count": len(self_loops),
    }


def analyze_simulation(shop, rule_name: str = "ATC") -> dict:
    """实跑仿真，定位未排工序的真实状态与资源日历瓶颈。"""
    print(f"\n=== [仿真分析] 实跑 {rule_name} 仿真，定位未排工序 ===", flush=True)
    from llm4drd_platform.core.simulator import Simulator
    from llm4drd_platform.core.rules import BUILTIN_RULES

    func = BUILTIN_RULES.get(rule_name, BUILTIN_RULES["ATC"])
    sim = Simulator(shop, func)
    r = sim.run()
    completed = {e["op_id"] for e in r.schedule}
    total = len(shop.operations)
    unscheduled = [op for op in shop.operations.values() if op.id not in completed]
    unschedulable = set(getattr(sim, "_unschedulable_ops", set()))
    print(f"  · 总工序 {total}，已排 {len(completed)}，未排 {len(unscheduled)}")
    print(f"  · 其中被仿真器判定为'已就绪但排不下(资源日历耗尽)'的: {len(unschedulable)} 道")

    # 分类未排工序
    task_op_ids: dict[str, list[str]] = defaultdict(list)
    for op in shop.operations.values():
        task_op_ids[op.task_id].append(op.id)
    completed_tasks = {
        t for t, ops in task_op_ids.items()
        if ops and all(o in completed for o in ops)
    }

    cat = Counter()
    rows = []
    # 统计"哪类资源日历最短"——按排不下的工序聚合
    short_resource_counter = Counter()
    for op in unscheduled:
        preds_done = all(p in completed for p in op.predecessor_ops) and \
            all(t in completed_tasks for t in op.predecessor_tasks)
        if op.id in unschedulable:
            kind = "排不下(资源日历覆盖不到)"
            cat[kind] += 1
            # 逐资源定位: 哪类资源的日历最短
            bottleneck = _locate_bottleneck_resource(shop, op)
            for bt in bottleneck:
                short_resource_counter[bt] += 1
        elif preds_done:
            kind = "就绪但未被派工(资源竞争/挂起)"
            cat[kind] += 1
        else:
            kind = "被上游未完成工序阻塞(级联)"
            cat[kind] += 1
        rows.append({
            "op_id": op.id,
            "task_id": op.task_id,
            "process_type": op.process_type,
            "predecessors_done": preds_done,
            "in_unschedulable_set": op.id in unschedulable,
            "classification": kind,
            "processing_time": round(float(getattr(op, "processing_time", 0.0) or 0.0), 3),
        })

    print("  · 未排工序分类:")
    for k, v in cat.most_common():
        print(f"      {k}: {v} 道")

    if unschedulable:
        print("\n  · 排不下的工序中，最短资源类型统计(★ 即应重点核查的日历):")
        for res_desc, cnt in short_resource_counter.most_common(15):
            print(f"      {res_desc}: 出现在 {cnt} 道排不下的工序中")
        # 打印前若干道排不下工序的资源明细
        print("\n  · 排不下工序资源明细(前 20 道):")
        shown = 0
        for op in unscheduled:
            if op.id not in unschedulable:
                continue
            detail = _resource_detail(shop, op)
            print(f"      op={op.id} 工艺={op.process_type} 工时={detail['processing_time']}h")
            print(f"          机器: {detail['machine']}")
            for td in detail["tooling"]:
                print(f"          工装[{td['type']}]: {td['summary']}")
            for pd in detail["personnel"]:
                print(f"          人员[{pd['type']}]: {pd['summary']}")
            shown += 1
            if shown >= 20:
                break
    return {"rows": rows, "cat": dict(cat), "unschedulable": len(unschedulable)}


def _locate_bottleneck_resource(shop, op) -> list[str]:
    """粗略判断哪类资源的日历最短(可能排不下)。

    返回形如 ['工装:TOOL_A(覆盖 5 天)', '人员:SKILL_B(覆盖 3 天)'] 的描述。
    比较口径: 各资源类型中"覆盖天数最小"的那台/个，与工序工时对比。
    """
    out = []
    try:
        pt = float(getattr(op, "processing_time", 0.0) or 0.0)
    except Exception:
        pt = 0.0

    machines = shop.get_eligible_machines(op)
    if machines:
        # 单道工序同一时刻只占一台机器，故与"单台最长可用工时"比较
        max_avail = max((_available_hours(m) or 0.0) for m in machines)
        if pt > max_avail:
            out.append(f"机器(单台最长可用 {max_avail:.0f}h < 需要 {pt:.1f}h)")
    for t in getattr(op, "required_tooling_types", []) or []:
        tools = shop.get_toolings_for_type(t)
        if tools:
            max_avail = max((_available_hours(x) or 0.0) for x in tools)
            if pt > max_avail:
                out.append(f"工装:{t}(单件最长可用 {max_avail:.0f}h < 需要 {pt:.1f}h)")
    for s in getattr(op, "required_personnel_skills", []) or []:
        people = shop.get_personnel_for_skill(s)
        if people:
            max_avail = max((_available_hours(x) or 0.0) for x in people)
            if pt > max_avail:
                out.append(f"人员:{s}(单人最长可用 {max_avail:.0f}h < 需要 {pt:.1f}h)")
    return out


def _resource_detail(shop, op) -> dict:
    """收集某排不下工序所需的各类资源的日历覆盖摘要。"""
    pt = round(float(getattr(op, "processing_time", 0.0) or 0.0), 2)
    machines = shop.get_eligible_machines(op)
    machine_summary = "无可用机器" if not machines else \
        f"{len(machines)} 台, 单台可用工时 min={min((_available_hours(m) or 0.0) for m in machines):.0f} " \
        f"max={max((_available_hours(m) or 0.0) for m in machines):.0f}"
    tooling = []
    for t in getattr(op, "required_tooling_types", []) or []:
        tools = shop.get_toolings_for_type(t)
        if tools:
            hours = [(_available_hours(x) or 0.0) for x in tools]
            tooling.append({"type": t, "summary": f"{len(tools)} 个, 单件可用工时 min={min(hours):.0f} max={max(hours):.0f}"})
        else:
            tooling.append({"type": t, "summary": "无实例"})
    personnel = []
    for s in getattr(op, "required_personnel_skills", []) or []:
        people = shop.get_personnel_for_skill(s)
        if people:
            hours = [(_available_hours(x) or 0.0) for x in people]
            personnel.append({"type": s, "summary": f"{len(people)} 人, 单人可用工时 min={min(hours):.0f} max={max(hours):.0f}"})
        else:
            personnel.append({"type": s, "summary": "无实例"})
    return {"processing_time": pt, "machine": machine_summary, "tooling": tooling, "personnel": personnel}


def main() -> int:
    parser = argparse.ArgumentParser(description="分析仿真中未排产的工序并定位根因")
    parser.add_argument("--db", default="llm4drd.db", help="实例数据库路径(默认 ./llm4drd.db)")
    parser.add_argument("--mode", choices=["structural", "simulate"],
                        default="structural", help="分析模式: structural(默认,快) 或 simulate(实跑仿真)")
    parser.add_argument("--simulate", action="store_true",
                        help="等价于 --mode simulate，额外捕获'已就绪但排不下'的工序")
    parser.add_argument("--rule", default="ATC", help="仿真使用的派工规则(默认 ATC)")
    parser.add_argument("--out", default=None, help="将未排工序清单导出为 CSV 的路径")
    args = parser.parse_args()

    if args.simulate:
        args.mode = "simulate"

    t0 = time.time()
    _bootstrap_package()
    from llm4drd_platform.data.db import InstanceStore
    from llm4drd_platform.core.models import ShopFloor

    db_path = pathlib.Path(args.db)
    if not db_path.exists():
        print(f"[错误] 数据库不存在: {db_path}", file=sys.stderr)
        return 2
    store = InstanceStore(str(db_path))
    if not store.has_data():
        print(f"[错误] 数据库无实例数据: {db_path}", file=sys.stderr)
        return 2
    shop: ShopFloor = store.build_shopfloor()
    print(f"[%.1fs] 已加载实例: 工序 {len(shop.operations)} / 任务 {len(shop.tasks)} "
          f"/ 机器 {len(shop.machines)} / 工装 {len(shop.toolings)} / 人员 {len(shop.personnel)}"
          % (time.time() - t0), flush=True)

    structural = analyze_structural(shop)

    sim_rows = None
    if args.mode == "simulate":
        sim_result = analyze_simulation(shop, args.rule)
        sim_rows = sim_result["rows"]

    # 导出 CSV
    if args.out and sim_rows is not None:
        try:
            with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "op_id", "task_id", "process_type", "predecessors_done",
                    "in_unschedulable_set", "classification", "processing_time"])
                writer.writeheader()
                writer.writerows(sim_rows)
            print(f"\n[导出] 未排工序清单已写入: {args.out} ({len(sim_rows)} 行)")
        except Exception as exc:
            print(f"[警告] 导出 CSV 失败: {exc}", file=sys.stderr)

    # 结论提示
    print("\n=== 结论与建议 ===", flush=True)
    if structural["cyclic_count"] or structural["self_loop_count"]:
        print("  ⚠ 发现依赖环！这些工序互相阻塞、永远无法就绪，是'无根因的级联受阻'的真正源头。")
        print("    处理: 在数据里打断环(去掉互相矛盾的前驱关系)，或到'实例与约束'页运行数据校验。")
    if structural["dangling_ops"] or structural["dangling_tasks"]:
        print("  ⚠ 发现断链(前驱指向不存在的工序/任务)，需修正数据中的前驱引用。")
    if structural["empty_task_preds"]:
        print("  ⚠ 发现空任务前驱(作为前驱的任务没有工序)，需补工序或移除该前驱关系。")
    if args.mode == "simulate" and sim_rows is not None:
        unschedulable = sum(1 for r in sim_rows if r["in_unschedulable_set"])
        if unschedulable:
            print(f"  ⚠ 有 {unschedulable} 道工序已就绪却排不下——检查上面'最短资源类型统计'，")
            print("    重点看工装/人员的日历是否比机器短(常见: 机器班次设很长，工装/人员忘了设)。")
    print(f"\n总耗时 {time.time() - t0:.1f}s。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
