# -*- coding: utf-8 -*-
"""统一迷你案例：3 订单 / 7 工序 / 2 机器。

用项目真实代码（core.models + core.simulator + optimization.exact）跑通，
为《LLM4DRD 排产算法基础理论》各章提供自洽数字。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 可用环境变量覆盖交期做敏感性实验：DUES="13,10,15"
_dues = os.environ.get("DUES")
if _dues:
    _d1, _d2, _d3 = (float(x) for x in _dues.split(","))
else:
    _d1, _d2, _d3 = 11.0, 8.0, 14.0

from core.models import Machine, MachineType, Operation, Order, ShopFloor, Task
from core.rules import BUILTIN_RULES
from core.simulator import Simulator

CASE_SPEC = {
    "machines": [
        {"id": "M1", "name": "机床1号", "type": "CNC"},
        {"id": "M2", "name": "机床2号", "type": "CNC"},
    ],
    "orders": [
        {"id": "O1", "name": "订单A(齿轮轴)", "due": _d1, "priority": 1,
         "task": "T1", "ops": [
             {"id": "A1", "name": "A1 粗车", "pt": 3.0, "pred": []},
             {"id": "A2", "name": "A2 精车", "pt": 2.0, "pred": ["A1"]},
             {"id": "A3", "name": "A3 磨削", "pt": 4.0, "pred": ["A2"]},
         ]},
        {"id": "O2", "name": "订单B(法兰盘)", "due": _d2, "priority": 1,
         "task": "T2", "ops": [
             {"id": "B1", "name": "B1 铣面", "pt": 5.0, "pred": []},
             {"id": "B2", "name": "B2 钻孔", "pt": 3.0, "pred": ["B1"]},
         ]},
        {"id": "O3", "name": "订单C(支架)", "due": _d3, "priority": 1,
         "task": "T3", "ops": [
             {"id": "C1", "name": "C1 下料", "pt": 2.0, "pred": []},
             {"id": "C2", "name": "C2 折弯", "pt": 6.0, "pred": ["C1"]},
         ]},
    ],
}


def build_shop() -> ShopFloor:
    shop = ShopFloor()
    shop.machine_types["CNC"] = MachineType(id="CNC", name="数控机床", is_critical=True)
    for m in CASE_SPEC["machines"]:
        shop.machines[m["id"]] = Machine(id=m["id"], name=m["name"], type_id=m["type"])
    for o in CASE_SPEC["orders"]:
        task_id = o["task"]
        shop.orders[o["id"]] = Order(
            id=o["id"], name=o["name"], release_time=0.0,
            due_date=o["due"], priority=o["priority"],
            task_ids=[task_id], main_task_id=task_id,
        )
        task = Task(
            id=task_id, order_id=o["id"], name=f"{o['name']}·主任务",
            is_main=True, release_time=0.0, due_date=o["due"],
        )
        for spec in o["ops"]:
            op = Operation(
                id=spec["id"], task_id=task_id, name=spec["name"],
                process_type="CNC", processing_time=spec["pt"],
                predecessor_ops=list(spec["pred"]),
            )
            task.operations.append(op)
            shop.operations[op.id] = op
        shop.tasks[task_id] = task
    shop.build_indexes()
    return shop


def run_rule(rule_name: str) -> dict:
    shop = build_shop()
    sim = Simulator(shop, BUILTIN_RULES[rule_name])
    result = sim.run()
    derived = {
        op.id: {
            "derived_due": op.derived_due_date,
            "derived_start": op.derived_start_time,
            "est": op.earliest_start_time,
            "eft": op.earliest_finish_time,
            "critical_slack": op.critical_slack,
        }
        for op in shop.operations.values()
    }
    return {
        "rule": rule_name,
        "kpi": result.to_dict(),
        "schedule": sorted(result.schedule, key=lambda e: (e["start"], e["machine_id"])),
        "derived": derived,
    }


def run_exact(time_limit: float = 60.0) -> dict:
    # optimization 包内用了相对导入（from ..core...），需以 llm4drd 包形式导入
    parent = str(Path(__file__).resolve().parent.parent.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    import inspect
    from llm4drd.optimization.exact import ExactSolver

    shop = build_shop()
    solver = ExactSolver(shop)
    solve_fn = getattr(solver, "solve", None) or getattr(solver, "solve_makespan", None)
    sig = inspect.signature(solve_fn) if solve_fn else None
    info = {"solve_signature": str(sig)}
    if solve_fn:
        try:
            result = solve_fn()
            info["result_type"] = type(result).__name__
            info["result"] = str(result)[:2000]
            for attr in ("makespan", "status", "objective", "schedule"):
                if hasattr(result, attr):
                    info[attr] = str(getattr(result, attr))[:1500]
        except TypeError:
            result = solve_fn(time_limit=time_limit)
            info["result"] = str(result)[:2000]
        except Exception as exc:  # noqa: BLE001
            info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def main() -> None:
    out = {"case": CASE_SPEC, "rules": {}}
    shop = build_shop()
    out["derived"] = {
        op.id: {
            "derived_due": op.derived_due_date,
            "derived_start": op.derived_start_time,
            "est": op.earliest_start_time,
            "eft": op.earliest_finish_time,
        }
        for op in shop.operations.values()
    }
    for name in ["FIFO", "SPT", "LPT", "EDD", "MST", "CR", "ATC"]:
        out["rules"][name] = run_rule(name)
    Path("docgen/case_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str)
    )
    for name, r in out["rules"].items():
        k = r["kpi"]
        print(f"{name:5s} makespan={k['makespan']:5.1f} tard={k['total_tardiness']:5.1f} "
              f"flow={k['avg_flowtime']:5.2f} util={k['avg_utilization']:.2f} "
              f"tardy={k['tardy_job_count']}")
    print("\n派生交期/最早时刻:")
    for op_id, d in out["derived"].items():
        print(f"  {op_id}: derived_due={d['derived_due']}, derived_start={d['derived_start']}, "
              f"EST={d['est']}, EFT={d['eft']}")
    print("\nATC 排程明细:")
    for e in out["rules"]["ATC"]["schedule"]:
        print(f"  {e['op_id']} on {e['machine_id']}: {e['start']} -> {e['end']}")
    print(run_exact())


if __name__ == "__main__":
    main()
