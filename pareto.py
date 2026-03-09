"""
帕累托多目标优化 v3
===================
支持全部工业排产常用目标:
  - 主订单延误数/延误时间/延误比例
  - 总延迟 / Makespan / 最大延迟
  - 关键资源利用率 / 平均利用率
  - 总等待时间 / 平均流程时间
"""
from dataclasses import dataclass, field
from typing import Optional, Callable
from .models import ShopFloor
from .simulator import Simulator, SimResult
from .dispatching_rules import BUILTIN_RULES, compile_rule_from_code


@dataclass
class Objective:
    key: str           # SimResult中的字段名
    label: str
    direction: str     # "min" or "max"

OBJECTIVES = {
    "total_tardiness": Objective("total_tardiness", "总延迟", "min"),
    "makespan": Objective("makespan", "Makespan", "min"),
    "main_order_tardy_count": Objective("main_order_tardy_count", "延误主订单数", "min"),
    "main_order_tardy_total_time": Objective("main_order_tardy_total_time", "主订单延误总时间", "min"),
    "main_order_tardy_ratio": Objective("main_order_tardy_ratio", "主订单延误比例", "min"),
    "avg_utilization": Objective("avg_utilization", "平均利用率", "max"),
    "critical_utilization": Objective("critical_utilization", "关键资源利用率", "max"),
    "total_wait_time": Objective("total_wait_time", "总等待时间", "min"),
    "avg_flowtime": Objective("avg_flowtime", "平均流程时间", "min"),
    "max_tardiness": Objective("max_tardiness", "最大延迟", "min"),
    "tardy_job_count": Objective("tardy_job_count", "延迟任务数", "min"),
    "avg_tardiness": Objective("avg_tardiness", "平均延迟", "min"),
}


@dataclass
class ParetoSolution:
    rule_name: str
    objectives: dict
    rank: int = 0
    crowding: float = 0.0
    schedule: list = field(default_factory=list)


def dominates(a: dict, b: dict, objs: list[Objective]) -> bool:
    any_better = False
    for o in objs:
        va, vb = a.get(o.key, 0), b.get(o.key, 0)
        if o.direction == "min":
            if va > vb: return False
            if va < vb: any_better = True
        else:
            if va < vb: return False
            if va > vb: any_better = True
    return any_better


class ParetoOptimizer:
    def __init__(self, shop: ShopFloor, objective_keys: list[str]):
        self.shop = shop
        self.objs = [OBJECTIVES[k] for k in objective_keys if k in OBJECTIVES]
        if not self.objs:
            self.objs = [OBJECTIVES["total_tardiness"], OBJECTIVES["makespan"]]

    def evaluate(self, rules: dict[str, Callable] = None) -> list[ParetoSolution]:
        rules = rules or BUILTIN_RULES
        solutions = []
        for name, func in rules.items():
            sim = Simulator(self.shop, func)
            result = sim.run()
            obj_vals = {o.key: getattr(result, o.key, 0) for o in self.objs}
            solutions.append(ParetoSolution(
                rule_name=name, objectives=obj_vals, schedule=result.schedule
            ))

        # Non-dominated sort
        n = len(solutions)
        dom_count = [0] * n
        dom_set = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if dominates(solutions[i].objectives, solutions[j].objectives, self.objs):
                    dom_set[i].append(j); dom_count[j] += 1
                elif dominates(solutions[j].objectives, solutions[i].objectives, self.objs):
                    dom_set[j].append(i); dom_count[i] += 1
        front = [i for i in range(n) if dom_count[i] == 0]
        for i in front: solutions[i].rank = 0
        rank = 0
        while front:
            nxt = []
            for i in front:
                for j in dom_set[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        solutions[j].rank = rank + 1
                        nxt.append(j)
            front = nxt; rank += 1

        # Crowding distance for rank 0
        r0 = [i for i in range(n) if solutions[i].rank == 0]
        if len(r0) > 2:
            for o in self.objs:
                sr = sorted(r0, key=lambda i: solutions[i].objectives.get(o.key, 0))
                solutions[sr[0]].crowding = float('inf')
                solutions[sr[-1]].crowding = float('inf')
                vmin = solutions[sr[0]].objectives.get(o.key, 0)
                vmax = solutions[sr[-1]].objectives.get(o.key, 0)
                rng = vmax - vmin if vmax != vmin else 1
                for k in range(1, len(sr)-1):
                    vp = solutions[sr[k-1]].objectives.get(o.key, 0)
                    vn = solutions[sr[k+1]].objectives.get(o.key, 0)
                    solutions[sr[k]].crowding += (vn - vp) / rng

        return solutions

    def to_frontend(self, sols: list[ParetoSolution]) -> dict:
        obj_keys = [o.key for o in self.objs]
        obj_labels = [o.label for o in self.objs]
        obj_dirs = [o.direction for o in self.objs]
        pts = []
        for s in sols:
            pt = {"rule": s.rule_name, "rank": s.rank, "crowding": round(s.crowding, 3), "is_pareto": s.rank == 0}
            for k in obj_keys:
                pt[k] = round(s.objectives.get(k, 0), 4)
            pts.append(pt)
        return {"objectives": obj_keys, "labels": obj_labels, "directions": obj_dirs,
                "solutions": pts, "pareto_front": [p for p in pts if p["is_pareto"]]}
