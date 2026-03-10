"""
帕累托多目标优化 v3
===================
支持全部工业排产常用目标:
  - 主订单延误数/延误时间/延误比例
  - 总延迟 / Makespan / 最大延迟
  - 关键资源利用率 / 平均利用率
  - 总等待时间 / 平均流程时间
"""
import random as _random
import math
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

    def to_frontend(self, sols: list) -> dict:
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


class WeightedEnsembleRule:
    """Weighted combination of K dispatch rules."""
    def __init__(self, weights, rule_fns):
        total = sum(abs(w) for w in weights) or 1.0
        self.weights = [w / total for w in weights]
        self.rule_fns = rule_fns

    def __call__(self, op, machine, feat, shop):
        score = 0.0
        for w, fn in zip(self.weights, self.rule_fns):
            try:
                score += w * fn(op, machine, feat, shop)
            except Exception:
                pass
        return score


class NSGA2Optimizer:
    def __init__(self, shop, objective_keys, rule_fns=None, pop_size=30, generations=20, seed=42):
        self.shop = shop
        self.objs = [OBJECTIVES[k] for k in objective_keys if k in OBJECTIVES]
        if not self.objs:
            self.objs = [OBJECTIVES["total_tardiness"], OBJECTIVES["makespan"]]
        from .dispatching_rules import BUILTIN_RULES
        self.rule_fns = rule_fns or list(BUILTIN_RULES.values())
        self.K = len(self.rule_fns)
        self.pop_size = pop_size
        self.generations = generations
        self.seed = seed

    def _random_weights(self, rng):
        w = [rng.gauss(0, 1) for _ in range(self.K)]
        return w

    def _evaluate(self, weights):
        rule = WeightedEnsembleRule(weights, self.rule_fns)
        sim = Simulator(self.shop, rule)
        result = sim.run()
        return {o.key: getattr(result, o.key, 0) for o in self.objs}

    def _fast_nondominated_sort(self, pop_objs):
        n = len(pop_objs)
        dom_count = [0] * n
        dom_set = [[] for _ in range(n)]
        fronts = [[]]
        for i in range(n):
            for j in range(i+1, n):
                if self._dominates(pop_objs[i], pop_objs[j]):
                    dom_set[i].append(j); dom_count[j] += 1
                elif self._dominates(pop_objs[j], pop_objs[i]):
                    dom_set[j].append(i); dom_count[i] += 1
        fronts[0] = [i for i in range(n) if dom_count[i] == 0]
        rank = [0] * n
        fi = 0
        while fronts[fi]:
            nxt = []
            for i in fronts[fi]:
                for j in dom_set[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        rank[j] = fi + 1
                        nxt.append(j)
            fi += 1
            fronts.append(nxt)
        return rank, fronts[:-1]

    def _dominates(self, a, b):
        any_better = False
        for o in self.objs:
            va, vb = a.get(o.key, 0), b.get(o.key, 0)
            if o.direction == "min":
                if va > vb: return False
                if va < vb: any_better = True
            else:
                if va < vb: return False
                if va > vb: any_better = True
        return any_better

    def _crowding_distance(self, indices, pop_objs):
        dist = {i: 0.0 for i in indices}
        for o in self.objs:
            sorted_idx = sorted(indices, key=lambda i: pop_objs[i].get(o.key, 0))
            dist[sorted_idx[0]] = float('inf')
            dist[sorted_idx[-1]] = float('inf')
            vmin = pop_objs[sorted_idx[0]].get(o.key, 0)
            vmax = pop_objs[sorted_idx[-1]].get(o.key, 0)
            rng = vmax - vmin if vmax != vmin else 1.0
            for k in range(1, len(sorted_idx)-1):
                vp = pop_objs[sorted_idx[k-1]].get(o.key, 0)
                vn = pop_objs[sorted_idx[k+1]].get(o.key, 0)
                dist[sorted_idx[k]] += (vn - vp) / rng
        return dist

    def _sbx_crossover(self, p1, p2, rng, eta=15):
        child = []
        for v1, v2 in zip(p1, p2):
            if rng.random() < 0.5:
                u = rng.random()
                if u < 0.5:
                    beta = (2*u) ** (1/(eta+1))
                else:
                    beta = (1/(2*(1-u))) ** (1/(eta+1))
                c = 0.5 * ((v1 + v2) - beta * abs(v2 - v1))
            else:
                c = v1
            child.append(c)
        return child

    def _polynomial_mutation(self, ind, rng, eta=20, prob=None):
        if prob is None: prob = 1.0 / self.K
        result = []
        for v in ind:
            if rng.random() < prob:
                u = rng.random()
                if u < 0.5:
                    delta = (2*u) ** (1/(eta+1)) - 1
                else:
                    delta = 1 - (2*(1-u)) ** (1/(eta+1))
                result.append(v + delta)
            else:
                result.append(v)
        return result

    def run(self, callback=None) -> list:
        rng = _random.Random(self.seed)

        # Initialize population
        pop_weights = [self._random_weights(rng) for _ in range(self.pop_size)]

        # Add identity weights for each individual rule as seeds
        for i in range(min(self.K, self.pop_size)):
            w = [0.0] * self.K
            w[i] = 1.0
            pop_weights[i] = w

        pop_objs = [self._evaluate(w) for w in pop_weights]

        total_evals = self.pop_size * (1 + self.generations)
        done_evals = self.pop_size

        for gen in range(self.generations):
            if callback:
                callback(done_evals, total_evals, gen)

            # Tournament selection + crossover + mutation -> offspring
            offspring_weights = []
            while len(offspring_weights) < self.pop_size:
                # Binary tournament
                i1, i2 = rng.sample(range(len(pop_weights)), 2)
                j1, j2 = rng.sample(range(len(pop_weights)), 2)

                rank, _ = self._fast_nondominated_sort(pop_objs)
                p1 = pop_weights[i1 if rank[i1] <= rank[i2] else i2]
                p2 = pop_weights[j1 if rank[j1] <= rank[j2] else j2]

                child = self._sbx_crossover(p1, p2, rng)
                child = self._polynomial_mutation(child, rng)
                offspring_weights.append(child)

            offspring_objs = [self._evaluate(w) for w in offspring_weights]
            done_evals += self.pop_size

            # Combine parent + offspring
            combined_w = pop_weights + offspring_weights
            combined_o = pop_objs + offspring_objs

            # Non-dominated sort
            rank, fronts = self._fast_nondominated_sort(combined_o)

            # Select next generation
            new_pop_w = []
            new_pop_o = []
            for front in fronts:
                if len(new_pop_w) + len(front) <= self.pop_size:
                    for i in front:
                        new_pop_w.append(combined_w[i])
                        new_pop_o.append(combined_o[i])
                else:
                    # Fill with crowding distance
                    needed = self.pop_size - len(new_pop_w)
                    cd = self._crowding_distance(front, combined_o)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    for i in sorted_front[:needed]:
                        new_pop_w.append(combined_w[i])
                        new_pop_o.append(combined_o[i])
                    break

            pop_weights = new_pop_w
            pop_objs = new_pop_o

        if callback:
            callback(total_evals, total_evals, self.generations)

        # Extract Pareto front
        rank, fronts = self._fast_nondominated_sort(pop_objs)
        results = []
        for i in range(len(pop_weights)):
            w = pop_weights[i]
            total = sum(abs(x) for x in w) or 1.0
            norm_w = [x/total for x in w]
            sol = ParetoSolution(
                rule_name=f"nsga2_{i}",
                objectives=pop_objs[i],
                rank=rank[i],
            )
            sol.weights = norm_w  # attach weights for display
            results.append(sol)

        return results
