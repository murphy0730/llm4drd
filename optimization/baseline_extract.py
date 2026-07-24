"""从 Pareto 前沿抽取基准方案 + 质量门禁。

夜间冷搜索产出一批 OptimizationSolution（Pareto 近似前沿）。这里做两件事：
1. extract_baseline_solutions —— 自适应抽取"逐目标极端解 + 膝点均衡解 + 参考点簇代表"，
   数量随前沿形状变化（提案 §4.2）。
2. passes_quality_gate —— 每个候选落库前与 ATC 基线同口径比较，按 emphasis 分档判定
   （提案修订后 §4.3）：均衡/簇代表要多数目标不劣于 ATC；极端解只要强调目标显著更优、
   其余不灾难退化即可，避免"抽了极端解又被统一门禁拒掉"的矛盾。

抽取全程复用现成工具：objectives.objective_vector（方向归一，越小越好）、
nsga3_core.generate_reference_points / normalize_vectors / associate_to_reference。
"""
from __future__ import annotations

import math

from .nsga3_core import associate_to_reference, generate_reference_points, normalize_vectors
from .objectives import ObjectiveSpec, objective_vector
from .solution_model import OptimizationSolution

_EPS = 1e-9


def _dir_value(sol: OptimizationSolution, spec: ObjectiveSpec) -> float:
    """把该目标转成"越小越好"的标量，供极端解排序。"""
    value = float(sol.objectives.get(spec.key, 0.0))
    return value if spec.direction == "min" else -value


def _knee_solution(solutions: list[OptimizationSolution], specs: list[ObjectiveSpec]) -> OptimizationSolution:
    """膝点/均衡解：归一化目标空间里离理想点（原点，各维最优）最近者。"""
    vectors = [objective_vector(sol.objectives, specs) for sol in solutions]
    normalized = normalize_vectors(vectors)
    best_index = 0
    best_distance = float("inf")
    for index, norm in enumerate(normalized):
        distance = math.sqrt(sum(value * value for value in norm))
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return solutions[best_index]


def _cluster_representatives(
    solutions: list[OptimizationSolution],
    specs: list[ObjectiveSpec],
    target: int,
) -> list[OptimizationSolution]:
    """按 NSGA-III 参考点把前沿聚类，每个被占用的参考点取最近的解作代表。"""
    vectors = [objective_vector(sol.objectives, specs) for sol in solutions]
    normalized = normalize_vectors(vectors)
    references = generate_reference_points(len(specs), max(1, target))
    best_by_ref: dict[int, tuple[float, OptimizationSolution]] = {}
    for sol, norm in zip(solutions, normalized):
        index, distance = associate_to_reference(norm, references)
        current = best_by_ref.get(index)
        if current is None or distance < current[0]:
            best_by_ref[index] = (distance, sol)
    return [best_by_ref[index][1] for index in sorted(best_by_ref)]


def extract_baseline_solutions(
    solutions: list[OptimizationSolution],
    specs: list[ObjectiveSpec],
    cluster_target: int = 8,
) -> list[tuple[str, OptimizationSolution]]:
    """抽取 (emphasis, solution) 列表，按签名去重。

    极端解优先于膝点、膝点优先于簇代表——去重时先出现的、语义更明确的标签胜出。
    """
    feasible = [sol for sol in solutions if sol.feasible]
    if not feasible:
        return []

    ordered: list[tuple[str, OptimizationSolution]] = []
    # 1. 逐目标极端解
    for spec in specs:
        best = min(feasible, key=lambda sol, s=spec: _dir_value(sol, s))
        ordered.append((f"{spec.direction}_{spec.key}", best))
    # 2. 膝点均衡解
    ordered.append(("balanced", _knee_solution(feasible, specs)))
    # 3. 参考点簇代表
    for index, rep in enumerate(_cluster_representatives(feasible, specs, cluster_target)):
        ordered.append((f"cluster_{index}", rep))

    seen: set[str] = set()
    result: list[tuple[str, OptimizationSolution]] = []
    for emphasis, sol in ordered:
        signature = sol.candidate.signature()
        if signature in seen:
            continue
        seen.add(signature)
        result.append((emphasis, sol))
    return result


def _improvement_fraction(value: float, atc: float, direction: str) -> float:
    """相对 ATC 的改进比例（正 = 更优）。min 目标越小越好，max 目标越大越好。"""
    denom = max(abs(atc), _EPS)
    if direction == "min":
        return (atc - value) / denom
    return (value - atc) / denom


def _emphasized_key(emphasis: str, specs: list[ObjectiveSpec]) -> str | None:
    """从 "min_avg_flowtime" / "max_critical_..." 解析出被强调的目标键。"""
    for spec in specs:
        if emphasis == f"{spec.direction}_{spec.key}":
            return spec.key
    return None


def passes_quality_gate(
    sol: OptimizationSolution,
    atc_sol: OptimizationSolution,
    specs: list[ObjectiveSpec],
    emphasis: str,
    *,
    min_improve: float = 0.10,
    catastrophe_factor: float = 1.5,
    tolerance: float = 1e-6,
) -> tuple[bool, dict]:
    """按 emphasis 分档判定是否可入库，并返回相对 ATC 的逐目标对比证据。

    - 均衡/簇代表：多数目标（≥2/3）不劣于 ATC 且至少一个严格更优。
    - 极端解 min_*/max_*：强调目标改进 ≥ min_improve，其余目标不出现灾难退化
      （不劣于 ATC 的 catastrophe_factor 倍）。
    """
    compare: dict[str, dict] = {}
    for spec in specs:
        value = float(sol.objectives.get(spec.key, 0.0))
        atc = float(atc_sol.objectives.get(spec.key, 0.0))
        fraction = _improvement_fraction(value, atc, spec.direction)
        compare[spec.key] = {
            "value": value,
            "atc": atc,
            "direction": spec.direction,
            "improvement": fraction,
            "improved": fraction > tolerance,
            "not_worse": fraction >= -tolerance,
        }

    emphasized_key = _emphasized_key(emphasis, specs)
    if emphasized_key is not None:
        # 极端解档
        target = compare[emphasized_key]
        passed = target["improvement"] >= min_improve
        for spec in specs:
            if spec.key == emphasized_key:
                continue
            info = compare[spec.key]
            # 灾难退化：min 目标涨过 ATC*factor，或 max 目标跌破 ATC/factor。
            if spec.direction == "min":
                catastrophic = info["value"] > info["atc"] * catastrophe_factor + tolerance
            else:
                catastrophic = info["value"] < info["atc"] / catastrophe_factor - tolerance
            if catastrophic:
                passed = False
                break
        verdict = {"tier": "extreme", "emphasized_key": emphasized_key}
    else:
        # 均衡 / 簇代表档
        not_worse_count = sum(1 for info in compare.values() if info["not_worse"])
        strictly_better = any(info["improved"] for info in compare.values())
        majority = math.ceil(len(specs) * 2 / 3)
        passed = not_worse_count >= majority and strictly_better
        verdict = {
            "tier": "balanced",
            "not_worse_count": not_worse_count,
            "majority_threshold": majority,
            "strictly_better": strictly_better,
        }

    return passed, {"objectives": compare, "verdict": verdict}
