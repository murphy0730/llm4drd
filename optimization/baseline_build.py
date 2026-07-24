"""夜间基准方案库构建管线（CLI 与 API 端点共用）。

流程（提案 §4.1–§4.4）：读当前实例 → 跑 HybridNSGA3ALNSOptimizer 三维多目标大规模优化
→ 从 Pareto 前沿抽取方案 → 逐个过质量门禁（同口径 ATC 对比）→ 组装行 → 落库（批次轮转）。

自增强循环（提案 §4.1）：夜间构建与白天优化共用同一个 _seed_population，因此夜间任务也
从「当前库里 active 方案」（上一夜的结果）热启动，在预算内对昨日方案做精细化，而非每次从零
冷探索。首夜库空自动退化为冷启动（图谱引导+11 规则+ATC），之后自动转热启动、自驱动抬升。
质量门禁始终以 ATC 为对照，故自增强不会绕过"稳定优于 ATC"的护栏。为让构建读到正确的库，
把传入的 db_path 显式接到 HybridConfig.baseline_db_path。
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime

from ..data.db import DB_PATH, BaselineSolutionStore, get_instance_version
from .baseline_extract import extract_baseline_solutions, passes_quality_gate
from .hybrid_nsga3_alns import HybridConfig, HybridNSGA3ALNSOptimizer
from .objectives import validate_objective_selection
from .solution_model import FEATURE_NAMES

logger = logging.getLogger(__name__)

# 提案 §2 已定的三维目标（延误率 min / 关键资源活跃窗口利用率 max / 平均流程时间 min）。
DEFAULT_OBJECTIVE_KEYS = [
    "main_order_tardy_ratio",
    "critical_active_window_utilization",
    "avg_flowtime",
]

# 夜间专用默认（提案决策 #5）。time_limit_s 是真正预算闸；generations 仅上限帽；
# stagnation_generations 放宽到 5，避免长任务过早早停。
NIGHTLY_TIME_LIMIT_S = 10800
NIGHTLY_GENERATIONS = 1440
NIGHTLY_STAGNATION_GENERATIONS = 5
NIGHTLY_COARSE_TIME_RATIO = 0.72


def _row_from_solution(emphasis, sol, optimizer, specs, baseline_compare, objective_keys, snapshot_version):
    candidate = sol.candidate
    return {
        "id": str(uuid.uuid4()),
        "emphasis": emphasis,
        "objective_keys": list(objective_keys),
        "feature_names": list(FEATURE_NAMES),
        "feature_weights": {name: float(candidate.feature_weights.get(name, 0.0)) for name in FEATURE_NAMES},
        "scale_json": {
            "time_scale": optimizer.time_scale,
            "due_scale": optimizer.due_scale,
            "priority_scale": optimizer.priority_scale,
            "busy_scale": optimizer.busy_scale,
        },
        "op_bias": {str(k): float(v) for k, v in candidate.op_bias.items()},
        "destroy_weights": {k: float(v) for k, v in candidate.destroy_weights.items()},
        "repair_weights": {k: float(v) for k, v in candidate.repair_weights.items()},
        "destroy_fraction": float(candidate.destroy_fraction),
        "reached_objectives": {spec.key: float(sol.objectives.get(spec.key, 0.0)) for spec in specs},
        "baseline_compare": baseline_compare,
        "snapshot_id": None,
        "snapshot_version": snapshot_version,
        "source": "learned",
    }


def build_baseline_library(
    shop,
    *,
    objective_keys=None,
    time_limit_s: int = NIGHTLY_TIME_LIMIT_S,
    generations: int = NIGHTLY_GENERATIONS,
    stagnation_generations: int = NIGHTLY_STAGNATION_GENERATIONS,
    coarse_time_ratio: float = NIGHTLY_COARSE_TIME_RATIO,
    seed: int = 42,
    graph_context=None,
    graph_context_mode: str = "legacy",
    store: BaselineSolutionStore | None = None,
    db_path: str = DB_PATH,
    progress_callback=None,
    persist: bool = True,
) -> dict:
    """跑一次夜间构建，返回摘要 dict：batch_id / rows / extracted / passed / result。"""
    objective_keys = list(objective_keys or DEFAULT_OBJECTIVE_KEYS)
    specs = validate_objective_selection(objective_keys)

    config = HybridConfig(
        objective_keys=objective_keys,
        time_limit_s=time_limit_s,
        generations=generations,
        stagnation_generations=stagnation_generations,
        coarse_time_ratio=coarse_time_ratio,
        seed=seed,
        # 自增强循环（提案 §4.1）：夜间也从当前库热启动；db_path 显式传入，避免读到默认库。
        baseline_seeds_enabled=True,
        baseline_db_path=db_path,
    )
    optimizer = HybridNSGA3ALNSOptimizer(shop, config, graph_context, graph_context_mode)
    result = optimizer.run(progress_callback)

    solutions = optimizer.archive.solutions()
    extracted = extract_baseline_solutions(solutions, specs)

    # ATC 基线与候选都过同一 approx_evaluator，保证门禁比较同口径。
    atc_sol = optimizer.approx_evaluator.evaluate(optimizer._default_candidate("ATC"), "baseline_gate", 0)
    snapshot_version = get_instance_version(db_path)

    rows: list[dict] = []
    for emphasis, sol in extracted:
        approx_sol = optimizer.approx_evaluator.evaluate(sol.candidate, "baseline_gate", 0)
        passed, compare = passes_quality_gate(approx_sol, atc_sol, specs, emphasis)
        if not passed:
            logger.info("基准方案 %s 未过质量门禁，丢弃", emphasis)
            continue
        rows.append(
            _row_from_solution(emphasis, sol, optimizer, specs, compare, objective_keys, snapshot_version)
        )

    batch_id = datetime.now().strftime("%Y%m%d%H%M%S")
    if persist and rows:
        target = store or BaselineSolutionStore(db_path)
        target.save_batch(batch_id, rows)

    return {
        "batch_id": batch_id,
        "rows": rows,
        "extracted": len(extracted),
        "passed": len(rows),
        "objective_keys": objective_keys,
        "result": result,
    }
