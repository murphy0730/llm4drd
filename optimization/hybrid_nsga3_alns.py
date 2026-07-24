from __future__ import annotations

import logging
import math
import os
import pickle
import random
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable

from ..core.rules import BUILTIN_RULES
from ..core.sim_runtime import SimulationRuntimePool
from ..core.simulator import Simulator
from ..knowledge.context import GraphContext, compare_legacy_context
from ..knowledge.graph import HeterogeneousGraph
from .alns_core import ALNSCore
from .approx_eval import ApproximateScheduleEvaluator
from .archive import ParetoArchive
from .nsga3_core import fast_nondominated_sort, select_survivors
from .objectives import ScheduleAnalytics, build_schedule_analytics, objective_summary_payload, validate_objective_selection
from .parallel_eval import (
    build_candidate_rule,
    init_worker,
    run_approx_evaluation,
    run_exact_simulation,
)
from .solution_model import (
    DESTROY_OPERATORS,
    FEATURE_NAMES,
    REPAIR_OPERATORS,
    CandidateParameters,
    OptimizationSolution,
    schedule_signature,
)


logger = logging.getLogger(__name__)


class _ProcessBackendUnavailable(Exception):
    """进程池基础设施故障（起不来/连接断）；不含任务自身的异常。"""

    def __init__(self, cause: BaseException):
        super().__init__(str(cause))
        self.cause = cause


RULE_TEMPLATES: dict[str, dict[str, float]] = {
    "EDD": {"due_date": 2.8, "slack": 1.8, "urgency": 1.0},
    "SPT": {"processing_time": 2.8, "remaining": 1.0},
    "LPT": {"processing_time": -2.4, "remaining": -0.6},
    "CR": {"slack": 2.2, "urgency": 1.5, "due_date": 1.0},
    "ATC": {"urgency": 2.4, "slack": 1.4, "processing_time": 0.8, "due_date": 0.8},
    "FIFO": {"wait_time": 2.6},
    "MST": {"slack": 2.0, "urgency": 1.1, "remaining": 0.9},
    "PRIORITY": {"priority": 2.4, "is_main": 1.8, "assembly_criticality": 1.2},
    "KIT_AWARE": {"prereq_ratio": 1.8, "assembly_criticality": 1.5, "shared_resource_degree": 1.0},
    "BOTTLENECK": {"bottleneck_adjacency": 2.2, "is_main": 1.5, "urgency": 1.0},
    "COMPOSITE": {
        "urgency": 1.5,
        "slack": 1.2,
        "processing_time": 0.7,
        "priority": 1.1,
        "assembly_criticality": 1.2,
        "shared_resource_degree": 0.9,
        "bottleneck_adjacency": 0.8,
    },
}


@dataclass
class HybridConfig:
    objective_keys: list[str]
    target_solution_count: int = 12
    time_limit_s: int = 90
    population_size: int = 24
    generations: int = 12
    alns_iterations_per_candidate: int = 6
    candidate_filter_multiplier: int = 3
    coarse_pool_multiplier: int = 4
    elite_refine_ratio: float = 0.4
    elite_refine_min: int = 4
    coarse_time_ratio: float = 0.68
    promotion_pool_multiplier: int = 3
    random_promotion_ratio: float = 0.12
    refine_rounds: int = 1
    alns_aggression: float = 1.0
    stagnation_generations: int = 3
    parallel_workers: int = 0
    seed: int = 42
    baseline_rule_name: str = "ATC"
    # "process": 批量评估用进程池绕开 GIL；"thread": 沿用线程池。
    parallel_backend: str = "process"
    # 白天热启动：从基准方案库追加 active 方案作为种子。库空/加载失败自动退化为
    # 现有行为（图谱引导 + 11 规则 + ATC），见 _load_baseline_seeds / _seed_population。
    baseline_seeds_enabled: bool = True
    baseline_db_path: str | None = None  # None → 用 data.db 默认 DB_PATH
    baseline_scale_tolerance: float = 0.20  # 尺度校验阈值：偏差超此比例则跳过该方案


@dataclass
class HybridResult:
    objective_keys: list[str]
    baseline: dict
    solutions: list[dict]
    archive_size: int
    requested_solution_count: int
    found_solution_count: int
    coarse_pool_size: int
    promoted_solution_count: int
    refined_solution_count: int
    generations_completed: int
    total_evaluations: int
    approximate_evaluations: int
    exact_evaluations: int
    elapsed_s: float
    parallel_workers: dict[str, int] = field(default_factory=dict)
    hypervolume_history: list[dict] = field(default_factory=list)
    status_history: list[dict] = field(default_factory=list)
    baseline_export: dict = field(default_factory=dict, repr=False)
    solutions_export: list[dict] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "objective_keys": self.objective_keys,
            "objective_catalog": objective_summary_payload(),
            "baseline": self.baseline,
            "solutions": self.solutions,
            "archive_size": self.archive_size,
            "requested_solution_count": self.requested_solution_count,
            "found_solution_count": self.found_solution_count,
            "coarse_pool_size": self.coarse_pool_size,
            "promoted_solution_count": self.promoted_solution_count,
            "refined_solution_count": self.refined_solution_count,
            "generations_completed": self.generations_completed,
            "total_evaluations": self.total_evaluations,
            "approximate_evaluations": self.approximate_evaluations,
            "exact_evaluations": self.exact_evaluations,
            "elapsed_s": round(self.elapsed_s, 2),
            "parallel_workers": self.parallel_workers,
            "hypervolume_history": self.hypervolume_history,
            "status_history": self.status_history,
        }

    def to_export_dict(self) -> dict:
        return {
            **self.to_dict(),
            "baseline": self.baseline_export or self.baseline,
            "solutions": self.solutions_export or self.solutions,
        }


def build_legacy_graph_features(shop, graph) -> dict[str, dict[str, float]]:
    task_predecessors: dict[str, list[str]] = {
        task_id: list(task.predecessor_task_ids) for task_id, task in shop.tasks.items()
    }
    main_ancestors: set[str] = set()
    for order in shop.orders.values():
        if not order.main_task_id:
            continue
        stack = [order.main_task_id]
        while stack:
            current = stack.pop()
            if current in main_ancestors:
                continue
            main_ancestors.add(current)
            stack.extend(task_predecessors.get(current, []))

    op_predecessors: dict[str, set[str]] = {
        op_id: set(op.predecessor_ops) for op_id, op in shop.operations.items()
    }
    for op_id, op in shop.operations.items():
        for predecessor_task_id in op.predecessor_tasks:
            predecessor_task = shop.tasks.get(predecessor_task_id)
            if predecessor_task:
                for predecessor_op in predecessor_task.operations:
                    op_predecessors[op_id].add(predecessor_op.id)

    depth_cache: dict[str, float] = {}

    def predecessor_depth(op_id: str, trail: set[str] | None = None) -> float:
        if op_id in depth_cache:
            return depth_cache[op_id]
        trail = trail or set()
        if op_id in trail:
            return 0.0
        predecessors = [
            pid for pid in op_predecessors.get(op_id, set()) if pid in shop.operations
        ]
        if not predecessors:
            depth_cache[op_id] = 0.0
            return 0.0
        trail = set(trail)
        trail.add(op_id)
        depth_cache[op_id] = 1.0 + max(
            predecessor_depth(pid, trail) for pid in predecessors
        )
        return depth_cache[op_id]

    features: dict[str, dict[str, float]] = {}
    for op_id, op in shop.operations.items():
        task = shop.tasks[op.task_id]
        order = shop.orders[task.order_id]
        eligible_count = max(1, len(shop.get_eligible_machines(op)))
        tooling_scarcity = sum(
            1.0 / max(1, len(shop.get_toolings_for_type(tooling_type)))
            for tooling_type in op.required_tooling_types
        )
        personnel_scarcity = sum(
            1.0 / max(1, len(shop.get_personnel_for_skill(skill_id)))
            for skill_id in op.required_personnel_skills
        )
        shared_degree = (
            1.0 / eligible_count
            + tooling_scarcity
            + personnel_scarcity
            + 0.25 * len(op.required_tooling_types)
            + 0.25 * len(op.required_personnel_skills)
        )
        critical_hits = sum(
            1
            for machine in shop.get_eligible_machines(op)
            if shop.machine_types.get(machine.type_id)
            and shop.machine_types[machine.type_id].is_critical
        )
        machine_critical_ratio = critical_hits / eligible_count
        graph_node = f"OP:{op_id}"
        out_degree = graph.out_degree(graph_node) if graph.has_node(graph_node) else 0

        if task.is_main:
            assembly_criticality = 1.0
        elif task.id in main_ancestors:
            assembly_criticality = 0.78
        elif order.main_task_id:
            assembly_criticality = 0.32
        else:
            assembly_criticality = 0.18

        features[op_id] = {
            "predecessor_depth": predecessor_depth(op_id)
            / max(1.0, len(shop.operations)),
            "assembly_criticality": assembly_criticality,
            "shared_resource_degree": min(3.0, shared_degree),
            "bottleneck_adjacency": min(
                1.0,
                0.55 * machine_critical_ratio + 0.45 * (1.0 / eligible_count),
            ),
            "graph_out_degree": min(
                1.0, out_degree / max(1, graph.number_of_nodes())
            ),
        }
    return features


class HybridNSGA3ALNSOptimizer:
    def __init__(
        self,
        shop,
        config: HybridConfig,
        graph_context: GraphContext | None = None,
        graph_context_mode="legacy",
    ):
        self.shop = shop
        self.config = config
        self.graph_context_mode = getattr(
            graph_context_mode, "value", graph_context_mode
        )
        if self.graph_context_mode not in {"legacy", "shadow", "active"}:
            raise ValueError(f"invalid graph context mode: {self.graph_context_mode}")
        if self.graph_context_mode in {"shadow", "active"} and graph_context is None:
            raise ValueError(
                f"GraphContext is required for {self.graph_context_mode} mode"
            )
        self.shop.build_indexes()
        self.graph_context = graph_context
        self.graph_context_diff = None
        self.operation_order_rank = {
            op_id: rank for rank, op_id in enumerate(self.shop.operations)
        }
        self._direct_successors: dict[str, list[str]] = {
            op_id: [] for op_id in self.shop.operations
        }
        for successor_id, operation in self.shop.operations.items():
            for predecessor_id in dict.fromkeys(operation.predecessor_ops):
                if predecessor_id in self._direct_successors:
                    self._direct_successors[predecessor_id].append(successor_id)
        self.specs = validate_objective_selection(config.objective_keys)
        self.rng = random.Random(config.seed)
        self.archive = ParetoArchive(self.specs)
        if self.graph_context_mode == "active":
            self.graph_model = None
            self.graph = None
        else:
            self.graph_model = HeterogeneousGraph()
            self.graph_model.build_from_shopfloor(self.shop)
            self.graph = self.graph_model.graph
        self.exact_cache: dict[str, OptimizationSolution] = {}
        self.approx_cache: dict[str, OptimizationSolution] = {}
        self.solution_pool: dict[str, OptimizationSolution] = {}
        self.coarse_solution_pool: dict[str, OptimizationSolution] = {}
        self.total_evaluations = 0
        self.approximate_evaluations = 0
        self.exact_evaluations = 0
        self.exact_eval_time_total = 0.0
        self.time_started = 0.0
        self.hypervolume_history: list[dict] = []
        self.status_history: list[dict] = []
        self.baseline_solution: OptimizationSolution | None = None
        self.coarse_baseline_solution: OptimizationSolution | None = None
        self.cache_lock = Lock()
        self.parallel_workers = self._resolve_parallel_workers()
        self.approx_parallel_workers = self._phase_parallel_workers("approx")
        self.exact_parallel_workers = self._phase_parallel_workers("exact")
        self.refine_parallel_workers = self._phase_parallel_workers("refine")
        self._process_pool = None
        self._process_backend_failed = False
        self._payload_path: str | None = None
        # 池按各阶段的最大并发建一次：懒建的池会被首个批次的 worker 数永久定容，
        # 之后阶段要求更高并发时也拿不到。
        self._max_pool_workers = max(
            self.parallel_workers,
            self.approx_parallel_workers,
            self.exact_parallel_workers,
            self.refine_parallel_workers,
        )
        # 每个并发 worker 一个互不共享的 runtime；静态数据只构建一次。
        self._runtime_pool = SimulationRuntimePool(self.shop, self._max_pool_workers)

        processing_times = [op.processing_time for op in self.shop.operations.values()]
        due_dates = [task.due_date for task in self.shop.tasks.values()]
        priorities = [order.priority for order in self.shop.orders.values()]
        self.time_scale = max(1.0, sum(processing_times) / len(processing_times)) if processing_times else 1.0
        self.due_scale = max(due_dates, default=self.time_scale)
        self.priority_scale = max(priorities, default=1)
        self.busy_scale = max(self.time_scale * max(1, len(self.shop.machines)), 1.0)
        if self.graph_context_mode == "active":
            context_features = graph_context.feature_view_by_operation_id()
            self.graph_features = {
                op_id: context_features[op_id] for op_id in self.shop.operations
            }
        else:
            self.graph_features = self._build_graph_features()
        if self.graph_context_mode == "shadow":
            self.graph_context_diff = compare_legacy_context(
                self.shop, graph_context
            )
        self._graph_guides = (
            None
            if self.graph_context_mode == "active"
            else self._build_graph_guides()
        )
        self.approx_evaluator = ApproximateScheduleEvaluator(
            self.shop,
            self.graph_features,
            self.time_scale,
            self.due_scale,
            self.priority_scale,
            keep_schedule_limit=0,
        )

    @property
    def graph_guides(self) -> dict[str, dict]:
        if self._graph_guides is None:
            self._graph_guides = self._build_graph_guides()
        return self._graph_guides

    def _resolve_parallel_workers(self) -> int:
        if self.config.parallel_workers and self.config.parallel_workers > 0:
            return max(1, self.config.parallel_workers)

        cpu_total = max(1, os.cpu_count() or 1)
        if cpu_total <= 2:
            return 1

        reserve = 1 if cpu_total <= 6 else 2
        safe_cpu = max(1, cpu_total - reserve)
        if cpu_total >= 12:
            safe_cpu = max(2, min(safe_cpu, math.ceil(cpu_total * 0.7)))

        op_count = len(self.shop.operations)
        if op_count < 180:
            problem_cap = 2
        elif op_count < 900:
            problem_cap = 4
        elif op_count < 4000:
            problem_cap = 6
        else:
            problem_cap = 8

        budget = max(1, self.config.time_limit_s)
        if budget <= 20:
            budget_cap = 2
        elif budget <= 60:
            budget_cap = 4
        elif budget <= 180:
            budget_cap = 6
        else:
            budget_cap = 8

        return max(1, min(safe_cpu, problem_cap, budget_cap))

    def _phase_parallel_workers(self, phase: str) -> int:
        base = self.parallel_workers
        op_count = len(self.shop.operations)
        if phase == "approx":
            if op_count < 250:
                return 1
            return max(1, min(base, 4 if op_count < 1500 else base))
        if phase == "refine":
            return max(1, min(base, 6 if op_count < 3000 else base))
        return base

    def _worker_count_for_batch(self, phase: str, batch_size: int) -> int:
        if batch_size <= 1:
            return 1
        if phase == "approx":
            return min(batch_size, self.approx_parallel_workers)
        if phase == "refine":
            return min(batch_size, self.refine_parallel_workers)
        return min(batch_size, self.exact_parallel_workers)

    def _build_graph_features(self) -> dict[str, dict[str, float]]:
        return build_legacy_graph_features(self.shop, self.graph)

    def _legacy_related_operations(self, current_id: str) -> list[str]:
        op = self.shop.operations[current_id]
        task = self.shop.tasks[op.task_id]
        related = list(op.predecessor_ops)
        related.extend(
            other.id for other in task.operations if other.id != current_id
        )
        for other_id, other in self.shop.operations.items():
            if other_id == current_id:
                continue
            if other.process_type == op.process_type:
                related.append(other_id)
            if set(other.required_tooling_types) & set(op.required_tooling_types):
                related.append(other_id)
            if set(other.required_personnel_skills) & set(
                op.required_personnel_skills
            ):
                related.append(other_id)
        node_id = f"OP:{current_id}"
        if self.graph.has_node(node_id):
            for predecessor in self.graph.predecessors(node_id):
                if predecessor.startswith("OP:"):
                    related.append(predecessor[3:])
            for successor in self.graph.successors(node_id):
                if successor.startswith("OP:"):
                    related.append(successor[3:])
        return related

    def _active_related_operations(self, current_id: str) -> list[str]:
        op = self.shop.operations[current_id]
        task = self.shop.tasks[op.task_id]
        related = list(op.predecessor_ops)
        related.extend(
            other.id for other in task.operations if other.id != current_id
        )

        group_members = set(
            self.graph_context.operations_in_group("process_type", op.process_type)
        )
        for tooling_type in op.required_tooling_types:
            group_members.update(
                self.graph_context.operations_in_group("tooling_type", tooling_type)
            )
        for skill_id in op.required_personnel_skills:
            group_members.update(
                self.graph_context.operations_in_group("personnel_skill", skill_id)
            )
        tooling_types = set(op.required_tooling_types)
        personnel_skills = set(op.required_personnel_skills)
        for other_id in sorted(
            group_members, key=self.operation_order_rank.__getitem__
        ):
            if other_id == current_id:
                continue
            other = self.shop.operations[other_id]
            if other.process_type == op.process_type:
                related.append(other_id)
            if set(other.required_tooling_types) & tooling_types:
                related.append(other_id)
            if set(other.required_personnel_skills) & personnel_skills:
                related.append(other_id)

        related.extend(op.predecessor_ops)
        related.extend(self._direct_successors[current_id])
        return related

    def _expand_operation_cluster(self, seed_ops: list[str], max_size: int) -> list[str]:
        cluster: list[str] = []
        seen: set[str] = set()
        queue = [op_id for op_id in seed_ops if op_id in self.shop.operations]

        while queue and len(cluster) < max_size:
            current_id = queue.pop(0)
            if current_id in seen or current_id not in self.shop.operations:
                continue
            seen.add(current_id)
            cluster.append(current_id)
            if self.graph_context_mode == "active":
                related = self._active_related_operations(current_id)
            else:
                related = self._legacy_related_operations(current_id)
            for op_id in related:
                if len(cluster) >= max_size:
                    break
                if op_id not in seen:
                    seen.add(op_id)
                    cluster.append(op_id)
                    queue.append(op_id)

        return cluster[:max_size]

    def _top_operations(self, score_fn, limit: int) -> list[str]:
        ranked = sorted(
            self.shop.operations,
            key=score_fn,
            reverse=True,
        )
        return ranked[: max(1, min(limit, len(ranked)))]

    def _build_graph_guides(self) -> dict[str, dict]:
        op_count = max(1, len(self.shop.operations))
        cluster_size = max(8, min(op_count, math.ceil(op_count * 0.22)))

        due_focus = sorted(
            self.shop.operations,
            key=lambda op_id: (
                self.shop.tasks[self.shop.operations[op_id].task_id].due_date,
                -self.shop.orders[self.shop.tasks[self.shop.operations[op_id].task_id].order_id].priority,
            ),
        )
        assembly_ops = self._top_operations(
            lambda op_id: (
                self.graph_features[op_id]["predecessor_depth"] * 1.3
                + self.graph_features[op_id]["assembly_criticality"] * 1.8
            ),
            cluster_size,
        )
        bottleneck_ops = self._top_operations(
            lambda op_id: (
                self.graph_features[op_id]["bottleneck_adjacency"] * 1.8
                + self.graph_features[op_id]["predecessor_depth"] * 0.6
            ),
            cluster_size,
        )
        shared_ops = self._top_operations(
            lambda op_id: (
                self.graph_features[op_id]["shared_resource_degree"] * 1.9
                + len(self.shop.operations[op_id].required_tooling_types) * 0.4
                + len(self.shop.operations[op_id].required_personnel_skills) * 0.4
            ),
            cluster_size,
        )
        due_ops = due_focus[:cluster_size]
        balanced_seed = list(dict.fromkeys(assembly_ops[: cluster_size // 2] + bottleneck_ops[: cluster_size // 2] + shared_ops[: cluster_size // 2]))

        return {
            "balanced": {
                "seed_rule": "COMPOSITE",
                "allowed_ops": self._expand_operation_cluster(balanced_seed, cluster_size),
                "feature_targets": {
                    "urgency": 1.4,
                    "slack": 1.1,
                    "assembly_criticality": 1.2,
                    "bottleneck_adjacency": 1.0,
                    "shared_resource_degree": 0.8,
                    "machine_load": 0.8,
                },
                "destroy_targets": {
                    "assembly_chain_destroy": 1.25,
                    "bottleneck_machine_destroy": 1.15,
                    "shared_tooling_destroy": 1.1,
                },
                "repair_targets": {
                    "assembly_sync_repair": 1.2,
                    "bottleneck_smoothing_repair": 1.1,
                    "shared_resource_repair": 1.05,
                },
                "destroy_fraction": 0.16,
            },
            "assembly_focus": {
                "seed_rule": "PRIORITY",
                "allowed_ops": self._expand_operation_cluster(assembly_ops, cluster_size),
                "feature_targets": {
                    "assembly_criticality": 2.1,
                    "predecessor_depth": 1.2,
                    "is_main": 1.4,
                    "priority": 1.1,
                    "prereq_ratio": 0.9,
                },
                "destroy_targets": {
                    "assembly_chain_destroy": 1.6,
                    "critical_predecessor_destroy": 1.2,
                },
                "repair_targets": {
                    "main_order_repair": 1.4,
                    "assembly_sync_repair": 1.3,
                },
                "destroy_fraction": 0.18,
            },
            "bottleneck_focus": {
                "seed_rule": "BOTTLENECK",
                "allowed_ops": self._expand_operation_cluster(bottleneck_ops, cluster_size),
                "feature_targets": {
                    "bottleneck_adjacency": 2.0,
                    "machine_load": 1.4,
                    "processing_time": 0.7,
                    "urgency": 0.8,
                    "shared_resource_degree": 0.6,
                },
                "destroy_targets": {
                    "bottleneck_machine_destroy": 1.6,
                    "critical_predecessor_destroy": 1.05,
                },
                "repair_targets": {
                    "bottleneck_smoothing_repair": 1.45,
                    "due_date_repair": 1.05,
                },
                "destroy_fraction": 0.2,
            },
            "shared_resource_focus": {
                "seed_rule": "KIT_AWARE",
                "allowed_ops": self._expand_operation_cluster(shared_ops, cluster_size),
                "feature_targets": {
                    "shared_resource_degree": 1.8,
                    "tooling_demand": 0.8,
                    "personnel_demand": 0.8,
                    "urgency": 0.9,
                    "slack": 0.7,
                },
                "destroy_targets": {
                    "shared_tooling_destroy": 1.45,
                    "shared_personnel_destroy": 1.45,
                },
                "repair_targets": {
                    "shared_resource_repair": 1.45,
                    "assembly_sync_repair": 1.0,
                },
                "destroy_fraction": 0.22,
            },
            "due_focus": {
                "seed_rule": "EDD",
                "allowed_ops": self._expand_operation_cluster(due_ops, cluster_size),
                "feature_targets": {
                    "due_date": 2.0,
                    "urgency": 1.3,
                    "slack": 1.4,
                    "priority": 0.8,
                    "assembly_criticality": 0.8,
                },
                "destroy_targets": {
                    "tardy_order_destroy": 1.5,
                    "assembly_chain_destroy": 1.1,
                },
                "repair_targets": {
                    "due_date_repair": 1.5,
                    "main_order_repair": 1.05,
                },
                "destroy_fraction": 0.16,
            },
        }

    def _project_candidate_to_graph_space(
        self,
        candidate: CandidateParameters,
        graph_profile: str | None = None,
        blend: float = 0.38,
    ) -> CandidateParameters:
        profile_name = graph_profile or candidate.graph_profile or "balanced"
        guide = self.graph_guides.get(profile_name) or self.graph_guides["balanced"]
        candidate.graph_profile = profile_name

        for name, target in guide["feature_targets"].items():
            current = candidate.feature_weights.get(name, 0.0)
            candidate.feature_weights[name] = (1.0 - blend) * current + blend * target

        for name, target in guide["destroy_targets"].items():
            current = candidate.destroy_weights.get(name, 1.0)
            candidate.destroy_weights[name] = max(0.05, (1.0 - blend) * current + blend * target)
        for name, target in guide["repair_targets"].items():
            current = candidate.repair_weights.get(name, 1.0)
            candidate.repair_weights[name] = max(0.05, (1.0 - blend) * current + blend * target)

        candidate.destroy_fraction = min(
            0.4,
            max(
                0.05,
                (1.0 - blend) * candidate.destroy_fraction + blend * guide.get("destroy_fraction", candidate.destroy_fraction),
            ),
        )

        allowed_ops = set(guide.get("allowed_ops", []))
        candidate.op_bias = {op_id: bias for op_id, bias in candidate.op_bias.items() if op_id in allowed_ops}
        if allowed_ops:
            ranked_ops = list(guide["allowed_ops"])[: min(18, len(guide["allowed_ops"]))]
            for rank, op_id in enumerate(ranked_ops):
                default_bias = 0.42 - 0.02 * rank
                if abs(default_bias) <= 1e-6:
                    continue
                current = candidate.op_bias.get(op_id, 0.0)
                candidate.op_bias[op_id] = current + default_bias * blend

        candidate.prune_bias()
        return candidate

    def _graph_alignment_score(self, candidate: CandidateParameters) -> float:
        guide = self.graph_guides.get(candidate.graph_profile or "balanced") or self.graph_guides["balanced"]
        feature_alignment = sum(
            candidate.feature_weights.get(name, 0.0) * target
            for name, target in guide["feature_targets"].items()
        )
        destroy_alignment = sum(
            candidate.destroy_weights.get(name, 1.0) * target
            for name, target in guide["destroy_targets"].items()
        )
        repair_alignment = sum(
            candidate.repair_weights.get(name, 1.0) * target
            for name, target in guide["repair_targets"].items()
        )
        allowed_ops = set(guide.get("allowed_ops", []))
        bias_overlap = sum(1 for op_id in candidate.op_bias if op_id in allowed_ops)
        graph_focus = (
            abs(candidate.feature_weights.get("assembly_criticality", 0.0))
            + abs(candidate.feature_weights.get("predecessor_depth", 0.0))
            + abs(candidate.feature_weights.get("shared_resource_degree", 0.0))
            + abs(candidate.feature_weights.get("bottleneck_adjacency", 0.0))
        )
        return feature_alignment + 0.2 * destroy_alignment + 0.2 * repair_alignment + 0.18 * bias_overlap + 0.35 * graph_focus

    def _default_candidate(self, rule_name: str | None = None) -> CandidateParameters:
        feature_weights = {name: 0.0 for name in FEATURE_NAMES}
        for key, value in RULE_TEMPLATES.get(rule_name or "", {}).items():
            if key in feature_weights:
                feature_weights[key] = value
        destroy_weights = {name: 1.0 for name in DESTROY_OPERATORS}
        repair_weights = {name: 1.0 for name in REPAIR_OPERATORS}
        return CandidateParameters(
            feature_weights=feature_weights,
            destroy_weights=destroy_weights,
            repair_weights=repair_weights,
            seed_rule_name=rule_name,
        )

    def _candidate_from_guide(self, profile_name: str, base_rule_name: str | None = None, intensity: float = 0.55) -> CandidateParameters:
        guide = self.graph_guides.get(profile_name) or self.graph_guides["balanced"]
        candidate = self._default_candidate(base_rule_name or guide.get("seed_rule"))
        candidate.graph_profile = profile_name
        for name, target in guide["feature_targets"].items():
            candidate.feature_weights[name] = candidate.feature_weights.get(name, 0.0) + intensity * target
        for name, target in guide["destroy_targets"].items():
            candidate.destroy_weights[name] = max(0.05, candidate.destroy_weights.get(name, 1.0) + 0.35 * intensity * target)
        for name, target in guide["repair_targets"].items():
            candidate.repair_weights[name] = max(0.05, candidate.repair_weights.get(name, 1.0) + 0.35 * intensity * target)
        candidate.destroy_fraction = guide.get("destroy_fraction", candidate.destroy_fraction)
        return self._project_candidate_to_graph_space(candidate, profile_name, blend=0.55)

    def _mutate_candidate(self, candidate: CandidateParameters, scale: float = 0.45) -> CandidateParameters:
        child = candidate.clone()
        for name in FEATURE_NAMES:
            if self.rng.random() < 0.45:
                child.feature_weights[name] += self.rng.gauss(0.0, scale)
        for name in DESTROY_OPERATORS:
            child.destroy_weights[name] = max(0.05, child.destroy_weights.get(name, 1.0) + self.rng.gauss(0.0, 0.22))
        for name in REPAIR_OPERATORS:
            child.repair_weights[name] = max(0.05, child.repair_weights.get(name, 1.0) + self.rng.gauss(0.0, 0.22))
        child.destroy_fraction = min(0.4, max(0.05, child.destroy_fraction + self.rng.gauss(0.0, 0.03)))
        chosen_profile = child.graph_profile or self.rng.choice(list(self.graph_guides))
        if self.rng.random() < 0.18:
            chosen_profile = self.rng.choice(list(self.graph_guides))
        return self._project_candidate_to_graph_space(child, chosen_profile, blend=0.34)

    def _crossover(self, left: CandidateParameters, right: CandidateParameters) -> CandidateParameters:
        child = self._default_candidate(self.rng.choice([left.seed_rule_name, right.seed_rule_name]))
        child.graph_profile = self.rng.choice([left.graph_profile or "balanced", right.graph_profile or "balanced"])
        for name in FEATURE_NAMES:
            midpoint = 0.5 * (left.feature_weights.get(name, 0.0) + right.feature_weights.get(name, 0.0))
            spread = abs(left.feature_weights.get(name, 0.0) - right.feature_weights.get(name, 0.0))
            child.feature_weights[name] = midpoint + self.rng.gauss(0.0, 0.18 + 0.15 * spread)
        for name in DESTROY_OPERATORS:
            child.destroy_weights[name] = max(
                0.05,
                0.5 * (left.destroy_weights.get(name, 1.0) + right.destroy_weights.get(name, 1.0)) + self.rng.gauss(0.0, 0.08),
            )
        for name in REPAIR_OPERATORS:
            child.repair_weights[name] = max(
                0.05,
                0.5 * (left.repair_weights.get(name, 1.0) + right.repair_weights.get(name, 1.0)) + self.rng.gauss(0.0, 0.08),
            )
        child.destroy_fraction = min(0.4, max(0.05, 0.5 * (left.destroy_fraction + right.destroy_fraction)))

        merged_bias = {}
        for source in (left.op_bias, right.op_bias):
            for op_id, bias in source.items():
                if self.rng.random() < 0.5:
                    merged_bias[op_id] = merged_bias.get(op_id, 0.0) + 0.5 * bias
        child.op_bias = merged_bias
        return self._project_candidate_to_graph_space(child, child.graph_profile, blend=0.32)

    def _dispatch_rule(self, candidate: CandidateParameters):
        return build_candidate_rule(
            candidate, self.graph_features, self.time_scale,
            self.busy_scale, self.priority_scale, self.due_scale,
        )

    def _enrich_schedule(self, schedule: list[dict]) -> list[dict]:
        payload: list[dict] = []
        for entry in schedule:
            item = dict(entry)
            task = self.shop.tasks.get(entry.get("task_id", ""))
            order = self.shop.orders.get(task.order_id) if task else None
            item["start_at"] = self.shop.time_label(entry.get("start"))
            item["end_at"] = self.shop.time_label(entry.get("end"))
            item["order_id"] = order.id if order else None
            item["order_name"] = order.name if order else None
            item["priority"] = order.priority if order else None
            item["due_date"] = task.due_date if task else None
            item["due_at"] = self.shop.time_label(task.due_date) if task else None
            item["is_main"] = bool(task.is_main) if task else False
            payload.append(item)
        return payload

    def _make_solution(
        self,
        candidate: CandidateParameters,
        source: str,
        generation: int,
        schedule: list[dict],
        analytics: ScheduleAnalytics,
        metrics: dict[str, float],
    ) -> OptimizationSolution:
        signature = schedule_signature(schedule)
        solution_id = f"S-{signature[:10]}"
        analytics_summary = {
            "tardy_order_ids": list(analytics.tardy_order_ids),
            "tardy_task_ids": list(analytics.tardy_task_ids),
            "bottleneck_machine_ids": list(analytics.bottleneck_machine_ids),
            "order_tardiness": dict(analytics.order_tardiness),
            "task_completion": dict(analytics.task_completion),
            "order_main_gap": dict(analytics.order_main_gap),
            "machine_utilization": dict(analytics.machine_utilization),
            "tooling_utilization": dict(analytics.tooling_utilization),
            "personnel_utilization": dict(analytics.personnel_utilization),
            "completed_operations": analytics.completed_operations,
            "total_operations": len(self.shop.operations),
        }
        objectives = {spec.key: analytics.objective_values.get(spec.key, 0.0) for spec in self.specs}
        return OptimizationSolution(
            solution_id=solution_id,
            source=source,
            generation=generation,
            candidate=candidate.clone(),
            objectives=objectives,
            metrics=metrics,
            schedule=schedule,
            feasible=analytics.feasible,
            schedule_signature=signature,
            analytics_summary=analytics_summary,
        )

    def _clone_solution(self, solution: OptimizationSolution, source: str, generation: int) -> OptimizationSolution:
        cloned = solution.clone()
        cloned.source = source
        cloned.generation = generation
        return cloned

    def _simulate_candidate(self, candidate: CandidateParameters, source: str, generation: int) -> OptimizationSolution:
        runtime = self._runtime_pool.acquire()
        try:
            simulator = Simulator(self.shop, self._dispatch_rule(candidate), runtime=runtime)
            sim_result = simulator.run()
        finally:
            self._runtime_pool.release(runtime)
        return self._solution_from_sim_result(candidate, sim_result, source, generation)

    def _solution_from_sim_result(self, candidate: CandidateParameters, sim_result, source: str, generation: int) -> OptimizationSolution:
        schedule = self._enrich_schedule(sim_result.schedule)
        analytics = build_schedule_analytics(self.shop, sim_result)
        metrics = sim_result.to_dict()
        metrics.update({key: round(value, 6) for key, value in analytics.objective_values.items()})
        metrics["completed_operations"] = analytics.completed_operations
        metrics["total_operations"] = len(self.shop.operations)
        metrics["feasible"] = analytics.feasible
        metrics["evaluation_mode"] = "exact"
        return self._make_solution(candidate, source, generation, schedule, analytics, metrics)

    def _worker_payload_bytes(self) -> bytes:
        return pickle.dumps(
            {
                "shop": self.shop,
                "graph_features": self.graph_features,
                "scales": {
                    "time_scale": self.time_scale,
                    "busy_scale": self.busy_scale,
                    "priority_scale": self.priority_scale,
                    "due_scale": self.due_scale,
                },
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    def _write_worker_payload(self) -> str:
        """把 payload 落到临时文件，只把路径交给 worker。

        initargs 里的 bytes 会被每个 worker 的主循环栈帧引用到进程退出——大实例
        下那是每进程上百 MB 的常驻开销，且父进程自己也要一直留着同一份。
        """
        handle, path = tempfile.mkstemp(prefix="llm4drd_worker_", suffix=".pkl")
        try:
            with os.fdopen(handle, "wb") as payload_file:
                payload_file.write(self._worker_payload_bytes())
        except BaseException:
            os.unlink(path)
            raise
        return path

    def _ensure_process_pool(self, worker_count: int):
        """worker_count 只用于"要不要并行"的判断；池本身按全局最大并发定容。"""
        if self._process_backend_failed or self.config.parallel_backend != "process":
            return None
        if self._process_pool is None:
            try:
                self._payload_path = self._write_worker_payload()
                self._process_pool = ProcessPoolExecutor(
                    max_workers=self._max_pool_workers,
                    initializer=init_worker,
                    initargs=(self._payload_path,),
                )
            except Exception as exc:
                logging.warning("hybrid: process pool unavailable (%s), using threads", exc)
                self._process_backend_failed = True
                self._discard_worker_payload()
                return None
        return self._process_pool

    def _discard_worker_payload(self) -> None:
        if self._payload_path is None:
            return
        try:
            os.unlink(self._payload_path)
        except OSError:
            logging.warning("hybrid: worker payload %s not removed", self._payload_path)
        self._payload_path = None

    def _shutdown_process_pool(self) -> None:
        if self._process_pool is not None:
            # 必须等 worker 真正退出再删 payload：shutdown 默认 wait=True，
            # 此时不会再有新 worker 去读这个文件。
            self._process_pool.shutdown(cancel_futures=True)
            self._process_pool = None
        self._discard_worker_payload()

    def _abandon_process_backend(self, exc: BaseException) -> None:
        logging.warning("hybrid: process backend failed (%s), falling back to threads", exc)
        self._process_backend_failed = True
        self._shutdown_process_pool()

    def _submit_to_pool(self, executor, function, *args):
        """提交任务；把"进程池起不来"与"任务本身出错"分开。

        worker 进程往往到首次 submit() 才真正启动，届时抛的是 RuntimeError /
        OSError / AssertionError（如 daemon 进程内不允许再起子进程——本服务跑在
        FastAPI worker 里正是该场景），而不是 BrokenProcessPool。这些属于基础设施
        故障，应当回退线程池；任务内部的异常则必须原样上抛，不能被回退掩盖。
        """
        try:
            return executor.submit(function, *args)
        except (BrokenProcessPool, pickle.PicklingError, RuntimeError, OSError, AssertionError) as exc:
            raise _ProcessBackendUnavailable(exc) from exc

    def _evaluate_candidate(self, candidate: CandidateParameters, source: str, generation: int) -> OptimizationSolution:
        signature = candidate.signature()
        with self.cache_lock:
            cached = self.exact_cache.get(signature)
        if cached is not None:
            return self._clone_solution(cached, source, generation)

        started = time.time()
        solution = self._simulate_candidate(candidate, source, generation)
        with self.cache_lock:
            existing = self.exact_cache.get(signature)
            if existing is None:
                self.exact_cache[signature] = solution.clone()
                self.total_evaluations += 1
                self.exact_evaluations += 1
                self.exact_eval_time_total += time.time() - started
                return solution
            return self._clone_solution(existing, source, generation)

    def _evaluate_candidate_approx(self, candidate: CandidateParameters, source: str, generation: int) -> OptimizationSolution:
        signature = candidate.signature()
        with self.cache_lock:
            cached = self.approx_cache.get(signature)
        if cached is not None:
            return self._clone_solution(cached, source, generation)

        solution = self.approx_evaluator.evaluate(candidate, source, generation)
        with self.cache_lock:
            existing = self.approx_cache.get(signature)
            if existing is None:
                self.approx_cache[signature] = solution.clone()
                self.total_evaluations += 1
                self.approximate_evaluations += 1
                return solution
            return self._clone_solution(existing, source, generation)

    def _parallel_evaluate_candidates_exact(
        self,
        candidates: list[CandidateParameters],
        source: str,
        generation: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[OptimizationSolution]:
        if not candidates:
            return []

        ordered_signatures: list[str] = []
        unique_candidates: dict[str, CandidateParameters] = {}
        results_by_signature: dict[str, OptimizationSolution] = {}

        for candidate in candidates:
            signature = candidate.signature()
            ordered_signatures.append(signature)
            if signature in results_by_signature or signature in unique_candidates:
                continue
            with self.cache_lock:
                cached = self.exact_cache.get(signature)
            if cached is not None:
                results_by_signature[signature] = self._clone_solution(cached, source, generation)
            else:
                unique_candidates[signature] = candidate.clone()

        pending = list(unique_candidates.items())
        completed_count = len(results_by_signature)
        total_count = completed_count + len(pending)
        if completed_count and progress_callback is not None:
            progress_callback(completed_count, total_count)

        def notify_completed() -> None:
            nonlocal completed_count
            completed_count += 1
            if progress_callback is not None:
                progress_callback(completed_count, total_count)

        if pending:
            worker_count = self._worker_count_for_batch("exact", len(pending))
            executor = self._ensure_process_pool(worker_count) if worker_count > 1 else None
            if executor is not None:
                batch_started = time.time()
                new_exact = 0
                try:
                    futures = {
                        self._submit_to_pool(executor, run_exact_simulation, candidate): signature
                        for signature, candidate in pending
                    }
                    for future in as_completed(futures):
                        signature = futures[future]
                        try:
                            sim_result = future.result()
                        except (BrokenProcessPool, pickle.PicklingError) as exc:
                            # 池在任务执行期间崩掉——基础设施故障，回退。
                            raise _ProcessBackendUnavailable(exc) from exc
                        solution = self._solution_from_sim_result(
                            unique_candidates[signature], sim_result, source, generation,
                        )
                        with self.cache_lock:
                            existing = self.exact_cache.get(signature)
                            if existing is None:
                                self.exact_cache[signature] = solution.clone()
                                self.total_evaluations += 1
                                self.exact_evaluations += 1
                                new_exact += 1
                                results_by_signature[signature] = solution
                            else:
                                results_by_signature[signature] = self._clone_solution(existing, source, generation)
                        notify_completed()
                except _ProcessBackendUnavailable as exc:
                    self._abandon_process_backend(exc.cause)
                if new_exact > 0:
                    self.exact_eval_time_total += max(0.0, time.time() - batch_started)
                # 进程池中途失败时，未完成的候选走缓存感知的串行路径补齐
                for signature, candidate in pending:
                    if signature not in results_by_signature:
                        results_by_signature[signature] = self._evaluate_candidate(candidate, source, generation)
                        notify_completed()
            elif worker_count > 1:
                batch_started = time.time()
                new_exact = 0
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = {
                        executor.submit(self._simulate_candidate, candidate, source, generation): signature
                        for signature, candidate in pending
                    }
                    for future in as_completed(futures):
                        signature = futures[future]
                        solution = future.result()
                        with self.cache_lock:
                            existing = self.exact_cache.get(signature)
                            if existing is None:
                                self.exact_cache[signature] = solution.clone()
                                self.total_evaluations += 1
                                self.exact_evaluations += 1
                                new_exact += 1
                                results_by_signature[signature] = solution
                            else:
                                results_by_signature[signature] = self._clone_solution(existing, source, generation)
                        notify_completed()
                if new_exact > 0:
                    self.exact_eval_time_total += max(0.0, time.time() - batch_started)
            else:
                for signature, candidate in pending:
                    started = time.time()
                    solution = self._simulate_candidate(candidate, source, generation)
                    with self.cache_lock:
                        existing = self.exact_cache.get(signature)
                        if existing is None:
                            self.exact_cache[signature] = solution.clone()
                            self.total_evaluations += 1
                            self.exact_evaluations += 1
                            self.exact_eval_time_total += time.time() - started
                            results_by_signature[signature] = solution
                        else:
                            results_by_signature[signature] = self._clone_solution(existing, source, generation)
                    notify_completed()

        return [self._clone_solution(results_by_signature[signature], source, generation) for signature in ordered_signatures]

    def _register_approx_solution(self, signature: str, solution: OptimizationSolution, source: str, generation: int) -> OptimizationSolution:
        with self.cache_lock:
            existing = self.approx_cache.get(signature)
            if existing is None:
                self.approx_cache[signature] = solution.clone()
                self.total_evaluations += 1
                self.approximate_evaluations += 1
                return solution
            return self._clone_solution(existing, source, generation)

    def _parallel_evaluate_candidates_approx(
        self,
        candidates: list[CandidateParameters],
        source: str,
        generation: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[OptimizationSolution]:
        if not candidates:
            return []
        completed_count = 0
        total_count = len(candidates)

        def notify_completed() -> None:
            nonlocal completed_count
            completed_count += 1
            if progress_callback is not None:
                progress_callback(completed_count, total_count)

        worker_count = self._worker_count_for_batch("approx", len(candidates))
        executor = self._ensure_process_pool(worker_count) if worker_count > 1 else None
        if executor is not None:
            results: list[OptimizationSolution] = []
            remaining: list[CandidateParameters] = []
            for candidate in candidates:
                signature = candidate.signature()
                with self.cache_lock:
                    cached = self.approx_cache.get(signature)
                if cached is not None:
                    results.append(self._clone_solution(cached, source, generation))
                    notify_completed()
                else:
                    remaining.append(candidate)
            submitted = list(remaining)
            try:
                futures = {
                    self._submit_to_pool(executor, run_approx_evaluation, candidate, source, generation): candidate
                    for candidate in remaining
                }
                for future in as_completed(futures):
                    candidate = futures[future]
                    try:
                        solution = future.result()
                    except (BrokenProcessPool, pickle.PicklingError) as exc:
                        raise _ProcessBackendUnavailable(exc) from exc
                    results.append(self._register_approx_solution(candidate.signature(), solution, source, generation))
                    submitted.remove(candidate)
                    notify_completed()
            except _ProcessBackendUnavailable as exc:
                self._abandon_process_backend(exc.cause)
                for candidate in submitted:
                    results.append(self._evaluate_candidate_approx(candidate, source, generation))
                    notify_completed()
        elif worker_count > 1:
            results = []
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(self._evaluate_candidate_approx, candidate, source, generation)
                    for candidate in candidates
                ]
                for future in as_completed(futures):
                    results.append(future.result())
                    notify_completed()
        else:
            results = []
            for candidate in candidates:
                results.append(
                    self._evaluate_candidate_approx(candidate, source, generation)
                )
                notify_completed()
        return [self._clone_solution(solution, source, generation) for solution in results]

    def _record_solution(self, solution: OptimizationSolution) -> None:
        self.archive.consider(solution)
        if solution.feasible and solution.schedule_signature not in self.solution_pool:
            self.solution_pool[solution.schedule_signature] = solution.clone()

    def _record_coarse_solution(self, solution: OptimizationSolution) -> None:
        if solution.feasible and solution.schedule_signature not in self.coarse_solution_pool:
            self.coarse_solution_pool[solution.schedule_signature] = solution.clone()

    def _evaluate_builtin_rule(self, rule_name: str, source: str, generation: int) -> OptimizationSolution:
        candidate = self._default_candidate(rule_name)
        cache_key = f"builtin::{rule_name}"
        with self.cache_lock:
            cached = self.exact_cache.get(cache_key)
        if cached is not None:
            return self._clone_solution(cached, source, generation)

        started = time.time()
        runtime = self._runtime_pool.acquire()
        try:
            simulator = Simulator(self.shop, BUILTIN_RULES[rule_name], runtime=runtime)
            sim_result = simulator.run()
        finally:
            self._runtime_pool.release(runtime)
        schedule = self._enrich_schedule(sim_result.schedule)
        analytics = build_schedule_analytics(self.shop, sim_result)
        metrics = sim_result.to_dict()
        metrics.update({key: round(value, 6) for key, value in analytics.objective_values.items()})
        metrics["completed_operations"] = analytics.completed_operations
        metrics["total_operations"] = len(self.shop.operations)
        metrics["feasible"] = analytics.feasible
        metrics["evaluation_mode"] = "exact"
        solution = self._make_solution(candidate, source, generation, schedule, analytics, metrics)
        with self.cache_lock:
            existing = self.exact_cache.get(cache_key)
            if existing is None:
                self.exact_cache[cache_key] = solution.clone()
                self.total_evaluations += 1
                self.exact_evaluations += 1
                self.exact_eval_time_total += time.time() - started
                return solution
            return self._clone_solution(existing, source, generation)

    def _context_for_solution(self, solution: OptimizationSolution) -> dict:
        return {
            "solution": solution,
            "analytics": solution.analytics_summary,
            "graph_features": self.graph_features,
            "shop": self.shop,
        }

    def _destroy_ops(self) -> dict[str, callable]:
        def choose_subset(op_ids: list[str], fraction: float, rng: random.Random) -> list[str]:
            if not op_ids:
                return []
            count = max(1, int(math.ceil(len(op_ids) * fraction)))
            ranked = list(dict.fromkeys(op_ids))
            if count >= len(ranked):
                return ranked
            rng.shuffle(ranked)
            return ranked[:count]

        def tardy_order_destroy(candidate: CandidateParameters, context: dict, rng: random.Random) -> list[str]:
            tardy_orders = context["analytics"].get("tardy_order_ids", [])
            ops = [op.id for op in self.shop.operations.values() if self.shop.tasks[op.task_id].order_id in tardy_orders]
            touched = choose_subset(ops, candidate.destroy_fraction, rng)
            for op_id in touched:
                candidate.op_bias.pop(op_id, None)
            return touched

        def assembly_chain_destroy(candidate: CandidateParameters, context: dict, rng: random.Random) -> list[str]:
            ops: list[str] = []
            for order_id in context["analytics"].get("tardy_order_ids", []):
                order = self.shop.orders.get(order_id)
                if not order or not order.main_task_id:
                    continue
                stack = [order.main_task_id]
                seen: set[str] = set()
                while stack:
                    task_id = stack.pop()
                    if task_id in seen:
                        continue
                    seen.add(task_id)
                    task = self.shop.tasks.get(task_id)
                    if task:
                        ops.extend(op.id for op in task.operations)
                        stack.extend(task.predecessor_task_ids)
            touched = choose_subset(ops, candidate.destroy_fraction, rng)
            for op_id in touched:
                candidate.op_bias.pop(op_id, None)
            return touched

        def bottleneck_machine_destroy(candidate: CandidateParameters, context: dict, rng: random.Random) -> list[str]:
            bottlenecks = set(context["analytics"].get("bottleneck_machine_ids", [])[:2])
            ops = [entry["op_id"] for entry in context["solution"].schedule if entry.get("machine_id") in bottlenecks]
            touched = choose_subset(ops, candidate.destroy_fraction, rng)
            for op_id in touched:
                candidate.op_bias.pop(op_id, None)
            return touched

        def shared_tooling_destroy(candidate: CandidateParameters, context: dict, rng: random.Random) -> list[str]:
            ranked = [
                op_id
                for op_id, values in sorted(
                    self.graph_features.items(),
                    key=lambda item: (len(self.shop.operations[item[0]].required_tooling_types), item[1]["shared_resource_degree"]),
                    reverse=True,
                )
                if self.shop.operations[op_id].required_tooling_types
            ]
            touched = choose_subset(ranked, candidate.destroy_fraction, rng)
            for op_id in touched:
                candidate.op_bias.pop(op_id, None)
            return touched

        def shared_personnel_destroy(candidate: CandidateParameters, context: dict, rng: random.Random) -> list[str]:
            ranked = [
                op_id
                for op_id, values in sorted(
                    self.graph_features.items(),
                    key=lambda item: (len(self.shop.operations[item[0]].required_personnel_skills), item[1]["shared_resource_degree"]),
                    reverse=True,
                )
                if self.shop.operations[op_id].required_personnel_skills
            ]
            touched = choose_subset(ranked, candidate.destroy_fraction, rng)
            for op_id in touched:
                candidate.op_bias.pop(op_id, None)
            return touched

        def critical_predecessor_destroy(candidate: CandidateParameters, context: dict, rng: random.Random) -> list[str]:
            ranked = [
                op_id
                for op_id, _ in sorted(
                    self.graph_features.items(),
                    key=lambda item: (item[1]["predecessor_depth"] + item[1]["assembly_criticality"], item[1]["bottleneck_adjacency"]),
                    reverse=True,
                )
            ]
            touched = choose_subset(ranked, candidate.destroy_fraction, rng)
            for op_id in touched:
                candidate.op_bias.pop(op_id, None)
            return touched

        return {
            "tardy_order_destroy": tardy_order_destroy,
            "assembly_chain_destroy": assembly_chain_destroy,
            "bottleneck_machine_destroy": bottleneck_machine_destroy,
            "shared_tooling_destroy": shared_tooling_destroy,
            "shared_personnel_destroy": shared_personnel_destroy,
            "critical_predecessor_destroy": critical_predecessor_destroy,
        }

    def _repair_ops(self) -> dict[str, callable]:
        def due_date_repair(candidate: CandidateParameters, context: dict, touched: list[str], rng: random.Random) -> None:
            candidate.feature_weights["due_date"] += 0.18
            candidate.feature_weights["urgency"] += 0.12
            candidate.feature_weights["slack"] += 0.08
            due_pairs = sorted(
                touched,
                key=lambda op_id: self.shop.tasks[self.shop.operations[op_id].task_id].due_date,
            )
            for rank, op_id in enumerate(due_pairs[: max(3, len(due_pairs) // 2)]):
                candidate.op_bias[op_id] = candidate.op_bias.get(op_id, 0.0) + 0.4 - 0.05 * rank

        def main_order_repair(candidate: CandidateParameters, context: dict, touched: list[str], rng: random.Random) -> None:
            candidate.feature_weights["priority"] += 0.08
            candidate.feature_weights["is_main"] += 0.16
            candidate.feature_weights["assembly_criticality"] += 0.16
            for op_id in touched:
                weight = self.graph_features.get(op_id, {}).get("assembly_criticality", 0.0)
                candidate.op_bias[op_id] = candidate.op_bias.get(op_id, 0.0) + 0.45 * weight

        def bottleneck_smoothing_repair(candidate: CandidateParameters, context: dict, touched: list[str], rng: random.Random) -> None:
            candidate.feature_weights["machine_load"] += 0.18
            candidate.feature_weights["bottleneck_adjacency"] += 0.14
            for op_id in touched:
                weight = self.graph_features.get(op_id, {}).get("bottleneck_adjacency", 0.0)
                candidate.op_bias[op_id] = candidate.op_bias.get(op_id, 0.0) + 0.3 * weight

        def assembly_sync_repair(candidate: CandidateParameters, context: dict, touched: list[str], rng: random.Random) -> None:
            order_gap = context["analytics"].get("order_main_gap", {})
            candidate.feature_weights["assembly_criticality"] += 0.14
            candidate.feature_weights["prereq_ratio"] += 0.08
            for op_id in touched:
                order_id = self.shop.tasks[self.shop.operations[op_id].task_id].order_id
                gap = min(1.0, order_gap.get(order_id, 0.0) / max(1.0, self.time_scale))
                candidate.op_bias[op_id] = candidate.op_bias.get(op_id, 0.0) + 0.35 * gap

        def shared_resource_repair(candidate: CandidateParameters, context: dict, touched: list[str], rng: random.Random) -> None:
            candidate.feature_weights["shared_resource_degree"] += 0.18
            candidate.feature_weights["tooling_demand"] += 0.06
            candidate.feature_weights["personnel_demand"] += 0.06
            for op_id in touched:
                scarcity = self.graph_features.get(op_id, {}).get("shared_resource_degree", 0.0)
                candidate.op_bias[op_id] = candidate.op_bias.get(op_id, 0.0) + 0.22 * max(0.0, 1.2 - scarcity)

        return {
            "due_date_repair": due_date_repair,
            "main_order_repair": main_order_repair,
            "bottleneck_smoothing_repair": bottleneck_smoothing_repair,
            "assembly_sync_repair": assembly_sync_repair,
            "shared_resource_repair": shared_resource_repair,
        }

    def _load_baseline_seeds(self) -> list[CandidateParameters]:
        """从基准方案库加载 active 方案作为热启动种子候选。

        库空 / 未建表 / 加载失败 / 版本或尺度护栏不过 → 返回 []，白天优化退化为现有行为。
        护栏（提案 §4.4 / §4.7.2）：
          - feature_names 与当前 FEATURE_NAMES 不一致 → 跳过（特征定义已变，权重会错位）；
          - scale_json 与当前实例现算尺度偏差超阈值 → 跳过（尺度耦合导致等效行为漂移）。
        """
        if not self.config.baseline_seeds_enabled:
            return []
        try:
            from ..data.db import DB_PATH, BaselineSolutionStore

            store = BaselineSolutionStore(self.config.baseline_db_path or DB_PATH)
            rows = store.load_active()
        except Exception as exc:  # 未建表 / DB 缺失 / 损坏：静默退化，不影响优化
            logger.debug("基准方案库加载失败，退化为现有种子：%s", exc)
            return []

        current_names = list(FEATURE_NAMES)
        current_scales = {
            "time_scale": self.time_scale,
            "due_scale": self.due_scale,
            "priority_scale": self.priority_scale,
        }
        tolerance = self.config.baseline_scale_tolerance
        candidates: list[CandidateParameters] = []
        for row in rows:
            try:
                if list(row.get("feature_names") or []) != current_names:
                    logger.debug("基准方案 %s feature_names 不匹配，跳过", row.get("id"))
                    continue
                if not self._scale_within_tolerance(row.get("scale_json") or {}, current_scales, tolerance):
                    logger.debug("基准方案 %s 尺度偏差超阈值，跳过", row.get("id"))
                    continue
                candidates.append(self._candidate_from_baseline_row(row))
            except Exception as exc:
                logger.debug("基准方案 %s 重建失败，跳过：%s", row.get("id"), exc)
                continue
        if candidates:
            logger.info("基准方案库热启动：注入 %d 个种子", len(candidates))
        return candidates

    @staticmethod
    def _scale_within_tolerance(stored: dict, current: dict, tolerance: float) -> bool:
        for key, current_value in current.items():
            stored_value = stored.get(key)
            if stored_value is None:
                continue
            denom = max(abs(float(current_value)), 1e-9)
            if abs(float(stored_value) - float(current_value)) / denom > tolerance:
                return False
        return True

    def _candidate_from_baseline_row(self, row: dict) -> CandidateParameters:
        weights = row.get("feature_weights") or {}
        feature_weights = {name: float(weights.get(name, 0.0)) for name in FEATURE_NAMES}
        destroy = row.get("destroy_weights") or {}
        repair = row.get("repair_weights") or {}
        destroy_weights = {name: float(destroy.get(name, 1.0)) for name in DESTROY_OPERATORS}
        repair_weights = {name: float(repair.get(name, 1.0)) for name in REPAIR_OPERATORS}
        op_bias = {str(k): float(v) for k, v in (row.get("op_bias") or {}).items()}
        return CandidateParameters(
            feature_weights=feature_weights,
            destroy_weights=destroy_weights,
            repair_weights=repair_weights,
            op_bias=op_bias,
            destroy_fraction=float(row.get("destroy_fraction", 0.3)),
            seed_rule_name=None,
            graph_profile="balanced",
        )

    def _seed_population(self) -> list[CandidateParameters]:
        seeds: list[CandidateParameters] = []
        # 基准方案插在最前：去重与截断时优先保留（提案修订后 §4.5）。
        baseline_seeds = self._load_baseline_seeds()
        for candidate in baseline_seeds:
            seeds.append(self._project_candidate_to_graph_space(candidate, "balanced", blend=0.22))
        for profile_name, guide in self.graph_guides.items():
            seeds.append(self._candidate_from_guide(profile_name, guide.get("seed_rule"), intensity=0.6))
            seeds.append(self._candidate_from_guide(profile_name, "COMPOSITE", intensity=0.4))
        for rule_name in BUILTIN_RULES:
            candidate = self._default_candidate(rule_name)
            candidate.graph_profile = "balanced"
            seeds.append(self._project_candidate_to_graph_space(candidate, "balanced", blend=0.22))

        unique: dict[str, CandidateParameters] = {}
        for seed in seeds:
            unique.setdefault(seed.signature(), seed)
        # 防截断：有基线时上调 population_size，确保基线与现有种子都进种群、互不挤占。
        # 库空时不触发，population_size 维持不变，行为与改前逐字节一致。
        if baseline_seeds and len(unique) > self.config.population_size:
            self.config.population_size = len(unique)
        population = list(unique.values())[: self.config.population_size]
        while len(population) < self.config.population_size:
            population.append(self._mutate_candidate(self.rng.choice(list(unique.values())), scale=0.45))
        return population

    def _filter_candidate_batch(self, candidates: list[CandidateParameters], limit: int) -> list[CandidateParameters]:
        if len(candidates) <= limit:
            return candidates

        unique: dict[str, CandidateParameters] = {}
        for candidate in candidates:
            unique.setdefault(candidate.signature(), candidate)
        ranked = sorted(
            unique.values(),
            key=lambda candidate: (
                self._graph_alignment_score(candidate),
                len(candidate.op_bias),
                candidate.seed_rule_name or "",
            ),
            reverse=True,
        )

        selected: list[CandidateParameters] = []
        seen_profiles: set[str] = set()
        for candidate in ranked:
            profile = candidate.graph_profile or "balanced"
            if profile in seen_profiles:
                continue
            selected.append(candidate)
            seen_profiles.add(profile)
            if len(selected) >= limit:
                return selected
        for candidate in ranked:
            if len(selected) >= limit:
                break
            if candidate.signature() not in {item.signature() for item in selected}:
                selected.append(candidate)
        return selected[:limit]

    def _build_offspring_batch(self, population: list[OptimizationSolution], limit: int) -> list[CandidateParameters]:
        if not population:
            return []
        if limit <= 0:
            return []
        target = max(limit, limit * max(1, self.config.candidate_filter_multiplier))
        raw_children: list[CandidateParameters] = []
        scales = self._objective_scale_map()
        while len(raw_children) < target:
            parent_a = self._tournament_pick(population, scales)
            parent_b = self._tournament_pick(population, scales)
            raw_children.append(self._mutate_candidate(self._crossover(parent_a.candidate, parent_b.candidate)))
        return self._filter_candidate_batch(raw_children, limit)

    def _objective_scale_map(self, reference: OptimizationSolution | None = None) -> dict[str, float]:
        baseline = reference or self.baseline_solution or self.coarse_baseline_solution
        return {
            spec.key: max(1.0, abs((baseline.objectives if baseline else {}).get(spec.key, 0.0)))
            for spec in self.specs
        }

    def _candidate_pool_target(self) -> int:
        return max(
            self.config.population_size * 2,
            self.config.target_solution_count * max(2, self.config.coarse_pool_multiplier),
        )

    def _time_remaining(self) -> float:
        return max(0.0, self.config.time_limit_s - (time.time() - self.time_started))

    def _average_exact_eval_time(self) -> float:
        if self.exact_evaluations <= 0:
            return 0.0
        return max(0.05, self.exact_eval_time_total / self.exact_evaluations)

    def _budget_limited_exact_count(self, desired: int, minimum: int = 0) -> int:
        if desired <= 0:
            return 0
        remaining = self._time_remaining()
        if remaining <= 0:
            return 0
        average = self._average_exact_eval_time()
        if average <= 0:
            return desired
        parallel_factor = max(1, self.exact_parallel_workers)
        allowed = int((max(0.0, remaining * 0.92) / average) * parallel_factor)
        if allowed <= 0:
            return 0
        limited = min(desired, allowed)
        return max(minimum if remaining > average * 0.6 else 0, limited)

    def _select_candidate_pool(self) -> list[OptimizationSolution]:
        feasible = list(self.solution_pool.values())
        if not feasible:
            return [self.baseline_solution] if self.baseline_solution is not None else []
        limit = min(len(feasible), self._candidate_pool_target())
        return select_survivors(feasible, self.specs, limit, self.config.seed + 451)

    def _select_coarse_candidate_pool(self) -> list[OptimizationSolution]:
        feasible = list(self.coarse_solution_pool.values())
        if not feasible:
            return [self.coarse_baseline_solution] if self.coarse_baseline_solution is not None else []
        limit = min(len(feasible), self._candidate_pool_target())
        return select_survivors(feasible, self.specs, limit, self.config.seed + 417)

    def _promotion_pool_target(self, coarse_pool_size: int) -> int:
        requested = max(
            self.config.elite_refine_min,
            self.config.population_size,
            self.config.target_solution_count * max(2, self.config.promotion_pool_multiplier),
        )
        return max(1, min(coarse_pool_size, requested))

    def _select_elites(self, candidate_pool: list[OptimizationSolution]) -> list[OptimizationSolution]:
        if not candidate_pool:
            return []
        requested = max(
            self.config.elite_refine_min,
            math.ceil(len(candidate_pool) * self.config.elite_refine_ratio),
            min(self.config.target_solution_count * 2, len(candidate_pool)),
        )
        limit = min(len(candidate_pool), requested)
        return select_survivors(candidate_pool, self.specs, limit, self.config.seed + 661)

    def _select_promotions(self, coarse_pool: list[OptimizationSolution]) -> list[OptimizationSolution]:
        if not coarse_pool:
            return []
        limit = self._promotion_pool_target(len(coarse_pool))
        elite_limit = max(1, min(limit, int(math.ceil(limit * (1.0 - self.config.random_promotion_ratio)))))
        promoted = select_survivors(coarse_pool, self.specs, elite_limit, self.config.seed + 577)
        used = {solution.schedule_signature for solution in promoted}
        remaining = [solution for solution in coarse_pool if solution.schedule_signature not in used]
        self.rng.shuffle(remaining)
        while len(promoted) < limit and remaining:
            promoted.append(remaining.pop())
        return promoted[:limit]

    def _refine_solution(
        self,
        solution: OptimizationSolution,
        scale_map: dict[str, float],
        generation: int,
        seed_offset: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> OptimizationSolution:
        if self.config.alns_iterations_per_candidate <= 0:
            return solution
        incumbent = solution.clone()
        incumbent.candidate.destroy_fraction = min(
            0.5,
            max(0.05, incumbent.candidate.destroy_fraction * max(0.35, self.config.alns_aggression)),
        )
        local_alns = ALNSCore(self._destroy_ops(), self._repair_ops(), self.config.seed + 97 + seed_offset)
        _, refined_solution = local_alns.refine(
            incumbent.candidate,
            incumbent,
            self._evaluate_candidate,
            self._context_for_solution,
            self.specs,
            scale_map,
            self.config.alns_iterations_per_candidate,
            generation,
            progress_callback=progress_callback,
        )
        return refined_solution if refined_solution.feasible else solution

    def _parallel_refine_elites(
        self,
        elites: list[OptimizationSolution],
        scale_map: dict[str, float],
        generation: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[OptimizationSolution]:
        if not elites:
            return []
        iteration_count = max(0, self.config.alns_iterations_per_candidate)
        total_count = len(elites) * iteration_count
        completed_count = 0
        progress_lock = Lock()

        def notify_iteration(_done: int, _total: int) -> None:
            nonlocal completed_count
            with progress_lock:
                completed_count += 1
                if progress_callback is not None:
                    progress_callback(completed_count, total_count)

        worker_count = self._worker_count_for_batch("refine", len(elites))
        if worker_count > 1:
            refined: list[OptimizationSolution] = []
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        self._refine_solution,
                        solution,
                        scale_map,
                        generation,
                        index * 53,
                        notify_iteration,
                    ): solution.solution_id
                    for index, solution in enumerate(elites)
                }
                for future in as_completed(futures):
                    refined.append(future.result())
            return refined
        return [
            self._refine_solution(
                solution,
                scale_map,
                generation,
                index * 53,
                notify_iteration,
            )
            for index, solution in enumerate(elites)
        ]

    def _tournament_pick(self, population: list[OptimizationSolution], scales: dict[str, float]) -> OptimizationSolution:
        if len(population) == 1:
            return population[0]
        left, right = self.rng.sample(population, 2)
        if left.rank < right.rank:
            return left
        if right.rank < left.rank:
            return right
        left_score = sum(
            (left.objectives[spec.key] if spec.direction == "min" else -left.objectives[spec.key]) / scales.get(spec.key, 1.0)
            for spec in self.specs
        )
        right_score = sum(
            (right.objectives[spec.key] if spec.direction == "min" else -right.objectives[spec.key]) / scales.get(spec.key, 1.0)
            for spec in self.specs
        )
        return left if left_score <= right_score else right

    def _snapshot_status(
        self,
        generation: int,
        population: list[OptimizationSolution],
        elapsed: float,
        phase: str = "coarse",
        *,
        phase_completed: int = 0,
        phase_total: int = 1,
        activity: str = "",
        calculate_hypervolume: bool = False,
    ) -> dict:
        feasible_ratio = sum(1 for solution in population if solution.feasible) / len(population) if population else 0.0
        if calculate_hypervolume or not self.hypervolume_history:
            hypervolume = self.archive.approximate_hypervolume(
                seed=self.config.seed + generation
            )
        else:
            hypervolume = self.hypervolume_history[-1]["hypervolume"]
        safe_total = max(1, int(phase_total))
        safe_completed = min(safe_total, max(0, int(phase_completed)))
        phase_progress = safe_completed / safe_total
        phase_ranges = {
            "coarse": (0.02, 0.62),
            "exact_promotion": (0.62, 0.82),
            "elite_refine": (0.82, 0.96),
            "finalize": (0.96, 1.0),
        }
        phase_start, phase_end = phase_ranges.get(phase, (0.0, 0.02))
        real_progress = phase_start + (phase_end - phase_start) * phase_progress
        snapshot = {
            "phase": phase,
            "phase_completed": safe_completed,
            "phase_total": safe_total,
            "phase_progress": round(phase_progress, 6),
            "real_progress": round(real_progress, 6),
            "activity": activity,
            "generation": generation,
            "archive_size": len(self.archive),
            "population_size": len(population),
            "candidate_pool_size": len(self.solution_pool),
            "coarse_pool_size": len(self.coarse_solution_pool),
            "feasible_ratio": round(feasible_ratio, 4),
            "hypervolume": round(hypervolume, 6),
            "elapsed_s": round(elapsed, 2),
            "total_evaluations": self.total_evaluations,
            "approximate_evaluations": self.approximate_evaluations,
            "exact_evaluations": self.exact_evaluations,
            "parallel_workers": {
                "base": self.parallel_workers,
                "approx": self.approx_parallel_workers,
                "exact": self.exact_parallel_workers,
                "refine": self.refine_parallel_workers,
            },
            "bottleneck_machine_ids": population[0].analytics_summary.get("bottleneck_machine_ids", []) if population else [],
        }
        if calculate_hypervolume or not self.hypervolume_history:
            self.hypervolume_history.append(
                {
                    "generation": generation,
                    "hypervolume": snapshot["hypervolume"],
                }
            )
        self.status_history.append(snapshot)
        return snapshot

    def _format_metric_payload(self, metrics: dict) -> dict:
        payload: dict = {}
        for key, value in (metrics or {}).items():
            if isinstance(value, bool):
                payload[key] = value
            elif isinstance(value, int):
                payload[key] = value
            elif isinstance(value, float):
                payload[key] = round(value, 4)
            elif isinstance(value, str):
                payload[key] = value
        return payload

    def _format_solution_payload(
        self,
        solution: OptimizationSolution,
        baseline: OptimizationSolution,
        schedule_limit: int | None = 120,
    ) -> dict:
        deltas = {
            spec.key: round(solution.objectives.get(spec.key, 0.0) - baseline.objectives.get(spec.key, 0.0), 4)
            for spec in self.specs
        }
        analytics = solution.analytics_summary
        return {
            "solution_id": solution.solution_id,
            "source": solution.source,
            "generation": solution.generation,
            "rank": solution.rank,
            "feasible": solution.feasible,
            "evaluation_mode": solution.metrics.get("evaluation_mode", "exact"),
            "objectives": {spec.key: round(solution.objectives.get(spec.key, 0.0), 4) for spec in self.specs},
            "metrics": self._format_metric_payload(solution.metrics),
            "delta_vs_baseline": deltas,
            "candidate": solution.candidate.summary(),
            "summary": {
                "completed_operations": analytics.get("completed_operations"),
                "total_operations": analytics.get("total_operations"),
                "tardy_order_ids": analytics.get("tardy_order_ids", []),
                "tardy_task_ids": analytics.get("tardy_task_ids", []),
                "bottleneck_machine_ids": analytics.get("bottleneck_machine_ids", []),
                "avg_utilization": round(solution.metrics.get("avg_utilization", 0.0), 4),
                "critical_utilization": round(solution.metrics.get("critical_utilization", 0.0), 4),
                "avg_active_window_utilization": round(solution.metrics.get("avg_active_window_utilization", 0.0), 4),
                "critical_active_window_utilization": round(solution.metrics.get("critical_active_window_utilization", 0.0), 4),
                "avg_net_available_utilization": round(solution.metrics.get("avg_net_available_utilization", 0.0), 4),
                "critical_net_available_utilization": round(solution.metrics.get("critical_net_available_utilization", 0.0), 4),
                "tooling_utilization": round(solution.metrics.get("tooling_utilization", 0.0), 4),
                "personnel_utilization": round(solution.metrics.get("personnel_utilization", 0.0), 4),
                "evaluation_mode": solution.metrics.get("evaluation_mode", "exact"),
            },
            "schedule": solution.schedule if schedule_limit is None else solution.schedule[:schedule_limit],
        }

    def _format_baseline_payload(self, solution: OptimizationSolution, schedule_limit: int | None = 120) -> dict:
        return {
            "solution_id": solution.solution_id,
            "rule_name": self.config.baseline_rule_name,
            "evaluation_mode": solution.metrics.get("evaluation_mode", "exact"),
            "objectives": {spec.key: round(solution.objectives.get(spec.key, 0.0), 4) for spec in self.specs},
            "metrics": self._format_metric_payload(solution.metrics),
            "schedule": solution.schedule if schedule_limit is None else solution.schedule[:schedule_limit],
            "summary": solution.analytics_summary,
        }

    def run(self, progress_callback=None) -> HybridResult:
        try:
            self.time_started = time.time()

            def emit(
                phase: str,
                generation: int,
                population: list[OptimizationSolution],
                completed: int,
                total: int,
                activity: str,
                *,
                calculate_hypervolume: bool = False,
            ) -> None:
                if progress_callback is None:
                    return
                progress_callback(
                    self._snapshot_status(
                        generation,
                        population,
                        time.time() - self.time_started,
                        phase=phase,
                        phase_completed=completed,
                        phase_total=total,
                        activity=activity,
                        calculate_hypervolume=calculate_hypervolume,
                    )
                )

            baseline_name = self.config.baseline_rule_name if self.config.baseline_rule_name in BUILTIN_RULES else "ATC"
            baseline_candidate = self._project_candidate_to_graph_space(
                self._default_candidate(baseline_name),
                "balanced",
                blend=0.18,
            )
            init_candidates = [baseline_candidate] + self._seed_population()
            init_limit = min(len(init_candidates), max(6, self.config.population_size))
            init_candidates = self._filter_candidate_batch(
                init_candidates[: init_limit * 2], init_limit
            )
            coarse_total = max(
                1,
                1 + len(init_candidates)
                + self.config.generations * self.config.population_size,
            )
            emit(
                "coarse",
                0,
                [],
                0,
                coarse_total,
                "正在评估近似基线 0 / 1",
            )
            self.coarse_baseline_solution = self._evaluate_candidate_approx(baseline_candidate, "baseline_approx", 0)
            self._record_coarse_solution(self.coarse_baseline_solution)
            coarse_completed = 1
            emit(
                "coarse",
                0,
                [self.coarse_baseline_solution],
                coarse_completed,
                coarse_total,
                "近似基线评估完成",
            )

            population = self._parallel_evaluate_candidates_approx(
                init_candidates,
                "init",
                0,
                progress_callback=lambda done, total: emit(
                    "coarse",
                    0,
                    [],
                    coarse_completed + done,
                    coarse_total,
                    f"正在初始化候选 {done} / {total}",
                ),
            )
            coarse_completed += len(init_candidates)
            if not population:
                population = [self.coarse_baseline_solution]
            for solution in population:
                self._record_coarse_solution(solution)
            population = select_survivors(population, self.specs, self.config.population_size, self.config.seed)
            scale_map = self._objective_scale_map(self.coarse_baseline_solution)

            generations_completed = 0
            coarse_deadline = self.time_started + self.config.time_limit_s * min(max(self.config.coarse_time_ratio, 0.45), 0.9)
            stagnation = 0
            previous_pool_size = len(self.coarse_solution_pool)
            emit(
                "coarse",
                0,
                population,
                coarse_completed,
                coarse_total,
                "候选池初始化完成",
                calculate_hypervolume=True,
            )

            for generation in range(1, self.config.generations + 1):
                if time.time() - self.time_started >= self.config.time_limit_s:
                    break
                if time.time() >= coarse_deadline and generation > 1:
                    break

                candidate_batch = self._build_offspring_batch(population, self.config.population_size)
                if not candidate_batch:
                    break
                generation_offset = coarse_completed
                offspring = self._parallel_evaluate_candidates_approx(
                    candidate_batch,
                    "coarse_offspring",
                    generation,
                    progress_callback=lambda done, total, generation=generation, offset=generation_offset: emit(
                        "coarse",
                        generation,
                        population,
                        offset + done,
                        coarse_total,
                        f"第 {generation} 代近似评估 {done} / {total}",
                    ),
                )
                coarse_completed += len(candidate_batch)
                for solution in offspring:
                    self._record_coarse_solution(solution)

                combined = population + offspring
                population = select_survivors(combined, self.specs, self.config.population_size, self.config.seed + generation)
                generations_completed = generation
                emit(
                    "coarse",
                    generation,
                    population,
                    coarse_completed,
                    coarse_total,
                    f"第 {generation} 代筛选完成",
                    calculate_hypervolume=True,
                )
                current_pool_size = len(self.coarse_solution_pool)
                if current_pool_size <= previous_pool_size:
                    stagnation += 1
                else:
                    stagnation = 0
                previous_pool_size = current_pool_size
                if (
                    generations_completed >= 2
                    and len(self.coarse_solution_pool) >= self._candidate_pool_target()
                    and stagnation >= max(1, self.config.stagnation_generations)
                ):
                    break

            coarse_pool = self._select_coarse_candidate_pool()
            promotions = self._select_promotions(coarse_pool)
            emit(
                "coarse",
                generations_completed,
                population,
                coarse_total,
                coarse_total,
                "候选广搜阶段完成",
                calculate_hypervolume=True,
            )

            desired_exact_total = 1 + len(promotions)
            emit(
                "exact_promotion",
                generations_completed,
                promotions or population,
                0,
                desired_exact_total,
                "正在执行精确基线评估 0 / 1",
            )
            self.baseline_solution = self._evaluate_builtin_rule(baseline_name, "baseline", 0)
            self._record_solution(self.baseline_solution)
            scale_map = self._objective_scale_map(self.baseline_solution)

            promoted_candidates = [solution.candidate.clone() for solution in promotions]
            requested_promotions = self._budget_limited_exact_count(
                len(promoted_candidates),
                minimum=min(2, len(promoted_candidates)),
            )
            promoted_candidates = promoted_candidates[:requested_promotions]
            exact_total = 1 + len(promoted_candidates)
            emit(
                "exact_promotion",
                generations_completed,
                promotions or population,
                1,
                exact_total,
                f"精确基线完成，候选评估 0 / {len(promoted_candidates)}",
                calculate_hypervolume=True,
            )
            promoted_exact = self._parallel_evaluate_candidates_exact(
                promoted_candidates,
                "promoted_exact",
                generations_completed + 1,
                progress_callback=lambda done, total: emit(
                    "exact_promotion",
                    generations_completed,
                    promotions or population,
                    1 + done,
                    1 + total,
                    f"正在精确评估候选 {done} / {total}",
                ),
            )
            for solution in promoted_exact:
                self._record_solution(solution)
            emit(
                "exact_promotion",
                generations_completed,
                promoted_exact or [self.baseline_solution],
                exact_total,
                exact_total,
                "精确候选评估完成",
                calculate_hypervolume=True,
            )

            candidate_pool = self._select_candidate_pool()
            elites = self._select_elites(candidate_pool)
            refined_solutions: list[OptimizationSolution] = []
            refine_budget = self._budget_limited_exact_count(
                len(elites) * max(1, self.config.alns_iterations_per_candidate) * max(1, self.config.refine_rounds),
                minimum=0,
            )
            if refine_budget > 0 and elites:
                max_elites = max(
                    1,
                    refine_budget // max(1, self.config.alns_iterations_per_candidate * max(1, self.config.refine_rounds)),
                )
                elites = elites[:max_elites]
            planned_refine_total = (
                len(elites)
                * max(1, self.config.alns_iterations_per_candidate)
                * max(1, self.config.refine_rounds)
                if elites and refine_budget > 0
                else 0
            )
            emit(
                "elite_refine",
                generations_completed,
                elites or promoted_exact or [self.baseline_solution],
                0,
                max(1, planned_refine_total),
                (
                    f"正在精修精英方案 0 / {planned_refine_total}"
                    if planned_refine_total
                    else "当前预算不足，跳过精英精修"
                ),
            )
            refine_completed = 0
            if elites and refine_budget > 0 and time.time() - self.time_started < self.config.time_limit_s:
                working = elites
                for round_index in range(max(1, self.config.refine_rounds)):
                    if time.time() - self.time_started >= self.config.time_limit_s:
                        break
                    working = self._parallel_refine_elites(
                        working,
                        scale_map,
                        generations_completed + 1 + round_index,
                        progress_callback=lambda done, total, offset=refine_completed: emit(
                            "elite_refine",
                            generations_completed,
                            working,
                            offset + done,
                            planned_refine_total,
                            f"正在执行 ALNS 精修 {offset + done} / {planned_refine_total}",
                        ),
                    )
                    refine_completed += (
                        len(working)
                        * max(0, self.config.alns_iterations_per_candidate)
                    )
                    for solution in working:
                        self._record_solution(solution)
                    refined_solutions = working
            emit(
                "elite_refine",
                generations_completed,
                refined_solutions or elites or [self.baseline_solution],
                max(1, planned_refine_total),
                max(1, planned_refine_total),
                "精英精修阶段完成",
                calculate_hypervolume=True,
            )

            emit(
                "finalize",
                generations_completed,
                refined_solutions or promoted_exact or [self.baseline_solution],
                0,
                1,
                "正在整理 Pareto 前沿",
            )
            selected = self.archive.select_diverse(self.config.target_solution_count, self.config.seed + 701)
            if selected:
                vectors = [
                    [solution.objectives[spec.key] if spec.direction == "min" else -solution.objectives[spec.key] for spec in self.specs]
                    for solution in selected
                ]
                ranks, _ = fast_nondominated_sort(vectors)
                for solution, rank in zip(selected, ranks):
                    solution.rank = rank
            selected.sort(
                key=lambda solution: (
                    solution.rank,
                    tuple(solution.objectives[spec.key] if spec.direction == "min" else -solution.objectives[spec.key] for spec in self.specs),
                )
            )

            finalize_total = max(1, 3 + len(selected) * 2)
            finalize_completed = 1
            emit(
                "finalize",
                generations_completed,
                selected,
                finalize_completed,
                finalize_total,
                "Pareto 排序完成，正在生成方案预览",
            )
            baseline_payload = self._format_baseline_payload(self.baseline_solution)
            finalize_completed += 1
            solution_payloads = []
            for solution in selected:
                solution_payloads.append(
                    self._format_solution_payload(solution, self.baseline_solution)
                )
                finalize_completed += 1
                emit(
                    "finalize",
                    generations_completed,
                    selected,
                    finalize_completed,
                    finalize_total,
                    f"正在生成方案预览 {len(solution_payloads)} / {len(selected)}",
                )
            baseline_export = self._format_baseline_payload(
                self.baseline_solution, schedule_limit=None
            )
            finalize_completed += 1
            solutions_export = []
            for solution in selected:
                solutions_export.append(
                    self._format_solution_payload(
                        solution,
                        self.baseline_solution,
                        schedule_limit=None,
                    )
                )
                finalize_completed += 1
                emit(
                    "finalize",
                    generations_completed,
                    selected,
                    finalize_completed,
                    finalize_total,
                    f"正在生成完整排程 {len(solutions_export)} / {len(selected)}",
                )
            emit(
                "finalize",
                generations_completed,
                selected,
                finalize_total,
                finalize_total,
                "Pareto 解与排程整理完成",
                calculate_hypervolume=True,
            )

            return HybridResult(
                objective_keys=[spec.key for spec in self.specs],
                baseline=baseline_payload,
                solutions=solution_payloads,
                archive_size=len(self.archive),
                requested_solution_count=self.config.target_solution_count,
                found_solution_count=len(selected),
                coarse_pool_size=len(self.coarse_solution_pool),
                promoted_solution_count=len(promoted_exact),
                refined_solution_count=len(refined_solutions),
                generations_completed=generations_completed,
                total_evaluations=self.total_evaluations,
                approximate_evaluations=self.approximate_evaluations,
                exact_evaluations=self.exact_evaluations,
                elapsed_s=time.time() - self.time_started,
                parallel_workers={
                    "base": self.parallel_workers,
                    "approx": self.approx_parallel_workers,
                    "exact": self.exact_parallel_workers,
                    "refine": self.refine_parallel_workers,
                },
                hypervolume_history=self.hypervolume_history,
                status_history=self.status_history,
                baseline_export=baseline_export,
                solutions_export=solutions_export,
            )
        finally:
            # 进程池的生命周期与一次 run() 对齐——避免 worker 泄漏到调用方。
            self._shutdown_process_pool()
