from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field

from .objectives import ObjectiveSpec


FEATURE_NAMES = [
    "urgency",
    "slack",
    "remaining",
    "processing_time",
    "priority",
    "is_main",
    "wait_time",
    "prereq_ratio",
    "machine_load",
    "tooling_demand",
    "personnel_demand",
    "predecessor_depth",
    "assembly_criticality",
    "shared_resource_degree",
    "bottleneck_adjacency",
    "due_date",
]


DESTROY_OPERATORS = [
    "tardy_order_destroy",
    "assembly_chain_destroy",
    "bottleneck_machine_destroy",
    "shared_tooling_destroy",
    "shared_personnel_destroy",
    "critical_predecessor_destroy",
]


REPAIR_OPERATORS = [
    "due_date_repair",
    "main_order_repair",
    "bottleneck_smoothing_repair",
    "assembly_sync_repair",
    "shared_resource_repair",
]


@dataclass
class CandidateParameters:
    feature_weights: dict[str, float]
    destroy_weights: dict[str, float]
    repair_weights: dict[str, float]
    op_bias: dict[str, float] = field(default_factory=dict)
    destroy_fraction: float = 0.18
    seed_rule_name: str | None = None
    graph_profile: str | None = None

    def clone(self) -> "CandidateParameters":
        return CandidateParameters(
            feature_weights=dict(self.feature_weights),
            destroy_weights=dict(self.destroy_weights),
            repair_weights=dict(self.repair_weights),
            op_bias=dict(self.op_bias),
            destroy_fraction=self.destroy_fraction,
            seed_rule_name=self.seed_rule_name,
            graph_profile=self.graph_profile,
        )

    def prune_bias(self, max_items: int = 120, epsilon: float = 1e-6) -> None:
        self.op_bias = {
            op_id: bias for op_id, bias in self.op_bias.items() if abs(bias) > epsilon
        }
        if len(self.op_bias) <= max_items:
            return
        keep = sorted(self.op_bias.items(), key=lambda item: abs(item[1]), reverse=True)[:max_items]
        self.op_bias = dict(keep)

    def signature(self) -> str:
        payload = {
            "feature_weights": {key: round(self.feature_weights.get(key, 0.0), 6) for key in sorted(self.feature_weights)},
            "destroy_weights": {key: round(self.destroy_weights.get(key, 0.0), 6) for key in sorted(self.destroy_weights)},
            "repair_weights": {key: round(self.repair_weights.get(key, 0.0), 6) for key in sorted(self.repair_weights)},
            "destroy_fraction": round(self.destroy_fraction, 6),
            "seed_rule_name": self.seed_rule_name,
            "graph_profile": self.graph_profile,
            "op_bias": {key: round(value, 4) for key, value in sorted(self.op_bias.items())},
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def summary(self) -> dict:
        return {
            "seed_rule_name": self.seed_rule_name,
            "graph_profile": self.graph_profile,
            "destroy_fraction": round(self.destroy_fraction, 3),
            "feature_weights": {key: round(value, 3) for key, value in sorted(self.feature_weights.items())},
            "bias_count": len(self.op_bias),
        }


def schedule_signature(schedule: list[dict]) -> str:
    normalized: list[tuple] = []
    for entry in sorted(schedule, key=lambda item: (item.get("start", 0.0), item.get("machine_id", ""), item.get("op_id", ""))):
        normalized.append(
            (
                entry.get("op_id"),
                entry.get("machine_id"),
                tuple(entry.get("tooling_ids", []) or []),
                tuple(entry.get("personnel_ids", []) or []),
                round(float(entry.get("start", 0.0)), 3),
                round(float(entry.get("end", 0.0)), 3),
            )
        )
    return hashlib.sha1(repr(normalized).encode("utf-8")).hexdigest()


@dataclass
class OptimizationSolution:
    solution_id: str
    source: str
    generation: int
    candidate: CandidateParameters
    objectives: dict[str, float]
    metrics: dict[str, float]
    schedule: list[dict]
    feasible: bool
    schedule_signature: str
    analytics_summary: dict = field(default_factory=dict)
    rank: int = 0
    reference_index: int | None = None
    niche_distance: float | None = None

    def clone(self) -> "OptimizationSolution":
        return OptimizationSolution(
            solution_id=self.solution_id,
            source=self.source,
            generation=self.generation,
            candidate=self.candidate.clone(),
            objectives=dict(self.objectives),
            metrics=dict(self.metrics),
            schedule=copy.deepcopy(self.schedule),
            feasible=self.feasible,
            schedule_signature=self.schedule_signature,
            analytics_summary=copy.deepcopy(self.analytics_summary),
            rank=self.rank,
            reference_index=self.reference_index,
            niche_distance=self.niche_distance,
        )

    def dominates(self, other: "OptimizationSolution", specs: list[ObjectiveSpec]) -> bool:
        any_better = False
        for spec in specs:
            left = float(self.objectives.get(spec.key, 0.0))
            right = float(other.objectives.get(spec.key, 0.0))
            if spec.direction == "min":
                if left > right + 1e-9:
                    return False
                if left < right - 1e-9:
                    any_better = True
            else:
                if left < right - 1e-9:
                    return False
                if left > right + 1e-9:
                    any_better = True
        return any_better
