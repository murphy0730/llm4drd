from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from .objectives import ObjectiveSpec, objective_vector
from .solution_model import CandidateParameters, OptimizationSolution


DestroyOperator = Callable[[CandidateParameters, dict, random.Random], list[str]]
RepairOperator = Callable[[CandidateParameters, dict, list[str], random.Random], None]
EvaluateFunc = Callable[[CandidateParameters, str, int], OptimizationSolution]
ContextFunc = Callable[[OptimizationSolution], dict]


@dataclass
class OperatorState:
    name: str
    weight: float = 1.0
    uses: int = 0
    successes: int = 0


def scalar_score(solution: OptimizationSolution, specs: list[ObjectiveSpec], scales: dict[str, float]) -> float:
    values = objective_vector(solution.objectives, specs)
    total = 0.0
    for value, spec in zip(values, specs):
        total += value / scales.get(spec.key, 1.0)
    return total


class ALNSCore:
    def __init__(self, destroy_ops: dict[str, DestroyOperator], repair_ops: dict[str, RepairOperator], seed: int):
        self.destroy_ops = destroy_ops
        self.repair_ops = repair_ops
        self.destroy_state = {name: OperatorState(name=name) for name in destroy_ops}
        self.repair_state = {name: OperatorState(name=name) for name in repair_ops}
        self.rng = random.Random(seed)

    def _choose_operator(
        self,
        states: dict[str, OperatorState],
        candidate_weights: dict[str, float],
    ) -> str:
        names = list(states.keys())
        weights: list[float] = []
        for name in names:
            state_weight = max(0.05, states[name].weight)
            mix_weight = max(0.01, float(candidate_weights.get(name, 1.0)))
            weights.append(state_weight * mix_weight)
        total = sum(weights)
        if total <= 0:
            return names[0]
        pick = self.rng.random() * total
        upto = 0.0
        for name, weight in zip(names, weights):
            upto += weight
            if pick <= upto:
                return name
        return names[-1]

    def _update_operator(self, state: OperatorState, accepted: bool, improved: bool) -> None:
        state.uses += 1
        if improved:
            state.successes += 1
            state.weight = min(6.0, state.weight * 1.12 + 0.08)
        elif accepted:
            state.weight = min(6.0, state.weight * 1.04 + 0.03)
        else:
            state.weight = max(0.2, state.weight * 0.94)

    def refine(
        self,
        candidate: CandidateParameters,
        incumbent: OptimizationSolution,
        evaluate_fn: EvaluateFunc,
        context_fn: ContextFunc,
        specs: list[ObjectiveSpec],
        scales: dict[str, float],
        iterations: int,
        generation: int,
    ) -> tuple[CandidateParameters, OptimizationSolution]:
        best_candidate = candidate.clone()
        best_solution = incumbent
        current_candidate = candidate.clone()
        current_solution = incumbent
        temperature = 1.0

        for _ in range(max(0, iterations)):
            destroy_name = self._choose_operator(self.destroy_state, current_candidate.destroy_weights)
            repair_name = self._choose_operator(self.repair_state, current_candidate.repair_weights)

            working = current_candidate.clone()
            context = context_fn(current_solution)
            touched = self.destroy_ops[destroy_name](working, context, self.rng)
            self.repair_ops[repair_name](working, context, touched, self.rng)
            working.prune_bias()

            candidate_solution = evaluate_fn(working, f"alns:{destroy_name}+{repair_name}", generation)
            accepted = False
            improved = False

            if candidate_solution.feasible:
                if candidate_solution.dominates(current_solution, specs):
                    accepted = True
                    improved = True
                elif not current_solution.dominates(candidate_solution, specs):
                    current_score = scalar_score(current_solution, specs, scales)
                    candidate_score = scalar_score(candidate_solution, specs, scales)
                    delta = candidate_score - current_score
                    if delta <= 0 or self.rng.random() < math.exp(-delta / max(0.05, temperature)):
                        accepted = True
                        improved = delta < 0

            self._update_operator(self.destroy_state[destroy_name], accepted, improved)
            self._update_operator(self.repair_state[repair_name], accepted, improved)

            if accepted:
                current_candidate = working
                current_solution = candidate_solution
                if candidate_solution.dominates(best_solution, specs) or (
                    not best_solution.dominates(candidate_solution, specs)
                    and scalar_score(candidate_solution, specs, scales) < scalar_score(best_solution, specs, scales)
                ):
                    best_candidate = working.clone()
                    best_solution = candidate_solution

            temperature *= 0.92

        return best_candidate, best_solution
