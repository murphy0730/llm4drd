from __future__ import annotations

import random

from .nsga3_core import select_survivors
from .objectives import ObjectiveSpec, objective_vector
from .solution_model import OptimizationSolution


class ParetoArchive:
    def __init__(self, specs: list[ObjectiveSpec]):
        self.specs = specs
        self._solutions: list[OptimizationSolution] = []

    def __len__(self) -> int:
        return len(self._solutions)

    def solutions(self) -> list[OptimizationSolution]:
        return list(self._solutions)

    def consider(self, solution: OptimizationSolution) -> bool:
        if not solution.feasible:
            return False
        if any(existing.schedule_signature == solution.schedule_signature for existing in self._solutions):
            return False

        objective_signature = tuple(round(solution.objectives.get(spec.key, 0.0), 6) for spec in self.specs)
        if any(
            tuple(round(existing.objectives.get(spec.key, 0.0), 6) for spec in self.specs) == objective_signature
            for existing in self._solutions
        ):
            return False

        if any(existing.dominates(solution, self.specs) for existing in self._solutions):
            return False

        survivors = [existing for existing in self._solutions if not solution.dominates(existing, self.specs)]
        survivors.append(solution)
        self._solutions = survivors
        return True

    def select_diverse(self, limit: int, seed: int) -> list[OptimizationSolution]:
        if limit <= 0:
            return []
        if len(self._solutions) <= limit:
            return list(self._solutions)
        return select_survivors(self._solutions, self.specs, limit, seed)

    def approximate_hypervolume(self, samples: int = 1500, seed: int = 0) -> float:
        if not self._solutions:
            return 0.0
        vectors = [objective_vector(solution.objectives, self.specs) for solution in self._solutions]
        dimensions = len(vectors[0])
        mins = [min(vector[d] for vector in vectors) for d in range(dimensions)]
        maxs = [max(vector[d] for vector in vectors) for d in range(dimensions)]

        normalized: list[list[float]] = []
        for vector in vectors:
            normalized.append(
                [
                    0.0 if abs(maxs[d] - mins[d]) <= 1e-12 else (vector[d] - mins[d]) / (maxs[d] - mins[d])
                    for d in range(dimensions)
                ]
            )

        ref = [1.05] * dimensions
        rng = random.Random(seed)
        dominated = 0
        for _ in range(samples):
            probe = [rng.random() * value for value in ref]
            if any(all(point[d] <= probe[d] + 1e-12 for d in range(dimensions)) for point in normalized):
                dominated += 1
        volume = 1.0
        for value in ref:
            volume *= value
        return volume * dominated / samples
