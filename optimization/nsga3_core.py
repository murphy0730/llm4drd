from __future__ import annotations

import math
import random
from math import comb

from .objectives import ObjectiveSpec, objective_vector


def dominates_vector(left: list[float], right: list[float]) -> bool:
    any_better = False
    for l_value, r_value in zip(left, right):
        if l_value > r_value + 1e-9:
            return False
        if l_value < r_value - 1e-9:
            any_better = True
    return any_better


def fast_nondominated_sort(vectors: list[list[float]]) -> tuple[list[int], list[list[int]]]:
    n = len(vectors)
    domination_count = [0] * n
    domination_set = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]
    rank = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            if dominates_vector(vectors[i], vectors[j]):
                domination_set[i].append(j)
                domination_count[j] += 1
            elif dominates_vector(vectors[j], vectors[i]):
                domination_set[j].append(i)
                domination_count[i] += 1

    fronts[0] = [i for i in range(n) if domination_count[i] == 0]
    front_index = 0
    while front_index < len(fronts) and fronts[front_index]:
        next_front: list[int] = []
        for i in fronts[front_index]:
            for j in domination_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    rank[j] = front_index + 1
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        front_index += 1
    return rank, fronts


def _recursive_reference_points(
    objective_count: int,
    left: int,
    total: int,
    prefix: list[float],
    out: list[list[float]],
) -> None:
    if objective_count == 1:
        out.append(prefix + [left / total])
        return
    for value in range(left + 1):
        _recursive_reference_points(objective_count - 1, left - value, total, prefix + [value / total], out)


def reference_point_divisions(objective_count: int, target_count: int) -> int:
    divisions = 1
    while comb(divisions + objective_count - 1, objective_count - 1) < target_count:
        divisions += 1
    return divisions


def generate_reference_points(objective_count: int, target_count: int) -> list[list[float]]:
    divisions = max(1, reference_point_divisions(objective_count, target_count))
    points: list[list[float]] = []
    _recursive_reference_points(objective_count, divisions, divisions, [], points)
    return points


def normalize_vectors(vectors: list[list[float]]) -> list[list[float]]:
    if not vectors:
        return []
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
    return normalized


def associate_to_reference(point: list[float], reference_points: list[list[float]]) -> tuple[int, float]:
    best_index = 0
    best_distance = float("inf")
    point_norm = math.sqrt(sum(value * value for value in point))
    if point_norm <= 1e-12:
        return 0, 0.0

    for index, reference in enumerate(reference_points):
        ref_norm = math.sqrt(sum(value * value for value in reference))
        if ref_norm <= 1e-12:
            continue
        projection_scale = sum(p * r for p, r in zip(point, reference)) / (ref_norm ** 2)
        projection = [projection_scale * value for value in reference]
        distance = math.sqrt(sum((p - q) ** 2 for p, q in zip(point, projection)))
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index, best_distance


def _niche_select(
    selected: list[int],
    last_front: list[int],
    normalized_vectors: list[list[float]],
    population_size: int,
    rng: random.Random,
) -> list[int]:
    if not last_front:
        return selected[:population_size]

    objective_count = len(normalized_vectors[0]) if normalized_vectors else 0
    reference_points = generate_reference_points(objective_count, population_size)
    selected_by_ref: dict[int, int] = {index: 0 for index in range(len(reference_points))}

    associations: dict[int, tuple[int, float]] = {}
    for index, vector in enumerate(normalized_vectors):
        associations[index] = associate_to_reference(vector, reference_points)

    for index in selected:
        ref_index, _ = associations[index]
        selected_by_ref[ref_index] = selected_by_ref.get(ref_index, 0) + 1

    remaining = set(last_front)
    while len(selected) < population_size and remaining:
        niche_min = min(selected_by_ref.get(index, 0) for index in range(len(reference_points)))
        niche_indices = [index for index in range(len(reference_points)) if selected_by_ref.get(index, 0) == niche_min]
        rng.shuffle(niche_indices)

        chosen_candidate = None
        chosen_reference = None
        for ref_index in niche_indices:
            niche_candidates = [index for index in remaining if associations[index][0] == ref_index]
            if not niche_candidates:
                continue
            if selected_by_ref.get(ref_index, 0) == 0:
                niche_candidates.sort(key=lambda index: associations[index][1])
                chosen_candidate = niche_candidates[0]
            else:
                chosen_candidate = rng.choice(niche_candidates)
            chosen_reference = ref_index
            break

        if chosen_candidate is None:
            chosen_candidate = rng.choice(list(remaining))
            chosen_reference = associations[chosen_candidate][0]

        selected.append(chosen_candidate)
        remaining.remove(chosen_candidate)
        selected_by_ref[chosen_reference] = selected_by_ref.get(chosen_reference, 0) + 1

    return selected[:population_size]


def select_survivors(candidates: list, specs: list[ObjectiveSpec], population_size: int, seed: int) -> list:
    if len(candidates) <= population_size:
        return list(candidates)

    vectors = [objective_vector(candidate.objectives, specs) for candidate in candidates]
    ranks, fronts = fast_nondominated_sort(vectors)
    for index, rank in enumerate(ranks):
        candidates[index].rank = rank

    selected_indices: list[int] = []
    last_front: list[int] = []
    for front in fronts:
        if len(selected_indices) + len(front) <= population_size:
            selected_indices.extend(front)
        else:
            last_front = front
            break

    if len(selected_indices) < population_size and last_front:
        normalized = normalize_vectors(vectors)
        rng = random.Random(seed)
        selected_indices = _niche_select(selected_indices, last_front, normalized, population_size, rng)

    return [candidates[index] for index in selected_indices]
