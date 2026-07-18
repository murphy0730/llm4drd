from __future__ import annotations

import json
import statistics
import time
from copy import deepcopy

from llm4drd.api.review_read import (
    build_review_solution_index,
    search_order_facets,
)
from llm4drd.core.models import Machine
from llm4drd.tests.shop_fixtures import make_graph_context_shop


ORDER_COUNT = 5_000
ENTRIES_PER_ORDER = 20
MACHINE_COUNT = 40
SOLUTION_COUNT = 4
LOOKUP_ITERATIONS = 100
SEARCH_ITERATIONS = 100
TARGET_ORDER_ID = "ORD-03456"


def _fixture_shop():
    shop = make_graph_context_shop()
    template_shifts = deepcopy(next(iter(shop.machines.values())).shifts)
    shop.machines.clear()
    for index in range(MACHINE_COUNT):
        type_id = "cut" if index % 2 == 0 else "asm"
        machine_id = f"M-{index:02d}"
        shop.machines[machine_id] = Machine(
            machine_id,
            f"Benchmark machine {index:02d}",
            type_id,
            shifts=deepcopy(template_shifts),
        )
    shop.build_indexes()
    return shop


def _solution_schedule(solution_index: int) -> list[dict]:
    schedule: list[dict] = []
    for order_index in range(ORDER_COUNT):
        order_id = f"ORD-{order_index:05d}"
        for entry_index in range(ENTRIES_PER_ORDER):
            start = order_index * 0.25 + entry_index * 1.5 + solution_index * 0.1
            schedule.append({
                "op_id": f"S{solution_index}-OP-{order_index:05d}-{entry_index:02d}",
                "order_id": order_id,
                "order_name": f"Benchmark order {order_index:05d}",
                "machine_id": f"M-{(order_index + entry_index + solution_index) % MACHINE_COUNT:02d}",
                "start": start,
                "end": start + 1.0 + (entry_index % 3) * 0.25,
            })
    return schedule


def _timed_ms(callable_):
    started = time.perf_counter()
    result = callable_()
    return result, (time.perf_counter() - started) * 1000


def run_benchmark() -> dict:
    shop = _fixture_shop()
    schedules = [
        _solution_schedule(solution_index)
        for solution_index in range(SOLUTION_COUNT)
    ]
    indexes = []
    cold_samples_ms = []
    for solution_index, schedule in enumerate(schedules):
        index, elapsed_ms = _timed_ms(
            lambda solution_index=solution_index, schedule=schedule:
            build_review_solution_index(shop, f"S-{solution_index + 1}", schedule)
        )
        indexes.append(index)
        cold_samples_ms.append(elapsed_ms)

    lookup_samples_ms = []
    lookup_results = []
    for iteration in range(LOOKUP_ITERATIONS):
        result, elapsed_ms = _timed_ms(
            lambda iteration=iteration:
            indexes[iteration % SOLUTION_COUNT].entries_by_order[TARGET_ORDER_ID]
        )
        lookup_results.append(result)
        lookup_samples_ms.append(elapsed_ms)

    search_samples_ms = []
    search_results = []
    for _ in range(SEARCH_ITERATIONS):
        result, elapsed_ms = _timed_ms(
            lambda: search_order_facets(indexes, TARGET_ORDER_ID, 50)
        )
        search_results.append(result)
        search_samples_ms.append(elapsed_ms)

    extracted, extraction_ms = _timed_ms(
        lambda: {
            index.solution_id: index.entries_by_order[TARGET_ORDER_ID]
            for index in indexes
        }
    )

    expected_entries = ORDER_COUNT * ENTRIES_PER_ORDER
    assert len(shop.machines) == MACHINE_COUNT
    assert len(schedules) == SOLUTION_COUNT
    assert all(len(schedule) == expected_entries for schedule in schedules)
    assert all(index.total_operations == expected_entries for index in indexes)
    assert all(len(index.order_facets) == ORDER_COUNT for index in indexes)
    assert all(
        len(result) == ENTRIES_PER_ORDER
        and all(entry["order_id"] == TARGET_ORDER_ID for entry in result)
        for result in lookup_results
    )
    assert all(
        result and result[0]["order_id"] == TARGET_ORDER_ID
        for result in search_results
    )
    assert list(extracted) == [f"S-{index}" for index in range(1, 5)]
    assert all(len(entries) == ENTRIES_PER_ORDER for entries in extracted.values())

    cold_total_ms = sum(cold_samples_ms)
    warm_interaction_total_ms = (
        sum(lookup_samples_ms) + sum(search_samples_ms) + extraction_ms
    )
    return {
        "fixture": {
            "orders": ORDER_COUNT,
            "entries_per_order": ENTRIES_PER_ORDER,
            "entries_per_solution": expected_entries,
            "machines": MACHINE_COUNT,
            "solutions": SOLUTION_COUNT,
            "total_schedule_entries": expected_entries * SOLUTION_COUNT,
            "target_order_id": TARGET_ORDER_ID,
        },
        "cold_index": {
            "calls": SOLUTION_COUNT,
            "samples_ms": [round(value, 3) for value in cold_samples_ms],
            "median_ms": round(statistics.median(cold_samples_ms), 3),
            "total_ms": round(cold_total_ms, 3),
        },
        "warm_lookup": {
            "calls": LOOKUP_ITERATIONS,
            "median_ms": round(statistics.median(lookup_samples_ms), 6),
            "total_ms": round(sum(lookup_samples_ms), 3),
        },
        "search": {
            "calls": SEARCH_ITERATIONS,
            "median_ms": round(statistics.median(search_samples_ms), 3),
            "total_ms": round(sum(search_samples_ms), 3),
            "first_result": search_results[-1][0]["order_id"],
        },
        "four_scheme_extraction": {
            "schemes": len(extracted),
            "entries": sum(len(entries) for entries in extracted.values()),
            "total_ms": round(extraction_ms, 6),
        },
        "targets": {
            "cold_index_total_lt_ms": 1000,
            "cold_index_total_met": cold_total_ms < 1000,
            "warm_interaction_total_lt_ms": 300,
            "warm_interaction_total_ms": round(warm_interaction_total_ms, 3),
            "warm_interaction_total_met": warm_interaction_total_ms < 300,
            "timing_thresholds_are_informational": True,
        },
    }


def main() -> None:
    print(json.dumps(run_benchmark(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
