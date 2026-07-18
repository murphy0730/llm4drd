from __future__ import annotations

import math
import re
import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

from ..core.models import ShopFloor


@dataclass
class ReviewSolutionIndex:
    solution_id: str
    total_operations: int
    order_facets: list[dict]
    entries_by_order: dict[str, list[dict]]
    machine_type_utilization: dict[str, dict]


T = TypeVar("T")


class ReviewReadCache:
    def __init__(self, max_entries: int = 24):
        self.max_entries = max(1, int(max_entries))
        self._items: OrderedDict[tuple[int, str, str], T] = OrderedDict()
        self._lock = threading.RLock()

    def get_or_build(self, key: tuple[int, str, str], builder: Callable[[], T]) -> T:
        with self._lock:
            cached = self._items.get(key)
            if cached is not None:
                self._items.move_to_end(key)
                return cached
        built = builder()
        with self._lock:
            cached = self._items.get(key)
            if cached is not None:
                self._items.move_to_end(key)
                return cached
            self._items[key] = built
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)
            return built

    def retain_version(self, version: int) -> None:
        with self._lock:
            for key in [key for key in self._items if key[0] != version]:
                del self._items[key]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def keys(self) -> list[tuple[int, str, str]]:
        with self._lock:
            return list(self._items)


def build_review_solution_index(
    shop: ShopFloor,
    solution_id: str,
    schedule: Iterable[dict],
) -> ReviewSolutionIndex:
    entries_by_order: dict[str, list[dict]] = defaultdict(list)
    order_facets: dict[str, dict] = {}
    machine_spans: dict[str, list[float]] = {}
    total = 0
    for entry in schedule:
        total += 1
        order_id = str(entry.get("order_id") or "-")
        entries_by_order[order_id].append(entry)
        facet = order_facets.setdefault(order_id, {
            "order_id": order_id,
            "order_name": entry.get("order_name") or "",
            "op_count": 0,
        })
        facet["op_count"] += 1
        machine_id = entry.get("machine_id")
        start, end = entry.get("start"), entry.get("end")
        if machine_id not in shop.machines or start is None or end is None:
            continue
        acc = machine_spans.setdefault(machine_id, [0.0, math.inf, -math.inf])
        acc[0] += max(0.0, float(end) - float(start))
        acc[1] = min(acc[1], float(start))
        acc[2] = max(acc[2], float(end))
    by_type: dict[str, list[float]] = defaultdict(list)
    for machine_id, (busy, first, last) in machine_spans.items():
        window = last - first
        util = min(1.0, max(0.0, busy / window)) if window > 1e-9 else 0.0
        by_type[shop.machines[machine_id].type_id].append(util)
    utilization = {
        type_id: {"utilization": round(sum(values) / len(values), 4), "used_machines": len(values)}
        for type_id, values in by_type.items()
    }
    return ReviewSolutionIndex(
        solution_id=solution_id,
        total_operations=total,
        order_facets=sorted(order_facets.values(), key=lambda item: _natural_key(item["order_id"])),
        entries_by_order=dict(entries_by_order),
        machine_type_utilization=utilization,
    )


def _natural_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value)))


def search_order_facets(
    indexes: Iterable[ReviewSolutionIndex],
    query: str,
    limit: int = 50,
) -> list[dict]:
    merged: dict[str, dict] = {}
    for index in indexes:
        for item in index.order_facets:
            current = merged.get(item["order_id"])
            if current is None or item["op_count"] > current["op_count"]:
                merged[item["order_id"]] = dict(item)
    needle = str(query or "").strip().lower()

    def rank(item: dict) -> tuple:
        order_id = item["order_id"].lower()
        name = str(item.get("order_name") or "").lower()
        bucket = 0 if order_id == needle else 1 if order_id.startswith(needle) else 2 if needle in order_id else 3
        return bucket, _natural_key(item["order_id"]), _natural_key(name)

    matches = [
        item for item in merged.values()
        if not needle or needle in item["order_id"].lower() or needle in str(item.get("order_name") or "").lower()
    ]
    return sorted(matches, key=rank)[:max(1, min(int(limit), 50))]
