# Gantt Review Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Gantt charts open at a useful four-day window, add scalable fuzzy order selection, simplify review tables, and make scheme/order linkage read cached indexed data without synchronous simulation or repeated full-page rendering.

**Architecture:** Keep the existing FastAPI + vanilla JavaScript application, but introduce a focused backend review-read module and a small UMD frontend runtime helper. The backend resolves only already-materialized schedules, builds bounded per-solution indexes lazily, and serves one batch response per review interaction; the frontend owns cancellation/caches, updates only stable review regions, and reuses vis Timeline datasets.

**Tech Stack:** Python 3.14, FastAPI, SQLite-backed rule cache, pytest/unittest, vanilla JavaScript, Node.js CommonJS tests, vis-timeline 7.7.3, Playwright.

## Global Constraints

- Do not change scheduling algorithms, objective functions, utilization formulas, or the maximum selection count of 4.
- Initial Gantt visibility is 96 hours: prefer 12 hours before and 84 hours after the schedule progress line, clamp to the full schedule, and show the full range when shorter than 96 hours.
- Review reads must never call `_rule_reference_solution`; an uncached `RULE:*` read returns HTTP 409.
- A selection change sends one batch review-data request; an order-only change sends one batch request with `include_utilization=false`.
- Fuzzy order search ranks exact ID, ID prefix, ID substring, then name substring; debounce is 200ms and the result limit is 50.
- The utilization formula and response compatibility fields remain unchanged, although the UI displays only percentages.
- No new database tables or frontend framework dependencies.
- New cold read work must run in FastAPI's threadpool through synchronous `def` endpoints.

---

## File Structure

- Create `api/review_read.py`: per-solution schedule index, bounded cache, utilization aggregation, and order search.
- Modify `api/server.py`: cache-first solution resolution and the two synchronous batch review endpoints.
- Create `frontend/review_runtime.js`: tested pure window/key/ranking functions and abortable cached request client.
- Modify `frontend/index_v2.html`: load `review_runtime.js` before `app_v2.js` and bump affected asset cache keys.
- Modify `frontend/app_v2.js`: integrate batch reads, stable dynamic review regions, Timeline reuse, and order combobox behavior.
- Modify `frontend/app_v2.css`: searchable combobox and compact comparison-table styling.
- Create `tests/test_review_solution_resolution.py`: prove review reads never simulate rules.
- Create `tests/test_review_read.py`: index, LRU, utilization, and fuzzy-search unit tests.
- Create `tests/test_review_api.py`: batch data/search endpoint contract and partial-failure tests.
- Create `tests/test_review_frontend_runtime.py`: execute UMD helper behavior in Node.
- Create `tests/test_review_frontend_contract.py`: verify the main app integrates the helper and removes old N-request/full-render paths.
- Create `tools/benchmark_review_read.py`: reproducible 5000-order/100000-entry/4-scheme read benchmark.
- Create `tools/verify_review_ui.py`: browser verification of table, combobox, one-request linkage, and Gantt window.

---

### Task 1: Make solution resolution cache-only for review reads

**Files:**
- Create: `tests/test_review_solution_resolution.py`
- Modify: `api/server.py:4216-4241`

**Interfaces:**
- Consumes: `RuleReferenceCacheStore.get(rule_name) -> dict | None`
- Produces: `_resolve_export_solution(current_shop, task, solution_id) -> dict`, with HTTP 409 for uncached `RULE:*`

- [ ] **Step 1: Write the failing resolver tests**

```python
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException

from llm4drd.api import server
from llm4drd.data.db import InstanceStore, RuleReferenceCacheStore, init_db
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class ReviewSolutionResolutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        db_path = str(Path(self.tmp.name) / "review.db")
        init_db(db_path)
        InstanceStore(db_path).save_from_shopfloor(make_graph_context_shop())
        self.cache = RuleReferenceCacheStore(db_path)

    def test_export_reference_wins_over_truncated_task_view(self):
        full = {"solution_id": "R-1", "schedule": [{"op_id": "full"}]}
        truncated = {
            "solution_id": "R-1",
            "schedule": [],
            "summary": {"total_operations": 1},
        }
        task = {
            "export_result": {"reference_solutions": [full], "solutions": []},
            "reference_solutions": [truncated],
        }
        self.assertIs(server._resolve_export_solution(None, task, "R-1"), full)

    def test_rule_cache_hit_never_runs_simulation(self):
        cached = {"solution_id": "RULE:EDD", "schedule": [{"op_id": "cached"}]}
        self.cache.put("EDD", cached)
        with (
            patch.object(server, "rule_reference_cache_store", self.cache),
            patch.object(server, "_rule_reference_solution", side_effect=AssertionError("must not simulate")),
        ):
            resolved = server._resolve_export_solution(
                make_graph_context_shop(),
                {"export_result": {"solutions": [], "reference_solutions": []}},
                "RULE:EDD",
            )
        self.assertEqual(resolved["schedule"], [{"op_id": "cached"}])

    def test_rule_cache_miss_returns_409_without_simulation(self):
        with (
            patch.object(server, "rule_reference_cache_store", self.cache),
            patch.object(server, "_rule_reference_solution", side_effect=AssertionError("must not simulate")),
            self.assertRaises(HTTPException) as ctx,
        ):
            server._resolve_export_solution(
                make_graph_context_shop(),
                {"export_result": {"solutions": [], "reference_solutions": []}},
                "RULE:SPT",
            )
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("尚未计算", str(ctx.exception.detail))
```

- [ ] **Step 2: Run the tests and confirm the current resolver fails**

Run: `PYTHONPATH=.. .venv/bin/python -m pytest -q tests/test_review_solution_resolution.py`

Expected: FAIL because `RULE:EDD` and `RULE:SPT` call `_rule_reference_solution`.

- [ ] **Step 3: Replace resolver precedence and remove implicit simulation**

Add this helper immediately before `_resolve_export_solution`:

```python
def _has_complete_schedule(solution: dict) -> bool:
    schedule = solution.get("schedule")
    if not isinstance(schedule, list):
        return False
    expected = (solution.get("summary") or {}).get("total_operations")
    if expected is None:
        return bool(schedule)
    try:
        return len(schedule) >= int(expected)
    except (TypeError, ValueError):
        return False
```

Replace `_resolve_export_solution` with:

```python
def _resolve_export_solution(current_shop: Optional[ShopFloor], task: dict, solution_id: str) -> dict:
    export_result = task.get("export_result") or task.get("result") or {}
    baseline = export_result.get("baseline")
    if solution_id == "BASELINE" or (baseline and solution_id == baseline.get("solution_id")):
        if not baseline:
            raise HTTPException(404, "未找到基线方案")
        return baseline
    for item in export_result.get("reference_solutions", []) or []:
        if item.get("solution_id") == solution_id:
            return item
    for item in export_result.get("solutions", []) or []:
        if item.get("solution_id") == solution_id:
            return item
    for item in task.get("reference_solutions", []) or []:
        if item.get("solution_id") == solution_id and _has_complete_schedule(item):
            return item
    if solution_id.startswith("RULE:"):
        rule_name = solution_id.split(":", 1)[1]
        cached = rule_reference_cache_store.get(rule_name)
        if cached is not None:
            return cached
        raise HTTPException(409, f"规则方案 {rule_name} 尚未计算完整排程，请先计算规则参照")
    raise HTTPException(404, f"未找到方案 {solution_id}")
```

- [ ] **Step 4: Run resolver and existing endpoint tests**

Run: `PYTHONPATH=.. .venv/bin/python -m pytest -q tests/test_review_solution_resolution.py tests/test_optimize_solution_schedule_api.py tests/test_machine_type_utilization_api.py`

Expected: PASS, 14 tests or more.

- [ ] **Step 5: Commit**

```bash
git add api/server.py tests/test_review_solution_resolution.py
git commit -m "fix: keep review reads on cached schedules"
```

---

### Task 2: Build the bounded per-solution review index

**Files:**
- Create: `api/review_read.py`
- Create: `tests/test_review_read.py`

**Interfaces:**
- Produces: `ReviewSolutionIndex`, `build_review_solution_index(shop, solution_id, schedule)`, `ReviewReadCache.get_or_build(key, builder)`, `search_order_facets(indexes, query, limit)`
- Cache key type: `(instance_version: int, task_id: str, solution_id: str)`

- [ ] **Step 1: Write failing index, reference-identity, cache, and ranking tests**

```python
import unittest

from llm4drd.api.review_read import (
    ReviewReadCache,
    build_review_solution_index,
    search_order_facets,
)
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class CountingSchedule(list):
    def __init__(self, values):
        super().__init__(values)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()


class ReviewReadIndexTests(unittest.TestCase):
    def test_build_scans_once_and_keeps_entry_references(self):
        entry = {
            "order_id": "O-100",
            "order_name": "订单100",
            "machine_id": "M-C1",
            "start": 0.0,
            "end": 10.0,
        }
        schedule = CountingSchedule([entry])
        index = build_review_solution_index(make_graph_context_shop(), "S-1", schedule)
        self.assertEqual(schedule.iterations, 1)
        self.assertIs(index.entries_by_order["O-100"][0], entry)
        self.assertEqual(index.machine_type_utilization["cut"]["utilization"], 1.0)

    def test_cache_reuses_value_and_evicts_lru(self):
        cache = ReviewReadCache(max_entries=2)
        builds = []

        def build(label):
            builds.append(label)
            return label

        self.assertEqual(cache.get_or_build((1, "t", "a"), lambda: build("a")), "a")
        self.assertEqual(cache.get_or_build((1, "t", "a"), lambda: build("again")), "a")
        cache.get_or_build((1, "t", "b"), lambda: build("b"))
        cache.get_or_build((1, "t", "c"), lambda: build("c"))
        self.assertEqual(builds, ["a", "b", "c"])
        self.assertNotIn((1, "t", "a"), cache.keys())

    def test_search_ranks_id_before_name_and_limits(self):
        shop = make_graph_context_shop()
        first = build_review_solution_index(shop, "S-1", [
            {"order_id": "X-001", "order_name": "普通订单", "machine_id": "M-C1", "start": 0, "end": 1},
            {"order_id": "001-X", "order_name": "普通订单", "machine_id": "M-C1", "start": 1, "end": 2},
            {"order_id": "X-900", "order_name": "名称001", "machine_id": "M-C1", "start": 2, "end": 3},
        ])
        result = search_order_facets([first], "001", 2)
        self.assertEqual([item["order_id"] for item in result], ["001-X", "X-001"])
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run: `PYTHONPATH=.. .venv/bin/python -m pytest -q tests/test_review_read.py`

Expected: collection FAIL with `ModuleNotFoundError: llm4drd.api.review_read`.

- [ ] **Step 3: Implement `api/review_read.py`**

Implement these exact public types and functions:

```python
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
```

- [ ] **Step 4: Run unit tests**

Run: `PYTHONPATH=.. .venv/bin/python -m pytest -q tests/test_review_read.py`

Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add api/review_read.py tests/test_review_read.py
git commit -m "feat: index review schedules for fast reads"
```

---

### Task 3: Add batch review-data and fuzzy order APIs

**Files:**
- Modify: `api/server.py:1-80, 3324-3395`
- Create: `tests/test_review_api.py`

**Interfaces:**
- Produces: `GET /api/optimize/hybrid/result/{task_id}/review-data`
- Produces: `GET /api/optimize/hybrid/result/{task_id}/review-orders`
- Consumes: `ReviewReadCache`, `build_review_solution_index`, `search_order_facets`

- [ ] **Step 1: Write failing API contract tests**

Create a fixture matching the existing schedule endpoint tests, but call the new synchronous endpoint directly and decode `response.body`:

```python
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm4drd.api import server
from llm4drd.api.review_read import ReviewReadCache
from llm4drd.data.db import InstanceStore, WorkflowProgressStore, init_db
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class ReviewApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        db_path = str(Path(self.tmp.name) / "review-api.db")
        init_db(db_path)
        originals = (
            server.inst_store, server.workflow_store, server.shop,
            server._active_shop_cache, server._hybrid_tasks,
            server._latest_hybrid_task_id, server.review_read_cache,
        )
        self.addCleanup(self._restore, originals)
        server.inst_store = InstanceStore(db_path)
        server.workflow_store = WorkflowProgressStore(db_path)
        server.inst_store.save_from_shopfloor(make_graph_context_shop())
        server.shop = None
        server._active_shop_cache = None
        server.review_read_cache = ReviewReadCache(8)
        entry = lambda sid, oid, mid, start: {
            "op_id": f"{sid}-{oid}",
            "order_id": oid,
            "order_name": f"订单{oid}",
            "machine_id": mid,
            "start": start,
            "end": start + 1,
        }
        server._hybrid_tasks = {
            "t1": {
                "status": "done",
                "result": {"solutions": [{"solution_id": "S-1"}, {"solution_id": "S-2"}]},
                "export_result": {"solutions": [
                    {"solution_id": "S-1", "schedule": [entry("S1", "O-001", "M-C1", 0)]},
                    {"solution_id": "S-2", "schedule": [entry("S2", "O-001", "M-C2", 2)]},
                ]},
                "reference_solutions": [],
            }
        }
        server._latest_hybrid_task_id = "t1"

    def _restore(self, originals):
        (
            server.inst_store, server.workflow_store, server.shop,
            server._active_shop_cache, server._hybrid_tasks,
            server._latest_hybrid_task_id, server.review_read_cache,
        ) = originals

    def test_batch_returns_two_schemes_and_utilization(self):
        response = server.optimize_review_data("t1", "S-1,S-2", "O-001", True)
        payload = json.loads(response.body)
        self.assertEqual(payload["solutions"], ["S-1", "S-2"])
        self.assertEqual(set(payload["schemes"]), {"S-1", "S-2"})
        self.assertIsNotNone(payload["type_utilization"])

    def test_order_only_request_omits_utilization(self):
        payload = json.loads(server.optimize_review_data("t1", "S-1,S-2", "O-001", False).body)
        self.assertIsNone(payload["type_utilization"])

    def test_search_returns_ranked_union(self):
        payload = json.loads(server.optimize_review_orders("t1", "S-1,S-2", "001", 50).body)
        self.assertEqual(payload["orders"][0]["order_id"], "O-001")
```

- [ ] **Step 2: Run tests and verify both endpoint names are missing**

Run: `PYTHONPATH=.. .venv/bin/python -m pytest -q tests/test_review_api.py`

Expected: FAIL with missing `review_read_cache` or endpoint attributes.

- [ ] **Step 3: Wire the cache and endpoint helpers**

Import from `api.review_read`, create `review_read_cache = ReviewReadCache(max_entries=24)`, and add:

```python
def _requested_solution_ids(raw: str) -> list[str]:
    return list(dict.fromkeys(part.strip() for part in str(raw or "").split(",") if part.strip()))[:4]


def _review_index(
    current_shop: ShopFloor,
    task_id: str,
    task: dict,
    solution_id: str,
    timings: dict[str, float],
):
    version = get_instance_version(inst_store.db_path)
    review_read_cache.retain_version(version)
    resolve_started = time.perf_counter()
    solution = _resolve_export_solution(current_shop, task, solution_id)
    timings["resolve_solution_ms"] += (time.perf_counter() - resolve_started) * 1000
    resolved_id = str(solution.get("solution_id") or solution_id)
    key = (version, task_id, resolved_id)

    def build():
        build_started = time.perf_counter()
        result = build_review_solution_index(current_shop, resolved_id, solution.get("schedule") or [])
        timings["build_index_ms"] += (time.perf_counter() - build_started) * 1000
        return result

    return review_read_cache.get_or_build(
        key,
        build,
    )


def _review_utilization_payload(current_shop: ShopFloor, ids: list[str], indexes: dict) -> dict:
    types = []
    for type_id in sorted(current_shop.machine_types):
        machine_type = current_shop.machine_types[type_id]
        per_solution = {
            solution_id: indexes[solution_id].machine_type_utilization[type_id]
            for solution_id in ids
            if solution_id in indexes and type_id in indexes[solution_id].machine_type_utilization
        }
        types.append({
            "type_id": type_id,
            "type_name": machine_type.name,
            "machines_total": len(current_shop.get_machines_for_type(type_id)),
            "is_critical": machine_type.is_critical,
            "per_solution": per_solution,
        })
    return {"solutions": [item for item in ids if item in indexes], "types": types}


def _json_response_with_timing(
    payload: dict,
    started: float,
    label: str,
    fields: dict,
    timings: dict[str, float],
) -> Response:
    serialize_started = time.perf_counter()
    body = json.dumps(_json_safe(payload), ensure_ascii=False, separators=(",", ":"))
    timings["serialize_ms"] = (time.perf_counter() - serialize_started) * 1000
    timings["total_ms"] = (time.perf_counter() - started) * 1000
    logging.info("%s %s timings=%s", label, fields, {
        key: round(value, 3) for key, value in timings.items()
    })
    return Response(content=body, media_type="application/json")
```

- [ ] **Step 4: Implement both synchronous endpoints**

```python
@app.get("/api/optimize/hybrid/result/{task_id}/review-data")
def optimize_review_data(
    task_id: str,
    solution_ids: str,
    order_id: Optional[str] = None,
    include_utilization: bool = True,
):
    started = time.perf_counter()
    resolved_task_id, task = _resolve_hybrid_task(task_id)
    current_shop = _active_shop()
    if current_shop is None:
        raise HTTPException(400, "当前没有可用实例")
    requested = _requested_solution_ids(solution_ids)
    timings = {
        "resolve_solution_ms": 0.0,
        "build_index_ms": 0.0,
        "lookup_order_ms": 0.0,
        "utilization_ms": 0.0,
        "serialize_ms": 0.0,
        "total_ms": 0.0,
    }
    indexes, failures = {}, {}
    for solution_id in requested:
        try:
            indexes[solution_id] = _review_index(
                current_shop, resolved_task_id, task, solution_id, timings
            )
        except HTTPException as exc:
            failures[solution_id] = str(exc.detail)
    lookup_started = time.perf_counter()
    order_union = search_order_facets(indexes.values(), "", 50)
    selected_order = order_id or (order_union[0]["order_id"] if order_union else None)
    schemes = {
        solution_id: list(index.entries_by_order.get(selected_order, []))
        for solution_id, index in indexes.items()
    }
    timings["lookup_order_ms"] = (time.perf_counter() - lookup_started) * 1000
    utilization_started = time.perf_counter()
    utilization = _review_utilization_payload(current_shop, requested, indexes) if include_utilization else None
    timings["utilization_ms"] = (time.perf_counter() - utilization_started) * 1000
    payload = {
        "task_id": resolved_task_id,
        "order_id": selected_order,
        "solutions": [item for item in requested if item in indexes],
        "schemes": schemes,
        "type_utilization": utilization,
        "failed_solution_ids": list(failures),
        "failure_messages": failures,
    }
    return _json_response_with_timing(payload, started, "review-data", {
        "task_id": resolved_task_id,
        "solutions": len(requested),
        "order_id": selected_order,
    }, timings)


@app.get("/api/optimize/hybrid/result/{task_id}/review-orders")
def optimize_review_orders(task_id: str, solution_ids: str, q: str = "", limit: int = 50):
    started = time.perf_counter()
    resolved_task_id, task = _resolve_hybrid_task(task_id)
    current_shop = _active_shop()
    if current_shop is None:
        raise HTTPException(400, "当前没有可用实例")
    timings = {
        "resolve_solution_ms": 0.0,
        "build_index_ms": 0.0,
        "lookup_order_ms": 0.0,
        "utilization_ms": 0.0,
        "serialize_ms": 0.0,
        "total_ms": 0.0,
    }
    indexes = []
    for solution_id in _requested_solution_ids(solution_ids):
        try:
            indexes.append(_review_index(
                current_shop, resolved_task_id, task, solution_id, timings
            ))
        except HTTPException:
            continue
    lookup_started = time.perf_counter()
    orders = search_order_facets(indexes, q, limit)
    timings["lookup_order_ms"] = (time.perf_counter() - lookup_started) * 1000
    payload = {
        "task_id": resolved_task_id,
        "query": q,
        "orders": orders,
    }
    return _json_response_with_timing(payload, started, "review-orders", {
        "task_id": resolved_task_id,
        "query_length": len(q),
    }, timings)
```

- [ ] **Step 5: Run new and existing API tests**

Run: `PYTHONPATH=.. .venv/bin/python -m pytest -q tests/test_review_api.py tests/test_review_read.py tests/test_optimize_solution_schedule_api.py tests/test_machine_type_utilization_api.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add api/server.py tests/test_review_api.py
git commit -m "feat: batch review schedule and order reads"
```

---

### Task 4: Add a tested frontend runtime for windows, keys, ranking, cancellation, and cache

**Files:**
- Create: `frontend/review_runtime.js`
- Create: `tests/test_review_frontend_runtime.py`
- Modify: `frontend/index_v2.html:8-12, 145-147`

**Interfaces:**
- Produces global/CommonJS `ReviewRuntime`
- Produces `computeInitialWindow`, `selectionKey`, `scheduleKey`, `rankOrders`, `createClient`

- [ ] **Step 1: Write Node-backed failing tests**

Follow `tests/test_real_optimize_progress.py` and run Node from Python. Test:

```javascript
const assert = require("assert");
const runtime = require("./frontend/review_runtime.js");
const fourDays = runtime.computeInitialWindow(
  "2026-01-01T00:00:00.000Z",
  "2026-01-11T00:00:00.000Z",
  "2026-01-05T00:00:00.000Z"
);
assert.strictEqual(new Date(fourDays.end) - new Date(fourDays.start), 96 * 3600000);
assert.strictEqual(fourDays.start, "2026-01-04T12:00:00.000Z");
assert.strictEqual(runtime.selectionKey("t1", ["S-2", "S-1"]), "t1::S-1,S-2");
assert.deepStrictEqual(
  runtime.rankOrders([
    {order_id: "X-001", order_name: "普通"},
    {order_id: "001-X", order_name: "普通"},
    {order_id: "X-900", order_name: "名称001"}
  ], "001", 2).map(x => x.order_id),
  ["001-X", "X-001"]
);
```

Add an async test with an injected fetcher that confirms the second `loadData` aborts the first and a repeated key returns the cached payload without a third fetch.

```javascript
let calls = 0;
const client = runtime.createClient({
  fetchReviewData: (_args, signal) => new Promise((resolve, reject) => {
    calls += 1;
    const timer = setTimeout(() => resolve({order_id: "O-1", schemes: {}}), calls === 1 ? 30 : 1);
    signal.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(new DOMException("aborted", "AbortError"));
    });
  }),
  fetchOrders: async () => ({orders: []})
});
const first = client.loadData({taskId: "t1", ids: ["S-1"], orderId: "O-1", includeUtilization: true});
const second = client.loadData({taskId: "t1", ids: ["S-1"], orderId: "O-2", includeUtilization: false});
assert.deepStrictEqual(await first, {cancelled: true});
assert.strictEqual((await second).payload.order_id, "O-1");
await client.loadData({taskId: "t1", ids: ["S-1"], orderId: "O-2", includeUtilization: false});
assert.strictEqual(calls, 2);
```

- [ ] **Step 2: Run and confirm the module is missing**

Run: `PYTHONPATH=.. .venv/bin/python -m pytest -q tests/test_review_frontend_runtime.py`

Expected: FAIL because `frontend/review_runtime.js` does not exist.

- [ ] **Step 3: Implement the UMD module**

Use the same wrapper as `optimize_progress.js`. Implement:

```javascript
function computeInitialWindow(fullStart, fullEnd, now, totalHours = 96, historyHours = 12) {
  const startMs = new Date(fullStart).getTime();
  const endMs = new Date(fullEnd).getTime();
  const spanMs = Math.max(0, endMs - startMs);
  const desiredMs = totalHours * 3600000;
  if (spanMs <= desiredMs) return { start: new Date(startMs).toISOString(), end: new Date(endMs).toISOString() };
  const anchor = Number.isFinite(new Date(now).getTime()) ? new Date(now).getTime() : startMs;
  let visibleStart = Math.max(startMs, Math.min(endMs, anchor) - historyHours * 3600000);
  let visibleEnd = visibleStart + desiredMs;
  if (visibleEnd > endMs) {
    visibleEnd = endMs;
    visibleStart = endMs - desiredMs;
  }
  return { start: new Date(visibleStart).toISOString(), end: new Date(visibleEnd).toISOString() };
}

function normalizeIds(ids) {
  return Array.from(new Set((ids || []).filter(Boolean))).sort().slice(0, 4);
}

function selectionKey(taskId, ids) {
  return `${taskId}::${normalizeIds(ids).join(",")}`;
}

function scheduleKey(taskId, ids, orderId) {
  return `${selectionKey(taskId, ids)}::${orderId || ""}`;
}

function rankOrders(orders, query, limit = 50) {
  const needle = String(query || "").trim().toLowerCase();
  const bucket = (item) => {
    const id = String(item.order_id || "").toLowerCase();
    const name = String(item.order_name || "").toLowerCase();
    if (id === needle) return 0;
    if (id.startsWith(needle)) return 1;
    if (id.includes(needle)) return 2;
    if (name.includes(needle)) return 3;
    return 4;
  };
  return (orders || [])
    .filter((item) => bucket(item) < 4)
    .slice()
    .sort((a, b) => bucket(a) - bucket(b)
      || String(a.order_id).localeCompare(String(b.order_id), "zh-CN", {numeric: true}))
    .slice(0, Math.max(1, Math.min(Number(limit) || 50, 50)));
}

function createClient({fetchReviewData, fetchOrders}) {
  const dataCache = new Map();
  const orderCache = new Map();
  let dataController = null;
  let orderController = null;
  async function loadData(args) {
    const key = `${scheduleKey(args.taskId, args.ids, args.orderId)}::${args.includeUtilization ? "u1" : "u0"}`;
    if (dataCache.has(key)) return {payload: dataCache.get(key), fromCache: true};
    if (dataController) dataController.abort();
    dataController = new AbortController();
    try {
      const payload = await fetchReviewData(args, dataController.signal);
      dataCache.set(key, payload);
      return {payload, fromCache: false};
    } catch (error) {
      if (error?.name === "AbortError") return {cancelled: true};
      throw error;
    }
  }
  async function searchOrders(args) {
    const key = `${selectionKey(args.taskId, args.ids)}::${String(args.query || "").trim().toLowerCase()}`;
    if (orderCache.has(key)) return {orders: orderCache.get(key), fromCache: true};
    if (orderController) orderController.abort();
    orderController = new AbortController();
    try {
      const payload = await fetchOrders(args, orderController.signal);
      const orders = payload.orders || [];
      orderCache.set(key, orders);
      return {orders, fromCache: false};
    } catch (error) {
      if (error?.name === "AbortError") return {cancelled: true};
      throw error;
    }
  }
  function reset() {
    if (dataController) dataController.abort();
    if (orderController) orderController.abort();
    dataCache.clear();
    orderCache.clear();
  }
  return {loadData, searchOrders, reset};
}
```

Return all six functions from the UMD factory:

```javascript
return {computeInitialWindow, normalizeIds, selectionKey, scheduleKey, rankOrders, createClient};
```

- [ ] **Step 4: Load the runtime before the main app and bump asset versions**

Add:

```html
<script src="/static/review_runtime.js?v=20260719-1"></script>
```

before `app_v2.js`, and bump `app_v2.js`/`app_v2.css` to `v=20260719-1`.

- [ ] **Step 5: Run frontend runtime tests**

Run: `PYTHONPATH=.. .venv/bin/python -m pytest -q tests/test_review_frontend_runtime.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add frontend/review_runtime.js frontend/index_v2.html tests/test_review_frontend_runtime.py
git commit -m "feat: add abortable review frontend runtime"
```

---

### Task 5: Apply the 96-hour initial window to every Gantt

**Files:**
- Modify: `frontend/app_v2.js:1478-1782, 2855-2940, 4243-4295`
- Create: `tests/test_review_frontend_contract.py`

**Interfaces:**
- Consumes: `ReviewRuntime.computeInitialWindow(fullStart, fullEnd, now)`
- Produces Gantt data fields: `fullWindow`, `initialWindow`, `viewKey`

- [ ] **Step 1: Write failing source-contract tests**

Assert that `app_v2.js` contains `fullWindow`, `initialWindow`, `ReviewRuntime.computeInitialWindow`, and Timeline options using `initialWindow`; assert both `buildGanttData` and `buildReviewGanttData` call the same helper.

```python
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "frontend" / "app_v2.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend" / "app_v2.css").read_text(encoding="utf-8")


class ReviewFrontendContractTests(unittest.TestCase):
    def test_all_gantts_use_shared_initial_window(self):
        self.assertIn("function ganttWindowPayload(", JS)
        self.assertGreaterEqual(JS.count("ganttWindowPayload("), 3)
        self.assertIn("ReviewRuntime.computeInitialWindow", JS)
        self.assertIn("fullWindow", JS)
        self.assertIn("initialWindow", JS)
        self.assertIn("data.initialWindow", JS)
```

- [ ] **Step 2: Run the contract test and confirm failure**

Run: `PYTHONPATH=.. .venv/bin/python -m pytest -q tests/test_review_frontend_contract.py`

Expected: FAIL because the current data only has `window`.

- [ ] **Step 3: Add one shared window builder**

```javascript
function ganttWindowPayload(horizonStart, horizonEnd, base, nowOffset, viewKey) {
  const fullWindow = {
    start: ganttOffsetToISO(horizonStart, base),
    end: ganttOffsetToISO(horizonEnd, base),
  };
  const effectiveNow = nowOffset !== null && nowOffset >= horizonStart && nowOffset <= horizonEnd
    ? ganttOffsetToISO(nowOffset, base)
    : fullWindow.start;
  return {
    fullWindow,
    initialWindow: ReviewRuntime.computeInitialWindow(fullWindow.start, fullWindow.end, effectiveNow),
    nowISO: effectiveNow,
    viewKey,
  };
}
```

In both ordinary Gantt return branches and `buildReviewGanttData`, spread this payload and remove padded full-range `window`. Use a `viewKey` containing canvas ID, selected order, solution IDs, and grouping mode; exclude machine filter/page so those operations preserve a manually changed window.

- [ ] **Step 4: Preserve manual ranges and mount with initial/full bounds**

Add `ganttViewWindows: {}` to app state. In `mountGantts`, choose:

```javascript
const stored = app.ganttViewWindows[el.id];
const selectedWindow = stored?.viewKey === data.viewKey ? stored.window : data.initialWindow;
```

Set Timeline `start/end` from `selectedWindow`, `min/max` from `data.fullWindow`, and register:

```javascript
timeline.on("rangechanged", (props) => {
  app.ganttViewWindows[el.id] = {
    viewKey: data.viewKey,
    window: { start: props.start.toISOString(), end: props.end.toISOString() },
  };
});
```

Clear `ganttViewWindows` in `resetInstanceDerivedState`.

- [ ] **Step 5: Run helper and contract tests**

Run: `PYTHONPATH=.. .venv/bin/python -m pytest -q tests/test_review_frontend_runtime.py tests/test_review_frontend_contract.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add frontend/app_v2.js tests/test_review_frontend_contract.py
git commit -m "feat: open gantt charts at four-day window"
```

---

### Task 6: Replace N requests and full-page reloads with batch state and local region updates

**Files:**
- Modify: `frontend/app_v2.js:180-197, 217-345, 2790-3045, 3178-3185, 4243-4295, 4639-4722, 5380-5405`
- Modify: `tests/test_review_frontend_contract.py`

**Interfaces:**
- Consumes: `GET review-data`, `ReviewRuntime.createClient`
- Produces: `loadReviewData(selected, orderId, includeUtilization)`, `refreshReviewDynamicRegions()`, `upsertReviewTimeline(data)`

- [ ] **Step 1: Tighten failing contract tests**

Require:

- `getReviewData` exists and accepts `signal`.
- Review linkage does not contain `Promise.all(ids.map`.
- Review loading functions do not call `renderCurrentPage()`.
- Stable IDs `review-comparison-region`, `review-utilization-region`, and `review-gantt-region` exist.
- Timeline entries store `items`, `groups`, and reuse them with `clear()`/`add()`.

Append:

```python
    def test_review_linkage_is_batched_and_locally_rendered(self):
        self.assertIn("getReviewData(", JS)
        self.assertNotIn("Promise.all(ids.map((id) =>", JS)
        self.assertIn('id="review-comparison-region"', JS)
        self.assertIn('id="review-utilization-region"', JS)
        self.assertIn('id="review-gantt-region"', JS)
        self.assertIn("refreshReviewDynamicRegions()", JS)
        self.assertIn(".items.clear()", JS)
        self.assertIn(".groups.clear()", JS)
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `PYTHONPATH=.. .venv/bin/python -m pytest -q tests/test_review_frontend_contract.py`

Expected: FAIL on the old N-request and full-render patterns.

- [ ] **Step 3: Add API methods and state**

Add `api.getReviewData(taskId, ids, orderId, includeUtilization, signal)` and `api.searchReviewOrders(taskId, ids, query, signal)` using `this.request(url, { signal })`.

```javascript
getReviewData(taskId, solutionIds, orderId, includeUtilization, signal) {
  const params = new URLSearchParams({
    solution_ids: ReviewRuntime.normalizeIds(solutionIds).join(","),
    include_utilization: includeUtilization ? "true" : "false",
  });
  if (orderId) params.set("order_id", orderId);
  return this.request(`/optimize/hybrid/result/${encodeURIComponent(taskId)}/review-data?${params}`, {signal});
},
searchReviewOrders(taskId, solutionIds, query, signal) {
  const params = new URLSearchParams({
    solution_ids: ReviewRuntime.normalizeIds(solutionIds).join(","),
    q: query || "",
    limit: "50",
  });
  return this.request(`/optimize/hybrid/result/${encodeURIComponent(taskId)}/review-orders?${params}`, {signal});
},
```

Replace `reviewGantt`/`typeUtilization` state with:

```javascript
reviewRead: {
  selectionKey: null,
  scheduleKey: null,
  orderId: null,
  schemes: {},
  utilization: null,
  failedIds: [],
  failureMessages: {},
  loading: false,
  error: null,
},
```

After `api`, instantiate one client with injected API methods. Reset it and `reviewRead` in `resetInstanceDerivedState`.

- [ ] **Step 4: Make render functions pure and add stable regions**

`renderReviewTypeUtilization` and `renderReviewGantt` must only render from `app.reviewRead`; they must not start requests. Wrap the comparison/utilization/Gantt cards in the three stable region IDs. After `renderReview()` sets `innerHTML`, call `ensureReviewData(getSelectedReviewCandidates())` once.

- [ ] **Step 5: Implement the single-request load path**

```javascript
async function loadReviewData(selected, orderId = null, includeUtilization = true) {
  const taskId = app.optimizeResult?.task_id;
  const ids = selected.map((item) => item.id).filter(Boolean);
  if (!taskId || !ids.length) return;
  app.reviewRead = { ...app.reviewRead, loading: true, error: null };
  refreshReviewDynamicRegions();
  try {
    const result = await reviewDataClient.loadData({ taskId, ids, orderId, includeUtilization });
    if (result.cancelled) return;
    const payload = result.payload;
    app.reviewRead = {
      selectionKey: ReviewRuntime.selectionKey(taskId, ids),
      scheduleKey: ReviewRuntime.scheduleKey(taskId, ids, payload.order_id),
      orderId: payload.order_id,
      schemes: payload.schemes || {},
      utilization: payload.type_utilization || app.reviewRead.utilization,
      failedIds: payload.failed_solution_ids || [],
      failureMessages: payload.failure_messages || {},
      loading: false,
      error: null,
    };
  } catch (error) {
    app.reviewRead = { ...app.reviewRead, loading: false, error: error.message || String(error) };
  }
  refreshReviewDynamicRegions();
}
```

`ensureReviewData` compares the selection key and calls `loadReviewData` only when it changes. Order selection calls it with `includeUtilization=false`.

- [ ] **Step 6: Reuse the review Timeline**

Extend each `ganttInstances` entry with `items` and `groups`. `upsertReviewTimeline` finds `gantt-review-compare`, clears/adds both DataSets, calls `timeline.setOptions({min,max})`, calls `timeline.setWindow(initialStart, initialEnd, {animation:false})`, and updates/adds custom time `sched-now`. Do not replace the canvas DOM during loading.

- [ ] **Step 7: Make candidate toggling local**

After updating `app.reviewSelection`, rerender only `review-comparison-region`, update the selected-count element, and call `ensureReviewData`. Do not call `renderCurrentPage()` from `toggle-candidate`.

- [ ] **Step 8: Run frontend and backend regression tests**

Run: `PYTHONPATH=.. .venv/bin/python -m pytest -q tests/test_review_frontend_contract.py tests/test_review_frontend_runtime.py tests/test_review_api.py`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add frontend/app_v2.js tests/test_review_frontend_contract.py
git commit -m "perf: batch and reuse review linkage views"
```

---

### Task 7: Replace order selects with a reusable fuzzy combobox

**Files:**
- Modify: `frontend/app_v2.js:1785-1874, 2942-2974, 5470-5650`
- Modify: `frontend/app_v2.css:2259-2275`
- Modify: `tests/test_review_frontend_contract.py`

**Interfaces:**
- Consumes: `ReviewRuntime.rankOrders`, `reviewDataClient.searchOrders`
- Produces markup with ARIA combobox/listbox semantics and one selection callback

- [ ] **Step 1: Add failing combobox contract assertions**

Require `role="combobox"`, `role="listbox"`, `aria-expanded`, a 200ms debounce constant, a 50-result limit, ArrowDown/ArrowUp/Enter/Escape handlers, and absence of `data-review-gantt-order` and `data-gantt-order-select`.

```python
    def test_order_selectors_use_accessible_fuzzy_combobox(self):
        for token in (
            'role="combobox"', 'role="listbox"', "aria-expanded",
            "ORDER_SEARCH_DEBOUNCE_MS = 200", "ORDER_SEARCH_LIMIT = 50",
            '"ArrowDown"', '"ArrowUp"', '"Enter"', '"Escape"',
        ):
            self.assertIn(token, JS)
        self.assertNotIn("data-review-gantt-order", JS)
        self.assertNotIn("data-gantt-order-select", JS)
```

- [ ] **Step 2: Run and verify failure**

Run: `PYTHONPATH=.. .venv/bin/python -m pytest -q tests/test_review_frontend_contract.py`

Expected: FAIL because both native selects still exist.

- [ ] **Step 3: Add the reusable renderer and mount function**

Add:

```javascript
const ORDER_SEARCH_DEBOUNCE_MS = 200;
const ORDER_SEARCH_LIMIT = 50;
app.orderComboboxSources = new Map();

function renderOrderCombobox(config) {
  app.orderComboboxSources.set(config.id, config);
  const selectedLabel = config.selected
    ? [config.selected.order_id, config.selected.order_name].filter(Boolean).join(" · ")
    : "";
  return `
    <div class="order-combobox" data-order-combobox="${escapeHtml(config.id)}">
      <input type="search" role="combobox" aria-autocomplete="list"
        aria-expanded="false" aria-controls="${escapeHtml(config.id)}-list"
        value="${escapeHtml(selectedLabel)}" placeholder="输入订单号模糊搜索">
      <div class="order-combobox-list" id="${escapeHtml(config.id)}-list"
        role="listbox" hidden></div>
    </div>`;
}
```

`mountOrderComboboxes` binds each unbound container, debounces input by 200ms, calls `source.search(query)` or local `ReviewRuntime.rankOrders`, renders at most 50 options, supports the four approved keys, and calls `source.select(order)` exactly once.

Use this key handler inside the mount function:

```javascript
input.addEventListener("keydown", async (event) => {
  if (event.key === "Escape") {
    list.hidden = true;
    input.setAttribute("aria-expanded", "false");
    return;
  }
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    const delta = event.key === "ArrowDown" ? 1 : -1;
    activeIndex = Math.max(0, Math.min(results.length - 1, activeIndex + delta));
    renderResults();
    return;
  }
  if (event.key === "Enter" && results[activeIndex]) {
    event.preventDefault();
    await choose(results[activeIndex]);
  }
});

input.addEventListener("input", () => {
  window.clearTimeout(timer);
  timer = window.setTimeout(async () => {
    results = (await source.search(input.value)).slice(0, ORDER_SEARCH_LIMIT);
    activeIndex = results.length ? 0 : -1;
    renderResults();
    list.hidden = false;
    input.setAttribute("aria-expanded", "true");
  }, ORDER_SEARCH_DEBOUNCE_MS);
});
```

- [ ] **Step 4: Integrate all Gantt order selectors**

- Ordinary client-side Gantt: search its local `orderOptions`.
- Single-plan server Gantt: call `api.searchReviewOrders` with one solution ID.
- Review linked Gantt: call `reviewDataClient.searchOrders` with all selected IDs.
- On selection, ordinary Gantt updates its filter; plan Gantt calls `loadPlanGantt(taskId, solutionId, order.order_id)`; review Gantt calls `loadReviewData(getSelectedReviewCandidates(), order.order_id, false)`.
- Call `mountOrderComboboxes` after ordinary Gantt mount and after dynamic review-region refresh.

- [ ] **Step 5: Add combobox styling**

Use a 280px input, an absolutely positioned scrollable list with max-height 280px, visible active/hover states, and no fixed option count in the DOM.

- [ ] **Step 6: Run frontend tests**

Run: `PYTHONPATH=.. .venv/bin/python -m pytest -q tests/test_review_frontend_runtime.py tests/test_review_frontend_contract.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add frontend/app_v2.js frontend/app_v2.css tests/test_review_frontend_contract.py
git commit -m "feat: add fuzzy order combobox to gantt views"
```

---

### Task 8: Simplify the comparison and utilization tables

**Files:**
- Modify: `frontend/app_v2.js:2736-2840`
- Modify: `frontend/app_v2.css:807-907`
- Modify: `tests/test_review_frontend_contract.py`

**Interfaces:**
- Preserves: checkbox selection and best-value highlighting
- Changes visible utilization cells to percentage only

- [ ] **Step 1: Add failing table contract assertions**

Assert:

- The first comparison header contains `sr-only` text but no visible `选`.
- Utilization cells do not contain `used_machines`, `machines_total`, `cell-sub`, or `util-bar`.
- `.util-col-type` is between 120px and 140px.
- `.util-plan-name` has normal wrapping and no ellipsis/max-width.

```python
    def test_review_tables_use_compact_approved_content(self):
        self.assertIn('<span class="sr-only">选择方案</span>', JS)
        utilization_start = JS.index("function renderReviewTypeUtilization")
        utilization_end = JS.index("function buildReviewGanttData")
        utilization_source = JS[utilization_start:utilization_end]
        for token in ("used_machines", "machines_total", "util-bar", "cell-sub"):
            self.assertNotIn(token, utilization_source)
        self.assertIn(".util-col-type { width: 132px; }", CSS)
        self.assertIn("white-space: normal", CSS)
        self.assertNotIn("text-overflow: ellipsis", CSS[CSS.index(".util-table"):CSS.index(".table-shell")])
```

- [ ] **Step 2: Run and verify failure**

Run: `PYTHONPATH=.. .venv/bin/python -m pytest -q tests/test_review_frontend_contract.py`

Expected: FAIL on all current table details.

- [ ] **Step 3: Make the first header accessible but visually blank**

Build headers separately so the first cell is:

```html
<th class="compare-check"><span class="sr-only">选择方案</span></th>
```

Keep the row checkbox `aria-label`.

- [ ] **Step 4: Reduce utilization cells to the approved content**

Render the first column as `<strong>${escapeHtml(type.type_name)}</strong>` and every plan cell as only `formatPercent(entry.utilization)`, with `<strong>` for the row maximum. Remove the bar, used/total count, type ID, and critical marker.

- [ ] **Step 5: Apply compact responsive CSS**

Set `.util-col-type { width: 132px; }`; let plan columns share remaining width; make `.util-plan-name` `white-space: normal`, `overflow: visible`, `word-break: break-word`; remove unused bar and sublabel rules.

- [ ] **Step 6: Run frontend contract tests**

Run: `PYTHONPATH=.. .venv/bin/python -m pytest -q tests/test_review_frontend_contract.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add frontend/app_v2.js frontend/app_v2.css tests/test_review_frontend_contract.py
git commit -m "fix: simplify review comparison tables"
```

---

### Task 9: Add performance and browser acceptance verification

**Files:**
- Create: `tools/benchmark_review_read.py`
- Create: `tools/verify_review_ui.py`
- Modify: `docs/superpowers/specs/2026-07-18-gantt-review-performance-design.md`

**Interfaces:**
- Benchmark prints cold index, warm lookup, search, and four-scheme batch timings as JSON.
- Browser verifier exits nonzero when any approved UI contract fails.

- [ ] **Step 1: Create the deterministic read benchmark**

Build 5000 order IDs, 20 entries per order, 40 machine IDs mapped onto the fixture shop, and four solution schedules. Measure with `time.perf_counter()`:

- four cold `build_review_solution_index` calls;
- 100 warm `entries_by_order[target]` lookups;
- 100 fuzzy `search_order_facets` calls;
- combined extraction of four scheme lists.

Print JSON containing the fixture sizes and median/total milliseconds. Assert data counts and returned order IDs, but do not hard-fail on wall-clock thresholds; print target comparisons for `<1000ms` cold and `<300ms` warm interaction so hardware variance remains visible.

- [ ] **Step 2: Create the Playwright review verifier**

Against a running local server with restored optimization data:

1. Navigate to `solution-review`.
2. Assert first comparison header visible text is empty.
3. Assert utilization headers are not visually clipped and cells match `^-?\\d+(\\.\\d+)?%$` or `-`.
4. Open the order combobox, type a substring, select a result with keyboard, and count exactly one `/review-data` request.
5. Read the active Timeline `getWindow()` through `page.evaluate`; assert its span is at most 96 hours plus one minute when the full range exceeds 96 hours.
6. Rapidly select two different orders and assert the final input/order matches the second choice with no console error.

- [ ] **Step 3: Run the complete automated suite**

Run:

```bash
PYTHONPATH=.. .venv/bin/python -m pytest -q
PYTHONPATH=.. .venv/bin/python -m llm4drd.tools.benchmark_review_read
```

Expected: all pytest tests PASS; benchmark reports correct sizes and prints all timing fields.

- [ ] **Step 4: Run the browser verifier**

Start the server in one terminal:

```bash
cd /Users/zhouwentao/Desktop
PYTHONPATH=. llm4drd/.venv/bin/uvicorn llm4drd.api.server:app --host 127.0.0.1 --port 8888
```

Run in another:

```bash
cd /Users/zhouwentao/Desktop
PYTHONPATH=. llm4drd/.venv/bin/python -m llm4drd.tools.verify_review_ui --base-url http://127.0.0.1:8888/
```

Expected: JSON output with every check `ok: true` and process exit 0.

- [ ] **Step 5: Record measured results in the approved spec**

Append a dated “Implementation verification” section containing the pytest count, benchmark fixture/timings, browser result, and any hardware note. Do not change the approved requirements.

- [ ] **Step 6: Run final verification**

Run:

```bash
git diff --check
PYTHONPATH=.. .venv/bin/python -m pytest -q
git status --short
```

Expected: no whitespace errors, all tests PASS, and only intended files are modified.

- [ ] **Step 7: Commit**

```bash
git add tools/benchmark_review_read.py tools/verify_review_ui.py docs/superpowers/specs/2026-07-18-gantt-review-performance-design.md
git commit -m "test: verify review performance and interaction"
```

---

## Final Review Gate

Before claiming completion:

1. Confirm `git log --oneline -9` contains one focused commit per task.
2. Confirm `git diff HEAD~9 --check` is clean.
3. Run the full pytest suite fresh and report the exact pass/fail count.
4. Run the 5000-order/100000-entry/4-scheme benchmark and report cold/warm/search timings.
5. Run the Playwright verifier and report each check.
6. Search `api/server.py` to prove review endpoints cannot reach `_rule_reference_solution`.
7. Search `frontend/app_v2.js` to prove the old `Promise.all(ids.map((id) => api.getOptimizeSolutionSchedule` path is gone.
