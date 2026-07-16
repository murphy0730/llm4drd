# Unified Graph Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the duplicated display/optimizer graph construction paths with one canonical graph builder and a versioned SQLite-backed `GraphContext` that cuts warm optimizer initialization time by at least 50% without changing deterministic optimization results.

**Architecture:** A pure domain layer builds `CanonicalGraph` and `GraphFingerprint` from `ShopFloor`. Two projections derive the existing display graph rows and an immutable compute context; `GraphArtifactStore` persists both atomically, while `GraphContextService` provides L1 memory, L2 SQLite, and rebuild-on-miss behavior. The hybrid optimizer supports `legacy`, `shadow`, and `active` modes so graph equivalence and optimization output can be proven before the new path becomes default.

**Tech Stack:** Python 3.10+, standard library (`dataclasses`, `hashlib`, `json`, `threading`, `time`), NetworkX 3.x compatibility adapter, SQLite/WAL, FastAPI, `unittest`.

## Global Constraints

- Do not change `Simulator` event progression or make ordinary simulation depend on `GraphContext`.
- Do not add NumPy, GNN, surrogate-model, external cache, or binary artifact dependencies.
- Persist all cross-process graph cache data in the existing SQLite database.
- Keep the current graph API response contract, order subgraph behavior, fuzzy order search, and `OS_` resource filtering.
- Keep candidate parameters, random-number call order, graph profile names, objective formulas, NSGA-III selection, ALNS operators, approximate evaluation, and exact simulation semantics unchanged.
- A cache miss or stale/corrupt cache rebuilds automatically; users are never required to prebuild the graph.
- Any fingerprint, schema, builder, or integrity mismatch conservatively invalidates both projections.
- A corrupt cache gets exactly one automatic rebuild attempt; a second validation failure terminates the operation explicitly.
- `GraphContext` is immutable after construction and no optimization hot loop may query SQLite or NetworkX.
- Fixed instance, configuration, seed, worker count, and evaluation count must produce identical candidate sequences and final serialized solutions across legacy, cold-build, SQLite-hit, and L1-hit paths.
- Warm initialization on medium and large fixtures must be at least 50% faster than legacy; fixed-evaluation end-to-end runtime may not regress by more than 3%; peak memory may not exceed 1.25x legacy.
- Use TDD for every behavior change and commit after every task.

---

## File Structure

### New files

- `knowledge/canonical.py` — fingerprint types, canonical nodes/edges, normalization, hashing, and the single business-relation builder.
- `knowledge/context.py` — immutable `GraphContext`, display/compute projections, validation, accessors, and shadow diffing.
- `knowledge/context_service.py` — L1/L2/rebuild orchestration, single-flight locking, invalidation, diagnostics, and mode parsing.
- `data/graph_artifact_store.py` — SQLite loading and atomic dual-projection persistence.
- `tests/shop_fixtures.py` — deterministic graph/optimizer fixtures and stable result-signature helpers.
- `tests/test_canonical_graph.py` — fingerprint and canonical edge semantics.
- `tests/test_graph_context.py` — compute projection, accessors, validation, and shadow parity.
- `tests/test_graph_artifact_store.py` — schema, round trip, rollback, stale, and corruption tests.
- `tests/test_graph_context_service.py` — L1/L2/miss, rebuild, invalidation, and single-flight tests.
- `tests/test_hybrid_graph_context.py` — legacy/shadow/active deterministic optimizer equivalence.
- `tools/benchmark_graph_context.py` — repeatable initialization, end-to-end, and memory benchmark.
- `docs/benchmarks/graph-context-baseline.json` — measured legacy baseline.
- `docs/benchmarks/graph-context-result.json` — measured cold/L2/L1 results.
- `docs/benchmarks/graph-context-report.md` — acceptance comparison and environment metadata.

### Modified files

- `knowledge/graph.py` — retain public NetworkX wrapper but delegate business semantics to `CanonicalGraphBuilder`.
- `knowledge/__init__.py` — export stable canonical/context/service public interfaces.
- `data/db.py` — add idempotent compute-graph tables, indexes, and display meta columns.
- `data/__init__.py` — export `GraphArtifactStore`.
- `optimization/hybrid_nsga3_alns.py` — accept `GraphContext`, add mode-specific graph access, and remove NetworkX from the active hot path.
- `api/server.py` — instantiate the service, unify graph build and optimization initialization, expose diagnostics, and invalidate through one method.
- `README.md` — document graph modes, cache behavior, and operational rollback.

---

### Task 1: Freeze Legacy Behavior and Add Deterministic Fixtures

**Files:**
- Create: `tests/shop_fixtures.py`
- Create: `tests/test_graph_legacy_baseline.py`
- Test: `tests/test_graph.py`
- Test: `tests/test_simulator_robustness.py`

**Interfaces:**
- Consumes: existing `ShopFloor`, `HeterogeneousGraph`, `HybridConfig`, and `HybridNSGA3ALNSOptimizer`.
- Produces: `make_graph_context_shop() -> ShopFloor`, `canonical_graph_signature(graph) -> tuple`, and `hybrid_result_signature(result) -> dict` for all later tests.

- [ ] **Step 1: Add a deterministic shared fixture**

Create `tests/shop_fixtures.py` with explicit IDs and insertion order. The fixture must include two orders, a main task, task and operation predecessors, an explicit machine choice, a process-type-derived machine choice, tooling, personnel, a critical machine type, and finite shift calendars.

```python
from __future__ import annotations

from llm4drd.core.models import (
    Machine, MachineType, Operation, Order, Personnel, Shift,
    ShopFloor, Task, Tooling, ToolingType,
)


def make_graph_context_shop() -> ShopFloor:
    shop = ShopFloor()
    shop.machine_types["cut"] = MachineType("cut", "Cut", is_critical=True)
    shop.machine_types["asm"] = MachineType("asm", "Assembly", is_critical=False)
    shifts = [Shift(day=day, start_hour=0.0, hours=24.0) for day in range(14)]
    shop.machines["M-C1"] = Machine("M-C1", "Cutter 1", "cut", shifts=list(shifts))
    shop.machines["M-C2"] = Machine("M-C2", "Cutter 2", "cut", shifts=list(shifts))
    shop.machines["M-A1"] = Machine("M-A1", "Assembly 1", "asm", shifts=list(shifts))
    shop.tooling_types["TL-CUT"] = ToolingType("TL-CUT", "Cut fixture")
    shop.toolings["TL-1"] = Tooling("TL-1", "Fixture 1", "TL-CUT", shifts=list(shifts))
    shop.personnel["P-1"] = Personnel("P-1", "Assembler", ["SK-ASM"], shifts=list(shifts))

    shop.orders["O-1"] = Order("O-1", "Order 1", release_time=0.0, due_date=40.0, priority=3)
    shop.orders["O-2"] = Order("O-2", "Order 2", release_time=1.0, due_date=48.0, priority=1)
    t11 = Task("T-11", "O-1", "Part", False, [], release_time=0.0, due_date=24.0)
    t12 = Task("T-12", "O-1", "Main", True, ["T-11"], release_time=0.0, due_date=40.0)
    t21 = Task("T-21", "O-2", "Second", True, [], release_time=1.0, due_date=48.0)
    shop.tasks.update({task.id: task for task in (t11, t12, t21)})

    op11 = Operation("OP-11", "T-11", "Cut", "cut", 4.0, eligible_machine_ids=["M-C1"])
    op12 = Operation("OP-12", "T-11", "Finish", "cut", 2.0, predecessor_ops=["OP-11"])
    op13 = Operation(
        "OP-13", "T-12", "Assemble", "asm", 3.0,
        predecessor_tasks=["T-11"], required_tooling_types=["TL-CUT"],
        required_personnel_skills=["SK-ASM"],
    )
    op21 = Operation("OP-21", "T-21", "Other cut", "cut", 5.0)
    for task, operations in ((t11, [op11, op12]), (t12, [op13]), (t21, [op21])):
        task.operations.extend(operations)
        shop.orders[task.order_id].task_ids.append(task.id)
        if task.is_main:
            shop.orders[task.order_id].main_task_id = task.id
        for operation in operations:
            shop.operations[operation.id] = operation
    shop.build_indexes()
    return shop


def canonical_graph_signature(graph) -> tuple:
    nodes = tuple(sorted((node_id, tuple(sorted(attrs.items()))) for node_id, attrs in graph.nodes(data=True)))
    edges = tuple(sorted((source, target, tuple(sorted(attrs.items()))) for source, target, attrs in graph.edges(data=True)))
    return nodes, edges


def hybrid_result_signature(result) -> dict:
    payload = result.to_export_dict()
    return {
        "baseline": payload["baseline"],
        "solutions": payload["solutions"],
        "archive_size": payload["archive_size"],
        "found_solution_count": payload["found_solution_count"],
        "generations_completed": payload["generations_completed"],
        "total_evaluations": payload["total_evaluations"],
        "approximate_evaluations": payload["approximate_evaluations"],
        "exact_evaluations": payload["exact_evaluations"],
    }
```

- [ ] **Step 2: Add legacy characterization tests before refactoring**

Create `tests/test_graph_legacy_baseline.py`:

```python
import unittest

from llm4drd.knowledge.graph import HeterogeneousGraph
from llm4drd.optimization.hybrid_nsga3_alns import HybridConfig, HybridNSGA3ALNSOptimizer
from llm4drd.tests.shop_fixtures import (
    canonical_graph_signature, hybrid_result_signature, make_graph_context_shop,
)


class LegacyGraphBaselineTests(unittest.TestCase):
    def test_legacy_graph_is_deterministic(self):
        left = HeterogeneousGraph(); left.build_from_shopfloor(make_graph_context_shop())
        right = HeterogeneousGraph(); right.build_from_shopfloor(make_graph_context_shop())
        self.assertEqual(canonical_graph_signature(left.graph), canonical_graph_signature(right.graph))

    def test_legacy_hybrid_is_deterministic(self):
        config = HybridConfig(
            objective_keys=["total_tardiness", "makespan"],
            target_solution_count=2, population_size=4, generations=1,
            alns_iterations_per_candidate=0, time_limit_s=60,
            parallel_workers=1, seed=17,
        )
        left = HybridNSGA3ALNSOptimizer(make_graph_context_shop(), config).run()
        right = HybridNSGA3ALNSOptimizer(make_graph_context_shop(), config).run()
        self.assertEqual(hybrid_result_signature(left), hybrid_result_signature(right))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the characterization suite**

Run from the repository root:

```bash
cd .. && llm4drd/.venv/bin/python -m unittest \
  llm4drd.tests.test_graph_legacy_baseline \
  llm4drd.tests.test_graph \
  llm4drd.tests.test_simulator_robustness -v
```

Expected: all tests pass. If deterministic optimizer output fails before refactoring, reduce no semantic assertion; instead fix the fixture/config so wall-clock termination cannot affect evaluation count.

- [ ] **Step 4: Commit the baseline**

```bash
git add tests/shop_fixtures.py tests/test_graph_legacy_baseline.py
git commit -m "test: freeze legacy graph optimizer behavior"
```

---

### Task 2: Implement Stable Fingerprints and the Canonical Graph Builder

**Files:**
- Create: `knowledge/canonical.py`
- Create: `tests/test_canonical_graph.py`
- Modify: `knowledge/__init__.py`
- Modify: `knowledge/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `ShopFloor` and the existing graph node/edge semantics.
- Produces: `GraphFingerprint`, `CanonicalNode`, `CanonicalEdge`, `CanonicalGraph`, `compute_graph_fingerprint(shop)`, and `CanonicalGraphBuilder.build(shop, progress_callback=None, deadline=None)`.

- [ ] **Step 1: Write failing fingerprint tests**

Create `tests/test_canonical_graph.py` with these first tests:

```python
import copy
import math
import unittest

from llm4drd.knowledge.canonical import CanonicalGraphBuilder, compute_graph_fingerprint
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class GraphFingerprintTests(unittest.TestCase):
    def test_fingerprint_is_independent_of_dictionary_order(self):
        left = make_graph_context_shop()
        right = make_graph_context_shop()
        right.operations = dict(reversed(list(right.operations.items())))
        right.tasks = dict(reversed(list(right.tasks.items())))
        self.assertEqual(compute_graph_fingerprint(left), compute_graph_fingerprint(right))

    def test_processing_time_changes_feature_not_topology(self):
        left = make_graph_context_shop()
        right = copy.deepcopy(left)
        right.operations["OP-11"].processing_time += 1.0
        a = compute_graph_fingerprint(left); b = compute_graph_fingerprint(right)
        self.assertNotEqual(a.instance_hash, b.instance_hash)
        self.assertEqual(a.topology_hash, b.topology_hash)
        self.assertNotEqual(a.feature_hash, b.feature_hash)

    def test_predecessor_change_changes_topology(self):
        left = make_graph_context_shop()
        right = copy.deepcopy(left)
        right.operations["OP-21"].predecessor_ops.append("OP-12")
        self.assertNotEqual(
            compute_graph_fingerprint(left).topology_hash,
            compute_graph_fingerprint(right).topology_hash,
        )

    def test_non_finite_graph_input_is_rejected(self):
        shop = make_graph_context_shop()
        shop.operations["OP-11"].processing_time = math.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compute_graph_fingerprint(shop)
```

- [ ] **Step 2: Run tests and verify the import failure**

```bash
cd .. && llm4drd/.venv/bin/python -m unittest llm4drd.tests.test_canonical_graph -v
```

Expected: FAIL with `ModuleNotFoundError: llm4drd.knowledge.canonical`.

- [ ] **Step 3: Implement canonical types and stable hashing**

Create `knowledge/canonical.py` with frozen dataclasses, SHA-256 hashing, sorted entity serialization, finite-number validation, constants `GRAPH_SCHEMA_VERSION = 1` and `GRAPH_BUILDER_VERSION = "canonical-v1"`, and this public surface:

```python
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Callable, Mapping, TypeAlias

from ..core.models import ShopFloor

ScalarValue: TypeAlias = str | int | float | bool | None
GRAPH_SCHEMA_VERSION = 1
GRAPH_BUILDER_VERSION = "canonical-v1"


@dataclass(frozen=True)
class GraphFingerprint:
    instance_hash: str
    topology_hash: str
    feature_hash: str
    schema_version: int = GRAPH_SCHEMA_VERSION
    builder_version: str = GRAPH_BUILDER_VERSION


@dataclass(frozen=True)
class CanonicalNode:
    node_id: str
    node_type: str
    entity_id: str
    attrs: Mapping[str, ScalarValue]


@dataclass(frozen=True)
class CanonicalEdge:
    source: str
    target: str
    edge_type: str
    attrs: Mapping[str, ScalarValue]


@dataclass(frozen=True)
class CanonicalGraph:
    nodes: tuple[CanonicalNode, ...]
    edges: tuple[CanonicalEdge, ...]
    fingerprint: GraphFingerprint

    def stats(self) -> dict:
        node_types: dict[str, int] = {}
        edge_types: dict[str, int] = {}
        for node in self.nodes:
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
        for edge in self.edges:
            edge_types[edge.edge_type] = edge_types.get(edge.edge_type, 0) + 1
        return {
            "total_nodes": len(self.nodes), "total_edges": len(self.edges),
            "node_types": node_types, "edge_types": edge_types,
        }


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compute_graph_fingerprint(shop: ShopFloor) -> GraphFingerprint:
    # Build three explicitly sorted payloads. Include all calendars/downtimes in
    # instance_payload, all relation-driving fields in topology_payload, and all
    # static feature inputs in feature_payload. Pass every float through
    # math.isfinite and raise ValueError("non-finite graph input: <field>") on failure.
    instance_payload = _normalized_instance_payload(shop)
    topology_payload = _normalized_topology_payload(shop)
    feature_payload = _normalized_feature_payload(shop)
    return GraphFingerprint(_digest(instance_payload), _digest(topology_payload), _digest(feature_payload))


```

Implement `_normalized_instance_payload`, `_normalized_topology_payload`, and `_normalized_feature_payload` as concrete sorted lists of primitive dicts. Do not serialize `__dict__`; enumerate the approved fields from the design spec so runtime-only resource state cannot leak into hashes.

Implement `CanonicalGraphBuilder.build()` with this exact migration map from `knowledge/graph.py:39-214`:

1. Retain the current task-predecessor aggregation from both `Task.predecessor_task_ids` and `Operation.predecessor_tasks`.
2. Replace every `self.graph.add_node(id, node_type=..., entity_id=..., **attrs)` with `nodes.append(CanonicalNode(id, node_type, entity_id, MappingProxyType(attrs)))`.
3. Replace every `self.graph.add_edge(source, target, edge_type=..., **attrs)` with `edges.append(CanonicalEdge(source, target, edge_type, MappingProxyType(attrs)))`.
4. Preserve every current attribute name and value, including derived dates, slack, status, required tooling, and required personnel.
5. Preserve existing progress/deadline checks after every 500 orders/tasks and 250 operations.
6. Return `CanonicalGraph(tuple(nodes), tuple(edges), compute_graph_fingerprint(shop))` in the exact legacy construction order. Fingerprint payloads and compute ordinals are independently sorted; canonical row order stays legacy-compatible so NetworkX adjacency iteration and display ordering do not change.

- [ ] **Step 4: Add canonical edge parity tests**

Append tests that assert the canonical node/edge sets equal the legacy graph and specifically verify all task-predecessor edges induced by `Operation.predecessor_tasks`.

```python
class CanonicalGraphBuilderTests(unittest.TestCase):
    def test_canonical_edges_match_legacy_graph(self):
        from llm4drd.knowledge.graph import HeterogeneousGraph
        shop = make_graph_context_shop()
        canonical = CanonicalGraphBuilder().build(shop)
        legacy = HeterogeneousGraph(); legacy.build_from_shopfloor(shop)
        expected = {(a, b, d["edge_type"]) for a, b, d in legacy.graph.edges(data=True)}
        actual = {(edge.source, edge.target, edge.edge_type) for edge in canonical.edges}
        self.assertEqual(actual, expected)
```

- [ ] **Step 5: Convert `HeterogeneousGraph` to a compatibility adapter**

Modify `knowledge/graph.py` so `build_from_shopfloor()` calls the canonical builder and only translates canonical rows into NetworkX:

```python
def build_from_shopfloor(self, shop, progress_callback=None, deadline=None):
    canonical = CanonicalGraphBuilder().build(shop, progress_callback, deadline)
    self.graph.clear()
    for node in canonical.nodes:
        self.graph.add_node(
            node.node_id, node_type=node.node_type, entity_id=node.entity_id,
            **dict(node.attrs),
        )
    for edge in canonical.edges:
        self.graph.add_edge(edge.source, edge.target, edge_type=edge.edge_type, **dict(edge.attrs))
```

Export canonical types from `knowledge/__init__.py`.

- [ ] **Step 6: Run graph tests**

```bash
cd .. && llm4drd/.venv/bin/python -m unittest \
  llm4drd.tests.test_canonical_graph \
  llm4drd.tests.test_graph \
  llm4drd.tests.test_graph_legacy_baseline -v
```

Expected: all tests pass and legacy/canonical edge signatures match.

- [ ] **Step 7: Commit**

```bash
git add knowledge/canonical.py knowledge/graph.py knowledge/__init__.py tests/test_canonical_graph.py
git commit -m "feat: add canonical graph builder and fingerprints"
```

---

### Task 3: Build the Immutable Compute Projection

**Files:**
- Create: `knowledge/context.py`
- Create: `tests/test_graph_context.py`
- Modify: `knowledge/__init__.py`

**Interfaces:**
- Consumes: `CanonicalGraph`, `GraphFingerprint`, and `ShopFloor`.
- Produces: `DisplayGraphProjection`, `GraphContext`, `ComputeGraphProjection.build(shop, canonical)`, `validate_graph_context(shop, context)`, and `compare_legacy_context(shop, context)`.

- [ ] **Step 1: Write failing accessor and feature tests**

Create `tests/test_graph_context.py`:

```python
import unittest

from llm4drd.knowledge.canonical import CanonicalGraphBuilder
from llm4drd.knowledge.context import ComputeGraphProjection, validate_graph_context
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class GraphContextTests(unittest.TestCase):
    def setUp(self):
        self.shop = make_graph_context_shop()
        canonical = CanonicalGraphBuilder().build(self.shop)
        self.context = ComputeGraphProjection().build(self.shop, canonical)

    def test_operation_index_is_stable_and_complete(self):
        self.assertEqual(self.context.operation_ids, tuple(sorted(self.shop.operations)))
        self.assertEqual(set(self.context.operation_index), set(self.shop.operations))

    def test_predecessor_successor_accessors_are_inverse(self):
        self.assertEqual(self.context.predecessors("OP-12"), ("OP-11",))
        self.assertIn("OP-12", self.context.successors("OP-11"))

    def test_explicit_and_derived_machine_edges_are_available(self):
        self.assertEqual(self.context.eligible_machines("OP-11"), ("M-C1",))
        self.assertEqual(set(self.context.eligible_machines("OP-12")), {"M-C1", "M-C2"})

    def test_feature_view_contains_legacy_feature_names(self):
        values = self.context.operation_features("OP-13")
        self.assertEqual(
            set(values),
            {"predecessor_depth", "assembly_criticality", "shared_resource_degree", "bottleneck_adjacency", "graph_out_degree"},
        )

    def test_groups_support_profile_expansion(self):
        self.assertEqual(set(self.context.operations_in_group("process_type", "cut")), {"OP-11", "OP-12", "OP-21"})
        self.assertEqual(self.context.operations_in_group("personnel_skill", "SK-ASM"), ("OP-13",))

    def test_context_validates_against_shop(self):
        validate_graph_context(self.shop, self.context)
```

- [ ] **Step 2: Run tests and verify failure**

```bash
cd .. && llm4drd/.venv/bin/python -m unittest llm4drd.tests.test_graph_context -v
```

Expected: FAIL because `knowledge.context` does not exist.

- [ ] **Step 3: Implement projections and immutable accessors**

Create `knowledge/context.py` with frozen tuples and read-only `MappingProxyType` maps. Use one helper to convert sorted `(source, target)` pairs to CSR offsets/indices.

```python
@dataclass(frozen=True)
class DisplayGraphProjection:
    nodes: tuple[CanonicalNode, ...]
    edges: tuple[CanonicalEdge, ...]
    fingerprint: GraphFingerprint
    stats: Mapping[str, object]

    @classmethod
    def from_canonical(cls, graph: CanonicalGraph) -> "DisplayGraphProjection":
        return cls(
            nodes=graph.nodes,
            edges=graph.edges,
            fingerprint=graph.fingerprint,
            stats=MappingProxyType(graph.stats()),
        )


@dataclass(frozen=True)
class GraphContext:
    fingerprint: GraphFingerprint
    operation_ids: tuple[str, ...]
    operation_index: Mapping[str, int]
    machine_ids: tuple[str, ...]
    machine_index: Mapping[str, int]
    predecessor_offsets: tuple[int, ...]
    predecessor_indices: tuple[int, ...]
    successor_offsets: tuple[int, ...]
    successor_indices: tuple[int, ...]
    eligible_machine_offsets: tuple[int, ...]
    eligible_machine_indices: tuple[int, ...]
    feature_names: tuple[str, ...]
    feature_matrix: tuple[tuple[float, ...], ...]
    operation_groups: Mapping[tuple[str, str], tuple[int, ...]]

    def operation_features(self, op_id: str) -> dict[str, float]:
        row = self.feature_matrix[self.operation_index[op_id]]
        return dict(zip(self.feature_names, row))

    def feature_view_by_operation_id(self) -> dict[str, dict[str, float]]:
        return {op_id: self.operation_features(op_id) for op_id in self.operation_ids}

    def _operation_relation(self, op_id, offsets, indices) -> tuple[str, ...]:
        ordinal = self.operation_index[op_id]
        return tuple(self.operation_ids[index] for index in indices[offsets[ordinal]:offsets[ordinal + 1]])

    def predecessors(self, op_id: str) -> tuple[str, ...]:
        return self._operation_relation(op_id, self.predecessor_offsets, self.predecessor_indices)

    def successors(self, op_id: str) -> tuple[str, ...]:
        return self._operation_relation(op_id, self.successor_offsets, self.successor_indices)

    def eligible_machines(self, op_id: str) -> tuple[str, ...]:
        ordinal = self.operation_index[op_id]
        indexes = self.eligible_machine_indices[
            self.eligible_machine_offsets[ordinal]:self.eligible_machine_offsets[ordinal + 1]
        ]
        return tuple(self.machine_ids[index] for index in indexes)

    def operations_in_group(self, group_type: str, group_key: str) -> tuple[str, ...]:
        indexes = self.operation_groups.get((group_type, group_key), ())
        return tuple(self.operation_ids[index] for index in indexes)


class ComputeGraphProjection:
    FEATURE_NAMES = (
        "predecessor_depth", "assembly_criticality", "shared_resource_degree",
        "bottleneck_adjacency", "graph_out_degree",
    )

```

Implement `ComputeGraphProjection.build()` in six explicit passes:

1. Create sorted `operation_ids` and `machine_ids`, plus `MappingProxyType` reverse maps.
2. Extract `operation_sequence` canonical edges into predecessor pairs and reverse them for successor pairs. Expand `op_depends_task` through the predecessor task's operations using `shop.tasks[task_id].operations`, matching the legacy feature formula.
3. Convert sorted pairs to CSR with `_pairs_to_csr(size, pairs)`: initialize `buckets = [[] for _ in range(size)]`, append each target, sort/deduplicate each bucket, extend `indices`, and append the new length to `offsets`.
4. Port `HybridNSGA3ALNSOptimizer._build_graph_features()` line-for-line into a pure `_build_feature_rows(shop, canonical)` helper. Keep every coefficient and denominator unchanged.
5. Build group sets from each operation's `process_type`, `required_tooling_types`, and `required_personnel_skills`; freeze group ordinal tuples in sorted order.
6. Construct `GraphContext`, immediately call `validate_graph_context(shop, context)`, and return it.

`validate_graph_context()` must raise `GraphContextCorruptError` for non-contiguous ordinals, out-of-bounds indices, non-monotonic offsets, non-finite features, missing operations, or metadata count mismatches. Define all five exception classes from the design in this module.

- [ ] **Step 4: Add shadow parity tests against the legacy optimizer features**

Add a pure `build_legacy_graph_features(shop, graph)` helper in `hybrid_nsga3_alns.py` by moving the current `_build_graph_features()` body without formula changes. Both the legacy optimizer method and the shadow test call this helper. Compare every operation and feature with `self.assertAlmostEqual(actual, expected, delta=1e-12)`.

- [ ] **Step 5: Run projection tests**

```bash
cd .. && llm4drd/.venv/bin/python -m unittest \
  llm4drd.tests.test_graph_context \
  llm4drd.tests.test_canonical_graph -v
```

Expected: all tests pass; no NetworkX object is stored in `GraphContext`.

- [ ] **Step 6: Commit**

```bash
git add knowledge/context.py knowledge/__init__.py tests/test_graph_context.py
git commit -m "feat: compile canonical graph into immutable context"
```

---

### Task 4: Add SQLite Schema and Atomic Dual-Projection Persistence

**Files:**
- Modify: `data/db.py`
- Create: `data/graph_artifact_store.py`
- Create: `tests/test_graph_artifact_store.py`
- Modify: `data/__init__.py`

**Interfaces:**
- Consumes: `DisplayGraphProjection`, `GraphContext`, and `GraphFingerprint`.
- Produces: `GraphArtifactStore.load_context(fingerprint)`, `save_artifacts(display, context, precommit_check=None)`, `clear_all()`, `mark_invalid(reason)`, and `load_context_meta()`.

- [ ] **Step 1: Write failing schema and round-trip tests**

Create `tests/test_graph_artifact_store.py` using `TemporaryDirectory`:

```python
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm4drd.data.db import init_db
from llm4drd.data.graph_artifact_store import GraphArtifactStore
from llm4drd.knowledge.canonical import CanonicalGraphBuilder
from llm4drd.knowledge.context import ComputeGraphProjection, DisplayGraphProjection
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class GraphArtifactStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "graph.db")
        init_db(self.db_path)
        shop = make_graph_context_shop()
        canonical = CanonicalGraphBuilder().build(shop)
        self.display = DisplayGraphProjection.from_canonical(canonical)
        self.context = ComputeGraphProjection().build(shop, canonical)
        self.store = GraphArtifactStore(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_schema_is_idempotent_and_context_round_trips(self):
        init_db(self.db_path)
        self.store.save_artifacts(self.display, self.context)
        loaded = self.store.load_context(self.context.fingerprint)
        self.assertEqual(loaded, self.context)

    def test_mismatched_fingerprint_is_a_cache_miss(self):
        self.store.save_artifacts(self.display, self.context)
        stale = self.context.fingerprint.__class__("x", "y", "z", 1, "canonical-v1")
        self.assertIsNone(self.store.load_context(stale))

    def test_precommit_failure_rolls_back_every_projection(self):
        self.store.save_artifacts(self.display, self.context)
        before = self.store.load_context_meta()
        with self.assertRaisesRegex(RuntimeError, "instance changed"):
            self.store.save_artifacts(
                self.display, self.context,
                precommit_check=lambda: (_ for _ in ()).throw(RuntimeError("instance changed")),
            )
        self.assertEqual(self.store.load_context_meta(), before)
```

- [ ] **Step 2: Run tests and verify failure**

```bash
cd .. && llm4drd/.venv/bin/python -m unittest llm4drd.tests.test_graph_artifact_store -v
```

Expected: FAIL because `GraphArtifactStore` and compute tables do not exist.

- [ ] **Step 3: Extend `init_db()` idempotently**

Add the four approved tables and indexes to `data/db.py`: `graph_context_meta`, `graph_entity_index`, `graph_context_relations`, `graph_operation_features`, and `graph_operation_groups`. Add fingerprint/build columns to `graph_meta` using `_safe_add_column()` so existing databases migrate without table replacement.

Use exact column names from the design spec. Add `invalid_reason TEXT DEFAULT ''` to both meta tables for diagnostics. Existing `GraphStore` query methods must keep working.

- [ ] **Step 4: Implement `GraphArtifactStore`**

Implement load/save using `get_db(self.db_path)`. `save_artifacts()` must:

1. Verify display/context fingerprints are equal.
2. Enter one `BEGIN IMMEDIATE` transaction via a dedicated connection helper that does not auto-commit before the method ends.
3. Execute `precommit_check` after acquiring the write transaction and before deleting rows.
4. Delete both old projections.
5. Insert display nodes/edges and all compute rows with `executemany` batches.
6. Insert `graph_meta` and `graph_context_meta` last.
7. Roll back on every exception.

`load_context()` must load all rows in deterministic SQL order and reconstruct the exact frozen `GraphContext`.

- [ ] **Step 5: Add corruption and rollback fault-injection tests**

After a successful save, directly delete one feature row and assert `load_context()` raises `GraphContextCorruptError`. Add an injected failure hook after relation insertion and verify neither meta table exposes a partial new fingerprint.

- [ ] **Step 6: Run storage and legacy graph-store tests**

```bash
cd .. && llm4drd/.venv/bin/python -m unittest \
  llm4drd.tests.test_graph_artifact_store \
  llm4drd.tests.test_graph -v
```

Expected: all tests pass; current order subgraph/search tests remain unchanged.

- [ ] **Step 7: Commit**

```bash
git add data/db.py data/graph_artifact_store.py data/__init__.py tests/test_graph_artifact_store.py
git commit -m "feat: persist graph display and compute artifacts atomically"
```

---

### Task 5: Implement L1/L2/Rebuild GraphContextService

**Files:**
- Create: `knowledge/context_service.py`
- Create: `tests/test_graph_context_service.py`
- Modify: `knowledge/__init__.py`

**Interfaces:**
- Consumes: `CanonicalGraphBuilder`, projections, `GraphArtifactStore`, and `ShopFloor`.
- Produces: `GraphContextMode`, `GraphContextDiagnostics`, `GraphContextService.get_or_build(...)`, `invalidate(reason)`, `clear_memory_cache()`, and `resolve_graph_context_mode()`.

- [ ] **Step 1: Write failing cache-path tests**

Create `tests/test_graph_context_service.py`:

```python
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm4drd.data.db import init_db
from llm4drd.data.graph_artifact_store import GraphArtifactStore
from llm4drd.knowledge.context_service import GraphContextService
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class GraphContextServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        db_path = str(Path(self.tmp.name) / "graph.db")
        init_db(db_path)
        self.service = GraphContextService(GraphArtifactStore(db_path))
        self.shop = make_graph_context_shop()

    def tearDown(self):
        self.tmp.cleanup()

    def test_miss_then_l1_hit_then_sqlite_hit(self):
        first, a = self.service.get_or_build(self.shop)
        self.assertEqual(a.cache_level, "built")
        second, b = self.service.get_or_build(self.shop)
        self.assertIs(second, first)
        self.assertEqual(b.cache_level, "l1")
        self.service.clear_memory_cache()
        third, c = self.service.get_or_build(self.shop)
        self.assertEqual(third, first)
        self.assertEqual(c.cache_level, "sqlite")

    def test_invalidation_forces_rebuild(self):
        _, first = self.service.get_or_build(self.shop)
        self.service.invalidate("operation_updated")
        _, second = self.service.get_or_build(self.shop)
        self.assertEqual(first.cache_level, "built")
        self.assertEqual(second.cache_level, "built")

    def test_same_fingerprint_builds_once_under_concurrency(self):
        barrier = threading.Barrier(3)
        results = []
        def run():
            barrier.wait(); results.append(self.service.get_or_build(self.shop)[1].cache_level)
        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads: thread.start()
        barrier.wait()
        for thread in threads: thread.join()
        self.assertEqual(results.count("built"), 1)
        self.assertEqual(len(results), 2)
```

- [ ] **Step 2: Run tests and verify failure**

```bash
cd .. && llm4drd/.venv/bin/python -m unittest llm4drd.tests.test_graph_context_service -v
```

Expected: FAIL because `knowledge.context_service` does not exist.

- [ ] **Step 3: Implement modes, diagnostics, and cache orchestration**

Create exact public types:

```python
class GraphContextMode(str, Enum):
    LEGACY = "legacy"
    SHADOW = "shadow"
    ACTIVE = "active"


@dataclass(frozen=True)
class GraphContextDiagnostics:
    cache_level: str
    cache_hit: bool
    fingerprint: GraphFingerprint
    load_time_ms: float
    build_time_ms: float
    validation_time_ms: float
    operation_count: int
    relation_count: int
    invalid_reason: str = ""


class GraphContextService:
    def __init__(self, store: GraphArtifactStore, builder=None, compute_projection=None):
        self.store = store
        self.builder = builder or CanonicalGraphBuilder()
        self.compute_projection = compute_projection or ComputeGraphProjection()
        self._lock = threading.Condition()
        self._memory_context = None
        self._building: set[GraphFingerprint] = set()
        self._failures: dict[GraphFingerprint, BaseException] = {}

    def get_or_build(
        self, shop: ShopFloor, *, force_rebuild: bool = False,
        progress_callback=None, deadline: float | None = None,
        current_fingerprint_provider=None,
    ) -> tuple[GraphContext, GraphContextDiagnostics]:
        """Return L1, L2, or a newly built validated context."""

    def invalidate(self, reason: str) -> None:
        with self._lock:
            self._memory_context = None
        self.store.mark_invalid(reason)

    def clear_memory_cache(self) -> None:
        with self._lock:
            self._memory_context = None
```

Use a lock plus condition variable keyed by fingerprint. Never hold the service lock while building or doing SQLite I/O. Waiters must receive the builder's result or exception and must not rebuild independently.

Implement `get_or_build()` as this state machine, keeping all timers in `time.perf_counter()`:

1. Compute fingerprint; under the condition lock, return `_memory_context` when it matches and `force_rebuild` is false.
2. If the same fingerprint is in `_building`, wait until it leaves `_building`; then return the matching L1 value or raise the stored failure.
3. Add fingerprint to `_building`, release the lock, and attempt `store.load_context()` unless forced.
4. Validate an L2 hit, publish it to L1 under the lock, notify waiters, and return `cache_level="sqlite"`.
5. On a miss, build canonical/display/compute projections and validate. When `current_fingerprint_provider` is supplied, call it immediately before save and raise `GraphContextStaleError("instance changed during graph build")` unless it equals the captured fingerprint; only then call `save_artifacts()`.
6. Publish the built context, remove the fingerprint from `_building`, notify waiters, and return `cache_level="built"`.
7. On any exception, store it in `_failures`, remove `_building`, notify waiters, and re-raise. For `GraphContextCorruptError`, clear the artifacts and repeat the build path exactly once before publishing failure.

On load corruption: clear artifacts, rebuild once, validate once, and propagate the second failure as `GraphContextBuildError` with the original corruption chained.

`resolve_graph_context_mode()` reads `LLM4DRD_GRAPH_CONTEXT_MODE`, accepts only `legacy|shadow|active`, logs and returns `legacy` for an invalid value.

- [ ] **Step 4: Add corruption-rebuild and builder-failure tests**

Inject a fake store that raises `GraphContextCorruptError` on first load and succeeds after `save_artifacts`; assert one rebuild. Inject a builder that fails twice and assert the service does not loop.

- [ ] **Step 5: Run service and store tests**

```bash
cd .. && llm4drd/.venv/bin/python -m unittest \
  llm4drd.tests.test_graph_context_service \
  llm4drd.tests.test_graph_artifact_store -v
```

Expected: all tests pass and concurrency test records one `built` result.

- [ ] **Step 6: Commit**

```bash
git add knowledge/context_service.py knowledge/__init__.py tests/test_graph_context_service.py
git commit -m "feat: add versioned graph context cache service"
```

---

### Task 6: Move the Display Graph API onto the Unified Build Path

**Files:**
- Modify: `api/server.py`
- Modify: `data/db.py`
- Modify: `tests/test_graph.py`
- Create: `tests/test_graph_api_context.py`

**Interfaces:**
- Consumes: `GraphArtifactStore`, `GraphContextService`, existing `GraphStore` read methods, and graph build task status.
- Produces: one process-wide `graph_context_service`; graph build uses `force_rebuild=True`; all instance mutations call `_invalidate_graph_context(reason)`.

- [ ] **Step 1: Write failing API integration tests**

Avoid binding the production database. Patch module globals with temporary stores and call endpoint coroutines via `asyncio.run()`.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from llm4drd.api import server
from llm4drd.data.db import GraphStore, init_db
from llm4drd.data.graph_artifact_store import GraphArtifactStore
from llm4drd.knowledge.context_service import GraphContextService
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class GraphApiContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        db_path = str(Path(self.tmp.name) / "graph.db")
        init_db(db_path)
        self.originals = (
            server.graph_store,
            server.graph_artifact_store,
            server.graph_context_service,
        )
        server.graph_store = GraphStore(db_path)
        server.graph_artifact_store = GraphArtifactStore(db_path)
        server.graph_context_service = GraphContextService(server.graph_artifact_store)

    def tearDown(self):
        (
            server.graph_store,
            server.graph_artifact_store,
            server.graph_context_service,
        ) = self.originals
        self.tmp.cleanup()

    def test_graph_build_writes_display_and_compute_meta(self):
        context, diagnostics = server._build_graph_artifacts(
            make_graph_context_shop(), force_rebuild=True,
        )
        display_meta = server.graph_store.load_meta()
        compute_meta = server.graph_artifact_store.load_context_meta()
        self.assertEqual(display_meta["instance_hash"], compute_meta["instance_hash"])
        self.assertEqual(diagnostics.cache_level, "built")

    def test_instance_operation_update_invalidates_one_service(self):
        calls = []
        original = server.graph_context_service
        server.graph_context_service = SimpleNamespace(invalidate=calls.append)
        try:
            server._invalidate_graph_context("operation_updated")
        finally:
            server.graph_context_service = original
        self.assertEqual(calls, ["operation_updated"])
```

- [ ] **Step 2: Run tests and verify failure**

```bash
cd .. && llm4drd/.venv/bin/python -m unittest llm4drd.tests.test_graph_api_context -v
```

Expected: FAIL because the API still constructs `HeterogeneousGraph` and has no context service.

- [ ] **Step 3: Instantiate shared stores and service**

In `api/server.py`:

```python
graph_store = GraphStore()
graph_artifact_store = GraphArtifactStore()
graph_context_service = GraphContextService(graph_artifact_store)


def _invalidate_graph_context(reason: str) -> None:
    graph_context_service.invalidate(reason)
```

Replace every direct `graph_store.clear_all()` after instance generation/import/update with `_invalidate_graph_context()` using explicit reasons: `instance_generated`, `order_updated`, `task_updated`, `operation_updated`, `machine_updated`, and `instance_imported`.

- [ ] **Step 4: Replace `/api/graph/build` internals**

Preserve timeout, size preflight, background status, progress messages, and response shapes. Replace direct `HeterogeneousGraph` plus `graph_store.save_graph()` with:

```python
context, diagnostics = graph_context_service.get_or_build(
    build_shop,
    force_rebuild=True,
    progress_callback=build_progress,
    deadline=deadline,
    current_fingerprint_provider=lambda: compute_graph_fingerprint(_active_shop()),
)
stats = graph_store.load_meta()
```

Add the tested helper used by the endpoint:

```python
def _build_graph_artifacts(current_shop, *, force_rebuild=False, progress_callback=None, deadline=None):
    return graph_context_service.get_or_build(
        current_shop,
        force_rebuild=force_rebuild,
        progress_callback=progress_callback,
        deadline=deadline,
    )
```

Record cache/build diagnostics in the graph task status. Do not remove legacy `GraphStore.save_graph()` yet; existing direct library callers and tests still depend on it during rollout.

- [ ] **Step 5: Extend graph meta without breaking existing clients**

Return existing fields plus fingerprint prefixes, schema/build versions, cache readiness, build time, and invalid reason. Do not rename `total_nodes`, `total_edges`, `node_type_counts`, or `edge_type_counts`.

- [ ] **Step 6: Run API and graph regression tests**

```bash
cd .. && llm4drd/.venv/bin/python -m unittest \
  llm4drd.tests.test_graph_api_context \
  llm4drd.tests.test_graph \
  llm4drd.tests.test_canonical_graph -v
```

Expected: all tests pass and graph/query response compatibility is preserved.

- [ ] **Step 7: Commit**

```bash
git add api/server.py data/db.py tests/test_graph.py tests/test_graph_api_context.py
git commit -m "feat: unify display graph build and invalidation"
```

---

### Task 7: Integrate GraphContext into Hybrid NSGA-III/ALNS

**Files:**
- Modify: `optimization/hybrid_nsga3_alns.py`
- Modify: `api/server.py`
- Create: `tests/test_hybrid_graph_context.py`
- Modify: `tests/test_simulator_robustness.py`

**Interfaces:**
- Consumes: `GraphContext`, `GraphContextMode`, `GraphContextDiagnostics`, and `GraphContextService`.
- Produces: `HybridNSGA3ALNSOptimizer(shop, config, graph_context=None, graph_context_mode="legacy")`, shadow comparison diagnostics, active context access, and graph context task-status payloads.

- [ ] **Step 1: Write failing legacy/shadow/active equivalence tests**

Create `tests/test_hybrid_graph_context.py`:

```python
import unittest

from llm4drd.knowledge.canonical import CanonicalGraphBuilder
from llm4drd.knowledge.context import ComputeGraphProjection
from llm4drd.optimization.hybrid_nsga3_alns import HybridConfig, HybridNSGA3ALNSOptimizer
from llm4drd.tests.shop_fixtures import hybrid_result_signature, make_graph_context_shop


class HybridGraphContextTests(unittest.TestCase):
    def setUp(self):
        self.config = HybridConfig(
            objective_keys=["total_tardiness", "makespan"], target_solution_count=2,
            population_size=4, generations=1, alns_iterations_per_candidate=0,
            time_limit_s=60, parallel_workers=1, seed=17,
        )

    def context(self, shop):
        canonical = CanonicalGraphBuilder().build(shop)
        return ComputeGraphProjection().build(shop, canonical)

    def test_active_matches_legacy(self):
        legacy_shop = make_graph_context_shop()
        active_shop = make_graph_context_shop()
        legacy = HybridNSGA3ALNSOptimizer(legacy_shop, self.config, graph_context_mode="legacy").run()
        active = HybridNSGA3ALNSOptimizer(
            active_shop, self.config, self.context(active_shop), "active",
        ).run()
        self.assertEqual(hybrid_result_signature(active), hybrid_result_signature(legacy))

    def test_shadow_uses_legacy_solver_and_reports_zero_diff(self):
        shop = make_graph_context_shop()
        optimizer = HybridNSGA3ALNSOptimizer(shop, self.config, self.context(shop), "shadow")
        self.assertEqual(optimizer.graph_context_diff.total_differences, 0)
        self.assertEqual(
            hybrid_result_signature(optimizer.run()),
            hybrid_result_signature(HybridNSGA3ALNSOptimizer(make_graph_context_shop(), self.config).run()),
        )

    def test_active_requires_context(self):
        with self.assertRaisesRegex(ValueError, "GraphContext is required"):
            HybridNSGA3ALNSOptimizer(make_graph_context_shop(), self.config, None, "active")
```

- [ ] **Step 2: Run tests and verify constructor failure**

```bash
cd .. && llm4drd/.venv/bin/python -m unittest llm4drd.tests.test_hybrid_graph_context -v
```

Expected: FAIL because the optimizer constructor has no context/mode parameters.

- [ ] **Step 3: Add mode-aware optimizer initialization**

Modify the constructor signature exactly as specified. Behavior:

- `legacy`: run current NetworkX and `_build_graph_features()` code unchanged.
- `shadow`: build the legacy state, compare context relations/features/profiles, store a `GraphContextDiff`, and solve with legacy state.
- `active`: do not instantiate `HeterogeneousGraph`; use `graph_context.feature_view_by_operation_id()` and context accessors.

Keep a single `self.operation_order_rank = {op_id: rank for rank, op_id in enumerate(self.shop.operations)}`. When group-index results replace legacy full scans, sort group members by this rank before extending clusters so tie order remains unchanged.

- [ ] **Step 4: Replace active neighbor and group lookups**

In `_expand_operation_cluster()` branch by mode:

```python
if self.graph_context_mode == "active":
    related.extend(self.graph_context.predecessors(current_id))
    related.extend(self.graph_context.successors(current_id))
    related.extend(self.graph_context.operations_in_group("process_type", op.process_type))
    for tooling_type in op.required_tooling_types:
        related.extend(self.graph_context.operations_in_group("tooling_type", tooling_type))
    for skill_id in op.required_personnel_skills:
        related.extend(self.graph_context.operations_in_group("personnel_skill", skill_id))
    related = sorted(dict.fromkeys(related), key=self.operation_order_rank.__getitem__)
else:
    related.extend(self._legacy_related_operations(current_id))
```

Create `_legacy_related_operations(current_id)` by moving the existing body from `hybrid_nsga3_alns.py:329-354` without reordering any append operation. This isolates the compatibility path and lets the active branch contain no NetworkX call.

Do not alter `_dispatch_rule`, approximate evaluation, exact simulation, candidate signatures, random sampling, or sorting keys beyond replacing the feature source.

- [ ] **Step 5: Integrate the service into `/api/optimize/hybrid`**

Read mode once per task. For `legacy`, initialize the optimizer directly. For `shadow` or `active`, update task phase to `graph_context_loading`, call `get_or_build()`, change phase to `graph_context_building` when diagnostics indicate a build, and include the diagnostics payload in status/result.

The background task must capture its `ShopFloor` and `GraphContext`; later global invalidation cannot mutate them.

- [ ] **Step 6: Run deterministic equivalence repeatedly**

```bash
cd .. && for run in 1 2 3; do llm4drd/.venv/bin/python -m unittest llm4drd.tests.test_hybrid_graph_context -v || exit 1; done
```

Expected: all three runs pass with identical signatures. If active differs, stop and fix ordering/formula parity; do not weaken assertions.

- [ ] **Step 7: Run optimizer and simulator regression tests**

```bash
cd .. && llm4drd/.venv/bin/python -m unittest \
  llm4drd.tests.test_hybrid_graph_context \
  llm4drd.tests.test_graph_legacy_baseline \
  llm4drd.tests.test_simulator_robustness -v
```

Expected: all tests pass and simulator tests are unchanged.

- [ ] **Step 8: Commit**

```bash
git add optimization/hybrid_nsga3_alns.py api/server.py tests/test_hybrid_graph_context.py tests/test_simulator_robustness.py
git commit -m "feat: run hybrid optimizer from cached graph context"
```

---

### Task 8: Benchmark, Document, Activate, and Verify the Rollout

**Files:**
- Create: `tools/benchmark_graph_context.py`
- Create: `docs/benchmarks/graph-context-baseline.json`
- Create: `docs/benchmarks/graph-context-result.json`
- Create: `docs/benchmarks/graph-context-report.md`
- Modify: `README.md`
- Modify: `api/server.py`
- Test: all `tests/`

**Interfaces:**
- Consumes: legacy/shadow/active optimizer modes and graph diagnostics.
- Produces: reproducible benchmark JSON/report, documented operational mode, structured log events, and default `active` mode after acceptance.

- [ ] **Step 1: Add structured logging assertions**

Extend service tests with `assertLogs()` for:

```text
graph_context.l1_hit
graph_context.sqlite_hit
graph_context.miss
graph_context.build_started
graph_context.build_completed
graph_context.invalidated
graph_context.corrupt
graph_context.rebuild_failed
```

Each record must include fingerprint prefix, cache level, operation count, relation count, and elapsed milliseconds where applicable.

- [ ] **Step 2: Implement the benchmark tool**

`tools/benchmark_graph_context.py` must accept:

```text
--sizes 80,500,2500
--runs 7
--warmup 2
--seed 42
--output-dir docs/benchmarks
```

For every size, measure with `time.perf_counter()` and `tracemalloc`:

- legacy optimizer initialization;
- canonical cold build and SQLite write;
- SQLite load and optimizer initialization;
- L1 hit and optimizer initialization;
- fixed one-generation end-to-end legacy and active optimization.

Write machine-readable JSON containing Python version, platform, CPU count, commit hash, parameters, all raw samples, medians, speedup, runtime regression percentage, and peak-memory ratio. Exit nonzero when medium/large warm initialization speedup is below 2.0x, runtime regression exceeds 3%, output signatures differ, or peak-memory ratio exceeds 1.25.

- [ ] **Step 3: Capture the legacy baseline before default activation**

```bash
cd .. && llm4drd/.venv/bin/python -m llm4drd.tools.benchmark_graph_context \
  --sizes 80,500,2500 --runs 7 --warmup 2 --seed 42 \
  --mode baseline --output-dir llm4drd/docs/benchmarks
```

Expected: creates `graph-context-baseline.json`; command exits 0.

- [ ] **Step 4: Capture cold/L2/L1 results and render the report**

```bash
cd .. && llm4drd/.venv/bin/python -m llm4drd.tools.benchmark_graph_context \
  --sizes 80,500,2500 --runs 7 --warmup 2 --seed 42 \
  --mode compare --output-dir llm4drd/docs/benchmarks
```

Expected: creates `graph-context-result.json` and `graph-context-report.md`, exits 0, and reports at least 2.0x warm initialization speedup for 500 and 2500 operations.

- [ ] **Step 5: Document operation and rollback**

Update `README.md` with:

- canonical graph versus display/compute projection explanation;
- automatic build behavior;
- L1/L2 cache semantics;
- `LLM4DRD_GRAPH_CONTEXT_MODE=legacy|shadow|active`;
- how to inspect graph context diagnostics in graph meta and optimize status;
- corruption/rebuild behavior;
- rollback command and the fact that no database downgrade is required;
- benchmark command and acceptance thresholds.

- [ ] **Step 6: Make active mode the default only after benchmark acceptance**

Change `resolve_graph_context_mode()` default from `legacy` to `active`. Keep explicit `legacy` and `shadow` support. Add a test proving an unset environment selects `active` and invalid values log a warning then select `legacy`.

- [ ] **Step 7: Run the full verification suite**

```bash
cd .. && llm4drd/.venv/bin/python -m unittest discover -s llm4drd/tests -t . -v
```

Expected: zero failures and zero errors.

Run syntax compilation:

```bash
cd .. && llm4drd/.venv/bin/python -m compileall -q llm4drd
```

Expected: exit 0 and no output.

Run diff checks:

```bash
git diff --check
git status --short
```

Expected: `git diff --check` exits 0; status contains only the intended benchmark, documentation, source, and test changes.

- [ ] **Step 8: Commit final rollout artifacts**

```bash
git add \
  tools/benchmark_graph_context.py \
  docs/benchmarks/graph-context-baseline.json \
  docs/benchmarks/graph-context-result.json \
  docs/benchmarks/graph-context-report.md \
  README.md api/server.py knowledge/context_service.py tests
git commit -m "perf: activate verified graph context cache"
```

---

## Final Acceptance Checklist

- [ ] `CanonicalGraphBuilder` is the only module that interprets `ShopFloor` graph relationships.
- [ ] `HeterogeneousGraph` is a compatibility adapter over canonical rows.
- [ ] Display and compute projections commit atomically with identical fingerprints.
- [ ] Missing, stale, or first-corrupt caches rebuild automatically.
- [ ] L1/L2 hits perform no canonical build or legacy feature build.
- [ ] Display graph APIs and existing filters remain compatible.
- [ ] Active hybrid mode creates no NetworkX graph.
- [ ] Legacy, shadow, cold active, SQLite active, and L1 active deterministic signatures match.
- [ ] Medium/large warm initialization is at least 2.0x faster.
- [ ] Fixed-evaluation end-to-end runtime regression is at most 3%.
- [ ] Peak memory is at most 1.25x legacy.
- [ ] Full unit suite and compileall pass.
- [ ] `legacy` rollback works without a schema downgrade.
