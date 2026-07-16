from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from ..core.models import ShopFloor
from .canonical import (
    CanonicalEdge,
    CanonicalGraph,
    CanonicalNode,
    GraphFingerprint,
    compute_graph_fingerprint,
)


class GraphContextError(Exception):
    pass


class GraphContextStaleError(GraphContextError):
    pass


class GraphContextCorruptError(GraphContextError):
    pass


class GraphContextBuildError(GraphContextError):
    pass


class GraphContextPersistenceError(GraphContextError):
    pass


@dataclass(frozen=True)
class DisplayGraphProjection:
    nodes: tuple[CanonicalNode, ...]
    edges: tuple[CanonicalEdge, ...]
    fingerprint: GraphFingerprint
    stats: Mapping[str, object]
    _node_order: tuple[str, ...] = field(default=(), repr=False, compare=False)

    @classmethod
    def from_canonical(cls, graph: CanonicalGraph) -> "DisplayGraphProjection":
        stats = {
            key: MappingProxyType(value) if isinstance(value, dict) else value
            for key, value in graph.stats().items()
        }
        return cls(
            nodes=graph.nodes,
            edges=graph.edges,
            fingerprint=graph.fingerprint,
            stats=MappingProxyType(stats),
            _node_order=graph._node_order,
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
        return {
            op_id: self.operation_features(op_id) for op_id in self.operation_ids
        }

    def _operation_relation(
        self,
        op_id: str,
        offsets: tuple[int, ...],
        indices: tuple[int, ...],
    ) -> tuple[str, ...]:
        ordinal = self.operation_index[op_id]
        start, end = offsets[ordinal], offsets[ordinal + 1]
        return tuple(self.operation_ids[index] for index in indices[start:end])

    def predecessors(self, op_id: str) -> tuple[str, ...]:
        return self._operation_relation(
            op_id, self.predecessor_offsets, self.predecessor_indices
        )

    def successors(self, op_id: str) -> tuple[str, ...]:
        return self._operation_relation(
            op_id, self.successor_offsets, self.successor_indices
        )

    def eligible_machines(self, op_id: str) -> tuple[str, ...]:
        ordinal = self.operation_index[op_id]
        start = self.eligible_machine_offsets[ordinal]
        end = self.eligible_machine_offsets[ordinal + 1]
        return tuple(
            self.machine_ids[index]
            for index in self.eligible_machine_indices[start:end]
        )

    def operations_in_group(
        self, group_type: str, group_key: str
    ) -> tuple[str, ...]:
        indexes = self.operation_groups.get((group_type, group_key), ())
        return tuple(self.operation_ids[index] for index in indexes)


def _pairs_to_csr(
    size: int, pairs: list[tuple[int, int]]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    buckets: list[list[int]] = [[] for _ in range(size)]
    for source, target in pairs:
        if not 0 <= source < size:
            raise GraphContextCorruptError("relation source ordinal out of bounds")
        buckets[source].append(target)

    offsets = [0]
    indices: list[int] = []
    for bucket in buckets:
        indices.extend(sorted(set(bucket)))
        offsets.append(len(indices))
    return tuple(offsets), tuple(indices)


def _build_feature_rows(
    shop: ShopFloor, canonical: CanonicalGraph
) -> dict[str, dict[str, float]]:
    task_predecessors: dict[str, list[str]] = {
        task_id: list(task.predecessor_task_ids)
        for task_id, task in shop.tasks.items()
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
        op_id: set(op.predecessor_ops)
        for op_id, op in shop.operations.items()
    }
    for op_id, op in shop.operations.items():
        for predecessor_task_id in op.predecessor_tasks:
            predecessor_task = shop.tasks.get(predecessor_task_id)
            if predecessor_task:
                for predecessor_op in predecessor_task.operations:
                    op_predecessors[op_id].add(predecessor_op.id)

    depth_cache: dict[str, float] = {}

    def predecessor_depth(
        op_id: str, trail: set[str] | None = None
    ) -> float:
        if op_id in depth_cache:
            return depth_cache[op_id]
        trail = trail or set()
        if op_id in trail:
            return 0.0
        predecessors = [
            predecessor_id
            for predecessor_id in op_predecessors.get(op_id, set())
            if predecessor_id in shop.operations
        ]
        if not predecessors:
            depth_cache[op_id] = 0.0
            return 0.0
        trail = set(trail)
        trail.add(op_id)
        depth_cache[op_id] = 1.0 + max(
            predecessor_depth(predecessor_id, trail)
            for predecessor_id in predecessors
        )
        return depth_cache[op_id]

    node_ids = set(canonical._node_order)
    if not node_ids:
        node_ids = {node.node_id for node in canonical.nodes}
        node_ids.update(edge.source for edge in canonical.edges)
        node_ids.update(edge.target for edge in canonical.edges)
    outgoing: dict[str, set[str]] = {}
    for edge in canonical.edges:
        outgoing.setdefault(edge.source, set()).add(edge.target)

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
        out_degree = len(outgoing.get(f"OP:{op_id}", ()))

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
                1.0, out_degree / max(1, len(node_ids))
            ),
        }
    return features


class ComputeGraphProjection:
    FEATURE_NAMES = (
        "predecessor_depth",
        "assembly_criticality",
        "shared_resource_degree",
        "bottleneck_adjacency",
        "graph_out_degree",
    )

    def build(self, shop: ShopFloor, canonical: CanonicalGraph) -> GraphContext:
        operation_ids = tuple(sorted(shop.operations))
        operation_index_data = {
            operation_id: index
            for index, operation_id in enumerate(operation_ids)
        }
        machine_ids = tuple(sorted(shop.machines))
        machine_index_data = {
            machine_id: index for index, machine_id in enumerate(machine_ids)
        }

        predecessor_pairs: list[tuple[int, int]] = []
        eligible_machine_pairs: list[tuple[int, int]] = []
        for edge in canonical.edges:
            if edge.edge_type == "operation_sequence":
                predecessor_id = edge.source.removeprefix("OP:")
                operation_id = edge.target.removeprefix("OP:")
                if (
                    operation_id in operation_index_data
                    and predecessor_id in operation_index_data
                ):
                    predecessor_pairs.append(
                        (
                            operation_index_data[operation_id],
                            operation_index_data[predecessor_id],
                        )
                    )
            elif edge.edge_type == "op_depends_task":
                task_id = edge.source.removeprefix("T:")
                operation_id = edge.target.removeprefix("OP:")
                predecessor_task = shop.tasks.get(task_id)
                if operation_id in operation_index_data and predecessor_task:
                    predecessor_pairs.extend(
                        (
                            operation_index_data[operation_id],
                            operation_index_data[predecessor_op.id],
                        )
                        for predecessor_op in predecessor_task.operations
                        if predecessor_op.id in operation_index_data
                    )
            elif edge.edge_type == "machine_eligible":
                operation_id = edge.source.removeprefix("OP:")
                machine_id = edge.target.removeprefix("M:")
                if (
                    operation_id in operation_index_data
                    and machine_id in machine_index_data
                ):
                    eligible_machine_pairs.append(
                        (
                            operation_index_data[operation_id],
                            machine_index_data[machine_id],
                        )
                    )

        successor_pairs = [
            (predecessor, operation)
            for operation, predecessor in predecessor_pairs
        ]
        predecessor_offsets, predecessor_indices = _pairs_to_csr(
            len(operation_ids), predecessor_pairs
        )
        successor_offsets, successor_indices = _pairs_to_csr(
            len(operation_ids), successor_pairs
        )
        eligible_machine_offsets, eligible_machine_indices = _pairs_to_csr(
            len(operation_ids), eligible_machine_pairs
        )

        feature_rows = _build_feature_rows(shop, canonical)
        feature_matrix = tuple(
            tuple(float(feature_rows[operation_id][name]) for name in self.FEATURE_NAMES)
            for operation_id in operation_ids
        )

        group_sets: dict[tuple[str, str], set[int]] = {}
        for operation_id in operation_ids:
            operation = shop.operations[operation_id]
            ordinal = operation_index_data[operation_id]
            group_sets.setdefault(
                ("process_type", operation.process_type), set()
            ).add(ordinal)
            for tooling_type in operation.required_tooling_types:
                group_sets.setdefault(("tooling_type", tooling_type), set()).add(
                    ordinal
                )
            for skill_id in operation.required_personnel_skills:
                group_sets.setdefault(("personnel_skill", skill_id), set()).add(
                    ordinal
                )
        operation_groups = MappingProxyType(
            {
                key: tuple(sorted(ordinals))
                for key, ordinals in sorted(group_sets.items())
            }
        )

        context = GraphContext(
            fingerprint=canonical.fingerprint,
            operation_ids=operation_ids,
            operation_index=MappingProxyType(operation_index_data),
            machine_ids=machine_ids,
            machine_index=MappingProxyType(machine_index_data),
            predecessor_offsets=predecessor_offsets,
            predecessor_indices=predecessor_indices,
            successor_offsets=successor_offsets,
            successor_indices=successor_indices,
            eligible_machine_offsets=eligible_machine_offsets,
            eligible_machine_indices=eligible_machine_indices,
            feature_names=self.FEATURE_NAMES,
            feature_matrix=feature_matrix,
            operation_groups=operation_groups,
        )
        validate_graph_context(shop, context)
        return context


def _validate_index(
    name: str, ids: tuple[str, ...], index: Mapping[str, int]
) -> None:
    if len(index) != len(ids) or set(index) != set(ids):
        raise GraphContextCorruptError(f"{name} metadata count mismatch")
    if tuple(sorted(index.values())) != tuple(range(len(ids))):
        raise GraphContextCorruptError(f"{name} ordinals are not contiguous")
    if any(index[entity_id] != ordinal for ordinal, entity_id in enumerate(ids)):
        raise GraphContextCorruptError(f"{name} index does not match IDs")


def _validate_csr(
    name: str,
    size: int,
    offsets: tuple[int, ...],
    indices: tuple[int, ...],
    target_size: int,
) -> None:
    if len(offsets) != size + 1 or not offsets or offsets[0] != 0:
        raise GraphContextCorruptError(f"{name} offset metadata mismatch")
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise GraphContextCorruptError(f"{name} offsets are not monotonic")
    if offsets[-1] != len(indices):
        raise GraphContextCorruptError(f"{name} terminal offset mismatch")
    if any(index < 0 or index >= target_size for index in indices):
        raise GraphContextCorruptError(f"{name} relation ordinal out of bounds")


def validate_graph_context(shop: ShopFloor, context: GraphContext) -> None:
    expected_operations = tuple(sorted(shop.operations))
    expected_machines = tuple(sorted(shop.machines))
    if context.operation_ids != expected_operations:
        raise GraphContextCorruptError("missing operations or operation order mismatch")
    if context.machine_ids != expected_machines:
        raise GraphContextCorruptError("machine metadata count mismatch")
    if context.fingerprint != compute_graph_fingerprint(shop):
        raise GraphContextCorruptError("graph fingerprint mismatch")

    _validate_index("operation", context.operation_ids, context.operation_index)
    _validate_index("machine", context.machine_ids, context.machine_index)
    operation_count = len(context.operation_ids)
    _validate_csr(
        "predecessor",
        operation_count,
        context.predecessor_offsets,
        context.predecessor_indices,
        operation_count,
    )
    _validate_csr(
        "successor",
        operation_count,
        context.successor_offsets,
        context.successor_indices,
        operation_count,
    )
    _validate_csr(
        "eligible machine",
        operation_count,
        context.eligible_machine_offsets,
        context.eligible_machine_indices,
        len(context.machine_ids),
    )

    if len(set(context.feature_names)) != len(context.feature_names):
        raise GraphContextCorruptError("duplicate feature names")
    if context.feature_names != ComputeGraphProjection.FEATURE_NAMES:
        raise GraphContextCorruptError("feature schema mismatch")
    if len(context.feature_matrix) != operation_count:
        raise GraphContextCorruptError("feature metadata count mismatch")
    for row in context.feature_matrix:
        if len(row) != len(context.feature_names):
            raise GraphContextCorruptError("feature row width mismatch")
        if any(not math.isfinite(value) for value in row):
            raise GraphContextCorruptError("non-finite graph context feature")
    for ordinals in context.operation_groups.values():
        if tuple(sorted(set(ordinals))) != ordinals:
            raise GraphContextCorruptError("group ordinals are not sorted and unique")
        if any(ordinal < 0 or ordinal >= operation_count for ordinal in ordinals):
            raise GraphContextCorruptError("group ordinal out of bounds")


@dataclass(frozen=True)
class GraphContextDiff:
    relation_differences: tuple[str, ...] = ()
    feature_differences: tuple[str, ...] = ()
    group_differences: tuple[str, ...] = ()

    @property
    def total_differences(self) -> int:
        return (
            len(self.relation_differences)
            + len(self.feature_differences)
            + len(self.group_differences)
        )


def compare_legacy_context(
    shop: ShopFloor, context: GraphContext
) -> GraphContextDiff:
    from ..optimization.hybrid_nsga3_alns import build_legacy_graph_features
    from .graph import HeterogeneousGraph

    graph_model = HeterogeneousGraph()
    graph_model.build_from_shopfloor(shop)
    legacy_features = build_legacy_graph_features(shop, graph_model.graph)
    relation_differences: list[str] = []
    feature_differences: list[str] = []
    group_differences: list[str] = []

    expected_predecessors: dict[str, set[str]] = {
        operation_id: {
            predecessor_id
            for predecessor_id in operation.predecessor_ops
            if predecessor_id in shop.operations
        }
        for operation_id, operation in shop.operations.items()
    }
    expected_successors: dict[str, set[str]] = {
        operation_id: set() for operation_id in shop.operations
    }
    for operation_id, operation in shop.operations.items():
        for task_id in operation.predecessor_tasks:
            predecessor_task = shop.tasks.get(task_id)
            if predecessor_task:
                expected_predecessors[operation_id].update(
                    predecessor.id
                    for predecessor in predecessor_task.operations
                    if predecessor.id in shop.operations
                )
    for operation_id, predecessors in expected_predecessors.items():
        for predecessor_id in predecessors:
            expected_successors[predecessor_id].add(operation_id)

    expected_groups: dict[tuple[str, str], set[str]] = {}
    for operation_id, operation in shop.operations.items():
        keys = [("process_type", operation.process_type)]
        keys.extend(("tooling_type", value) for value in operation.required_tooling_types)
        keys.extend(
            ("personnel_skill", value)
            for value in operation.required_personnel_skills
        )
        for key in keys:
            expected_groups.setdefault(key, set()).add(operation_id)

    for operation_id, operation in shop.operations.items():
        expected = tuple(sorted(expected_predecessors[operation_id]))
        if context.predecessors(operation_id) != expected:
            relation_differences.append(f"predecessors:{operation_id}")
        expected = tuple(sorted(expected_successors[operation_id]))
        if context.successors(operation_id) != expected:
            relation_differences.append(f"successors:{operation_id}")
        expected_machines = tuple(
            sorted(
                target.removeprefix("M:")
                for _, target, attrs in graph_model.graph.out_edges(
                    f"OP:{operation_id}", data=True
                )
                if attrs.get("edge_type") == "machine_eligible"
            )
        )
        if context.eligible_machines(operation_id) != expected_machines:
            relation_differences.append(f"eligible_machines:{operation_id}")

        actual_features = context.operation_features(operation_id)
        for name, expected_value in legacy_features[operation_id].items():
            if abs(actual_features.get(name, math.inf) - expected_value) > 1e-12:
                feature_differences.append(f"{operation_id}:{name}")

    for (group_type, group_key), members in expected_groups.items():
        if context.operations_in_group(group_type, group_key) != tuple(sorted(members)):
            group_differences.append(f"{group_type}:{group_key}")
    unexpected_groups = set(context.operation_groups) - set(expected_groups)
    group_differences.extend(
        f"{kind}:{key}" for kind, key in sorted(unexpected_groups)
    )

    return GraphContextDiff(
        relation_differences=tuple(relation_differences),
        feature_differences=tuple(feature_differences),
        group_differences=tuple(sorted(set(group_differences))),
    )
