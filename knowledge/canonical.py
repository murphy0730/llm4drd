from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from types import MappingProxyType
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
    _node_order: tuple[str, ...] = field(default=(), repr=False, compare=False)

    def stats(self) -> dict:
        node_types: dict[str, int] = {}
        edge_types: dict[str, int] = {}
        for node in self.nodes:
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
        for edge in self.edges:
            edge_types[edge.edge_type] = edge_types.get(edge.edge_type, 0) + 1
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": node_types,
            "edge_types": edge_types,
        }


def _finite(value: float, field: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"non-finite graph input: {field}")
    return numeric


def _shifts(resource_id: str, shifts: list) -> list[dict]:
    return [
        {
            "day": int(shift.day),
            "start_hour": _finite(shift.start_hour, f"{resource_id}.shifts.start_hour"),
            "hours": _finite(shift.hours, f"{resource_id}.shifts.hours"),
        }
        for shift in sorted(shifts, key=lambda shift: (shift.day, shift.start_hour, shift.hours))
    ]


def _downtimes(resource_id: str, downtimes: list) -> list[dict]:
    return [
        {
            "id": downtime.id,
            "machine_id": downtime.machine_id,
            "downtime_type": downtime.downtime_type,
            "start_time": _finite(downtime.start_time, f"{resource_id}.downtimes.start_time"),
            "end_time": _finite(downtime.end_time, f"{resource_id}.downtimes.end_time"),
        }
        for downtime in sorted(
            downtimes,
            key=lambda downtime: (
                downtime.id,
                downtime.machine_id,
                downtime.downtime_type,
                downtime.start_time,
                downtime.end_time,
            ),
        )
    ]


def _normalized_instance_payload(shop: ShopFloor) -> dict:
    return {
        "plan_start_at": shop.plan_start_at.isoformat(),
        "machine_types": [
            {
                "key": type_id,
                "id": machine_type.id,
                "name": machine_type.name,
                "is_critical": machine_type.is_critical,
            }
            for type_id, machine_type in sorted(shop.machine_types.items())
        ],
        "tooling_types": [
            {"key": type_id, "id": tooling_type.id, "name": tooling_type.name}
            for type_id, tooling_type in sorted(shop.tooling_types.items())
        ],
        "machines": [
            {
                "key": machine_id,
                "id": machine.id,
                "name": machine.name,
                "type_id": machine.type_id,
                "shifts": _shifts(machine_id, machine.shifts),
                "downtimes": _downtimes(machine_id, machine.downtimes),
            }
            for machine_id, machine in sorted(shop.machines.items())
        ],
        "toolings": [
            {
                "key": tooling_id,
                "id": tooling.id,
                "name": tooling.name,
                "type_id": tooling.type_id,
                "shifts": _shifts(tooling_id, tooling.shifts),
                "downtimes": _downtimes(tooling_id, tooling.downtimes),
            }
            for tooling_id, tooling in sorted(shop.toolings.items())
        ],
        "personnel": [
            {
                "key": person_id,
                "id": person.id,
                "name": person.name,
                "skills": sorted(person.skills),
                "shifts": _shifts(person_id, person.shifts),
                "downtimes": _downtimes(person_id, person.downtimes),
            }
            for person_id, person in sorted(shop.personnel.items())
        ],
        "orders": [
            {
                "key": order_id,
                "id": order.id,
                "name": order.name,
                "release_time": _finite(order.release_time, f"orders.{order_id}.release_time"),
                "due_date": _finite(order.due_date, f"orders.{order_id}.due_date"),
                "priority": order.priority,
                "task_ids": sorted(order.task_ids),
                "main_task_id": order.main_task_id,
                "status": order.status,
            }
            for order_id, order in sorted(shop.orders.items())
        ],
        "tasks": [
            {
                "key": task_id,
                "id": task.id,
                "order_id": task.order_id,
                "name": task.name,
                "is_main": task.is_main,
                "predecessor_task_ids": sorted(task.predecessor_task_ids),
                "operation_ids": sorted(operation.id for operation in task.operations),
                "release_time": _finite(task.release_time, f"tasks.{task_id}.release_time"),
                "due_date": _finite(task.due_date, f"tasks.{task_id}.due_date"),
            }
            for task_id, task in sorted(shop.tasks.items())
        ],
        "operations": [
            {
                "key": operation_id,
                "id": operation.id,
                "task_id": operation.task_id,
                "name": operation.name,
                "process_type": operation.process_type,
                "processing_time": _finite(
                    operation.processing_time, f"operations.{operation_id}.processing_time"
                ),
                "predecessor_ops": sorted(operation.predecessor_ops),
                "predecessor_tasks": sorted(operation.predecessor_tasks),
                "eligible_machine_ids": sorted(operation.eligible_machine_ids),
                "required_tooling_types": sorted(operation.required_tooling_types),
                "required_personnel_skills": sorted(operation.required_personnel_skills),
                "status": operation.status.value,
            }
            for operation_id, operation in sorted(shop.operations.items())
        ],
    }


def _normalized_topology_payload(shop: ShopFloor) -> dict:
    return {
        "machine_types": [
            {
                "key": type_id,
                "id": machine_type.id,
                "is_critical": machine_type.is_critical,
            }
            for type_id, machine_type in sorted(shop.machine_types.items())
        ],
        "machines": [
            {"key": machine_id, "id": machine.id, "type_id": machine.type_id}
            for machine_id, machine in sorted(shop.machines.items())
        ],
        "tooling_types": [
            {"key": type_id, "id": tooling_type.id}
            for type_id, tooling_type in sorted(shop.tooling_types.items())
        ],
        "toolings": [
            {"key": tooling_id, "id": tooling.id, "type_id": tooling.type_id}
            for tooling_id, tooling in sorted(shop.toolings.items())
        ],
        "personnel": [
            {"key": person_id, "id": person.id, "skills": sorted(person.skills)}
            for person_id, person in sorted(shop.personnel.items())
        ],
        "orders": [
            {"key": order_id, "id": order.id}
            for order_id, order in sorted(shop.orders.items())
        ],
        "tasks": [
            {
                "key": task_id,
                "id": task.id,
                "order_id": task.order_id,
                "predecessor_task_ids": sorted(task.predecessor_task_ids),
            }
            for task_id, task in sorted(shop.tasks.items())
        ],
        "operations": [
            {
                "key": operation_id,
                "id": operation.id,
                "task_id": operation.task_id,
                "process_type": operation.process_type,
                "predecessor_ops": sorted(operation.predecessor_ops),
                "predecessor_tasks": sorted(operation.predecessor_tasks),
                "eligible_machine_ids": sorted(operation.eligible_machine_ids),
                "required_tooling_types": sorted(operation.required_tooling_types),
                "required_personnel_skills": sorted(operation.required_personnel_skills),
            }
            for operation_id, operation in sorted(shop.operations.items())
        ],
    }


def _normalized_feature_payload(shop: ShopFloor) -> dict:
    return {
        "machine_types": [
            {
                "key": type_id,
                "id": machine_type.id,
                "is_critical": machine_type.is_critical,
            }
            for type_id, machine_type in sorted(shop.machine_types.items())
        ],
        "machines": [
            {"key": machine_id, "id": machine.id, "type_id": machine.type_id}
            for machine_id, machine in sorted(shop.machines.items())
        ],
        "toolings": [
            {"key": tooling_id, "id": tooling.id, "type_id": tooling.type_id}
            for tooling_id, tooling in sorted(shop.toolings.items())
        ],
        "personnel": [
            {"key": person_id, "id": person.id, "skills": sorted(person.skills)}
            for person_id, person in sorted(shop.personnel.items())
        ],
        "orders": [
            {
                "key": order_id,
                "id": order.id,
                "release_time": _finite(order.release_time, f"orders.{order_id}.release_time"),
                "due_date": _finite(order.due_date, f"orders.{order_id}.due_date"),
                "priority": order.priority,
                "main_task_id": order.main_task_id,
            }
            for order_id, order in sorted(shop.orders.items())
        ],
        "tasks": [
            {
                "key": task_id,
                "id": task.id,
                "order_id": task.order_id,
                "is_main": task.is_main,
                "predecessor_task_ids": sorted(task.predecessor_task_ids),
                "operation_ids": sorted(operation.id for operation in task.operations),
                "release_time": _finite(task.release_time, f"tasks.{task_id}.release_time"),
                "due_date": _finite(task.due_date, f"tasks.{task_id}.due_date"),
            }
            for task_id, task in sorted(shop.tasks.items())
        ],
        "operations": [
            {
                "key": operation_id,
                "id": operation.id,
                "task_id": operation.task_id,
                "process_type": operation.process_type,
                "processing_time": _finite(
                    operation.processing_time, f"operations.{operation_id}.processing_time"
                ),
                "predecessor_ops": sorted(operation.predecessor_ops),
                "predecessor_tasks": sorted(operation.predecessor_tasks),
                "eligible_machine_ids": sorted(operation.eligible_machine_ids),
                "required_tooling_types": sorted(operation.required_tooling_types),
                "required_personnel_skills": sorted(operation.required_personnel_skills),
                "status": operation.status.value,
            }
            for operation_id, operation in sorted(shop.operations.items())
        ],
    }


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compute_graph_fingerprint(shop: ShopFloor) -> GraphFingerprint:
    instance_payload = _normalized_instance_payload(shop)
    topology_payload = _normalized_topology_payload(shop)
    feature_payload = _normalized_feature_payload(shop)
    return GraphFingerprint(
        _digest(instance_payload),
        _digest(topology_payload),
        _digest(feature_payload),
    )


class CanonicalGraphBuilder:
    def build(
        self,
        shop: ShopFloor,
        progress_callback: Callable[[int, int, int, int], None] | None = None,
        deadline: float | None = None,
    ) -> CanonicalGraph:
        nodes: list[CanonicalNode] = []
        edges: list[CanonicalEdge] = []
        node_order: list[str] = []
        seen_nodes: set[str] = set()
        edge_pairs: set[tuple[str, str]] = set()

        def record_node_appearance(node_id: str) -> None:
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                node_order.append(node_id)

        def add_node(
            node_id: str,
            node_type: str,
            entity_id: str,
            attrs: dict[str, ScalarValue],
        ) -> None:
            record_node_appearance(node_id)
            nodes.append(
                CanonicalNode(node_id, node_type, entity_id, MappingProxyType(attrs))
            )

        def add_edge(
            source: str,
            target: str,
            edge_type: str,
            attrs: dict[str, ScalarValue] | None = None,
        ) -> None:
            record_node_appearance(source)
            record_node_appearance(target)
            edge_pairs.add((source, target))
            edges.append(
                CanonicalEdge(
                    source,
                    target,
                    edge_type,
                    MappingProxyType(attrs or {}),
                )
            )

        def report(processed: int, total: int) -> None:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("图谱内存构建超过时间限制")
            if progress_callback:
                progress_callback(processed, total, len(node_order), len(edge_pairs))

        for machine_id, machine in shop.machines.items():
            machine_type = shop.machine_types.get(machine.type_id)
            add_node(
                f"M:{machine_id}",
                "machine",
                machine_id,
                {
                    "label": machine.name,
                    "type_id": machine.type_id,
                    "type_name": machine_type.name if machine_type else "",
                    "is_critical": machine_type.is_critical if machine_type else False,
                },
            )

        for tooling_id, tooling in shop.toolings.items():
            tooling_type = shop.tooling_types.get(tooling.type_id)
            add_node(
                f"TL:{tooling_id}",
                "tooling",
                tooling_id,
                {
                    "label": tooling.name,
                    "type_id": tooling.type_id,
                    "type_name": tooling_type.name if tooling_type else "",
                },
            )

        for person_id, person in shop.personnel.items():
            add_node(
                f"P:{person_id}",
                "personnel",
                person_id,
                {"label": person.name, "skills": ";".join(person.skills)},
            )

        report(0, max(1, len(shop.orders) + len(shop.tasks) + len(shop.operations)))

        task_predecessors = {
            task_id: list(task.predecessor_task_ids)
            for task_id, task in shop.tasks.items()
        }
        for operation in shop.operations.values():
            predecessors = task_predecessors.get(operation.task_id)
            if predecessors is None:
                continue
            for predecessor_task_id in operation.predecessor_tasks:
                if predecessor_task_id not in predecessors:
                    predecessors.append(predecessor_task_id)

        processed = 0
        total_entities = max(1, len(shop.orders) + len(shop.tasks) + len(shop.operations))
        for order_id, order in shop.orders.items():
            add_node(
                f"O:{order_id}",
                "order",
                order_id,
                {
                    "label": order.name,
                    "due_date": order.due_date,
                    "due_at": shop.time_label(order.due_date),
                    "priority": order.priority,
                    "release_time": order.release_time,
                    "release_at": shop.time_label(order.release_time),
                },
            )
            processed += 1
            if processed % 500 == 0:
                report(processed, total_entities)

        for task_id, task in shop.tasks.items():
            add_node(
                f"T:{task_id}",
                "task",
                task_id,
                {
                    "label": task.name,
                    "order_id": task.order_id,
                    "is_main": task.is_main,
                    "due_date": task.due_date,
                    "release_time": task.release_time,
                    "due_at": shop.time_label(task.due_date),
                    "release_at": shop.time_label(task.release_time),
                    "derived_due_date": task.derived_due_date,
                    "derived_due_at": shop.time_label(task.derived_due_date),
                    "derived_start_time": task.derived_start_time,
                    "derived_start_at": shop.time_label(task.derived_start_time),
                    "critical_path_time": task.critical_path_time,
                    "critical_slack": task.critical_slack,
                },
            )
            add_edge(f"O:{task.order_id}", f"T:{task_id}", "order_has_task")
            for predecessor_task_id in task_predecessors[task_id]:
                add_edge(
                    f"T:{predecessor_task_id}",
                    f"T:{task_id}",
                    "task_predecessor",
                )
            processed += 1
            if processed % 500 == 0:
                report(processed, total_entities)

        for operation_id, operation in shop.operations.items():
            add_node(
                f"OP:{operation_id}",
                "operation",
                operation_id,
                {
                    "label": operation.name,
                    "task_id": operation.task_id,
                    "process_type": operation.process_type,
                    "processing_time": operation.processing_time,
                    "required_tooling_types": ";".join(operation.required_tooling_types),
                    "required_personnel_skills": ";".join(
                        operation.required_personnel_skills
                    ),
                    "status": operation.status.value,
                    "derived_due_date": operation.derived_due_date,
                    "derived_due_at": shop.time_label(operation.derived_due_date),
                    "derived_start_time": operation.derived_start_time,
                    "derived_start_at": shop.time_label(operation.derived_start_time),
                    "critical_slack": operation.critical_slack,
                },
            )
            add_edge(
                f"T:{operation.task_id}",
                f"OP:{operation_id}",
                "task_has_operation",
            )
            for predecessor_operation_id in operation.predecessor_ops:
                add_edge(
                    f"OP:{predecessor_operation_id}",
                    f"OP:{operation_id}",
                    "operation_sequence",
                )
            for predecessor_task_id in operation.predecessor_tasks:
                add_edge(
                    f"T:{predecessor_task_id}",
                    f"OP:{operation_id}",
                    "op_depends_task",
                )

            eligible_machine_ids = operation.eligible_machine_ids
            if not eligible_machine_ids:
                eligible_machine_ids = shop._machine_by_type.get(operation.process_type, [])
            for machine_id in eligible_machine_ids:
                add_edge(
                    f"OP:{operation_id}",
                    f"M:{machine_id}",
                    "machine_eligible",
                )
            for tooling_type in operation.required_tooling_types:
                for tooling in shop.get_toolings_for_type(tooling_type):
                    add_edge(
                        f"OP:{operation_id}",
                        f"TL:{tooling.id}",
                        "tooling_eligible",
                    )
            for skill_id in operation.required_personnel_skills:
                for person in shop.get_personnel_for_skill(skill_id):
                    add_edge(
                        f"OP:{operation_id}",
                        f"P:{person.id}",
                        "personnel_eligible",
                    )
            processed += 1
            if processed % 250 == 0:
                report(processed, total_entities)

        report(total_entities, total_entities)
        return CanonicalGraph(
            tuple(nodes),
            tuple(edges),
            compute_graph_fingerprint(shop),
            tuple(node_order),
        )
