from __future__ import annotations

import json
import math
import sqlite3
import time
from contextlib import contextmanager
from types import MappingProxyType
from typing import Callable, Iterator

from ..knowledge.canonical import CanonicalEdge, GraphFingerprint
from ..knowledge.context import (
    ComputeGraphProjection,
    DisplayGraphProjection,
    GraphContext,
    GraphContextCorruptError,
    GraphContextPersistenceError,
)
from .db import DB_PATH, get_db


class GraphArtifactStore:
    _COMPUTE_TABLES = (
        "graph_entity_index",
        "graph_context_relations",
        "graph_operation_features",
        "graph_operation_groups",
    )

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    @contextmanager
    def _write_connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _display_rows(display: DisplayGraphProjection):
        nodes_by_id = {node.node_id: node for node in display.nodes}
        node_order = list(display._node_order)
        seen_nodes = set(node_order)
        if not node_order:
            for node in display.nodes:
                if node.node_id not in seen_nodes:
                    seen_nodes.add(node.node_id)
                    node_order.append(node.node_id)
            for edge in display.edges:
                for node_id in (edge.source, edge.target):
                    if node_id not in seen_nodes:
                        seen_nodes.add(node_id)
                        node_order.append(node_id)
        for node in display.nodes:
            if node.node_id not in seen_nodes:
                seen_nodes.add(node.node_id)
                node_order.append(node.node_id)

        node_rows = []
        node_type_counts: dict[str, int] = {}
        for node_id in node_order:
            node = nodes_by_id.get(node_id)
            if node is None:
                node_type = ""
                entity_id = ""
                attrs = {}
                count_type = "unknown"
            else:
                node_type = node.node_type
                entity_id = node.entity_id
                attrs = dict(node.attrs)
                count_type = node_type or "unknown"
            node_type_counts[count_type] = node_type_counts.get(count_type, 0) + 1
            node_rows.append(
                (
                    node_id,
                    node_type,
                    entity_id,
                    json.dumps(attrs, ensure_ascii=False, sort_keys=True),
                )
            )

        edges_by_source: dict[str, dict[str, CanonicalEdge]] = {}
        for edge in display.edges:
            edges_by_source.setdefault(edge.source, {})[edge.target] = edge
        edge_rows = []
        edge_type_counts: dict[str, int] = {}
        ordered_sources = [*node_order]
        ordered_sources.extend(
            source for source in edges_by_source if source not in seen_nodes
        )
        for source in ordered_sources:
            for edge in edges_by_source.get(source, {}).values():
                edge_type = edge.edge_type or "unknown"
                edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
                edge_rows.append(
                    (
                        edge.source,
                        edge.target,
                        edge.edge_type,
                        json.dumps(
                            dict(edge.attrs), ensure_ascii=False, sort_keys=True
                        ),
                    )
                )
        return node_rows, edge_rows, node_type_counts, edge_type_counts

    @staticmethod
    def _relation_rows(context: GraphContext) -> list[tuple[str, int, int]]:
        rows: list[tuple[str, int, int]] = []

        def extend(
            relation_type: str,
            offsets: tuple[int, ...],
            indices: tuple[int, ...],
        ) -> None:
            for source in range(len(offsets) - 1):
                rows.extend(
                    (relation_type, source, target)
                    for target in indices[offsets[source] : offsets[source + 1]]
                )

        extend("predecessor", context.predecessor_offsets, context.predecessor_indices)
        extend("successor", context.successor_offsets, context.successor_indices)
        extend(
            "eligible_machine",
            context.eligible_machine_offsets,
            context.eligible_machine_indices,
        )
        return rows

    @staticmethod
    def _entity_rows(
        display: DisplayGraphProjection, context: GraphContext
    ) -> list[tuple[str, str, int]]:
        entity_ids: dict[str, set[str]] = {
            "order": set(),
            "task": set(),
            "tooling": set(),
            "personnel": set(),
        }
        for node in display.nodes:
            if node.node_type in entity_ids:
                entity_ids[node.node_type].add(node.entity_id)

        rows = [
            ("operation", entity_id, ordinal)
            for ordinal, entity_id in enumerate(context.operation_ids)
        ]
        rows.extend(
            ("machine", entity_id, ordinal)
            for ordinal, entity_id in enumerate(context.machine_ids)
        )
        for entity_type, ids in entity_ids.items():
            rows.extend(
                (entity_type, entity_id, ordinal)
                for ordinal, entity_id in enumerate(sorted(ids))
            )
        return rows

    def _after_relation_insert(self, conn: sqlite3.Connection) -> None:
        pass

    def save_artifacts(
        self,
        display: DisplayGraphProjection,
        context: GraphContext,
        precommit_check: Callable[[], None] | None = None,
    ) -> None:
        if display.fingerprint != context.fingerprint:
            raise GraphContextPersistenceError("display and compute fingerprints differ")

        fingerprint = context.fingerprint
        node_rows, edge_rows, node_type_counts, edge_type_counts = self._display_rows(
            display
        )
        entity_rows = self._entity_rows(display, context)
        relation_rows = self._relation_rows(context)
        feature_rows = [
            (ordinal, *row) for ordinal, row in enumerate(context.feature_matrix)
        ]
        group_rows = [
            (group_type, group_key, ordinal)
            for (group_type, group_key), ordinals in context.operation_groups.items()
            for ordinal in ordinals
        ]
        created_at = time.time()

        with self._write_connection() as conn:
            if precommit_check is not None:
                precommit_check()

            conn.execute("DELETE FROM graph_meta")
            conn.execute("DELETE FROM graph_context_meta")
            conn.execute("DELETE FROM graph_nodes")
            conn.execute("DELETE FROM graph_edges")
            for table in self._COMPUTE_TABLES:
                conn.execute(f"DELETE FROM {table}")

            conn.executemany(
                "INSERT INTO graph_nodes (node_id, node_type, entity_id, attrs) VALUES (?,?,?,?)",
                node_rows,
            )
            conn.executemany(
                "INSERT INTO graph_edges (source, target, edge_type, attrs) VALUES (?,?,?,?)",
                edge_rows,
            )
            conn.executemany(
                "INSERT INTO graph_entity_index (entity_type, entity_id, ordinal) VALUES (?,?,?)",
                entity_rows,
            )
            conn.executemany(
                """
                INSERT INTO graph_context_relations
                (relation_type, source_ordinal, target_ordinal) VALUES (?,?,?)
                """,
                relation_rows,
            )
            self._after_relation_insert(conn)
            conn.executemany(
                """
                INSERT INTO graph_operation_features
                (op_ordinal, predecessor_depth, assembly_criticality,
                 shared_resource_degree, bottleneck_adjacency, graph_out_degree)
                VALUES (?,?,?,?,?,?)
                """,
                feature_rows,
            )
            conn.executemany(
                """
                INSERT INTO graph_operation_groups
                (group_type, group_key, op_ordinal) VALUES (?,?,?)
                """,
                group_rows,
            )

            conn.execute(
                """
                INSERT INTO graph_meta
                (id, total_nodes, total_edges, node_type_counts, edge_type_counts,
                 created_at, instance_hash, topology_hash, feature_hash,
                 schema_version, builder_version, build_time_ms, invalid_reason)
                VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    len(node_rows),
                    len(edge_rows),
                    json.dumps(node_type_counts, sort_keys=True),
                    json.dumps(edge_type_counts, sort_keys=True),
                    created_at,
                    fingerprint.instance_hash,
                    fingerprint.topology_hash,
                    fingerprint.feature_hash,
                    fingerprint.schema_version,
                    fingerprint.builder_version,
                    0.0,
                    "",
                ),
            )
            conn.execute(
                """
                INSERT INTO graph_context_meta
                (id, instance_hash, topology_hash, feature_hash, schema_version,
                 builder_version, status, operation_count, relation_count,
                 feature_count, build_time_ms, created_at, invalid_reason)
                VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    fingerprint.instance_hash,
                    fingerprint.topology_hash,
                    fingerprint.feature_hash,
                    fingerprint.schema_version,
                    fingerprint.builder_version,
                    "ready",
                    len(context.operation_ids),
                    len(relation_rows),
                    len(feature_rows),
                    0.0,
                    created_at,
                    "",
                ),
            )

    @staticmethod
    def _rows_to_csr(
        size: int, rows: list[tuple[int, int]], target_size: int, name: str
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        buckets: list[list[int]] = [[] for _ in range(size)]
        for source_value, target_value in rows:
            source = int(source_value)
            target = int(target_value)
            if not 0 <= source < size or not 0 <= target < target_size:
                raise GraphContextCorruptError(f"{name} relation ordinal out of bounds")
            buckets[source].append(target)
        offsets = [0]
        indices: list[int] = []
        for bucket in buckets:
            indices.extend(bucket)
            offsets.append(len(indices))
        return tuple(offsets), tuple(indices)

    def load_context(self, fingerprint: GraphFingerprint) -> GraphContext | None:
        with get_db(self.db_path) as conn:
            conn.row_factory = None
            meta = conn.execute(
                """
                SELECT status, instance_hash, topology_hash, feature_hash,
                       schema_version, builder_version, operation_count,
                       relation_count, feature_count
                FROM graph_context_meta WHERE id=1
                """
            ).fetchone()
            if not meta or meta[0] != "ready":
                return None
            stored_fingerprint = GraphFingerprint(
                instance_hash=meta[1],
                topology_hash=meta[2],
                feature_hash=meta[3],
                schema_version=int(meta[4]),
                builder_version=meta[5],
            )
            if stored_fingerprint != fingerprint:
                return None

            display_meta = conn.execute(
                """
                SELECT instance_hash, topology_hash, feature_hash,
                       schema_version, builder_version, invalid_reason,
                       total_nodes, total_edges,
                       (SELECT COUNT(*) FROM graph_nodes),
                       (SELECT COUNT(*) FROM graph_edges)
                FROM graph_meta WHERE id=1
                """
            ).fetchone()
            if not display_meta:
                raise GraphContextCorruptError("display graph metadata is missing")
            display_fingerprint = GraphFingerprint(
                instance_hash=display_meta[0],
                topology_hash=display_meta[1],
                feature_hash=display_meta[2],
                schema_version=int(display_meta[3]),
                builder_version=display_meta[4],
            )
            if display_fingerprint != stored_fingerprint:
                raise GraphContextCorruptError(
                    "display and compute fingerprints differ"
                )
            if display_meta[5]:
                raise GraphContextCorruptError("display graph is marked invalid")
            if (
                int(display_meta[6]) != display_meta[8]
                or int(display_meta[7]) != display_meta[9]
            ):
                raise GraphContextCorruptError("display graph metadata count mismatch")

            entity_rows = conn.execute(
                """
                SELECT entity_type, entity_id, ordinal
                FROM graph_entity_index
                WHERE entity_type IN ('machine', 'operation')
                ORDER BY entity_type, ordinal
                """
            ).fetchall()
            entity_ids: dict[str, list[str]] = {"machine": [], "operation": []}
            for entity_type, entity_id, ordinal in entity_rows:
                values = entity_ids[entity_type]
                if int(ordinal) != len(values):
                    raise GraphContextCorruptError(
                        f"{entity_type} ordinals are not contiguous"
                    )
                values.append(entity_id)
            operation_ids = tuple(entity_ids["operation"])
            machine_ids = tuple(entity_ids["machine"])
            if len(operation_ids) != int(meta[6]):
                raise GraphContextCorruptError("operation metadata count mismatch")

            relation_rows = conn.execute(
                """
                SELECT relation_type, source_ordinal, target_ordinal
                FROM graph_context_relations
                ORDER BY relation_type, source_ordinal, target_ordinal
                """
            ).fetchall()
            if len(relation_rows) != int(meta[7]):
                raise GraphContextCorruptError("relation metadata count mismatch")
            relations: dict[str, list[tuple[int, int]]] = {
                "predecessor": [],
                "successor": [],
                "eligible_machine": [],
            }
            for relation_type, source, target in relation_rows:
                if relation_type not in relations:
                    raise GraphContextCorruptError("unknown graph relation type")
                relations[relation_type].append((source, target))

            predecessor_offsets, predecessor_indices = self._rows_to_csr(
                len(operation_ids),
                relations["predecessor"],
                len(operation_ids),
                "predecessor",
            )
            successor_offsets, successor_indices = self._rows_to_csr(
                len(operation_ids),
                relations["successor"],
                len(operation_ids),
                "successor",
            )
            eligible_machine_offsets, eligible_machine_indices = self._rows_to_csr(
                len(operation_ids),
                relations["eligible_machine"],
                len(machine_ids),
                "eligible machine",
            )

            feature_rows = conn.execute(
                """
                SELECT op_ordinal, predecessor_depth, assembly_criticality,
                       shared_resource_degree, bottleneck_adjacency,
                       graph_out_degree
                FROM graph_operation_features ORDER BY op_ordinal
                """
            ).fetchall()
            if (
                len(feature_rows) != len(operation_ids)
                or len(feature_rows) != int(meta[8])
                or any(
                    int(row[0]) != ordinal
                    for ordinal, row in enumerate(feature_rows)
                )
            ):
                raise GraphContextCorruptError("feature metadata count mismatch")
            feature_names = ComputeGraphProjection.FEATURE_NAMES
            feature_matrix = tuple(
                tuple(float(value) for value in row[1:])
                for row in feature_rows
            )
            if any(
                not math.isfinite(value)
                for feature_row in feature_matrix
                for value in feature_row
            ):
                raise GraphContextCorruptError("non-finite graph context feature")

            group_rows = conn.execute(
                """
                SELECT group_type, group_key, op_ordinal
                FROM graph_operation_groups
                ORDER BY group_type, group_key, op_ordinal
                """
            ).fetchall()
            groups: dict[tuple[str, str], list[int]] = {}
            for group_type, group_key, ordinal_value in group_rows:
                ordinal = int(ordinal_value)
                if not 0 <= ordinal < len(operation_ids):
                    raise GraphContextCorruptError("group ordinal out of bounds")
                groups.setdefault((group_type, group_key), []).append(ordinal)

            return GraphContext(
                fingerprint=stored_fingerprint,
                operation_ids=operation_ids,
                operation_index=MappingProxyType(
                    {
                        entity_id: ordinal
                        for ordinal, entity_id in enumerate(operation_ids)
                    }
                ),
                machine_ids=machine_ids,
                machine_index=MappingProxyType(
                    {
                        entity_id: ordinal
                        for ordinal, entity_id in enumerate(machine_ids)
                    }
                ),
                predecessor_offsets=predecessor_offsets,
                predecessor_indices=predecessor_indices,
                successor_offsets=successor_offsets,
                successor_indices=successor_indices,
                eligible_machine_offsets=eligible_machine_offsets,
                eligible_machine_indices=eligible_machine_indices,
                feature_names=feature_names,
                feature_matrix=feature_matrix,
                operation_groups=MappingProxyType(
                    {key: tuple(values) for key, values in groups.items()}
                ),
            )

    def load_context_meta(self) -> dict | None:
        with get_db(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM graph_context_meta WHERE id=1"
            ).fetchone()
            return dict(row) if row else None

    def mark_invalid(self, reason: str) -> None:
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE graph_context_meta
                SET status='invalid', invalid_reason=? WHERE id=1
                """,
                (reason,),
            )
            conn.execute(
                "UPDATE graph_meta SET invalid_reason=? WHERE id=1", (reason,)
            )

    def clear_all(self) -> None:
        with self._write_connection() as conn:
            conn.execute("DELETE FROM graph_meta")
            conn.execute("DELETE FROM graph_context_meta")
            conn.execute("DELETE FROM graph_nodes")
            conn.execute("DELETE FROM graph_edges")
            for table in self._COMPUTE_TABLES:
                conn.execute(f"DELETE FROM {table}")
