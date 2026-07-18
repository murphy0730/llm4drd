from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from ..core.time_utils import datetime_to_offset_hours, default_plan_start, ensure_aware, isoformat_or_none

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("LLM4DRD_DB", "llm4drd.db")


def _clean_scalar(value, default: str = ""):
    if value is None:
        return default
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else default
    return value


def _float_or_default(value, default: float = 0.0) -> float:
    cleaned = _clean_scalar(value, None)
    if cleaned is None:
        return float(default)
    return float(cleaned)


def _int_or_default(value, default: int = 0) -> int:
    cleaned = _clean_scalar(value, None)
    if cleaned is None:
        return int(default)
    return int(float(cleaned))


@contextmanager
def get_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _bump_instance_version(conn) -> None:
    """实例数据变更计数器。

    _active_shop() 按此版本号缓存整个 ShopFloor（大实例重建一次约 2s），
    所以每个写入 inst_* / planning_context / machine_downtime 的方法都必须调用它，
    否则接口会静默返回过期实例。参与 build_shopfloor() 的写入点都算。
    """
    conn.execute(
        "INSERT INTO inst_version (id, version) VALUES (1, 1) "
        "ON CONFLICT(id) DO UPDATE SET version = version + 1"
    )


def get_instance_version(db_path: str = DB_PATH) -> int:
    with get_db(db_path) as conn:
        row = conn.execute("SELECT version FROM inst_version WHERE id=1").fetchone()
        return int(row["version"]) if row else 0


def _safe_add_column(conn, table_name: str, column_name: str, ddl: str) -> None:
    try:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}")
    except sqlite3.OperationalError:
        pass


def init_db(db_path: str = DB_PATH):
    with get_db(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS rules (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                code TEXT NOT NULL,
                problem_type TEXT DEFAULT 'FAFSP',
                objective TEXT DEFAULT 'total_tardiness',
                fitness REAL,
                hybrid_score REAL,
                llm_score REAL,
                generation INTEGER DEFAULT 0,
                parent_ids TEXT DEFAULT '[]',
                features_used TEXT DEFAULT '[]',
                is_active INTEGER DEFAULT 1,
                is_builtin INTEGER DEFAULT 0,
                created_at REAL DEFAULT (strftime('%s','now')),
                updated_at REAL DEFAULT (strftime('%s','now')),
                metadata TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS evolution_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_type TEXT,
                objective TEXT,
                config TEXT,
                best_rule_id TEXT,
                best_fitness REAL,
                total_generations INTEGER,
                generation_history TEXT,
                started_at REAL,
                completed_at REAL,
                status TEXT DEFAULT 'running'
            );
            CREATE TABLE IF NOT EXISTS schedule_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_id TEXT,
                instance_id TEXT,
                total_tardiness REAL,
                makespan REAL,
                avg_utilization REAL,
                avg_flowtime REAL,
                tardy_count INTEGER,
                total_jobs INTEGER,
                schedule_data TEXT,
                created_at REAL DEFAULT (strftime('%s','now'))
            );
            CREATE TABLE IF NOT EXISTS reschedule_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger_reason TEXT,
                old_fitness REAL,
                new_fitness REAL,
                improvement REAL,
                changed_count INTEGER,
                total_count INTEGER,
                computation_time REAL,
                created_at REAL DEFAULT (strftime('%s','now'))
            );
            CREATE TABLE IF NOT EXISTS planning_context (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                plan_start_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS inst_version (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS inst_orders (
                order_id TEXT PRIMARY KEY,
                order_name TEXT,
                release_time REAL DEFAULT 0,
                due_date REAL,
                priority INTEGER DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS inst_tasks (
                task_id TEXT PRIMARY KEY,
                order_id TEXT,
                task_name TEXT,
                is_main INTEGER DEFAULT 0,
                predecessor_task_ids TEXT DEFAULT '',
                release_time REAL DEFAULT 0,
                due_date REAL
            );
            CREATE TABLE IF NOT EXISTS inst_operations (
                op_id TEXT PRIMARY KEY,
                task_id TEXT,
                op_name TEXT,
                process_type TEXT,
                processing_time REAL,
                turnover_time REAL DEFAULT 0,
                predecessor_ops TEXT DEFAULT '',
                predecessor_tasks TEXT DEFAULT '',
                eligible_machine_ids TEXT DEFAULT '',
                required_tooling_types TEXT DEFAULT '',
                required_personnel_skills TEXT DEFAULT '',
                initial_status TEXT DEFAULT '',
                initial_start_time REAL,
                initial_end_time REAL,
                initial_remaining_processing_time REAL,
                initial_assigned_machine_id TEXT DEFAULT '',
                initial_assigned_tooling_ids TEXT DEFAULT '',
                initial_assigned_personnel_ids TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS inst_machine_types (
                type_id TEXT PRIMARY KEY,
                type_name TEXT,
                is_critical INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS inst_tooling_types (
                type_id TEXT PRIMARY KEY,
                type_name TEXT
            );
            CREATE TABLE IF NOT EXISTS inst_machines (
                machine_id TEXT PRIMARY KEY,
                machine_name TEXT,
                type_id TEXT,
                shifts TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS inst_toolings (
                tooling_id TEXT PRIMARY KEY,
                tooling_name TEXT,
                type_id TEXT,
                shifts TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS inst_personnel (
                personnel_id TEXT PRIMARY KEY,
                personnel_name TEXT,
                skills TEXT DEFAULT '',
                shifts TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS graph_nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT,
                entity_id TEXT,
                attrs TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                target TEXT,
                edge_type TEXT,
                attrs TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS graph_meta (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                total_nodes INTEGER DEFAULT 0,
                total_edges INTEGER DEFAULT 0,
                node_type_counts TEXT DEFAULT '{}',
                edge_type_counts TEXT DEFAULT '{}',
                created_at REAL DEFAULT (strftime('%s','now')),
                instance_hash TEXT DEFAULT '',
                topology_hash TEXT DEFAULT '',
                feature_hash TEXT DEFAULT '',
                schema_version INTEGER DEFAULT 0,
                builder_version TEXT DEFAULT '',
                build_time_ms REAL DEFAULT 0,
                invalid_reason TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS graph_context_meta (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                instance_hash TEXT NOT NULL,
                topology_hash TEXT NOT NULL,
                feature_hash TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                builder_version TEXT NOT NULL,
                status TEXT NOT NULL,
                operation_count INTEGER NOT NULL,
                relation_count INTEGER NOT NULL,
                feature_count INTEGER NOT NULL,
                build_time_ms REAL NOT NULL,
                created_at REAL NOT NULL,
                invalid_reason TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS graph_entity_index (
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                PRIMARY KEY (entity_type, entity_id),
                UNIQUE (entity_type, ordinal)
            );
            CREATE TABLE IF NOT EXISTS graph_context_relations (
                relation_type TEXT NOT NULL,
                source_ordinal INTEGER NOT NULL,
                target_ordinal INTEGER NOT NULL,
                PRIMARY KEY (relation_type, source_ordinal, target_ordinal)
            );
            CREATE TABLE IF NOT EXISTS graph_operation_features (
                op_ordinal INTEGER PRIMARY KEY,
                predecessor_depth REAL NOT NULL,
                assembly_criticality REAL NOT NULL,
                shared_resource_degree REAL NOT NULL,
                bottleneck_adjacency REAL NOT NULL,
                graph_out_degree REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS graph_operation_groups (
                group_type TEXT NOT NULL,
                group_key TEXT NOT NULL,
                op_ordinal INTEGER NOT NULL,
                PRIMARY KEY (group_type, group_key, op_ordinal)
            );
            CREATE TABLE IF NOT EXISTS machine_downtime (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id TEXT NOT NULL,
                downtime_type TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS workflow_progress (
                step TEXT PRIMARY KEY,
                instance_version INTEGER NOT NULL,
                payload TEXT NOT NULL,
                updated_at REAL DEFAULT (strftime('%s','now'))
            );
            CREATE TABLE IF NOT EXISTS rule_reference_cache (
                inst_version INTEGER NOT NULL,
                rule_name    TEXT NOT NULL,
                payload      TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                PRIMARY KEY (inst_version, rule_name)
            );
            CREATE INDEX IF NOT EXISTS idx_rules_type ON rules(problem_type, objective);
            CREATE INDEX IF NOT EXISTS idx_rules_active ON rules(is_active);
            CREATE INDEX IF NOT EXISTS idx_results_rule ON schedule_results(rule_id);
            CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes(node_type);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_type ON graph_edges(edge_type);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_src ON graph_edges(source);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_tgt ON graph_edges(target);
            CREATE INDEX IF NOT EXISTS idx_graph_context_rel_src
                ON graph_context_relations(relation_type, source_ordinal);
            CREATE INDEX IF NOT EXISTS idx_graph_context_rel_tgt
                ON graph_context_relations(relation_type, target_ordinal);
            CREATE INDEX IF NOT EXISTS idx_graph_operation_groups_lookup
                ON graph_operation_groups(group_type, group_key);
            """
        )
        _safe_add_column(conn, "inst_operations", "required_tooling_types", "TEXT DEFAULT ''")
        _safe_add_column(conn, "inst_operations", "required_personnel_skills", "TEXT DEFAULT ''")
        _safe_add_column(conn, "inst_operations", "initial_status", "TEXT DEFAULT ''")
        _safe_add_column(conn, "inst_operations", "initial_start_time", "REAL")
        _safe_add_column(conn, "inst_operations", "initial_end_time", "REAL")
        _safe_add_column(conn, "inst_operations", "initial_remaining_processing_time", "REAL")
        _safe_add_column(conn, "inst_operations", "initial_assigned_machine_id", "TEXT DEFAULT ''")
        _safe_add_column(conn, "inst_operations", "initial_assigned_tooling_ids", "TEXT DEFAULT ''")
        _safe_add_column(conn, "inst_operations", "initial_assigned_personnel_ids", "TEXT DEFAULT ''")
        _safe_add_column(conn, "inst_operations", "turnover_time", "REAL DEFAULT 0")
        _safe_add_column(conn, "graph_meta", "instance_hash", "TEXT DEFAULT ''")
        _safe_add_column(conn, "graph_meta", "topology_hash", "TEXT DEFAULT ''")
        _safe_add_column(conn, "graph_meta", "feature_hash", "TEXT DEFAULT ''")
        _safe_add_column(conn, "graph_meta", "schema_version", "INTEGER DEFAULT 0")
        _safe_add_column(conn, "graph_meta", "builder_version", "TEXT DEFAULT ''")
        _safe_add_column(conn, "graph_meta", "build_time_ms", "REAL DEFAULT 0")
        _safe_add_column(conn, "graph_meta", "invalid_reason", "TEXT DEFAULT ''")
        _safe_add_column(conn, "graph_context_meta", "invalid_reason", "TEXT DEFAULT ''")
        conn.execute(
            "INSERT OR IGNORE INTO planning_context (id, plan_start_at) VALUES (1, ?)",
            (isoformat_or_none(default_plan_start()),),
        )
    logger.info("Database initialized: %s", db_path)


WORKFLOW_STEPS = ("validation", "simulation", "optimization", "review")


class WorkflowProgressStore:
    """各流程步骤（校验/仿真/优化/评审）的结果快照，供进程重启后恢复。

    每条快照记录写入时的 inst_version。读取时版本对不上就当作不存在：实例一旦被
    改动（导入、编辑、改停机/班次），旧的校验结论和排程结果都不再成立，必须重算。
    与 graph_meta 的失效判定同源，只是这里用版本号而非指纹——指纹要重建整个
    ShopFloor 才能算，而这些步骤本来就只在实例未变时才可复用。

    写入方法不调用 _bump_instance_version()：这张表不是实例数据，递增版本号会把
    刚存进去的快照连同 _active_shop() 缓存一起作废。
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def save(self, step: str, payload: dict):
        if step not in WORKFLOW_STEPS:
            raise ValueError(f"未知的流程步骤: {step}")
        with get_db(self.db_path) as conn:
            row = conn.execute("SELECT version FROM inst_version WHERE id=1").fetchone()
            conn.execute(
                "INSERT INTO workflow_progress (step, instance_version, payload, updated_at) "
                "VALUES (?, ?, ?, strftime('%s','now')) "
                "ON CONFLICT(step) DO UPDATE SET "
                "instance_version=excluded.instance_version, payload=excluded.payload, "
                "updated_at=excluded.updated_at",
                (step, int(row["version"]) if row else 0, json.dumps(payload, ensure_ascii=False)),
            )

    def load(self, step: str) -> Optional[dict]:
        with get_db(self.db_path) as conn:
            return self._load_valid(conn, step)

    def load_all(self) -> dict:
        with get_db(self.db_path) as conn:
            loaded = {step: self._load_valid(conn, step) for step in WORKFLOW_STEPS}
        return {step: payload for step, payload in loaded.items() if payload is not None}

    def clear(self, step: str):
        with get_db(self.db_path) as conn:
            conn.execute("DELETE FROM workflow_progress WHERE step=?", (step,))

    def clear_all(self):
        with get_db(self.db_path) as conn:
            conn.execute("DELETE FROM workflow_progress")

    def _load_valid(self, conn, step: str) -> Optional[dict]:
        row = conn.execute(
            "SELECT wp.payload FROM workflow_progress wp "
            "JOIN inst_version iv ON iv.id=1 AND iv.version=wp.instance_version "
            "WHERE wp.step=?",
            (step,),
        ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["payload"])
        except (TypeError, ValueError):
            logger.warning("workflow_progress[%s] 内容损坏，已忽略", step)
            return None


class RuleReferenceCacheStore:
    """规则参照方案的两级缓存（内存 dict + SQLite）。

    缓存键 = (inst_version, rule_name)。规则参照方案是"当前实例 + 规则名"的纯函数，
    实例一旦被改动 _bump_instance_version() 会让版本号递增，读取时用当前版本号取行，
    版本对不上就当作未命中，与 WorkflowProgressStore 同源失效，不产生脏数据。

    写入方法不调用 _bump_instance_version()：缓存不是实例数据，递增版本号会把刚写入
    的缓存连同 _active_shop() 缓存一起作废。存的是全量 payload（schedule 不截断、目标
    值覆盖全部 OBJECTIVE_SPECS），响应时再按请求裁剪，避免缓存被截断版本污染。
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._mem: dict[tuple[int, str], dict] = {}

    @staticmethod
    def _current_version(conn) -> int:
        row = conn.execute("SELECT version FROM inst_version WHERE id=1").fetchone()
        return int(row["version"]) if row else 0

    def _prune_old(self, version: int) -> None:
        # 只保留当前版本的内存条目：旧版本一旦被 _bump_instance_version() 作废就再也读不到，
        # 留在 _mem 里只会随实例编辑次数无界累积（每条含全量 schedule，可达 MB 级）。
        stale = [key for key in self._mem if key[0] != version]
        for key in stale:
            del self._mem[key]

    def get(self, rule_name: str) -> Optional[dict]:
        with get_db(self.db_path) as conn:
            version = self._current_version(conn)
            self._prune_old(version)
            key = (version, rule_name)
            if key in self._mem:
                return self._mem[key]
            row = conn.execute(
                "SELECT payload FROM rule_reference_cache WHERE inst_version=? AND rule_name=?",
                (version, rule_name),
            ).fetchone()
        if not row:
            return None
        try:
            payload = json.loads(row["payload"])
        except (TypeError, ValueError):
            logger.warning("rule_reference_cache[%s] 内容损坏，已忽略", rule_name)
            return None
        self._mem[key] = payload
        return payload

    def put(self, rule_name: str, payload: dict) -> None:
        with get_db(self.db_path) as conn:
            version = self._current_version(conn)
            conn.execute(
                "INSERT INTO rule_reference_cache (inst_version, rule_name, payload, created_at) "
                "VALUES (?,?,?,strftime('%s','now')) "
                "ON CONFLICT(inst_version, rule_name) DO UPDATE SET "
                "payload=excluded.payload, created_at=excluded.created_at",
                (version, rule_name, json.dumps(payload, ensure_ascii=False)),
            )
        self._prune_old(version)
        self._mem[(version, rule_name)] = payload


class DowntimeStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def clear_all(self):
        with get_db(self.db_path) as conn:
            conn.execute("DELETE FROM machine_downtime")
            _bump_instance_version(conn)

    def save(self, machine_id: str, downtime_type: str, start_time: float, end_time: float) -> int:
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO machine_downtime (machine_id, downtime_type, start_time, end_time) VALUES (?,?,?,?)",
                (machine_id, downtime_type, float(start_time), float(end_time)),
            )
            _bump_instance_version(conn)
            return cur.lastrowid

    def replace_all(self, rows: list[dict]):
        self.clear_all()
        for row in rows:
            self.save(
                row["machine_id"],
                row.get("downtime_type", "planned"),
                _float_or_default(row.get("start_time"), 0.0),
                _float_or_default(row.get("end_time"), 0.0),
            )

    def list_all(self) -> list:
        with get_db(self.db_path) as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM machine_downtime ORDER BY machine_id, start_time").fetchall()]

    def delete(self, downtime_id: int):
        with get_db(self.db_path) as conn:
            conn.execute("DELETE FROM machine_downtime WHERE id=?", (downtime_id,))
            _bump_instance_version(conn)

    def update(self, downtime_id: int, data: dict):
        with get_db(self.db_path) as conn:
            conn.execute(
                "UPDATE machine_downtime SET machine_id=?, downtime_type=?, start_time=?, end_time=? WHERE id=?",
                (data["machine_id"], data["downtime_type"], float(data["start_time"]), float(data["end_time"]), downtime_id),
            )
            _bump_instance_version(conn)

    def load_all_as_downtimes(self) -> dict:
        from ..core.models import Downtime

        result = {}
        for row in self.list_all():
            downtime = Downtime(
                id=str(row["id"]),
                machine_id=row["machine_id"],
                downtime_type=row["downtime_type"],
                start_time=float(row["start_time"]),
                end_time=float(row["end_time"]),
            )
            result.setdefault(downtime.machine_id, []).append(downtime)
        return result


class RuleStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def save_rule(self, rule_id: str, name: str, code: str, problem_type: str = "FAFSP", objective: str = "total_tardiness", fitness: float = None, hybrid_score: float = None, llm_score: float = None, generation: int = 0, is_builtin: bool = False, metadata: dict = None):
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO rules
                (id, name, code, problem_type, objective, fitness, hybrid_score, llm_score, generation, is_builtin, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (rule_id, name, code, problem_type, objective, fitness, hybrid_score, llm_score, generation, 1 if is_builtin else 0, time.time(), json.dumps(metadata or {})),
            )

    def get_rule(self, rule_id: str) -> Optional[dict]:
        with get_db(self.db_path) as conn:
            row = conn.execute("SELECT * FROM rules WHERE id=?", (rule_id,)).fetchone()
            return dict(row) if row else None

    def get_best_rules(self, problem_type: str = "FAFSP", objective: str = "total_tardiness", limit: int = 5) -> list[dict]:
        with get_db(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM rules WHERE problem_type=? AND objective=? AND is_active=1 ORDER BY COALESCE(hybrid_score, fitness, 999999) ASC LIMIT ?",
                (problem_type, objective, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_all_rules(self, active_only: bool = True) -> list[dict]:
        with get_db(self.db_path) as conn:
            sql = "SELECT * FROM rules WHERE is_active=1 ORDER BY updated_at DESC" if active_only else "SELECT * FROM rules ORDER BY updated_at DESC"
            return [dict(row) for row in conn.execute(sql).fetchall()]

    def deactivate_rule(self, rule_id: str):
        with get_db(self.db_path) as conn:
            conn.execute("UPDATE rules SET is_active=0, updated_at=? WHERE id=?", (time.time(), rule_id))

    def save_schedule_result(self, rule_id: str, instance_id: str, result_dict: dict):
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO schedule_results
                (rule_id, instance_id, total_tardiness, makespan, avg_utilization, avg_flowtime, tardy_count, total_jobs, schedule_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (rule_id, instance_id, result_dict.get("total_tardiness", 0), result_dict.get("makespan", 0), result_dict.get("avg_utilization", 0), result_dict.get("avg_flowtime", 0), result_dict.get("tardy_job_count", 0), result_dict.get("total_jobs", 0), json.dumps(result_dict.get("schedule", []))),
            )

    def save_reschedule_record(self, record: dict):
        with get_db(self.db_path) as conn:
            conn.execute(
                "INSERT INTO reschedule_history (trigger_reason, old_fitness, new_fitness, improvement, changed_count, total_count, computation_time) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (record.get("trigger_reason", ""), record.get("old_fitness", 0), record.get("new_fitness", 0), record.get("improvement_pct", 0), record.get("changed_assignments", 0), record.get("total_assignments", 0), record.get("computation_time_s", 0)),
            )

    def get_performance_trend(self, rule_id: str, limit: int = 50) -> list[dict]:
        with get_db(self.db_path) as conn:
            rows = conn.execute("SELECT total_tardiness, makespan, avg_utilization, created_at FROM schedule_results WHERE rule_id=? ORDER BY created_at DESC LIMIT ?", (rule_id, limit)).fetchall()
            return [dict(row) for row in rows]


class InstanceStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def clear_all(self):
        with get_db(self.db_path) as conn:
            for table_name in [
                "planning_context",
                "inst_orders",
                "inst_tasks",
                "inst_operations",
                "inst_machine_types",
                "inst_tooling_types",
                "inst_machines",
                "inst_toolings",
                "inst_personnel",
            ]:
                conn.execute(f"DELETE FROM {table_name}")
            _bump_instance_version(conn)

    def save_from_shopfloor(self, shop):
        self.clear_all()
        with get_db(self.db_path) as conn:
            _bump_instance_version(conn)
            conn.execute("INSERT INTO planning_context (id, plan_start_at) VALUES (1, ?)", (isoformat_or_none(ensure_aware(shop.plan_start_at)),))
            for order_id, order in shop.orders.items():
                conn.execute("INSERT INTO inst_orders (order_id, order_name, release_time, due_date, priority) VALUES (?,?,?,?,?)", (order_id, order.name, order.release_time, order.due_date, order.priority))
            for task_id, task in shop.tasks.items():
                conn.execute(
                    "INSERT INTO inst_tasks (task_id, order_id, task_name, is_main, predecessor_task_ids, release_time, due_date) VALUES (?,?,?,?,?,?,?)",
                    (task_id, task.order_id, task.name, 1 if task.is_main else 0, ";".join(task.predecessor_task_ids), task.release_time, task.due_date),
                )
            for op_id, op in shop.operations.items():
                conn.execute(
                    """
                    INSERT INTO inst_operations
                    (
                        op_id, task_id, op_name, process_type, processing_time, turnover_time, predecessor_ops, predecessor_tasks,
                        eligible_machine_ids, required_tooling_types, required_personnel_skills, initial_status,
                        initial_start_time, initial_end_time, initial_remaining_processing_time, initial_assigned_machine_id,
                        initial_assigned_tooling_ids, initial_assigned_personnel_ids
                    )
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        op_id,
                        op.task_id,
                        op.name,
                        op.process_type,
                        op.processing_time,
                        op.turnover_time,
                        ";".join(op.predecessor_ops),
                        ";".join(op.predecessor_tasks),
                        ";".join(op.eligible_machine_ids),
                        ";".join(op.required_tooling_types),
                        ";".join(op.required_personnel_skills),
                        op.status.value if str(op.status.value).lower() != "pending" else "",
                        op.start_time,
                        op.end_time,
                        op.remaining_processing_time,
                        op.assigned_machine_id or "",
                        ";".join(op.assigned_tooling_ids),
                        ";".join(op.assigned_personnel_ids),
                    ),
                )
            for type_id, machine_type in shop.machine_types.items():
                conn.execute("INSERT INTO inst_machine_types (type_id, type_name, is_critical) VALUES (?,?,?)", (type_id, machine_type.name, 1 if machine_type.is_critical else 0))
            for type_id, tooling_type in shop.tooling_types.items():
                conn.execute("INSERT INTO inst_tooling_types (type_id, type_name) VALUES (?,?)", (type_id, tooling_type.name))
            for machine_id, machine in shop.machines.items():
                conn.execute("INSERT INTO inst_machines (machine_id, machine_name, type_id, shifts) VALUES (?,?,?,?)", (machine_id, machine.name, machine.type_id, _shifts_to_str(machine.shifts)))
            for tooling_id, tooling in shop.toolings.items():
                conn.execute("INSERT INTO inst_toolings (tooling_id, tooling_name, type_id, shifts) VALUES (?,?,?,?)", (tooling_id, tooling.name, tooling.type_id, _shifts_to_str(tooling.shifts)))
            for person_id, person in shop.personnel.items():
                conn.execute("INSERT INTO inst_personnel (personnel_id, personnel_name, skills, shifts) VALUES (?,?,?,?)", (person_id, person.name, ";".join(person.skills), _shifts_to_str(person.shifts)))

    def save_from_csv(self, orders_rows, tasks_rows, ops_rows, machine_type_rows, machine_rows, tooling_type_rows=None, tooling_rows=None, personnel_rows=None, initial_state_rows=None, plan_start_at=None):
        self.clear_all()
        with get_db(self.db_path) as conn:
            _bump_instance_version(conn)
            conn.execute("INSERT INTO planning_context (id, plan_start_at) VALUES (1, ?)", (isoformat_or_none(ensure_aware(plan_start_at or default_plan_start())),))
            order_due_map: dict[str, float] = {}
            order_release_map: dict[str, float] = {}
            for row in orders_rows:
                order_id = _clean_scalar(row["order_id"])
                release_time = _float_or_default(row.get("release_time"), 0.0)
                due_date = _float_or_default(row.get("due_date"), 0.0)
                priority = _int_or_default(row.get("priority"), 1)
                order_due_map[order_id] = due_date
                order_release_map[order_id] = release_time
                conn.execute(
                    "INSERT INTO inst_orders (order_id, order_name, release_time, due_date, priority) VALUES (?,?,?,?,?)",
                    (order_id, _clean_scalar(row.get("order_name"), ""), release_time, due_date, priority),
                )
            for row in tasks_rows:
                order_id = _clean_scalar(row["order_id"])
                release_time = _float_or_default(row.get("release_time"), order_release_map.get(order_id, 0.0))
                due_raw = _clean_scalar(row.get("due_date"), None)
                due_date = _float_or_default(due_raw, order_due_map.get(order_id, 0.0)) if due_raw is not None else order_due_map.get(order_id, 0.0)
                conn.execute(
                    "INSERT INTO inst_tasks (task_id, order_id, task_name, is_main, predecessor_task_ids, release_time, due_date) VALUES (?,?,?,?,?,?,?)",
                    (
                        _clean_scalar(row["task_id"]),
                        order_id,
                        _clean_scalar(row.get("task_name"), ""),
                        1 if str(_clean_scalar(row.get("is_main"), "N")).upper() in {"Y", "1", "TRUE"} else 0,
                        _clean_scalar(row.get("predecessor_task_ids"), ""),
                        release_time,
                        due_date,
                    ),
                )
            initial_state_map: dict[str, dict] = {}
            for row in initial_state_rows or []:
                op_id = _clean_scalar(row.get("op_id"), "")
                if op_id:
                    initial_state_map[op_id] = row
            for row in ops_rows:
                op_id = _clean_scalar(row["op_id"])
                state_row = initial_state_map.get(op_id, {})
                conn.execute(
                    """
                    INSERT INTO inst_operations
                    (
                        op_id, task_id, op_name, process_type, processing_time, turnover_time, predecessor_ops, predecessor_tasks,
                        eligible_machine_ids, required_tooling_types, required_personnel_skills, initial_status,
                        initial_start_time, initial_end_time, initial_remaining_processing_time, initial_assigned_machine_id,
                        initial_assigned_tooling_ids, initial_assigned_personnel_ids
                    )
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        op_id,
                        _clean_scalar(row["task_id"]),
                        _clean_scalar(row.get("op_name"), ""),
                        _clean_scalar(row.get("process_type"), ""),
                        _float_or_default(row.get("processing_time_hrs", row.get("processing_time", 0)), 0.0),
                        _float_or_default(row.get("turnover_time_hrs", row.get("turnover_time", 0)), 0.0),
                        _clean_scalar(row.get("predecessor_ops"), ""),
                        _clean_scalar(row.get("predecessor_tasks"), ""),
                        _clean_scalar(row.get("eligible_machine_ids"), ""),
                        _clean_scalar(row.get("required_tooling_types"), ""),
                        _clean_scalar(row.get("required_personnel_skills"), ""),
                        _clean_scalar(state_row.get("initial_status"), ""),
                        _float_or_default(state_row.get("initial_start_time"), 0.0) if _clean_scalar(state_row.get("initial_start_time"), None) is not None else None,
                        _float_or_default(state_row.get("initial_end_time"), 0.0) if _clean_scalar(state_row.get("initial_end_time"), None) is not None else None,
                        _float_or_default(state_row.get("initial_remaining_processing_time"), 0.0) if _clean_scalar(state_row.get("initial_remaining_processing_time"), None) is not None else None,
                        _clean_scalar(state_row.get("initial_assigned_machine_id"), ""),
                        _clean_scalar(state_row.get("initial_assigned_tooling_ids"), ""),
                        _clean_scalar(state_row.get("initial_assigned_personnel_ids"), ""),
                    ),
                )
            for row in machine_type_rows:
                conn.execute("INSERT INTO inst_machine_types (type_id, type_name, is_critical) VALUES (?,?,?)", (_clean_scalar(row["type_id"]), _clean_scalar(row.get("type_name"), ""), 1 if str(_clean_scalar(row.get("is_critical"), "N")).upper() in {"Y", "1", "TRUE"} else 0))
            for row in machine_rows:
                conn.execute("INSERT INTO inst_machines (machine_id, machine_name, type_id, shifts) VALUES (?,?,?,?)", (_clean_scalar(row["machine_id"]), _clean_scalar(row.get("machine_name"), ""), _clean_scalar(row.get("type_id"), ""), _clean_scalar(row.get("shifts"), "")))
            for row in tooling_type_rows or []:
                conn.execute("INSERT INTO inst_tooling_types (type_id, type_name) VALUES (?,?)", (_clean_scalar(row["type_id"]), _clean_scalar(row.get("type_name"), "")))
            for row in tooling_rows or []:
                conn.execute("INSERT INTO inst_toolings (tooling_id, tooling_name, type_id, shifts) VALUES (?,?,?,?)", (_clean_scalar(row["tooling_id"]), _clean_scalar(row.get("tooling_name"), ""), _clean_scalar(row.get("type_id"), ""), _clean_scalar(row.get("shifts"), "")))
            for row in personnel_rows or []:
                conn.execute("INSERT INTO inst_personnel (personnel_id, personnel_name, skills, shifts) VALUES (?,?,?,?)", (_clean_scalar(row["personnel_id"]), _clean_scalar(row.get("personnel_name"), ""), _clean_scalar(row.get("skills"), ""), _clean_scalar(row.get("shifts"), "")))

    def load_all(self) -> dict:
        with get_db(self.db_path) as conn:
            planning_context = conn.execute("SELECT * FROM planning_context WHERE id=1").fetchone()
            return {
                "planning_context": dict(planning_context) if planning_context else {"plan_start_at": isoformat_or_none(default_plan_start())},
                "orders": [dict(row) for row in conn.execute("SELECT * FROM inst_orders").fetchall()],
                "tasks": [dict(row) for row in conn.execute("SELECT * FROM inst_tasks").fetchall()],
                "operations": [dict(row) for row in conn.execute("SELECT * FROM inst_operations").fetchall()],
                "machine_types": [dict(row) for row in conn.execute("SELECT * FROM inst_machine_types").fetchall()],
                "tooling_types": [dict(row) for row in conn.execute("SELECT * FROM inst_tooling_types").fetchall()],
                "machines": [dict(row) for row in conn.execute("SELECT * FROM inst_machines").fetchall()],
                "toolings": [dict(row) for row in conn.execute("SELECT * FROM inst_toolings").fetchall()],
                "personnel": [dict(row) for row in conn.execute("SELECT * FROM inst_personnel").fetchall()],
            }

    def get_plan_start_at(self):
        with get_db(self.db_path) as conn:
            row = conn.execute("SELECT plan_start_at FROM planning_context WHERE id=1").fetchone()
            if row and row["plan_start_at"]:
                return ensure_aware(datetime.fromisoformat(row["plan_start_at"].replace("Z", "+00:00")))
            return default_plan_start()

    def has_data(self) -> bool:
        with get_db(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM inst_orders").fetchone()[0] > 0

    def update_order(self, order_id: str, data: dict):
        plan_start_at = self.get_plan_start_at()
        release_time = datetime_to_offset_hours(plan_start_at, data.get("release_at", data.get("release_time", 0))) or 0.0
        due_date = datetime_to_offset_hours(plan_start_at, data.get("due_at", data.get("due_date", 0))) or 0.0
        with get_db(self.db_path) as conn:
            conn.execute(
                "UPDATE inst_orders SET order_name=?, release_time=?, due_date=?, priority=? WHERE order_id=?",
                (data["order_name"], float(release_time), float(due_date), int(data["priority"]), order_id),
            )
            _bump_instance_version(conn)

    def update_task(self, task_id: str, data: dict):
        plan_start_at = self.get_plan_start_at()
        release_time = datetime_to_offset_hours(plan_start_at, data.get("release_at", data.get("release_time", 0))) or 0.0
        due_date = datetime_to_offset_hours(plan_start_at, data.get("due_at", data.get("due_date", 0))) or 0.0
        with get_db(self.db_path) as conn:
            conn.execute(
                "UPDATE inst_tasks SET order_id=?, task_name=?, is_main=?, predecessor_task_ids=?, release_time=?, due_date=? WHERE task_id=?",
                (data["order_id"], data["task_name"], 1 if data.get("is_main") else 0, data.get("predecessor_task_ids", ""), float(release_time), float(due_date), task_id),
            )
            _bump_instance_version(conn)

    def update_operation(self, op_id: str, data: dict):
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE inst_operations
                SET task_id=?, op_name=?, process_type=?, processing_time=?, turnover_time=?, predecessor_ops=?, predecessor_tasks=?, eligible_machine_ids=?, required_tooling_types=?, required_personnel_skills=?, initial_status=?, initial_start_time=?, initial_end_time=?, initial_remaining_processing_time=?, initial_assigned_machine_id=?, initial_assigned_tooling_ids=?, initial_assigned_personnel_ids=?
                WHERE op_id=?
                """,
                (
                    data["task_id"],
                    data["op_name"],
                    data["process_type"],
                    float(data["processing_time"]),
                    _float_or_default(data.get("turnover_time"), 0.0),
                    data.get("predecessor_ops", ""),
                    data.get("predecessor_tasks", ""),
                    data.get("eligible_machine_ids", ""),
                    data.get("required_tooling_types", ""),
                    data.get("required_personnel_skills", ""),
                    data.get("initial_status", ""),
                    data.get("initial_start_time"),
                    data.get("initial_end_time"),
                    data.get("initial_remaining_processing_time"),
                    data.get("initial_assigned_machine_id", ""),
                    data.get("initial_assigned_tooling_ids", ""),
                    data.get("initial_assigned_personnel_ids", ""),
                    op_id,
                ),
            )
            _bump_instance_version(conn)

    def update_machine(self, machine_id: str, data: dict):
        with get_db(self.db_path) as conn:
            conn.execute("UPDATE inst_machines SET machine_name=?, type_id=?, shifts=? WHERE machine_id=?", (data["machine_name"], data["type_id"], normalize_shifts_field(data.get("shifts", "")), machine_id))
            _bump_instance_version(conn)

    def build_shopfloor(self):
        from ..core.models import Machine, MachineType, Operation, Order, Personnel, ShopFloor, Task, Tooling, ToolingType

        data = self.load_all()
        plan_start_at = ensure_aware(datetime.fromisoformat(data["planning_context"]["plan_start_at"].replace("Z", "+00:00")))
        shop = ShopFloor(plan_start_at=plan_start_at)
        for row in data["machine_types"]:
            shop.machine_types[row["type_id"]] = MachineType(id=row["type_id"], name=row["type_name"], is_critical=bool(row["is_critical"]))
        for row in data["tooling_types"]:
            shop.tooling_types[row["type_id"]] = ToolingType(id=row["type_id"], name=row["type_name"])
        for row in data["machines"]:
            shop.machines[row["machine_id"]] = Machine(id=row["machine_id"], name=row["machine_name"], type_id=row["type_id"], shifts=_parse_shifts(row["shifts"]))
        for row in data["toolings"]:
            shop.toolings[row["tooling_id"]] = Tooling(id=row["tooling_id"], name=row["tooling_name"], type_id=row["type_id"], shifts=_parse_shifts(row["shifts"]))
        for row in data["personnel"]:
            shop.personnel[row["personnel_id"]] = Personnel(id=row["personnel_id"], name=row["personnel_name"], skills=[token.strip() for token in row["skills"].split(";") if token.strip()], shifts=_parse_shifts(row["shifts"]))
        for row in data["orders"]:
            shop.orders[row["order_id"]] = Order(id=row["order_id"], name=row["order_name"], release_time=float(row["release_time"]), due_date=float(row["due_date"]), priority=int(row["priority"]))
        for row in data["tasks"]:
            task = Task(id=row["task_id"], order_id=row["order_id"], name=row["task_name"], is_main=bool(row["is_main"]), predecessor_task_ids=[token.strip() for token in row["predecessor_task_ids"].split(";") if token.strip()], release_time=float(row["release_time"]), due_date=float(row["due_date"]))
            shop.tasks[task.id] = task
            if task.order_id in shop.orders:
                shop.orders[task.order_id].task_ids.append(task.id)
                if task.is_main:
                    shop.orders[task.order_id].main_task_id = task.id
        for row in data["operations"]:
            op = Operation(id=row["op_id"], task_id=row["task_id"], name=row["op_name"], process_type=row["process_type"], processing_time=float(row["processing_time"]), turnover_time=_float_or_default(row.get("turnover_time"), 0.0), predecessor_ops=[token.strip() for token in row["predecessor_ops"].split(";") if token.strip()], predecessor_tasks=[token.strip() for token in row["predecessor_tasks"].split(";") if token.strip()], eligible_machine_ids=[token.strip() for token in row["eligible_machine_ids"].replace(",", ";").replace("，", ";").split(";") if token.strip()], required_tooling_types=[token.strip() for token in row.get("required_tooling_types", "").split(";") if token.strip()], required_personnel_skills=[token.strip() for token in row.get("required_personnel_skills", "").split(";") if token.strip()])
            op._initial_status = _clean_scalar(row.get("initial_status"), "")
            op._initial_start_time = row.get("initial_start_time")
            op._initial_end_time = row.get("initial_end_time")
            op._initial_remaining_processing_time = row.get("initial_remaining_processing_time")
            op._initial_assigned_machine_id = _clean_scalar(row.get("initial_assigned_machine_id"), "")
            op._initial_assigned_tooling_ids = _clean_scalar(row.get("initial_assigned_tooling_ids"), "")
            op._initial_assigned_personnel_ids = _clean_scalar(row.get("initial_assigned_personnel_ids"), "")
            shop.operations[op.id] = op
            if op.task_id in shop.tasks:
                shop.tasks[op.task_id].operations.append(op)
        self._load_downtimes_into_shop(shop)
        shop.build_indexes()
        shop.ensure_calendar_capacity(min_days=max(shop.calendar_days(), 14), safety_factor=1.45, max_days=720)
        _apply_initial_operation_states(shop)
        return shop

    def _load_downtimes_into_shop(self, shop):
        downtimes_by_machine = DowntimeStore(self.db_path).load_all_as_downtimes()
        for machine_id, machine in shop.machines.items():
            machine.downtimes = downtimes_by_machine.get(machine_id, [])


def _apply_initial_operation_states(shop):
    from ..core.models import OpStatus, ResourceState

    for resource in [*shop.machines.values(), *shop.toolings.values(), *shop.personnel.values()]:
        resource.state = ResourceState.IDLE
        resource.current_op_id = None
        resource.current_finish_time = 0.0

    for op in shop.operations.values():
        initial_status = str(getattr(op, "_initial_status", "") or "").strip().lower()
        op.assigned_machine_id = ""
        op.assigned_tooling_ids = []
        op.assigned_personnel_ids = []
        op.start_time = None
        op.end_time = None
        op.remaining_processing_time = None
        if initial_status == "completed":
            op.status = OpStatus.COMPLETED
            op.assigned_machine_id = str(getattr(op, "_initial_assigned_machine_id", "") or "").strip()
            op.assigned_tooling_ids = _split_ids(getattr(op, "_initial_assigned_tooling_ids", ""))
            op.assigned_personnel_ids = _split_ids(getattr(op, "_initial_assigned_personnel_ids", ""))
            op.start_time = _coalesce_number(getattr(op, "_initial_start_time", None), None)
            op.remaining_processing_time = 0.0
            op.end_time = float(_coalesce_number(getattr(op, "_initial_end_time", None), 0.0))
        elif initial_status == "ready":
            op.status = OpStatus.READY
        elif initial_status == "processing":
            op.status = OpStatus.PROCESSING
            op.assigned_machine_id = str(getattr(op, "_initial_assigned_machine_id", "") or "").strip()
            op.assigned_tooling_ids = _split_ids(getattr(op, "_initial_assigned_tooling_ids", ""))
            op.assigned_personnel_ids = _split_ids(getattr(op, "_initial_assigned_personnel_ids", ""))
            start_time = _coalesce_number(getattr(op, "_initial_start_time", None), 0.0)
            end_time = _coalesce_number(getattr(op, "_initial_end_time", None), None)
            remaining = _coalesce_number(getattr(op, "_initial_remaining_processing_time", None), None)
            resources = _operation_assigned_resources(shop, op)
            if resources:
                if remaining is None and end_time is not None and end_time > 0:
                    remaining = _joint_productive_time(resources, 0.0, end_time)
                if remaining is None:
                    remaining = op.processing_time
                op.remaining_processing_time = max(0.001, min(float(op.processing_time), float(remaining)))
                op.start_time = float(start_time or 0.0)
                if end_time is None or end_time <= 0:
                    end_time = _joint_compute_effective_end(resources, 0.0, op.remaining_processing_time)
                op.end_time = float(end_time)
                for resource in resources:
                    resource.state = ResourceState.BUSY
                    resource.current_op_id = op.id
                    resource.current_finish_time = op.end_time
            else:
                op.status = OpStatus.READY
                op.assigned_machine_id = ""
                op.assigned_tooling_ids = []
                op.assigned_personnel_ids = []
                op.start_time = None
                op.end_time = None
                op.remaining_processing_time = None
        else:
            op.status = OpStatus.PENDING

    for task in shop.tasks.values():
        if task.operations and all(op.status == OpStatus.COMPLETED for op in task.operations):
            task.completion_time = max((op.end_time or 0.0) for op in task.operations)
        else:
            task.completion_time = None


def _shifts_to_str(shifts) -> str:
    return ";".join(f"{shift.day}/{shift.start_hour}/{shift.hours}" for shift in shifts)


def _parse_shifts(raw: str):
    from ..core.models import Shift

    shifts = []
    if not raw:
        return shifts
    for segment in raw.split(";"):
        parts = segment.strip().split("/")
        if len(parts) == 3:
            shifts.append(Shift(day=int(float(parts[0])), start_hour=float(parts[1]), hours=float(parts[2])))
    return shifts


def shifts_to_payload(raw) -> list[dict]:
    """把存储的 "day/start/hours;..." 字符串解析为结构化数组，供前端展示 / 编辑。

    数据库里班次是紧凑字符串（如 "0/0/1;0/3/9.5;..."），但前端"机器维护"页把该字段
    当作 JSON 数组渲染（asArray + JSON.stringify），字符串会被 asArray 视为非数组而显示成
    []。这里统一转成 [{day, start_hour, hours}, ...]，让展示与编辑拿到真实数据。
    """
    return [
        {"day": shift.day, "start_hour": shift.start_hour, "hours": shift.hours}
        for shift in _parse_shifts(raw if isinstance(raw, str) else "")
    ]


def normalize_shifts_field(value) -> str:
    """把前端回传的班次（结构化数组或已是规范字符串）统一转成存储用的 "day/start/hours;..." 字符串。

    前端保存机器时会 JSON.parse 文本框内容并回传数组；若直接写入 TEXT 列会得到无法被
    _parse_shifts 识别的内容（甚至绑定报错）。这里做归一化，保证与导入格式一致。
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        segments = []
        for item in value:
            if not isinstance(item, dict):
                continue
            day = item.get("day")
            start_hour = item.get("start_hour", item.get("start"))
            hours = item.get("hours")
            if day is None or start_hour is None or hours is None:
                continue
            try:
                segments.append(f"{int(float(day))}/{float(start_hour)}/{float(hours)}")
            except (TypeError, ValueError):
                continue
        return ";".join(segments)
    return ""


def _split_ids(raw: str) -> list[str]:
    return [token.strip() for token in str(raw or "").split(";") if token and token.strip()]


def _coalesce_number(value, default):
    cleaned = _clean_scalar(value, None)
    if cleaned is None:
        return default
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return default


def _operation_assigned_resources(shop, op):
    machine = shop.machines.get(op.assigned_machine_id) if op.assigned_machine_id else None
    if machine is None:
        return []
    toolings = [shop.toolings[tooling_id] for tooling_id in op.assigned_tooling_ids if tooling_id in shop.toolings]
    people = [shop.personnel[person_id] for person_id in op.assigned_personnel_ids if person_id in shop.personnel]
    return [machine, *toolings, *people]


def _joint_next_available_time(resources: list, not_before: float) -> float:
    probe = not_before
    for _ in range(1000):
        shifted = False
        for resource in resources:
            ready = resource.next_available_time(probe)
            if ready == float("inf"):
                return ready
            if ready > probe + 1e-9:
                probe = ready
                shifted = True
        if not shifted:
            return probe
    return float("inf")


def _joint_next_unavailable_time(resources: list, at_time: float) -> float:
    return min(resource.next_unavailable_time(at_time) for resource in resources)


def _joint_compute_effective_end(resources: list, start: float, duration: float) -> float:
    when = _joint_next_available_time(resources, start)
    remaining = duration
    for _ in range(10000):
        if when == float("inf"):
            return when
        if remaining <= 1e-9:
            return when
        unavailable = _joint_next_unavailable_time(resources, when)
        if unavailable == float("inf"):
            return when + remaining
        workable = max(0.0, unavailable - when)
        if workable >= remaining - 1e-9:
            return when + remaining
        remaining -= workable
        when = _joint_next_available_time(resources, unavailable)
    return when


def _joint_productive_time(resources: list, start: float, end: float) -> float:
    when = _joint_next_available_time(resources, start)
    total = 0.0
    for _ in range(10000):
        if when == float("inf") or when >= end - 1e-9:
            return total
        unavailable = _joint_next_unavailable_time(resources, when)
        productive_end = min(unavailable, end)
        if productive_end > when:
            total += productive_end - when
        when = _joint_next_available_time(resources, productive_end)
    return total


class GraphStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def clear_all(self):
        with get_db(self.db_path) as conn:
            conn.execute("DELETE FROM graph_nodes")
            conn.execute("DELETE FROM graph_edges")
            conn.execute("DELETE FROM graph_meta")

    def save_graph(self, graph_wrapper, progress_callback=None, deadline: float | None = None, batch_size: int = 1000):
        graph = graph_wrapper.graph
        total_nodes = graph.number_of_nodes()
        total_edges = graph.number_of_edges()
        total_rows = max(1, total_nodes + total_edges)
        written = 0

        def ensure_time():
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("图谱数据库保存超过时间限制")

        with get_db(self.db_path) as conn:
            conn.execute("DELETE FROM graph_nodes")
            conn.execute("DELETE FROM graph_edges")
            conn.execute("DELETE FROM graph_meta")

            node_batch = []
            for node_id, attrs in graph.nodes(data=True):
                node_batch.append((node_id, attrs.get("node_type", ""), attrs.get("entity_id", ""), json.dumps({key: value for key, value in attrs.items() if key not in {"node_type", "entity_id"}}, ensure_ascii=False)))
                if len(node_batch) >= batch_size:
                    ensure_time()
                    conn.executemany("INSERT INTO graph_nodes VALUES (?,?,?,?)", node_batch)
                    written += len(node_batch)
                    node_batch.clear()
                    if progress_callback:
                        progress_callback(written, total_rows)
            if node_batch:
                ensure_time()
                conn.executemany("INSERT INTO graph_nodes VALUES (?,?,?,?)", node_batch)
                written += len(node_batch)
                if progress_callback:
                    progress_callback(written, total_rows)

            edge_batch = []
            for source, target, attrs in graph.edges(data=True):
                edge_batch.append((source, target, attrs.get("edge_type", ""), json.dumps({key: value for key, value in attrs.items() if key != "edge_type"}, ensure_ascii=False)))
                if len(edge_batch) >= batch_size:
                    ensure_time()
                    conn.executemany("INSERT INTO graph_edges (source, target, edge_type, attrs) VALUES (?,?,?,?)", edge_batch)
                    written += len(edge_batch)
                    edge_batch.clear()
                    if progress_callback:
                        progress_callback(written, total_rows)
            if edge_batch:
                ensure_time()
                conn.executemany("INSERT INTO graph_edges (source, target, edge_type, attrs) VALUES (?,?,?,?)", edge_batch)
                written += len(edge_batch)
                if progress_callback:
                    progress_callback(written, total_rows)

            stats = graph_wrapper.get_graph_stats()
            conn.execute(
                """
                INSERT OR REPLACE INTO graph_meta
                (id, total_nodes, total_edges, node_type_counts, edge_type_counts, created_at)
                VALUES (1,?,?,?,?,?)
                """,
                (
                    stats["total_nodes"],
                    stats["total_edges"],
                    json.dumps(stats["node_types"]),
                    json.dumps(stats["edge_types"]),
                    time.time(),
                ),
            )
            conn.execute(
                """
                UPDATE graph_context_meta
                SET status='invalid', invalid_reason='legacy_graph_saved'
                WHERE id=1
                """
            )

    def load_nodes(self, node_type: str = None, search: str = None, limit: int = 200, offset: int = 0) -> tuple[int, list[dict]]:
        with get_db(self.db_path) as conn:
            where = " WHERE 1=1"
            params = []
            if node_type:
                where += " AND node_type=?"
                params.append(node_type)
            if search:
                where += " AND (node_id LIKE ? OR entity_id LIKE ?)"
                params.extend([f"%{search}%", f"%{search}%"])
            total = conn.execute("SELECT COUNT(*) FROM graph_nodes" + where, params).fetchone()[0]
            rows = conn.execute("SELECT * FROM graph_nodes" + where + " ORDER BY rowid LIMIT ? OFFSET ?", [*params, max(1, limit), max(0, offset)]).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["attrs"] = json.loads(item["attrs"]) if item["attrs"] else {}
                result.append(item)
            return total, result

    def load_edges(self, edge_type: str = None, search: str = None, limit: int = 200, offset: int = 0) -> tuple[int, list[dict]]:
        with get_db(self.db_path) as conn:
            where = " WHERE 1=1"
            params = []
            if edge_type:
                where += " AND edge_type=?"
                params.append(edge_type)
            if search:
                where += " AND (source LIKE ? OR target LIKE ?)"
                params.extend([f"%{search}%", f"%{search}%"])
            total = conn.execute("SELECT COUNT(*) FROM graph_edges" + where, params).fetchone()[0]
            rows = conn.execute("SELECT * FROM graph_edges" + where + " ORDER BY id LIMIT ? OFFSET ?", [*params, max(1, limit), max(0, offset)]).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["attrs"] = json.loads(item["attrs"]) if item["attrs"] else {}
                result.append(item)
            return total, result

    def load_meta(self) -> Optional[dict]:
        with get_db(self.db_path) as conn:
            row = conn.execute("SELECT * FROM graph_meta WHERE id=1").fetchone()
            if not row:
                return None
            result = dict(row)
            result["node_type_counts"] = json.loads(result["node_type_counts"])
            result["edge_type_counts"] = json.loads(result["edge_type_counts"])
            return result

    def has_data(self) -> bool:
        with get_db(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0] > 0

    def get_node_neighbors(self, node_id: str) -> dict:
        with get_db(self.db_path) as conn:
            outgoing = [dict(row) for row in conn.execute("SELECT * FROM graph_edges WHERE source=?", (node_id,)).fetchall()]
            incoming = [dict(row) for row in conn.execute("SELECT * FROM graph_edges WHERE target=?", (node_id,)).fetchall()]
            for edge in outgoing + incoming:
                edge["attrs"] = json.loads(edge["attrs"]) if edge["attrs"] else {}
            return {"outgoing": outgoing, "incoming": incoming}

    def _build_order_subgraph(self, conn, order_node_id: str) -> dict:
        """根据订单节点 ID，拉取其任务/工序/资源与相关边；OS_ 开头的机器在 SQL 层直接过滤。"""
        task_rows = conn.execute(
            "SELECT target FROM graph_edges WHERE source=? AND edge_type='order_has_task'",
            (order_node_id,),
        ).fetchall()
        task_ids = [row["target"] for row in task_rows]

        operation_ids = []
        if task_ids:
            placeholders = ",".join("?" for _ in task_ids)
            operation_ids = [
                row["target"]
                for row in conn.execute(
                    f"SELECT target FROM graph_edges WHERE source IN ({placeholders}) AND edge_type='task_has_operation'",
                    task_ids,
                ).fetchall()
            ]

        resource_ids = []
        if operation_ids:
            placeholders = ",".join("?" for _ in operation_ids)
            # 机器：在 SQL 层排除 OS_ 开头的机器（按 entity_id 与 node_id 双判），其余资源不过滤
            machine_rows = conn.execute(
                f"""
                SELECT DISTINCT e.target FROM graph_edges e
                WHERE e.source IN ({placeholders})
                  AND e.edge_type = 'machine_eligible'
                  AND NOT EXISTS (
                    SELECT 1 FROM graph_nodes n
                    WHERE n.node_id = e.target
                      AND n.node_type = 'machine'
                      AND (n.entity_id LIKE 'OS_%' OR n.node_id LIKE 'OS_%')
                  )
                """,
                operation_ids,
            ).fetchall()
            resource_ids += [row["target"] for row in machine_rows]
            other_rows = conn.execute(
                f"""
                SELECT DISTINCT target FROM graph_edges
                WHERE source IN ({placeholders})
                  AND edge_type IN ('tooling_eligible','personnel_eligible')
                """,
                operation_ids,
            ).fetchall()
            resource_ids += [row["target"] for row in other_rows]

        node_ids = list(dict.fromkeys([order_node_id, *task_ids, *operation_ids, *resource_ids]))
        placeholders = ",".join("?" for _ in node_ids)
        nodes = [
            dict(row)
            for row in conn.execute(
                f"SELECT * FROM graph_nodes WHERE node_id IN ({placeholders}) ORDER BY rowid",
                node_ids,
            ).fetchall()
        ]
        edges = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT * FROM graph_edges
                WHERE source IN ({placeholders}) AND target IN ({placeholders})
                ORDER BY id
                """,
                [*node_ids, *node_ids],
            ).fetchall()
        ]
        for item in nodes + edges:
            item["attrs"] = json.loads(item["attrs"]) if item["attrs"] else {}
        return {"order_id": order_node_id, "nodes": nodes, "edges": edges}

    def _find_order_row(self, conn, query: str):
        """按订单号精确/模糊解析订单节点（精确 entity_id/node_id 优先，再 LIKE 模糊匹配）。"""
        q = (query or "").strip()
        if not q:
            return None
        exact = conn.execute(
            "SELECT * FROM graph_nodes WHERE node_type='order' AND (entity_id=? OR node_id=? OR node_id=?) LIMIT 1",
            (q, q, f"O:{q}"),
        ).fetchone()
        if exact:
            return exact
        like = f"%{q}%"
        return conn.execute(
            "SELECT * FROM graph_nodes WHERE node_type='order' AND (entity_id LIKE ? OR node_id LIKE ? OR attrs LIKE ?) ORDER BY entity_id LIMIT 1",
            (like, like, like),
        ).fetchone()

    def load_order_subgraph(self, order_id: str) -> dict:
        """Load one order, its tasks/operations, and resources used by those operations."""
        with get_db(self.db_path) as conn:
            order_row = conn.execute(
                """
                SELECT * FROM graph_nodes
                WHERE node_type='order' AND (node_id=? OR entity_id=? OR node_id=? )
                ORDER BY CASE WHEN node_id=? THEN 0 ELSE 1 END
                LIMIT 1
                """,
                (order_id, order_id, f"O:{order_id}", order_id),
            ).fetchone()
            if not order_row:
                return {"order_id": None, "nodes": [], "edges": []}
            return self._build_order_subgraph(conn, order_row["node_id"])

    def search_order_subgraph(self, query: str) -> dict:
        """按订单号/名称模糊解析订单并返回其子图（OS_ 机器在 SQL 层过滤）。无匹配返回空子图。"""
        with get_db(self.db_path) as conn:
            order_row = self._find_order_row(conn, query)
            if not order_row:
                return {"order_id": None, "nodes": [], "edges": []}
            return self._build_order_subgraph(conn, order_row["node_id"])
