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
                created_at REAL DEFAULT (strftime('%s','now'))
            );
            CREATE TABLE IF NOT EXISTS machine_downtime (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id TEXT NOT NULL,
                downtime_type TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_rules_type ON rules(problem_type, objective);
            CREATE INDEX IF NOT EXISTS idx_rules_active ON rules(is_active);
            CREATE INDEX IF NOT EXISTS idx_results_rule ON schedule_results(rule_id);
            CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes(node_type);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_type ON graph_edges(edge_type);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_src ON graph_edges(source);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_tgt ON graph_edges(target);
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
        conn.execute(
            "INSERT OR IGNORE INTO planning_context (id, plan_start_at) VALUES (1, ?)",
            (isoformat_or_none(default_plan_start()),),
        )
    logger.info("Database initialized: %s", db_path)


class DowntimeStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def clear_all(self):
        with get_db(self.db_path) as conn:
            conn.execute("DELETE FROM machine_downtime")

    def save(self, machine_id: str, downtime_type: str, start_time: float, end_time: float) -> int:
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO machine_downtime (machine_id, downtime_type, start_time, end_time) VALUES (?,?,?,?)",
                (machine_id, downtime_type, float(start_time), float(end_time)),
            )
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

    def update(self, downtime_id: int, data: dict):
        with get_db(self.db_path) as conn:
            conn.execute(
                "UPDATE machine_downtime SET machine_id=?, downtime_type=?, start_time=?, end_time=? WHERE id=?",
                (data["machine_id"], data["downtime_type"], float(data["start_time"]), float(data["end_time"]), downtime_id),
            )

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

    def save_from_shopfloor(self, shop):
        self.clear_all()
        with get_db(self.db_path) as conn:
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
                        op_id, task_id, op_name, process_type, processing_time, predecessor_ops, predecessor_tasks,
                        eligible_machine_ids, required_tooling_types, required_personnel_skills, initial_status,
                        initial_start_time, initial_end_time, initial_remaining_processing_time, initial_assigned_machine_id,
                        initial_assigned_tooling_ids, initial_assigned_personnel_ids
                    )
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        op_id,
                        op.task_id,
                        op.name,
                        op.process_type,
                        op.processing_time,
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
                        op_id, task_id, op_name, process_type, processing_time, predecessor_ops, predecessor_tasks,
                        eligible_machine_ids, required_tooling_types, required_personnel_skills, initial_status,
                        initial_start_time, initial_end_time, initial_remaining_processing_time, initial_assigned_machine_id,
                        initial_assigned_tooling_ids, initial_assigned_personnel_ids
                    )
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        op_id,
                        _clean_scalar(row["task_id"]),
                        _clean_scalar(row.get("op_name"), ""),
                        _clean_scalar(row.get("process_type"), ""),
                        _float_or_default(row.get("processing_time_hrs", row.get("processing_time", 0)), 0.0),
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

    def update_task(self, task_id: str, data: dict):
        plan_start_at = self.get_plan_start_at()
        release_time = datetime_to_offset_hours(plan_start_at, data.get("release_at", data.get("release_time", 0))) or 0.0
        due_date = datetime_to_offset_hours(plan_start_at, data.get("due_at", data.get("due_date", 0))) or 0.0
        with get_db(self.db_path) as conn:
            conn.execute(
                "UPDATE inst_tasks SET order_id=?, task_name=?, is_main=?, predecessor_task_ids=?, release_time=?, due_date=? WHERE task_id=?",
                (data["order_id"], data["task_name"], 1 if data.get("is_main") else 0, data.get("predecessor_task_ids", ""), float(release_time), float(due_date), task_id),
            )

    def update_operation(self, op_id: str, data: dict):
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE inst_operations
                SET task_id=?, op_name=?, process_type=?, processing_time=?, predecessor_ops=?, predecessor_tasks=?, eligible_machine_ids=?, required_tooling_types=?, required_personnel_skills=?, initial_status=?, initial_start_time=?, initial_end_time=?, initial_remaining_processing_time=?, initial_assigned_machine_id=?, initial_assigned_tooling_ids=?, initial_assigned_personnel_ids=?
                WHERE op_id=?
                """,
                (
                    data["task_id"],
                    data["op_name"],
                    data["process_type"],
                    float(data["processing_time"]),
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

    def update_machine(self, machine_id: str, data: dict):
        with get_db(self.db_path) as conn:
            conn.execute("UPDATE inst_machines SET machine_name=?, type_id=?, shifts=? WHERE machine_id=?", (data["machine_name"], data["type_id"], data.get("shifts", ""), machine_id))

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
            op = Operation(id=row["op_id"], task_id=row["task_id"], name=row["op_name"], process_type=row["process_type"], processing_time=float(row["processing_time"]), predecessor_ops=[token.strip() for token in row["predecessor_ops"].split(";") if token.strip()], predecessor_tasks=[token.strip() for token in row["predecessor_tasks"].split(";") if token.strip()], eligible_machine_ids=[token.strip() for token in row["eligible_machine_ids"].split(";") if token.strip()], required_tooling_types=[token.strip() for token in row.get("required_tooling_types", "").split(";") if token.strip()], required_personnel_skills=[token.strip() for token in row.get("required_personnel_skills", "").split(";") if token.strip()])
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

    def save_graph(self, graph_wrapper):
        self.clear_all()
        graph = graph_wrapper.graph
        with get_db(self.db_path) as conn:
            for node_id, attrs in graph.nodes(data=True):
                conn.execute("INSERT INTO graph_nodes VALUES (?,?,?,?)", (node_id, attrs.get("node_type", ""), attrs.get("entity_id", ""), json.dumps({key: value for key, value in attrs.items() if key not in {"node_type", "entity_id"}}, ensure_ascii=False)))
            for source, target, attrs in graph.edges(data=True):
                conn.execute("INSERT INTO graph_edges (source, target, edge_type, attrs) VALUES (?,?,?,?)", (source, target, attrs.get("edge_type", ""), json.dumps({key: value for key, value in attrs.items() if key != "edge_type"}, ensure_ascii=False)))
            stats = graph_wrapper.get_graph_stats()
            conn.execute("INSERT OR REPLACE INTO graph_meta VALUES (1,?,?,?,?,?)", (stats["total_nodes"], stats["total_edges"], json.dumps(stats["node_types"]), json.dumps(stats["edge_types"]), time.time()))

    def load_nodes(self, node_type: str = None, search: str = None) -> list[dict]:
        with get_db(self.db_path) as conn:
            sql = "SELECT * FROM graph_nodes WHERE 1=1"
            params = []
            if node_type:
                sql += " AND node_type=?"
                params.append(node_type)
            if search:
                sql += " AND (node_id LIKE ? OR entity_id LIKE ?)"
                params.extend([f"%{search}%", f"%{search}%"])
            rows = conn.execute(sql, params).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["attrs"] = json.loads(item["attrs"]) if item["attrs"] else {}
                result.append(item)
            return result

    def load_edges(self, edge_type: str = None, search: str = None) -> list[dict]:
        with get_db(self.db_path) as conn:
            sql = "SELECT * FROM graph_edges WHERE 1=1"
            params = []
            if edge_type:
                sql += " AND edge_type=?"
                params.append(edge_type)
            if search:
                sql += " AND (source LIKE ? OR target LIKE ?)"
                params.extend([f"%{search}%", f"%{search}%"])
            rows = conn.execute(sql, params).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["attrs"] = json.loads(item["attrs"]) if item["attrs"] else {}
                result.append(item)
            return result

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
