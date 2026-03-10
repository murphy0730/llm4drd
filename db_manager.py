"""
数据库管理 - SQLite 简易版本
============================
存储: 规则库、订单数据、调度结果、进化历史
"""
import sqlite3
import json
import time
import os
import logging
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("LLM4DRD_DB", "llm4drd.db")


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


def init_db(db_path: str = DB_PATH):
    """初始化数据库表"""
    with get_db(db_path) as conn:
        conn.executescript("""
        -- 规则库
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

        -- 进化历史
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

        -- 调度结果
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

        -- 重排历史
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

        -- 订单表
        CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY,
            products TEXT,
            release_time REAL,
            due_date REAL,
            priority INTEGER DEFAULT 1,
            status TEXT DEFAULT 'pending',
            created_at REAL DEFAULT (strftime('%s','now'))
        );

        -- 实例-订单表
        CREATE TABLE IF NOT EXISTS inst_orders (
            order_id TEXT PRIMARY KEY,
            order_name TEXT,
            release_time REAL DEFAULT 0,
            due_date REAL,
            priority INTEGER DEFAULT 1
        );

        -- 实例-任务表
        CREATE TABLE IF NOT EXISTS inst_tasks (
            task_id TEXT PRIMARY KEY,
            order_id TEXT,
            task_name TEXT,
            is_main INTEGER DEFAULT 0,
            predecessor_task_ids TEXT DEFAULT '',
            release_time REAL DEFAULT 0,
            due_date REAL
        );

        -- 实例-工序表
        CREATE TABLE IF NOT EXISTS inst_operations (
            op_id TEXT PRIMARY KEY,
            task_id TEXT,
            op_name TEXT,
            process_type TEXT,
            processing_time REAL,
            predecessor_ops TEXT DEFAULT '',
            predecessor_tasks TEXT DEFAULT '',
            eligible_machine_ids TEXT DEFAULT ''
        );

        -- 实例-机器类别表
        CREATE TABLE IF NOT EXISTS inst_machine_types (
            type_id TEXT PRIMARY KEY,
            type_name TEXT,
            is_critical INTEGER DEFAULT 0
        );

        -- 实例-机器表
        CREATE TABLE IF NOT EXISTS inst_machines (
            machine_id TEXT PRIMARY KEY,
            machine_name TEXT,
            type_id TEXT,
            shifts TEXT DEFAULT ''
        );

        -- 异构图-节点表
        CREATE TABLE IF NOT EXISTS graph_nodes (
            node_id TEXT PRIMARY KEY,
            node_type TEXT,
            entity_id TEXT,
            attrs TEXT DEFAULT '{}'
        );

        -- 异构图-边表
        CREATE TABLE IF NOT EXISTS graph_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            target TEXT,
            edge_type TEXT,
            attrs TEXT DEFAULT '{}'
        );

        -- 异构图-统计
        CREATE TABLE IF NOT EXISTS graph_meta (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            total_nodes INTEGER DEFAULT 0,
            total_edges INTEGER DEFAULT 0,
            node_type_counts TEXT DEFAULT '{}',
            edge_type_counts TEXT DEFAULT '{}',
            created_at REAL DEFAULT (strftime('%s','now'))
        );

        CREATE INDEX IF NOT EXISTS idx_rules_type ON rules(problem_type, objective);
        CREATE INDEX IF NOT EXISTS idx_rules_active ON rules(is_active);
        CREATE INDEX IF NOT EXISTS idx_results_rule ON schedule_results(rule_id);
        CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes(node_type);
        CREATE INDEX IF NOT EXISTS idx_graph_edges_type ON graph_edges(edge_type);
        CREATE INDEX IF NOT EXISTS idx_graph_edges_src ON graph_edges(source);
        CREATE INDEX IF NOT EXISTS idx_graph_edges_tgt ON graph_edges(target);

        CREATE TABLE IF NOT EXISTS machine_downtime (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id TEXT NOT NULL,
            downtime_type TEXT NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL
        );
        """)
    logger.info(f"Database initialized: {db_path}")


class DowntimeStore:
    """机器停机时间管理"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def save(self, machine_id: str, downtime_type: str, start_time: float, end_time: float) -> int:
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO machine_downtime (machine_id, downtime_type, start_time, end_time) VALUES (?,?,?,?)",
                (machine_id, downtime_type, start_time, end_time)
            )
            return cur.lastrowid

    def list_all(self) -> list:
        with get_db(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM machine_downtime ORDER BY machine_id, start_time").fetchall()
            return [dict(r) for r in rows]

    def delete(self, id: int):
        with get_db(self.db_path) as conn:
            conn.execute("DELETE FROM machine_downtime WHERE id=?", (id,))

    def update(self, id: int, data: dict):
        with get_db(self.db_path) as conn:
            conn.execute(
                "UPDATE machine_downtime SET machine_id=?, downtime_type=?, start_time=?, end_time=? WHERE id=?",
                (data["machine_id"], data["downtime_type"], float(data["start_time"]), float(data["end_time"]), id)
            )

    def load_for_machine(self, machine_id: str) -> list:
        with get_db(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM machine_downtime WHERE machine_id=? ORDER BY start_time",
                (machine_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def load_all_as_downtimes(self) -> dict:
        """Returns dict[machine_id -> list[Downtime]]"""
        from .models import Downtime
        result = {}
        for row in self.list_all():
            mid = row["machine_id"]
            dt = Downtime(
                id=str(row["id"]),
                machine_id=mid,
                downtime_type=row["downtime_type"],
                start_time=row["start_time"],
                end_time=row["end_time"]
            )
            result.setdefault(mid, []).append(dt)
        return result


class RuleStore:
    """规则库管理"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def save_rule(self, rule_id: str, name: str, code: str,
                  problem_type: str = "FAFSP",
                  objective: str = "total_tardiness",
                  fitness: float = None,
                  hybrid_score: float = None,
                  llm_score: float = None,
                  generation: int = 0,
                  is_builtin: bool = False,
                  metadata: dict = None):
        """保存规则到库"""
        with get_db(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO rules 
                (id, name, code, problem_type, objective, fitness, 
                 hybrid_score, llm_score, generation, is_builtin, 
                 updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rule_id, name, code, problem_type, objective,
                fitness, hybrid_score, llm_score, generation,
                1 if is_builtin else 0,
                time.time(),
                json.dumps(metadata or {}),
            ))

    def get_rule(self, rule_id: str) -> Optional[dict]:
        """获取单条规则"""
        with get_db(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM rules WHERE id = ?", (rule_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_best_rules(self, problem_type: str = "FAFSP",
                       objective: str = "total_tardiness",
                       limit: int = 5) -> list[dict]:
        """获取某类问题的最优规则"""
        with get_db(self.db_path) as conn:
            rows = conn.execute("""
                SELECT * FROM rules 
                WHERE problem_type = ? AND objective = ? AND is_active = 1
                ORDER BY COALESCE(hybrid_score, fitness, 999999) ASC
                LIMIT ?
            """, (problem_type, objective, limit)).fetchall()
            return [dict(r) for r in rows]

    def get_all_rules(self, active_only: bool = True) -> list[dict]:
        """获取所有规则"""
        with get_db(self.db_path) as conn:
            if active_only:
                rows = conn.execute(
                    "SELECT * FROM rules WHERE is_active = 1 ORDER BY updated_at DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM rules ORDER BY updated_at DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def deactivate_rule(self, rule_id: str):
        """停用规则"""
        with get_db(self.db_path) as conn:
            conn.execute(
                "UPDATE rules SET is_active = 0, updated_at = ? WHERE id = ?",
                (time.time(), rule_id),
            )

    def save_evolution_run(self, problem_type: str, objective: str,
                           config: dict, best_rule_id: str,
                           best_fitness: float, total_generations: int,
                           generation_history: list) -> int:
        """保存进化运行记录"""
        with get_db(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO evolution_runs 
                (problem_type, objective, config, best_rule_id, best_fitness,
                 total_generations, generation_history, started_at, completed_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                problem_type, objective, json.dumps(config),
                best_rule_id, best_fitness, total_generations,
                json.dumps(generation_history),
                generation_history[0].get("time", 0) if generation_history else 0,
                time.time(), "completed",
            ))
            return cursor.lastrowid

    def save_schedule_result(self, rule_id: str, instance_id: str,
                             result_dict: dict):
        """保存调度结果"""
        with get_db(self.db_path) as conn:
            conn.execute("""
                INSERT INTO schedule_results 
                (rule_id, instance_id, total_tardiness, makespan,
                 avg_utilization, avg_flowtime, tardy_count, total_jobs)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rule_id, instance_id,
                result_dict.get("total_tardiness", 0),
                result_dict.get("makespan", 0),
                result_dict.get("avg_utilization", 0),
                result_dict.get("avg_flowtime", 0),
                result_dict.get("tardy_job_count", 0),
                result_dict.get("total_jobs", 0),
            ))

    def save_reschedule_record(self, record: dict):
        """保存重排记录"""
        with get_db(self.db_path) as conn:
            conn.execute("""
                INSERT INTO reschedule_history
                (trigger_reason, old_fitness, new_fitness, improvement,
                 changed_count, total_count, computation_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get("trigger_reason", ""),
                record.get("old_fitness", 0),
                record.get("new_fitness", 0),
                record.get("improvement_pct", 0),
                record.get("changed_assignments", 0),
                record.get("total_assignments", 0),
                record.get("computation_time_s", 0),
            ))

    def get_performance_trend(self, rule_id: str, limit: int = 50) -> list[dict]:
        """获取规则性能趋势"""
        with get_db(self.db_path) as conn:
            rows = conn.execute("""
                SELECT total_tardiness, makespan, avg_utilization, created_at
                FROM schedule_results
                WHERE rule_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (rule_id, limit)).fetchall()
            return [dict(r) for r in rows]


class InstanceStore:
    """实例数据持久化"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def clear_all(self):
        """清空所有实例数据"""
        with get_db(self.db_path) as conn:
            conn.execute("DELETE FROM inst_orders")
            conn.execute("DELETE FROM inst_tasks")
            conn.execute("DELETE FROM inst_operations")
            conn.execute("DELETE FROM inst_machine_types")
            conn.execute("DELETE FROM inst_machines")

    def save_from_shopfloor(self, shop):
        """从 ShopFloor 对象保存到数据库(覆盖)"""
        self.clear_all()
        with get_db(self.db_path) as conn:
            for oid, o in shop.orders.items():
                conn.execute(
                    "INSERT INTO inst_orders VALUES (?,?,?,?,?)",
                    (oid, o.name, o.release_time, o.due_date, o.priority),
                )
            for tid, t in shop.tasks.items():
                conn.execute(
                    "INSERT INTO inst_tasks VALUES (?,?,?,?,?,?,?)",
                    (tid, t.order_id, t.name, 1 if t.is_main else 0,
                     ";".join(t.predecessor_task_ids), t.release_time, t.due_date),
                )
            for opid, op in shop.operations.items():
                conn.execute(
                    "INSERT INTO inst_operations VALUES (?,?,?,?,?,?,?,?)",
                    (opid, op.task_id, op.name, op.process_type, op.processing_time,
                     ";".join(op.predecessor_ops), ";".join(op.predecessor_tasks),
                     ";".join(op.eligible_machine_ids)),
                )
            for mtid, mt in shop.machine_types.items():
                conn.execute(
                    "INSERT INTO inst_machine_types VALUES (?,?,?)",
                    (mtid, mt.name, 1 if mt.is_critical else 0),
                )
            for mid, m in shop.machines.items():
                shifts_str = ";".join(
                    f"{s.day}/{s.start_hour}/{s.hours}" for s in m.shifts
                )
                conn.execute(
                    "INSERT INTO inst_machines VALUES (?,?,?,?)",
                    (mid, m.name, m.type_id, shifts_str),
                )

    def save_from_csv(self, orders_rows, tasks_rows, ops_rows,
                      mt_rows, machines_rows):
        """从 CSV 数据保存到数据库(覆盖)"""
        self.clear_all()
        with get_db(self.db_path) as conn:
            for r in orders_rows:
                conn.execute(
                    "INSERT INTO inst_orders VALUES (?,?,?,?,?)",
                    (r["order_id"], r.get("order_name", ""),
                     float(r.get("release_time", 0)),
                     float(r.get("due_date", 0)),
                     int(r.get("priority", 1))),
                )
            for r in tasks_rows:
                conn.execute(
                    "INSERT INTO inst_tasks VALUES (?,?,?,?,?,?,?)",
                    (r["task_id"], r["order_id"], r.get("task_name", ""),
                     1 if r.get("is_main", "N").upper() in ("Y", "1", "TRUE") else 0,
                     r.get("predecessor_task_ids", ""),
                     float(r.get("release_time", 0)),
                     float(r.get("due_date", 0))),
                )
            for r in ops_rows:
                conn.execute(
                    "INSERT INTO inst_operations VALUES (?,?,?,?,?,?,?,?)",
                    (r["op_id"], r["task_id"], r.get("op_name", ""),
                     r.get("process_type", ""),
                     float(r.get("processing_time_hrs", 0)),
                     r.get("predecessor_ops", ""),
                     r.get("predecessor_tasks", ""),
                     r.get("eligible_machine_ids", "")),
                )
            for r in mt_rows:
                conn.execute(
                    "INSERT INTO inst_machine_types VALUES (?,?,?)",
                    (r["type_id"], r.get("type_name", ""),
                     1 if r.get("is_critical", "N").upper() in ("Y", "1", "TRUE") else 0),
                )
            for r in machines_rows:
                conn.execute(
                    "INSERT INTO inst_machines VALUES (?,?,?,?)",
                    (r["machine_id"], r.get("machine_name", ""),
                     r.get("type_id", ""), r.get("shifts", "")),
                )

    def load_all(self) -> dict:
        """从数据库加载所有实例数据"""
        with get_db(self.db_path) as conn:
            orders = [dict(r) for r in conn.execute("SELECT * FROM inst_orders").fetchall()]
            tasks = [dict(r) for r in conn.execute("SELECT * FROM inst_tasks").fetchall()]
            operations = [dict(r) for r in conn.execute("SELECT * FROM inst_operations").fetchall()]
            machine_types = [dict(r) for r in conn.execute("SELECT * FROM inst_machine_types").fetchall()]
            machines = [dict(r) for r in conn.execute("SELECT * FROM inst_machines").fetchall()]
        return {
            "orders": orders, "tasks": tasks, "operations": operations,
            "machine_types": machine_types, "machines": machines,
        }

    def has_data(self) -> bool:
        with get_db(self.db_path) as conn:
            cnt = conn.execute("SELECT COUNT(*) FROM inst_orders").fetchone()[0]
            return cnt > 0

    def update_order(self, order_id: str, data: dict):
        with get_db(self.db_path) as conn:
            conn.execute("""
                UPDATE inst_orders SET order_name=?, release_time=?, due_date=?, priority=?
                WHERE order_id=?
            """, (data["order_name"], float(data["release_time"]),
                  float(data["due_date"]), int(data["priority"]), order_id))

    def update_task(self, task_id: str, data: dict):
        with get_db(self.db_path) as conn:
            conn.execute("""
                UPDATE inst_tasks SET order_id=?, task_name=?, is_main=?,
                predecessor_task_ids=?, release_time=?, due_date=?
                WHERE task_id=?
            """, (data["order_id"], data["task_name"],
                  1 if data.get("is_main") else 0,
                  data.get("predecessor_task_ids", ""),
                  float(data.get("release_time", 0)),
                  float(data.get("due_date", 0)), task_id))

    def update_operation(self, op_id: str, data: dict):
        with get_db(self.db_path) as conn:
            conn.execute("""
                UPDATE inst_operations SET task_id=?, op_name=?, process_type=?,
                processing_time=?, predecessor_ops=?, predecessor_tasks=?,
                eligible_machine_ids=?
                WHERE op_id=?
            """, (data["task_id"], data["op_name"], data["process_type"],
                  float(data["processing_time"]),
                  data.get("predecessor_ops", ""),
                  data.get("predecessor_tasks", ""),
                  data.get("eligible_machine_ids", ""), op_id))

    def update_machine(self, machine_id: str, data: dict):
        with get_db(self.db_path) as conn:
            conn.execute("""
                UPDATE inst_machines SET machine_name=?, type_id=?, shifts=?
                WHERE machine_id=?
            """, (data["machine_name"], data["type_id"],
                  data.get("shifts", ""), machine_id))

    def build_shopfloor(self) -> 'ShopFloor':
        """从数据库数据重建 ShopFloor 对象"""
        from .models import ShopFloor, MachineType, Machine, Shift, Order, Task, Operation
        data = self.load_all()
        shop = ShopFloor()

        for mt in data["machine_types"]:
            shop.machine_types[mt["type_id"]] = MachineType(
                id=mt["type_id"], name=mt["type_name"],
                is_critical=bool(mt["is_critical"]),
            )

        for m in data["machines"]:
            shifts = []
            if m["shifts"]:
                for seg in m["shifts"].split(";"):
                    parts = seg.strip().split("/")
                    if len(parts) == 3:
                        shifts.append(Shift(
                            day=int(float(parts[0])),
                            start_hour=float(parts[1]),
                            hours=float(parts[2]),
                        ))
            shop.machines[m["machine_id"]] = Machine(
                id=m["machine_id"], name=m["machine_name"],
                type_id=m["type_id"], shifts=shifts,
            )

        for o in data["orders"]:
            shop.orders[o["order_id"]] = Order(
                id=o["order_id"], name=o["order_name"],
                release_time=o["release_time"],
                due_date=o["due_date"], priority=o["priority"],
            )

        for t in data["tasks"]:
            pred = [x.strip() for x in t["predecessor_task_ids"].split(";") if x.strip()]
            task = Task(
                id=t["task_id"], order_id=t["order_id"],
                name=t["task_name"], is_main=bool(t["is_main"]),
                predecessor_task_ids=pred,
                release_time=t["release_time"], due_date=t["due_date"],
            )
            shop.tasks[t["task_id"]] = task
            if t["order_id"] in shop.orders:
                shop.orders[t["order_id"]].task_ids.append(t["task_id"])
                if task.is_main:
                    shop.orders[t["order_id"]].main_task_id = t["task_id"]

        for op in data["operations"]:
            pred_ops = [x.strip() for x in op["predecessor_ops"].split(";") if x.strip()]
            pred_tasks = [x.strip() for x in op["predecessor_tasks"].split(";") if x.strip()]
            eligible = [x.strip() for x in op["eligible_machine_ids"].split(";") if x.strip()]
            operation = Operation(
                id=op["op_id"], task_id=op["task_id"],
                name=op["op_name"], process_type=op["process_type"],
                processing_time=op["processing_time"],
                predecessor_ops=pred_ops, predecessor_tasks=pred_tasks,
                eligible_machine_ids=eligible,
            )
            shop.operations[op["op_id"]] = operation
            if op["task_id"] in shop.tasks:
                shop.tasks[op["task_id"]].operations.append(operation)

        shop.build_indexes()
        self._load_downtimes_into_shop(shop)
        return shop

    def _load_downtimes_into_shop(self, shop):
        """Load downtimes from DB and attach to machines."""
        dt_store = DowntimeStore(self.db_path)
        downtimes_by_machine = dt_store.load_all_as_downtimes()
        for mid, m in shop.machines.items():
            m.downtimes = downtimes_by_machine.get(mid, [])


class GraphStore:
    """异构图持久化"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def clear_all(self):
        with get_db(self.db_path) as conn:
            conn.execute("DELETE FROM graph_nodes")
            conn.execute("DELETE FROM graph_edges")
            conn.execute("DELETE FROM graph_meta")

    def save_graph(self, hg):
        """从 HeterogeneousGraph 保存到数据库(覆盖)"""
        self.clear_all()
        graph = hg.graph
        with get_db(self.db_path) as conn:
            for nid, attrs in graph.nodes(data=True):
                conn.execute(
                    "INSERT INTO graph_nodes VALUES (?,?,?,?)",
                    (nid, attrs.get("node_type", ""),
                     attrs.get("entity_id", ""),
                     json.dumps({k: v for k, v in attrs.items()
                                 if k not in ("node_type", "entity_id")},
                                ensure_ascii=False)),
                )
            for src, tgt, attrs in graph.edges(data=True):
                conn.execute(
                    "INSERT INTO graph_edges (source, target, edge_type, attrs) VALUES (?,?,?,?)",
                    (src, tgt, attrs.get("edge_type", ""),
                     json.dumps({k: v for k, v in attrs.items()
                                 if k != "edge_type"}, ensure_ascii=False)),
                )
            stats = hg.get_graph_stats()
            conn.execute(
                "INSERT OR REPLACE INTO graph_meta VALUES (1,?,?,?,?,?)",
                (stats["total_nodes"], stats["total_edges"],
                 json.dumps(stats["node_types"]),
                 json.dumps(stats["edge_types"]),
                 time.time()),
            )

    def load_nodes(self, node_type: str = None, search: str = None) -> list[dict]:
        with get_db(self.db_path) as conn:
            sql = "SELECT * FROM graph_nodes WHERE 1=1"
            params = []
            if node_type:
                sql += " AND node_type = ?"
                params.append(node_type)
            if search:
                sql += " AND (node_id LIKE ? OR entity_id LIKE ?)"
                params.extend([f"%{search}%", f"%{search}%"])
            rows = conn.execute(sql, params).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["attrs"] = json.loads(d["attrs"]) if d["attrs"] else {}
                result.append(d)
            return result

    def load_edges(self, edge_type: str = None, search: str = None) -> list[dict]:
        with get_db(self.db_path) as conn:
            sql = "SELECT * FROM graph_edges WHERE 1=1"
            params = []
            if edge_type:
                sql += " AND edge_type = ?"
                params.append(edge_type)
            if search:
                sql += " AND (source LIKE ? OR target LIKE ?)"
                params.extend([f"%{search}%", f"%{search}%"])
            rows = conn.execute(sql, params).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["attrs"] = json.loads(d["attrs"]) if d["attrs"] else {}
                result.append(d)
            return result

    def load_meta(self) -> Optional[dict]:
        with get_db(self.db_path) as conn:
            row = conn.execute("SELECT * FROM graph_meta WHERE id=1").fetchone()
            if not row:
                return None
            d = dict(row)
            d["node_type_counts"] = json.loads(d["node_type_counts"])
            d["edge_type_counts"] = json.loads(d["edge_type_counts"])
            return d

    def has_data(self) -> bool:
        with get_db(self.db_path) as conn:
            cnt = conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
            return cnt > 0

    def get_node_neighbors(self, node_id: str) -> dict:
        """获取节点的邻居(出边和入边)"""
        with get_db(self.db_path) as conn:
            out_edges = [dict(r) for r in conn.execute(
                "SELECT * FROM graph_edges WHERE source=?", (node_id,)).fetchall()]
            in_edges = [dict(r) for r in conn.execute(
                "SELECT * FROM graph_edges WHERE target=?", (node_id,)).fetchall()]
            for e in out_edges + in_edges:
                e["attrs"] = json.loads(e["attrs"]) if e["attrs"] else {}
            return {"outgoing": out_edges, "incoming": in_edges}
