"""
数据模型 v3 — 面向真实业务的工业排产数据结构
=============================================
核心变更:
  - 按业务表结构定义: 订单→任务→工序, 工序有前置关系(支持半成品+装配)
  - 机器按工艺类别分组: 车/铣/磨/钻/镗/装配等, 每类多台
  - 每台机器有班次日历: 白班/夜班/维修日
  - 工序指定加工时间和可选机器类别(同类机器中只分配一台)
  - 主订单 vs 子任务区分, 便于统计主订单级延误
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid, csv, io


class OpStatus(Enum):
    PENDING = "pending"
    READY = "ready"        # 前置全部完成, 可调度
    PROCESSING = "processing"
    COMPLETED = "completed"

class MachineState(Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFF = "off"            # 非工作时间
    MAINTENANCE = "maintenance"


@dataclass
class Shift:
    """班次定义"""
    day: int               # 第几天 (0-based)
    start_hour: float      # 班次开始 (0-24)
    hours: float           # 可用小时数, 0=不可用

@dataclass
class MachineType:
    """机器类别 — 如 车/铣/磨/钻/镗/装配/检测 等"""
    id: str
    name: str              # 如 "车床", "铣床", "装配站"
    is_critical: bool = False  # 是否关键资源(用于KPI计算)

@dataclass
class Machine:
    """具体一台机器"""
    id: str
    name: str
    type_id: str                           # 所属机器类别
    shifts: list[Shift] = field(default_factory=list)  # 班次日历
    state: MachineState = MachineState.IDLE
    current_op_id: Optional[str] = None
    current_finish_time: float = 0.0
    total_busy_time: float = 0.0
    last_op_type: Optional[str] = None

    def available_hours_on_day(self, day: int) -> float:
        """获取某天的可用小时数"""
        total = 0.0
        for s in self.shifts:
            if s.day == day:
                total += s.hours
        return total if total > 0 else 18.0  # 默认18小时

    def is_available_at(self, time_hours: float) -> bool:
        """检查某时刻是否在工作时间内"""
        day = int(time_hours / 24)
        hour_in_day = time_hours % 24
        day_shifts = [s for s in self.shifts if s.day == day]
        if not day_shifts:
            return True  # 无日历数据默认可用
        for s in day_shifts:
            if s.hours <= 0:
                continue
            if s.start_hour <= hour_in_day < s.start_hour + s.hours:
                return True
        return False

    def next_available_time(self, from_time: float) -> float:
        """从某时刻开始, 找到下一个可用时刻"""
        t = from_time
        for _ in range(100):  # 最多查100天
            if self.is_available_at(t):
                return t
            # 跳到下一个班次
            day = int(t / 24)
            next_day_start = (day + 1) * 24
            t = next_day_start
        return from_time  # fallback


@dataclass
class Operation:
    """
    工序 — 最小调度单元
    每个工序属于一个任务, 有前置工序约束
    """
    id: str
    task_id: str           # 所属任务
    name: str              # 工序名称, 如 "粗车", "精铣", "组装"
    process_type: str      # 工艺类型, 对应 MachineType.id
    processing_time: float # 加工时间(小时)
    predecessor_ops: list[str] = field(default_factory=list)  # 前置工序ID列表
    predecessor_tasks: list[str] = field(default_factory=list) # 前置任务ID列表
    eligible_machine_ids: list[str] = field(default_factory=list)  # 可选具体机器ID
    # 运行时状态
    status: OpStatus = OpStatus.PENDING
    assigned_machine_id: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def is_ready(self) -> bool:
        return self.status == OpStatus.READY


@dataclass
class Task:
    """
    任务 — 对应一个半成品/组件/成品的完整加工任务
    一个任务有多道工序(按前置关系排列, 不一定线性)
    """
    id: str
    order_id: str          # 所属订单
    name: str              # 如 "壳体加工", "活塞加工", "总装"
    is_main: bool = False  # 是否主任务 (用于统计主订单延误)
    predecessor_task_ids: list[str] = field(default_factory=list) # 前置任务
    operations: list[Operation] = field(default_factory=list)
    release_time: float = 0.0
    due_date: float = float('inf')
    completion_time: Optional[float] = None

    @property
    def is_completed(self) -> bool:
        return all(op.status == OpStatus.COMPLETED for op in self.operations)

    @property
    def remaining_time(self) -> float:
        return sum(op.processing_time for op in self.operations if op.status != OpStatus.COMPLETED)

    @property
    def progress(self) -> float:
        if not self.operations: return 0.0
        done = sum(1 for op in self.operations if op.status == OpStatus.COMPLETED)
        return done / len(self.operations)


@dataclass
class Order:
    """
    订单 — 包含一个或多个任务(含子任务和主任务)
    主任务延误才算订单延误
    """
    id: str
    name: str = ""
    release_time: float = 0.0
    due_date: float = float('inf')
    priority: int = 1
    task_ids: list[str] = field(default_factory=list)
    main_task_id: Optional[str] = None  # 主任务ID
    status: str = "pending"


@dataclass
class ShopFloor:
    """完整车间模型"""
    machine_types: dict[str, MachineType] = field(default_factory=dict)
    machines: dict[str, Machine] = field(default_factory=dict)
    orders: dict[str, Order] = field(default_factory=dict)
    tasks: dict[str, Task] = field(default_factory=dict)
    operations: dict[str, Operation] = field(default_factory=dict)

    # 索引 (调度时加速查找)
    _machine_by_type: dict[str, list[str]] = field(default_factory=dict)
    _ops_by_task: dict[str, list[str]] = field(default_factory=dict)
    _tasks_by_order: dict[str, list[str]] = field(default_factory=dict)

    def build_indexes(self):
        """构建加速索引"""
        self._machine_by_type.clear()
        for mid, m in self.machines.items():
            self._machine_by_type.setdefault(m.type_id, []).append(mid)
        self._ops_by_task.clear()
        for oid, op in self.operations.items():
            self._ops_by_task.setdefault(op.task_id, []).append(oid)
        self._tasks_by_order.clear()
        for tid, t in self.tasks.items():
            self._tasks_by_order.setdefault(t.order_id, []).append(tid)

    def get_machines_for_type(self, type_id: str) -> list[Machine]:
        return [self.machines[mid] for mid in self._machine_by_type.get(type_id, []) if mid in self.machines]

    def get_eligible_machines(self, op: Operation) -> list[Machine]:
        """获取工序可用机器: 优先用指定列表, 否则按工艺类型找"""
        if op.eligible_machine_ids:
            return [self.machines[mid] for mid in op.eligible_machine_ids if mid in self.machines]
        return self.get_machines_for_type(op.process_type)

    def get_critical_machines(self) -> list[Machine]:
        """获取关键资源机器"""
        critical_types = {tid for tid, mt in self.machine_types.items() if mt.is_critical}
        return [m for m in self.machines.values() if m.type_id in critical_types]

    def check_op_ready(self, op: Operation) -> bool:
        """检查工序是否就绪(所有前置完成)"""
        for pid in op.predecessor_ops:
            if pid in self.operations and self.operations[pid].status != OpStatus.COMPLETED:
                return False
        for tid in op.predecessor_tasks:
            if tid in self.tasks and not self.tasks[tid].is_completed:
                return False
        return True

    def get_ready_ops(self) -> list[Operation]:
        """获取所有就绪的工序 — 核心调度入口"""
        ready = []
        for op in self.operations.values():
            if op.status == OpStatus.PENDING and self.check_op_ready(op):
                op.status = OpStatus.READY
                ready.append(op)
            elif op.status == OpStatus.READY:
                ready.append(op)
        return ready

    def summary(self) -> dict:
        return {
            "orders": len(self.orders),
            "tasks": len(self.tasks),
            "operations": len(self.operations),
            "machines": len(self.machines),
            "machine_types": len(self.machine_types),
        }

    def to_csv(self) -> str:
        """导出所有数据到一个 CSV"""
        output = io.StringIO()
        w = csv.writer(output)
        # Header
        w.writerow([
            "record_type", "order_id", "order_name", "order_due_date", "order_priority",
            "task_id", "task_name", "is_main_task", "task_due_date",
            "predecessor_tasks",
            "op_id", "op_name", "process_type", "processing_time_hrs",
            "predecessor_ops", "eligible_machines",
            "machine_type_id", "machine_type_name", "is_critical",
            "machine_id", "machine_name", "day", "shift_start", "shift_hours",
        ])
        # Orders + Tasks + Ops
        for oid, order in self.orders.items():
            for tid in order.task_ids:
                task = self.tasks.get(tid)
                if not task: continue
                for op in task.operations:
                    w.writerow([
                        "operation", oid, order.name, order.due_date, order.priority,
                        tid, task.name, "Y" if task.is_main else "N", task.due_date,
                        ";".join(task.predecessor_task_ids),
                        op.id, op.name, op.process_type, op.processing_time,
                        ";".join(op.predecessor_ops), ";".join(op.eligible_machine_ids),
                        "", "", "", "", "", "", "", "",
                    ])
        # Machines
        for mid, m in self.machines.items():
            mt = self.machine_types.get(m.type_id)
            if m.shifts:
                for s in m.shifts:
                    w.writerow([
                        "machine", "", "", "", "",
                        "", "", "", "", "",
                        "", "", "", "",
                        "", "",
                        m.type_id, mt.name if mt else "", "Y" if (mt and mt.is_critical) else "N",
                        mid, m.name, s.day, s.start_hour, s.hours,
                    ])
            else:
                w.writerow([
                    "machine", "", "", "", "",
                    "", "", "", "", "",
                    "", "", "", "",
                    "", "",
                    m.type_id, mt.name if mt else "", "Y" if (mt and mt.is_critical) else "N",
                    mid, m.name, "", "", "",
                ])
        return output.getvalue()


def uid(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:6]}"
