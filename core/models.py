from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
from typing import Optional
import csv
import io
import uuid

from .time_utils import default_plan_start, ensure_aware, isoformat_or_none, offset_hours_to_datetime


class OpStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    PROCESSING = "processing"
    COMPLETED = "completed"


class ResourceState(Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFF = "off"
    MAINTENANCE = "maintenance"


MachineState = ResourceState


@dataclass
class Shift:
    day: int
    start_hour: float
    hours: float

    @property
    def start_offset(self) -> float:
        return self.day * 24.0 + self.start_hour

    @property
    def end_offset(self) -> float:
        return self.start_offset + self.hours


@dataclass
class Downtime:
    id: str
    machine_id: str
    downtime_type: str
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


class CalendarResourceMixin:
    shifts: list[Shift]
    downtimes: list[Downtime]
    calendar_anchor_hour: float

    def compile_calendar(self) -> None:
        anchor_hour = getattr(self, "calendar_anchor_hour", 0.0)
        shifts = getattr(self, "shifts", [])
        downtimes = sorted(getattr(self, "downtimes", []), key=lambda dt: (dt.start_time, dt.end_time))

        shift_windows = _merge_windows(
            [
                (
                    shift.day * 24.0 + shift.start_hour - anchor_hour,
                    shift.day * 24.0 + shift.start_hour - anchor_hour + shift.hours,
                )
                for shift in shifts
                if shift.hours > 0
            ]
        )
        downtime_windows = _merge_windows([(dt.start_time, dt.end_time) for dt in downtimes])
        base_windows = shift_windows if shift_windows else [(0.0, float("inf"))]
        available_windows = _subtract_windows(base_windows, downtime_windows)

        self._calendar_shift_windows = tuple(shift_windows)
        self._calendar_shift_starts = tuple(start for start, _ in shift_windows)
        self._calendar_downtimes = tuple(downtimes)
        self._calendar_downtime_starts = tuple(dt.start_time for dt in downtimes)
        self._calendar_available_windows = tuple(available_windows)
        self._calendar_available_starts = tuple(start for start, _ in available_windows)
        self._calendar_compiled = True

    def _ensure_calendar_cache(self) -> None:
        if not getattr(self, "_calendar_compiled", False):
            self.compile_calendar()

    def _sorted_shift_windows(self) -> list[tuple[float, float]]:
        self._ensure_calendar_cache()
        return list(self._calendar_shift_windows)

    def _sorted_downtimes(self) -> list[Downtime]:
        self._ensure_calendar_cache()
        return list(self._calendar_downtimes)

    def _active_downtime(self, when: float) -> Downtime | None:
        self._ensure_calendar_cache()
        downtimes = self._calendar_downtimes
        if not downtimes:
            return None
        starts = self._calendar_downtime_starts
        index = bisect_right(starts, when) - 1
        if index >= 0:
            downtime = downtimes[index]
            if downtime.start_time <= when < downtime.end_time:
                return downtime
        next_index = index + 1
        if 0 <= next_index < len(downtimes):
            downtime = downtimes[next_index]
            if downtime.start_time <= when < downtime.end_time:
                return downtime
        return None

    def _shift_available_at(self, when: float) -> bool:
        self._ensure_calendar_cache()
        windows = self._calendar_shift_windows
        if not windows:
            return True
        starts = self._calendar_shift_starts
        index = bisect_right(starts, when) - 1
        return index >= 0 and when < windows[index][1]

    def _next_shift_available(self, when: float) -> float:
        self._ensure_calendar_cache()
        windows = self._calendar_shift_windows
        if not windows:
            return when
        starts = self._calendar_shift_starts
        index = bisect_right(starts, when) - 1
        if index >= 0 and when < windows[index][1]:
            return when
        next_index = index + 1
        if next_index < len(windows):
            return windows[next_index][0]
        return float("inf")

    def _current_shift_end(self, when: float) -> float:
        self._ensure_calendar_cache()
        windows = self._calendar_shift_windows
        if not windows:
            return float("inf")
        starts = self._calendar_shift_starts
        index = bisect_right(starts, when) - 1
        if index >= 0 and when < windows[index][1]:
            return windows[index][1]
        return float("inf")

    def is_available_at(self, when: float) -> bool:
        self._ensure_calendar_cache()
        windows = self._calendar_available_windows
        if not windows:
            return False
        starts = self._calendar_available_starts
        index = bisect_right(starts, when) - 1
        return index >= 0 and when < windows[index][1]

    def next_available_time(self, when: float) -> float:
        self._ensure_calendar_cache()
        windows = self._calendar_available_windows
        if not windows:
            return float("inf")
        starts = self._calendar_available_starts
        index = bisect_right(starts, when) - 1
        if index >= 0 and when < windows[index][1]:
            return when
        next_index = index + 1
        if next_index < len(windows):
            return windows[next_index][0]
        return float("inf")

    def next_start_time(self, when: float) -> float:
        return self.next_available_time(when)

    def next_unavailable_time(self, when: float) -> float:
        self._ensure_calendar_cache()
        windows = self._calendar_available_windows
        if not windows:
            return when
        starts = self._calendar_available_starts
        index = bisect_right(starts, when) - 1
        if index >= 0 and when < windows[index][1]:
            return windows[index][1]
        return when

    def compute_effective_end(self, start: float, duration: float) -> float:
        self._ensure_calendar_cache()
        when = self.next_available_time(start)
        remaining = duration
        if when == float("inf"):
            return when
        windows = self._calendar_available_windows
        starts = self._calendar_available_starts
        index = bisect_right(starts, when) - 1
        while remaining > 1e-9 and 0 <= index < len(windows):
            _, end = windows[index]
            if end == float("inf"):
                return when + remaining
            workable = max(0.0, end - when)
            if workable >= remaining - 1e-9:
                return when + remaining
            remaining -= workable
            index += 1
            if index >= len(windows):
                return float("inf")
            when = max(end, windows[index][0])
        return when

    def unavailable_windows(self, horizon_end: float) -> list[tuple[float, float]]:
        self._ensure_calendar_cache()
        windows: list[tuple[float, float]] = []
        available_windows = self._calendar_available_windows
        cursor = 0.0
        for start, end in available_windows:
            if start >= horizon_end:
                break
            if start > cursor:
                windows.append((cursor, min(start, horizon_end)))
            if end == float("inf"):
                cursor = horizon_end
                break
            cursor = max(cursor, end)
            if cursor >= horizon_end:
                break
        if cursor < horizon_end:
            windows.append((cursor, horizon_end))
        return _merge_windows(windows)

    def available_time_between(self, start_time: float, end_time: float) -> float:
        self._ensure_calendar_cache()
        if end_time <= start_time:
            return 0.0
        total = 0.0
        for window_start, window_end in self._calendar_available_windows:
            if window_end <= start_time:
                continue
            if window_start >= end_time:
                break
            overlap_start = max(start_time, window_start)
            overlap_end = min(end_time, window_end)
            if overlap_end > overlap_start:
                total += overlap_end - overlap_start
        return total


@dataclass
class MachineType:
    id: str
    name: str
    is_critical: bool = False


@dataclass
class ToolingType:
    id: str
    name: str


@dataclass
class Machine(CalendarResourceMixin):
    id: str
    name: str
    type_id: str
    shifts: list[Shift] = field(default_factory=list)
    state: ResourceState = ResourceState.IDLE
    current_op_id: Optional[str] = None
    current_finish_time: float = 0.0
    total_busy_time: float = 0.0
    last_op_type: Optional[str] = None
    downtimes: list[Downtime] = field(default_factory=list)
    calendar_anchor_hour: float = 0.0
    _calendar_compiled: bool = field(default=False, init=False, repr=False)
    _calendar_shift_windows: tuple[tuple[float, float], ...] = field(default_factory=tuple, init=False, repr=False)
    _calendar_shift_starts: tuple[float, ...] = field(default_factory=tuple, init=False, repr=False)
    _calendar_downtimes: tuple[Downtime, ...] = field(default_factory=tuple, init=False, repr=False)
    _calendar_downtime_starts: tuple[float, ...] = field(default_factory=tuple, init=False, repr=False)
    _calendar_available_windows: tuple[tuple[float, float], ...] = field(default_factory=tuple, init=False, repr=False)
    _calendar_available_starts: tuple[float, ...] = field(default_factory=tuple, init=False, repr=False)


@dataclass
class Tooling(CalendarResourceMixin):
    id: str
    name: str
    type_id: str
    shifts: list[Shift] = field(default_factory=list)
    state: ResourceState = ResourceState.IDLE
    current_op_id: Optional[str] = None
    current_finish_time: float = 0.0
    total_busy_time: float = 0.0
    downtimes: list[Downtime] = field(default_factory=list)
    calendar_anchor_hour: float = 0.0
    _calendar_compiled: bool = field(default=False, init=False, repr=False)
    _calendar_shift_windows: tuple[tuple[float, float], ...] = field(default_factory=tuple, init=False, repr=False)
    _calendar_shift_starts: tuple[float, ...] = field(default_factory=tuple, init=False, repr=False)
    _calendar_downtimes: tuple[Downtime, ...] = field(default_factory=tuple, init=False, repr=False)
    _calendar_downtime_starts: tuple[float, ...] = field(default_factory=tuple, init=False, repr=False)
    _calendar_available_windows: tuple[tuple[float, float], ...] = field(default_factory=tuple, init=False, repr=False)
    _calendar_available_starts: tuple[float, ...] = field(default_factory=tuple, init=False, repr=False)


@dataclass
class Personnel(CalendarResourceMixin):
    id: str
    name: str
    skills: list[str] = field(default_factory=list)
    shifts: list[Shift] = field(default_factory=list)
    state: ResourceState = ResourceState.IDLE
    current_op_id: Optional[str] = None
    current_finish_time: float = 0.0
    total_busy_time: float = 0.0
    downtimes: list[Downtime] = field(default_factory=list)
    calendar_anchor_hour: float = 0.0
    _calendar_compiled: bool = field(default=False, init=False, repr=False)
    _calendar_shift_windows: tuple[tuple[float, float], ...] = field(default_factory=tuple, init=False, repr=False)
    _calendar_shift_starts: tuple[float, ...] = field(default_factory=tuple, init=False, repr=False)
    _calendar_downtimes: tuple[Downtime, ...] = field(default_factory=tuple, init=False, repr=False)
    _calendar_downtime_starts: tuple[float, ...] = field(default_factory=tuple, init=False, repr=False)
    _calendar_available_windows: tuple[tuple[float, float], ...] = field(default_factory=tuple, init=False, repr=False)
    _calendar_available_starts: tuple[float, ...] = field(default_factory=tuple, init=False, repr=False)

    def has_skill(self, skill_id: str) -> bool:
        return skill_id in self.skills


@dataclass
class Operation:
    id: str
    task_id: str
    name: str
    process_type: str
    processing_time: float
    predecessor_ops: list[str] = field(default_factory=list)
    predecessor_tasks: list[str] = field(default_factory=list)
    eligible_machine_ids: list[str] = field(default_factory=list)
    required_tooling_types: list[str] = field(default_factory=list)
    required_personnel_skills: list[str] = field(default_factory=list)
    status: OpStatus = OpStatus.PENDING
    assigned_machine_id: Optional[str] = None
    assigned_tooling_ids: list[str] = field(default_factory=list)
    assigned_personnel_ids: list[str] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    remaining_processing_time: Optional[float] = None
    derived_due_date: float = float("inf")
    derived_start_time: float = float("inf")
    earliest_start_time: float = 0.0
    earliest_finish_time: float = 0.0
    critical_slack: float = float("inf")

    @property
    def is_ready(self) -> bool:
        return self.status == OpStatus.READY

    @property
    def work_remaining(self) -> float:
        if self.status == OpStatus.COMPLETED:
            return 0.0
        return self.remaining_processing_time if self.remaining_processing_time is not None else self.processing_time


@dataclass
class Task:
    id: str
    order_id: str
    name: str
    is_main: bool = False
    predecessor_task_ids: list[str] = field(default_factory=list)
    operations: list[Operation] = field(default_factory=list)
    release_time: float = 0.0
    due_date: float = float("inf")
    completion_time: Optional[float] = None
    derived_due_date: float = float("inf")
    derived_start_time: float = float("inf")
    earliest_start_time: float = 0.0
    earliest_finish_time: float = 0.0
    critical_path_time: float = 0.0
    critical_slack: float = float("inf")

    @property
    def is_completed(self) -> bool:
        return all(op.status == OpStatus.COMPLETED for op in self.operations)

    @property
    def remaining_time(self) -> float:
        return sum(op.work_remaining for op in self.operations if op.status != OpStatus.COMPLETED)

    @property
    def progress(self) -> float:
        if not self.operations:
            return 0.0
        completed = sum(1 for op in self.operations if op.status == OpStatus.COMPLETED)
        return completed / len(self.operations)


@dataclass
class Order:
    id: str
    name: str = ""
    release_time: float = 0.0
    due_date: float = float("inf")
    priority: int = 1
    task_ids: list[str] = field(default_factory=list)
    main_task_id: Optional[str] = None
    status: str = "pending"


@dataclass
class ShopFloor:
    machine_types: dict[str, MachineType] = field(default_factory=dict)
    tooling_types: dict[str, ToolingType] = field(default_factory=dict)
    machines: dict[str, Machine] = field(default_factory=dict)
    toolings: dict[str, Tooling] = field(default_factory=dict)
    personnel: dict[str, Personnel] = field(default_factory=dict)
    orders: dict[str, Order] = field(default_factory=dict)
    tasks: dict[str, Task] = field(default_factory=dict)
    operations: dict[str, Operation] = field(default_factory=dict)
    plan_start_at: object = field(default_factory=default_plan_start)

    _machine_by_type: dict[str, list[str]] = field(default_factory=dict)
    _tooling_by_type: dict[str, list[str]] = field(default_factory=dict)
    _personnel_by_skill: dict[str, list[str]] = field(default_factory=dict)
    _ops_by_task: dict[str, list[str]] = field(default_factory=dict)
    _tasks_by_order: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.plan_start_at = ensure_aware(self.plan_start_at)

    def build_indexes(self) -> None:
        self.plan_start_at = ensure_aware(self.plan_start_at)
        anchor_hour = _clock_hours(self.plan_start_at)
        self._machine_by_type.clear()
        for mid, machine in self.machines.items():
            machine.shifts.sort(key=lambda shift: (shift.day, shift.start_hour))
            machine.downtimes.sort(key=lambda dt: (dt.start_time, dt.end_time))
            machine.calendar_anchor_hour = anchor_hour
            machine.compile_calendar()
            self._machine_by_type.setdefault(machine.type_id, []).append(mid)

        self._tooling_by_type.clear()
        for tooling_id, tooling in self.toolings.items():
            tooling.shifts.sort(key=lambda shift: (shift.day, shift.start_hour))
            tooling.downtimes.sort(key=lambda dt: (dt.start_time, dt.end_time))
            tooling.calendar_anchor_hour = anchor_hour
            tooling.compile_calendar()
            self._tooling_by_type.setdefault(tooling.type_id, []).append(tooling_id)

        self._personnel_by_skill.clear()
        for person_id, person in self.personnel.items():
            person.shifts.sort(key=lambda shift: (shift.day, shift.start_hour))
            person.downtimes.sort(key=lambda dt: (dt.start_time, dt.end_time))
            person.calendar_anchor_hour = anchor_hour
            person.compile_calendar()
            for skill in person.skills:
                self._personnel_by_skill.setdefault(skill, []).append(person_id)

        self._ops_by_task.clear()
        for op_id, op in self.operations.items():
            self._ops_by_task.setdefault(op.task_id, []).append(op_id)

        self._tasks_by_order.clear()
        for task_id, task in self.tasks.items():
            self._tasks_by_order.setdefault(task.order_id, []).append(task_id)
        self.derive_internal_targets()

    def calendar_days(self) -> int:
        max_day = -1
        for resource in [*self.machines.values(), *self.toolings.values(), *self.personnel.values()]:
            if getattr(resource, "shifts", None):
                max_day = max(max_day, max((shift.day for shift in resource.shifts), default=-1))
        return max_day + 1 if max_day >= 0 else 0

    def estimate_required_schedule_days(
        self,
        safety_factor: float = 1.45,
        min_days: int = 14,
        max_days: int = 720,
    ) -> int:
        current_days = max(1, self.calendar_days())
        process_work: dict[str, float] = {}
        required_tooling_by_process: dict[str, bool] = {}
        required_personnel_by_process: dict[str, bool] = {}

        for op in self.operations.values():
            process_work[op.process_type] = process_work.get(op.process_type, 0.0) + max(0.0, op.processing_time)
            if op.required_tooling_types:
                required_tooling_by_process[op.process_type] = True
            if op.required_personnel_skills:
                required_personnel_by_process[op.process_type] = True

        total_work = sum(process_work.values())
        if total_work <= 0:
            finite_due_dates = [order.due_date for order in self.orders.values() if math.isfinite(order.due_date)]
            due_based_days = int(math.ceil(max(finite_due_dates, default=0.0) / 24.0)) + 2
            return max(min_days, min(max_days, max(current_days, due_based_days)))

        process_hours_needed: list[float] = []
        for process_type, work_hours in process_work.items():
            machine_units = _resource_parallel_units(self.get_machines_for_type(process_type), current_days)
            units = machine_units

            if required_tooling_by_process.get(process_type):
                tooling_units = _resource_parallel_units(
                    self.get_toolings_for_type(f"tool_{process_type}"),
                    current_days,
                )
                units = min(units, tooling_units)

            if required_personnel_by_process.get(process_type):
                personnel_units = _resource_parallel_units(
                    self.get_personnel_for_skill(f"skill_{process_type}"),
                    current_days,
                )
                units = min(units, personnel_units)

            process_hours_needed.append(work_hours / max(units, 0.1))

        global_units = _resource_parallel_units(list(self.machines.values()), current_days)
        if any(required_tooling_by_process.values()):
            global_units = min(global_units, _resource_parallel_units(list(self.toolings.values()), current_days))
        if any(required_personnel_by_process.values()):
            global_units = min(global_units, _resource_parallel_units(list(self.personnel.values()), current_days))

        total_parallel_hours = total_work / max(global_units, 0.1)
        bottleneck_hours = max(process_hours_needed, default=0.0)
        required_hours = max(total_parallel_hours * 1.28, bottleneck_hours * 1.75)

        max_release = max(
            [order.release_time for order in self.orders.values()] +
            [task.release_time for task in self.tasks.values()] +
            [0.0]
        )
        finite_due_dates = [order.due_date for order in self.orders.values() if math.isfinite(order.due_date)]
        due_based_hours = max(finite_due_dates, default=0.0)
        complexity_buffer = max(
            24.0,
            0.02 * total_work + 0.9 * len(self.orders) + 0.18 * len(self.tasks),
        )
        target_hours = max(
            due_based_hours + 24.0,
            max_release + required_hours * max(1.0, safety_factor) + complexity_buffer,
        )
        estimated_days = int(math.ceil(max(target_hours, 0.0) / 24.0)) + 1
        return max(min_days, min(max_days, max(current_days, estimated_days)))

    def ensure_calendar_days(self, min_days: int) -> bool:
        target_days = max(1, int(math.ceil(min_days)))
        current_days = self.calendar_days()
        if target_days <= current_days:
            return False

        for resource in [*self.machines.values(), *self.toolings.values(), *self.personnel.values()]:
            resource.shifts = _extend_shift_calendar(resource.shifts, target_days)

        self.build_indexes()
        return True

    def ensure_calendar_capacity(
        self,
        safety_factor: float = 1.45,
        min_days: int = 14,
        max_days: int = 720,
    ) -> dict:
        current_days = self.calendar_days()
        estimated_days = self.estimate_required_schedule_days(
            safety_factor=safety_factor,
            min_days=min_days,
            max_days=max_days,
        )
        extended = self.ensure_calendar_days(estimated_days)
        return {
            "current_days": current_days,
            "required_days": estimated_days,
            "extended": extended,
            "final_days": self.calendar_days(),
        }

    def derive_internal_targets(self) -> None:
        task_predecessors: dict[str, set[str]] = {task_id: set() for task_id in self.tasks}
        task_successors: dict[str, set[str]] = {task_id: set() for task_id in self.tasks}

        def add_task_edge(predecessor_id: str, successor_id: str) -> None:
            if predecessor_id == successor_id:
                return
            if predecessor_id not in self.tasks or successor_id not in self.tasks:
                return
            if predecessor_id in task_predecessors[successor_id]:
                return
            task_predecessors[successor_id].add(predecessor_id)
            task_successors[predecessor_id].add(successor_id)

        for task_id, task in self.tasks.items():
            for predecessor_id in task.predecessor_task_ids:
                add_task_edge(predecessor_id, task_id)

        for op in self.operations.values():
            for predecessor_task_id in op.predecessor_tasks:
                add_task_edge(predecessor_task_id, op.task_id)
            for predecessor_op_id in op.predecessor_ops:
                predecessor_op = self.operations.get(predecessor_op_id)
                if predecessor_op and predecessor_op.task_id != op.task_id:
                    add_task_edge(predecessor_op.task_id, op.task_id)

        task_meta: dict[str, dict] = {}
        for task_id, task in self.tasks.items():
            op_ids = [op.id for op in task.operations if op.id in self.operations]
            predecessors = {op_id: set() for op_id in op_ids}
            successors = {op_id: set() for op_id in op_ids}
            for op_id in op_ids:
                op = self.operations[op_id]
                for predecessor_id in op.predecessor_ops:
                    if predecessor_id in predecessors:
                        predecessors[op_id].add(predecessor_id)
                        successors[predecessor_id].add(op_id)
            topo = _topological_order(predecessors, op_ids)
            earliest_offsets: dict[str, float] = {}
            critical_path = 0.0
            for op_id in topo:
                op = self.operations[op_id]
                offset = max(
                    (earliest_offsets[predecessor_id] + self.operations[predecessor_id].processing_time for predecessor_id in predecessors[op_id]),
                    default=0.0,
                )
                earliest_offsets[op_id] = offset
                critical_path = max(critical_path, offset + op.processing_time)
            task_meta[task_id] = {
                "op_ids": op_ids,
                "predecessors": predecessors,
                "successors": successors,
                "topo": topo,
                "earliest_offsets": earliest_offsets,
                "critical_path": critical_path,
            }

        task_topo = _topological_order(task_predecessors, list(self.tasks.keys()))
        for task_id in reversed(task_topo):
            task = self.tasks[task_id]
            order = self.orders.get(task.order_id)
            external_due = task.due_date
            if not math.isfinite(external_due) and order is not None:
                external_due = order.due_date
            successor_starts = [
                self.tasks[successor_id].derived_start_time
                for successor_id in task_successors.get(task_id, set())
                if math.isfinite(self.tasks[successor_id].derived_start_time)
            ]
            candidates = [value for value in [external_due, *successor_starts] if math.isfinite(value)]
            task.critical_path_time = task_meta.get(task_id, {}).get("critical_path", 0.0)
            task.derived_due_date = min(candidates) if candidates else float("inf")
            task.derived_start_time = (
                task.derived_due_date - task.critical_path_time
                if math.isfinite(task.derived_due_date)
                else float("inf")
            )

        for task_id in task_topo:
            task = self.tasks[task_id]
            order = self.orders.get(task.order_id)
            base_release = max(task.release_time, order.release_time if order else 0.0)
            predecessor_finish = max(
                (self.tasks[predecessor_id].earliest_finish_time for predecessor_id in task_predecessors.get(task_id, set())),
                default=base_release,
            )
            task.earliest_start_time = max(base_release, predecessor_finish)
            task.earliest_finish_time = task.earliest_start_time + task.critical_path_time
            task.critical_slack = (
                task.derived_due_date - task.earliest_finish_time
                if math.isfinite(task.derived_due_date)
                else float("inf")
            )

        for task_id, task in self.tasks.items():
            meta = task_meta.get(task_id, {})
            topo = meta.get("topo", [])
            successors = meta.get("successors", {})
            earliest_offsets = meta.get("earliest_offsets", {})

            for op_id in reversed(topo):
                op = self.operations[op_id]
                successor_starts = [
                    self.operations[successor_id].derived_start_time
                    for successor_id in successors.get(op_id, set())
                    if math.isfinite(self.operations[successor_id].derived_start_time)
                ]
                candidates = [value for value in [task.derived_due_date, *successor_starts] if math.isfinite(value)]
                op.derived_due_date = min(candidates) if candidates else float("inf")
                op.derived_start_time = (
                    op.derived_due_date - op.processing_time
                    if math.isfinite(op.derived_due_date)
                    else float("inf")
                )

            for op_id in topo:
                op = self.operations[op_id]
                op.earliest_start_time = task.earliest_start_time + earliest_offsets.get(op_id, 0.0)
                op.earliest_finish_time = op.earliest_start_time + op.processing_time
                op.critical_slack = (
                    op.derived_start_time - op.earliest_start_time
                    if math.isfinite(op.derived_start_time)
                    else float("inf")
                )

    def get_machines_for_type(self, type_id: str) -> list[Machine]:
        return [self.machines[mid] for mid in self._machine_by_type.get(type_id, []) if mid in self.machines]

    def get_toolings_for_type(self, type_id: str) -> list[Tooling]:
        return [self.toolings[tid] for tid in self._tooling_by_type.get(type_id, []) if tid in self.toolings]

    def get_personnel_for_skill(self, skill_id: str) -> list[Personnel]:
        return [self.personnel[pid] for pid in self._personnel_by_skill.get(skill_id, []) if pid in self.personnel]

    def get_eligible_machines(self, op: Operation) -> list[Machine]:
        if op.eligible_machine_ids:
            return [self.machines[mid] for mid in op.eligible_machine_ids if mid in self.machines]
        return self.get_machines_for_type(op.process_type)

    def get_critical_machines(self) -> list[Machine]:
        critical_types = {type_id for type_id, mt in self.machine_types.items() if mt.is_critical}
        return [machine for machine in self.machines.values() if machine.type_id in critical_types]

    def get_operation_release_time(self, op: Operation) -> float:
        task = self.tasks.get(op.task_id)
        order = self.orders.get(task.order_id) if task else None
        task_release = task.release_time if task else 0.0
        order_release = order.release_time if order else 0.0
        return max(task_release, order_release)

    def check_op_ready(self, op: Operation) -> bool:
        for predecessor_id in op.predecessor_ops:
            predecessor = self.operations.get(predecessor_id)
            if predecessor and predecessor.status != OpStatus.COMPLETED:
                return False
        for predecessor_task_id in op.predecessor_tasks:
            predecessor_task = self.tasks.get(predecessor_task_id)
            if predecessor_task and not predecessor_task.is_completed:
                return False
        return True

    def get_ready_ops(self, now: float = 0.0) -> list[Operation]:
        ready: list[Operation] = []
        for op in self.operations.values():
            if op.status == OpStatus.PENDING and self.check_op_ready(op) and self.get_operation_release_time(op) <= now:
                op.status = OpStatus.READY
                ready.append(op)
            elif op.status == OpStatus.READY:
                ready.append(op)
        return ready

    def offset_to_datetime(self, offset_hours: float | None):
        return offset_hours_to_datetime(self.plan_start_at, offset_hours)

    def time_label(self, offset_hours: float | None) -> str | None:
        return isoformat_or_none(self.offset_to_datetime(offset_hours))

    def summary(self) -> dict:
        return {
            "orders": len(self.orders),
            "tasks": len(self.tasks),
            "operations": len(self.operations),
            "ops_ready": sum(1 for op in self.operations.values() if op.status == OpStatus.READY),
            "ops_in_progress": sum(1 for op in self.operations.values() if op.status == OpStatus.PROCESSING),
            "ops_completed": sum(1 for op in self.operations.values() if op.status == OpStatus.COMPLETED),
            "machines": len(self.machines),
            "machine_types": len(self.machine_types),
            "toolings": len(self.toolings),
            "tooling_types": len(self.tooling_types),
            "personnel": len(self.personnel),
            "calendar_days": self.calendar_days(),
            "plan_start_at": self.time_label(0.0),
        }

    def to_csv(self) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "record_type",
                "plan_start_at",
                "order_id",
                "order_name",
                "order_due_offset_h",
                "order_release_offset_h",
                "order_priority",
                "task_id",
                "task_name",
                "is_main_task",
                "task_due_offset_h",
                "task_release_offset_h",
                "predecessor_tasks",
                "op_id",
                "op_name",
                "process_type",
                "processing_time_h",
                "predecessor_ops",
                "eligible_machines",
                "required_tooling_types",
                "required_personnel_skills",
                "machine_type_id",
                "machine_type_name",
                "is_critical",
                "machine_id",
                "machine_name",
                "calendar_day",
                "calendar_start_h",
                "calendar_hours",
                "tooling_type_id",
                "tooling_type_name",
                "tooling_id",
                "tooling_name",
                "personnel_id",
                "personnel_name",
                "personnel_skills",
            ]
        )

        writer.writerow(["meta", isoformat_or_none(self.plan_start_at)])

        for order_id, order in self.orders.items():
            for task_id in order.task_ids:
                task = self.tasks.get(task_id)
                if not task:
                    continue
                for op in task.operations:
                    writer.writerow(
                        [
                            "operation",
                            isoformat_or_none(self.plan_start_at),
                            order_id,
                            order.name,
                            order.due_date,
                            order.release_time,
                            order.priority,
                            task_id,
                            task.name,
                            "Y" if task.is_main else "N",
                            task.due_date,
                            task.release_time,
                            ";".join(task.predecessor_task_ids),
                            op.id,
                            op.name,
                            op.process_type,
                            op.processing_time,
                            ";".join(op.predecessor_ops),
                            ";".join(op.eligible_machine_ids),
                            ";".join(op.required_tooling_types),
                            ";".join(op.required_personnel_skills),
                        ]
                    )

        for machine_id, machine in self.machines.items():
            machine_type = self.machine_types.get(machine.type_id)
            if machine.shifts:
                for shift in machine.shifts:
                    writer.writerow(
                        [
                            "machine",
                            isoformat_or_none(self.plan_start_at),
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            machine.type_id,
                            machine_type.name if machine_type else "",
                            "Y" if machine_type and machine_type.is_critical else "N",
                            machine_id,
                            machine.name,
                            shift.day,
                            shift.start_hour,
                            shift.hours,
                        ]
                    )
            else:
                writer.writerow(
                    [
                        "machine",
                        isoformat_or_none(self.plan_start_at),
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        machine.type_id,
                        machine_type.name if machine_type else "",
                        "Y" if machine_type and machine_type.is_critical else "N",
                        machine_id,
                        machine.name,
                    ]
                )

        for tooling_type_id, tooling_type in self.tooling_types.items():
            writer.writerow(["tooling_type", isoformat_or_none(self.plan_start_at), "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", tooling_type_id, tooling_type.name])
        for tooling_id, tooling in self.toolings.items():
            writer.writerow(["tooling", isoformat_or_none(self.plan_start_at), "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", tooling.type_id, self.tooling_types.get(tooling.type_id).name if tooling.type_id in self.tooling_types else "", tooling_id, tooling.name])
        for personnel_id, person in self.personnel.items():
            writer.writerow(["personnel", isoformat_or_none(self.plan_start_at), "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", personnel_id, person.name, ";".join(person.skills)])

        return output.getvalue()


def _topological_order(predecessors: dict[str, set[str]], node_ids: list[str]) -> list[str]:
    remaining = {node_id: {pred for pred in predecessors.get(node_id, set()) if pred in node_ids} for node_id in node_ids}
    successors: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for node_id, preds in remaining.items():
        for predecessor_id in preds:
            successors.setdefault(predecessor_id, set()).add(node_id)

    queue = sorted([node_id for node_id in node_ids if not remaining[node_id]])
    ordered: list[str] = []

    while queue:
        current = queue.pop(0)
        ordered.append(current)
        for node_id in successors.get(current, set()):
            if current in remaining[node_id]:
                remaining[node_id].discard(current)
                if not remaining[node_id] and node_id not in ordered and node_id not in queue:
                    queue.append(node_id)
        queue.sort()

    if len(ordered) < len(node_ids):
        leftovers = [node_id for node_id in node_ids if node_id not in ordered]
        ordered.extend(sorted(leftovers))
    return ordered


def _merge_windows(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    normalized = sorted((start, end) for start, end in windows if end > start)
    merged: list[list[float]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _subtract_windows(
    base_windows: list[tuple[float, float]],
    blocked_windows: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if not base_windows:
        return []
    if not blocked_windows:
        return list(base_windows)

    available: list[tuple[float, float]] = []
    block_index = 0

    for base_start, base_end in base_windows:
        cursor = base_start
        while block_index < len(blocked_windows) and blocked_windows[block_index][1] <= cursor:
            block_index += 1

        probe = block_index
        while probe < len(blocked_windows):
            block_start, block_end = blocked_windows[probe]
            if block_start >= base_end:
                break
            if block_start > cursor:
                available.append((cursor, min(block_start, base_end)))
            cursor = max(cursor, block_end)
            if cursor >= base_end:
                break
            probe += 1
        if cursor < base_end:
            available.append((cursor, base_end))

    return [(start, end) for start, end in available if end > start]


def _clock_hours(value: datetime) -> float:
    aware = ensure_aware(value)
    return (
        aware.hour
        + aware.minute / 60.0
        + aware.second / 3600.0
        + aware.microsecond / 3_600_000_000.0
    )


def _resource_parallel_units(resources: list, schedule_days: int) -> float:
    if not resources:
        return 1.0
    horizon_hours = max(24.0 * max(1, schedule_days), 1.0)
    total_hours = 0.0
    for resource in resources:
        total_hours += sum(max(0.0, shift.hours) for shift in getattr(resource, "shifts", []))
    if total_hours <= 0:
        return float(len(resources))
    return max(total_hours / horizon_hours, 0.1)


def _extend_shift_calendar(shifts: list[Shift], target_days: int) -> list[Shift]:
    if not shifts:
        return list(shifts)

    grouped: dict[int, list[Shift]] = {}
    for shift in shifts:
        grouped.setdefault(int(shift.day), []).append(shift)
    current_days = max(grouped.keys(), default=-1) + 1
    if target_days <= current_days:
        return list(shifts)

    template_len = max(1, min(7, current_days))
    template_start = max(0, current_days - template_len)
    template: list[list[Shift]] = []
    for day in range(template_start, current_days):
        template.append(
            [
                Shift(day=0, start_hour=shift.start_hour, hours=shift.hours)
                for shift in sorted(grouped.get(day, []), key=lambda item: item.start_hour)
                if shift.hours > 0
            ]
        )

    if not any(template):
        fallback = sorted(shifts, key=lambda item: (item.day, item.start_hour))
        template = [[Shift(day=0, start_hour=shift.start_hour, hours=shift.hours) for shift in fallback if shift.hours > 0]]

    extended = list(shifts)
    for day in range(current_days, target_days):
        source = template[(day - template_start) % len(template)]
        for shift in source:
            extended.append(Shift(day=day, start_hour=shift.start_hour, hours=shift.hours))
    return extended


def uid(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:6]}"
