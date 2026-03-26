from __future__ import annotations

import copy
import heapq
import time as wall_time
from dataclasses import dataclass, field
from typing import Callable

from .models import OpStatus, Operation, ResourceState, ShopFloor

PDRFunc = Callable


@dataclass
class SimResult:
    makespan: float = 0.0
    total_tardiness: float = 0.0
    avg_tardiness: float = 0.0
    max_tardiness: float = 0.0
    tardy_job_count: int = 0
    total_jobs: int = 0
    avg_flowtime: float = 0.0
    main_order_tardy_count: int = 0
    main_order_tardy_total_time: float = 0.0
    main_order_tardy_ratio: float = 0.0
    total_main_orders: int = 0
    avg_utilization: float = 0.0
    critical_utilization: float = 0.0
    total_wait_time: float = 0.0
    avg_wait_time: float = 0.0
    schedule: list = field(default_factory=list)
    wall_time_ms: float = 0.0
    event_count: int = 0

    def to_dict(self) -> dict:
        return {
            key: (round(value, 2) if isinstance(value, float) else value)
            for key, value in self.__dict__.items()
            if key != "schedule"
        }


@dataclass(order=True)
class Event:
    time: float
    seq: int = field(compare=True, default=0)
    event_type: str = field(compare=False, default="")
    data: dict = field(compare=False, default_factory=dict)


class Simulator:
    _seq = 0

    def __init__(self, shop: ShopFloor, pdr: PDRFunc):
        self.orig_shop = shop
        self.pdr = pdr
        self._eligible_machine_ids: dict[str, set[str]] = {}
        self._tooling_candidates: dict[str, dict[str, list]] = {}
        self._personnel_candidates: dict[str, dict[str, list]] = {}
        self._release_time_cache: dict[str, float] = {}
        self._dependent_ops_by_op: dict[str, list[str]] = {}
        self._dependent_ops_by_task: dict[str, list[str]] = {}
        self._ready_by_type: dict[str, set[str]] = {}
        self._dispatch_scheduled_at: dict[str, float | None] = {}
        self._release_checks_scheduled: set[str] = set()
        self._completed_ops: set[str] = set()
        self._completed_tasks: set[str] = set()
        self._task_remaining_ops: dict[str, int] = {}

    def _init_runtime_caches(self, shop: ShopFloor) -> None:
        self._eligible_machine_ids = {}
        self._tooling_candidates = {}
        self._personnel_candidates = {}
        self._release_time_cache = {}
        self._dependent_ops_by_op = {op_id: [] for op_id in shop.operations}
        self._dependent_ops_by_task = {task_id: [] for task_id in shop.tasks}
        self._ready_by_type = {}
        self._dispatch_scheduled_at = {machine_id: None for machine_id in shop.machines}
        self._release_checks_scheduled = set()
        self._completed_ops = set()
        self._completed_tasks = set()
        self._task_remaining_ops = {task_id: len(task.operations) for task_id, task in shop.tasks.items()}

        for op_id, op in shop.operations.items():
            self._eligible_machine_ids[op_id] = {machine.id for machine in shop.get_eligible_machines(op)}
            self._tooling_candidates[op_id] = {
                tooling_type: list(shop.get_toolings_for_type(tooling_type))
                for tooling_type in op.required_tooling_types
            }
            self._personnel_candidates[op_id] = {
                skill_id: list(shop.get_personnel_for_skill(skill_id))
                for skill_id in op.required_personnel_skills
            }
            self._release_time_cache[op_id] = shop.get_operation_release_time(op)
            for predecessor_id in op.predecessor_ops:
                self._dependent_ops_by_op.setdefault(predecessor_id, []).append(op_id)
            for predecessor_task_id in op.predecessor_tasks:
                self._dependent_ops_by_task.setdefault(predecessor_task_id, []).append(op_id)

    def run(self, max_time: float = 999999) -> SimResult:
        started_at = wall_time.time()
        shop = copy.deepcopy(self.orig_shop)
        shop.build_indexes()
        self._init_runtime_caches(shop)

        event_queue: list[Event] = []
        ready_ops: set[str] = set()
        schedule: list[dict] = []
        event_count = 0
        completed_ops = 0
        self._seq = 0

        completed_ops += self._seed_initial_state(shop, ready_ops, event_queue, schedule)

        for op in shop.operations.values():
            if op.status == OpStatus.PENDING and self._is_op_ready(op):
                self._queue_release_or_ready(shop, op, ready_ops, event_queue)

        for machine_id in shop.machines:
            self._schedule_dispatch(event_queue, machine_id, 0.0)

        now = 0.0
        while event_queue and event_count < 5_000_000:
            event = heapq.heappop(event_queue)
            if event.time > max_time:
                break
            now = event.time
            event_count += 1

            if event.event_type == "release_check":
                op = shop.operations.get(event.data["op_id"])
                if op:
                    self._release_checks_scheduled.discard(op.id)
                if op and op.status == OpStatus.PENDING and self._is_op_ready(op):
                    if self._release_time_cache.get(op.id, shop.get_operation_release_time(op)) <= now:
                        self._mark_ready(op, ready_ops)
                        self._trigger_idle_dispatches(shop, event_queue, now, process_type=op.process_type)

            elif event.event_type == "dispatch":
                machine = shop.machines.get(event.data["machine_id"])
                scheduled = self._dispatch_scheduled_at.get(event.data["machine_id"])
                if scheduled is None or abs(scheduled - event.time) > 1e-9:
                    continue
                self._dispatch_scheduled_at[event.data["machine_id"]] = None
                if not machine or machine.state != ResourceState.IDLE:
                    continue
                if machine.current_finish_time > now:
                    self._schedule_dispatch(event_queue, machine.id, machine.current_finish_time)
                    continue

                machine_available = machine.next_available_time(now)
                if machine_available == float("inf"):
                    continue
                if machine_available > now + 1e-9:
                    self._schedule_dispatch(event_queue, machine.id, machine_available)
                    continue

                best: tuple[float, float, str, Operation, list, list] | None = None
                next_dispatch_time = float("inf")
                candidate_ids = list(self._ready_by_type.get(machine.type_id, set()))
                for op_id in candidate_ids:
                    op = shop.operations.get(op_id)
                    if not op or op.status != OpStatus.READY:
                        self._discard_ready(op_id, machine.type_id, ready_ops)
                        continue
                    if machine.id not in self._eligible_machine_ids.get(op.id, set()):
                        continue

                    start_time, toolings, people = self._earliest_feasible_start(shop, machine, op, now)
                    if start_time == float("inf"):
                        continue
                    if start_time > now + 1e-9:
                        next_dispatch_time = min(next_dispatch_time, start_time)
                        continue

                    task = shop.tasks.get(op.task_id)
                    order = shop.orders.get(task.order_id) if task else None
                    features = self._features(op, task, order, machine, shop, now)
                    try:
                        score = self.pdr(op, machine, features, shop)
                    except Exception:
                        score = 0.0
                    tie_break = op.work_remaining
                    candidate = (score, -tie_break, op.id, op, toolings, people)
                    if best is None or candidate[:3] > best[:3]:
                        best = candidate

                if best is None:
                    if next_dispatch_time < float("inf"):
                        self._schedule_dispatch(event_queue, machine.id, next_dispatch_time)
                    continue

                _, _, _, op, toolings, people = best
                resources = [machine, *toolings, *people]
                start_time = _joint_next_available_time(resources, now)
                productive_duration = op.work_remaining
                end_time = _joint_compute_effective_end(resources, start_time, productive_duration)

                op.status = OpStatus.PROCESSING
                op.remaining_processing_time = productive_duration
                op.assigned_machine_id = machine.id
                op.assigned_tooling_ids = [tool.id for tool in toolings]
                op.assigned_personnel_ids = [person.id for person in people]
                op.start_time = start_time
                op.end_time = end_time
                self._discard_ready(op.id, op.process_type, ready_ops)

                for resource in resources:
                    resource.state = ResourceState.BUSY
                    resource.current_op_id = op.id
                    resource.current_finish_time = end_time

                schedule.append(
                    {
                        "op_id": op.id,
                        "op_name": op.name,
                        "task_id": op.task_id,
                        "machine_id": machine.id,
                        "machine_name": machine.name,
                        "process_type": op.process_type,
                        "start": round(start_time, 3),
                        "end": round(end_time, 3),
                        "duration": round(productive_duration, 3),
                        "elapsed_duration": round(end_time - start_time, 3),
                        "tooling_ids": [tool.id for tool in toolings],
                        "personnel_ids": [person.id for person in people],
                    }
                )
                self._push(
                    event_queue,
                    end_time,
                    "op_done",
                    op_id=op.id,
                    machine_id=machine.id,
                    tooling_ids=[tool.id for tool in toolings],
                    personnel_ids=[person.id for person in people],
                    productive_duration=productive_duration,
                )

            elif event.event_type == "op_done":
                op = shop.operations.get(event.data["op_id"])
                machine = shop.machines.get(event.data["machine_id"])
                toolings = [shop.toolings[tid] for tid in event.data.get("tooling_ids", []) if tid in shop.toolings]
                people = [shop.personnel[pid] for pid in event.data.get("personnel_ids", []) if pid in shop.personnel]
                resources = [resource for resource in [machine, *toolings, *people] if resource is not None]
                newly_ready_types: set[str] = set()

                if op:
                    op.status = OpStatus.COMPLETED
                    op.end_time = now
                    op.remaining_processing_time = 0.0
                    self._completed_ops.add(op.id)
                    completed_ops += 1
                    task = shop.tasks.get(op.task_id)
                    impacted_ops = set(self._dependent_ops_by_op.get(op.id, []))
                    if task:
                        self._task_remaining_ops[task.id] = max(0, self._task_remaining_ops.get(task.id, 1) - 1)
                    if task and self._task_remaining_ops.get(task.id, 0) == 0:
                        task.completion_time = now
                        self._completed_tasks.add(task.id)
                        impacted_ops.update(self._dependent_ops_by_task.get(task.id, []))
                    for next_op_id in impacted_ops:
                        next_op = shop.operations.get(next_op_id)
                        if next_op and next_op.status == OpStatus.PENDING and self._is_op_ready(next_op):
                            self._queue_release_or_ready(shop, next_op, ready_ops, event_queue)
                            if next_op.status == OpStatus.READY:
                                newly_ready_types.add(next_op.process_type)

                productive_duration = event.data.get("productive_duration", 0.0)
                per_resource_work = productive_duration
                for resource in resources:
                    resource.total_busy_time += per_resource_work
                    resource.state = ResourceState.IDLE
                    resource.current_op_id = None
                    resource.current_finish_time = 0.0

                if machine:
                    self._schedule_dispatch(event_queue, machine.id, now)
                    for process_type in newly_ready_types:
                        self._trigger_idle_dispatches(shop, event_queue, now, process_type=process_type)

            if completed_ops >= len(shop.operations):
                break

        return self._compute_kpi(shop, schedule, wall_time.time() - started_at, event_count)

    def _seed_initial_state(
        self,
        shop: ShopFloor,
        ready_ops: set[str],
        event_queue: list[Event],
        schedule: list[dict],
    ) -> int:
        completed_count = 0

        for task in shop.tasks.values():
            self._task_remaining_ops[task.id] = len(task.operations)

        for op in shop.operations.values():
            if op.status == OpStatus.COMPLETED:
                completed_count += 1
                self._completed_ops.add(op.id)
                self._task_remaining_ops[op.task_id] = max(0, self._task_remaining_ops.get(op.task_id, 1) - 1)
                op.remaining_processing_time = 0.0

        for task in shop.tasks.values():
            if self._task_remaining_ops.get(task.id, len(task.operations)) == 0:
                self._completed_tasks.add(task.id)
                task.completion_time = max((op.end_time or 0.0) for op in task.operations) if task.operations else 0.0

        for op in shop.operations.values():
            if op.status == OpStatus.READY:
                if self._is_op_ready(op) and self._release_time_cache.get(op.id, shop.get_operation_release_time(op)) <= 0:
                    self._mark_ready(op, ready_ops)
                else:
                    op.status = OpStatus.PENDING

        for op in shop.operations.values():
            if op.status != OpStatus.PROCESSING:
                continue
            machine = shop.machines.get(op.assigned_machine_id) if op.assigned_machine_id else None
            toolings = [shop.toolings[tooling_id] for tooling_id in op.assigned_tooling_ids if tooling_id in shop.toolings]
            people = [shop.personnel[person_id] for person_id in op.assigned_personnel_ids if person_id in shop.personnel]
            resources = [resource for resource in [machine, *toolings, *people] if resource is not None]
            if machine is None or not resources:
                op.status = OpStatus.READY if self._is_op_ready(op) else OpStatus.PENDING
                op.assigned_machine_id = None
                op.assigned_tooling_ids = []
                op.assigned_personnel_ids = []
                op.start_time = None
                op.end_time = None
                op.remaining_processing_time = None
                if op.status == OpStatus.READY:
                    self._mark_ready(op, ready_ops)
                continue

            productive_duration = op.remaining_processing_time if op.remaining_processing_time is not None else op.processing_time
            productive_duration = max(0.001, productive_duration)
            start_time = op.start_time if op.start_time is not None else 0.0
            end_time = op.end_time if op.end_time is not None else _joint_compute_effective_end(resources, 0.0, productive_duration)
            op.remaining_processing_time = productive_duration
            op.start_time = start_time
            op.end_time = end_time

            for resource in resources:
                resource.state = ResourceState.BUSY
                resource.current_op_id = op.id
                resource.current_finish_time = end_time

            schedule.append(
                {
                    "op_id": op.id,
                    "op_name": op.name,
                    "task_id": op.task_id,
                    "machine_id": machine.id,
                    "machine_name": machine.name,
                    "process_type": op.process_type,
                    "start": round(start_time, 3),
                    "end": round(end_time, 3),
                    "duration": round(productive_duration, 3),
                    "elapsed_duration": round(end_time - start_time, 3),
                    "tooling_ids": [tool.id for tool in toolings],
                    "personnel_ids": [person.id for person in people],
                    "status": "in_progress",
                }
            )
            self._push(
                event_queue,
                end_time,
                "op_done",
                op_id=op.id,
                machine_id=machine.id,
                tooling_ids=[tool.id for tool in toolings],
                personnel_ids=[person.id for person in people],
                productive_duration=productive_duration,
            )

        return completed_count

    def _push(self, event_queue: list[Event], when: float, event_type: str, **data) -> None:
        self._seq += 1
        heapq.heappush(event_queue, Event(when, self._seq, event_type, data))

    def _schedule_dispatch(self, event_queue: list[Event], machine_id: str, when: float) -> None:
        existing = self._dispatch_scheduled_at.get(machine_id)
        if existing is not None and existing <= when + 1e-9:
            return
        self._dispatch_scheduled_at[machine_id] = when
        self._push(event_queue, when, "dispatch", machine_id=machine_id)

    def _mark_ready(self, op: Operation, ready_ops: set[str]) -> None:
        op.status = OpStatus.READY
        ready_ops.add(op.id)
        self._ready_by_type.setdefault(op.process_type, set()).add(op.id)

    def _discard_ready(self, op_id: str, process_type: str, ready_ops: set[str]) -> None:
        ready_ops.discard(op_id)
        bucket = self._ready_by_type.get(process_type)
        if bucket is not None:
            bucket.discard(op_id)

    def _is_op_ready(self, op: Operation) -> bool:
        for predecessor_id in op.predecessor_ops:
            if predecessor_id not in self._completed_ops:
                return False
        for predecessor_task_id in op.predecessor_tasks:
            if predecessor_task_id not in self._completed_tasks:
                return False
        return True

    def _queue_release_or_ready(
        self,
        shop: ShopFloor,
        op: Operation,
        ready_ops: set[str],
        event_queue: list[Event],
    ) -> None:
        release_time = self._release_time_cache.get(op.id, shop.get_operation_release_time(op))
        if release_time <= 0:
            self._mark_ready(op, ready_ops)
            return
        if op.id in self._release_checks_scheduled:
            return
        self._release_checks_scheduled.add(op.id)
        self._push(event_queue, release_time, "release_check", op_id=op.id)

    def _trigger_idle_dispatches(
        self,
        shop: ShopFloor,
        event_queue: list[Event],
        now: float,
        process_type: str | None = None,
    ) -> None:
        for machine in shop.machines.values():
            if machine.state != ResourceState.IDLE:
                continue
            if process_type and machine.type_id != process_type:
                continue
            self._schedule_dispatch(event_queue, machine.id, now)

    def _next_resource_ready_time(self, resource, now: float) -> float:
        base = max(now, getattr(resource, "current_finish_time", 0.0))
        return resource.next_available_time(base)

    def _resource_ready_at(self, resource, when: float, cache: dict[tuple[str, float], float]) -> float:
        key = (getattr(resource, "id", str(id(resource))), when)
        if key not in cache:
            cache[key] = self._next_resource_ready_time(resource, when)
        return cache[key]

    def _select_aux_resources(self, shop: ShopFloor, op: Operation, when: float, ready_cache: dict[tuple[str, float], float] | None = None):
        ready_cache = ready_cache or {}
        selected_toolings = []
        used_toolings: set[str] = set()
        for tooling_type in op.required_tooling_types:
            best_tooling = None
            best_key = None
            for tooling in self._tooling_candidates.get(op.id, {}).get(tooling_type, []):
                if tooling.id in used_toolings:
                    continue
                if tooling.current_finish_time > when + 1e-9:
                    continue
                ready_at = self._resource_ready_at(tooling, when, ready_cache)
                if ready_at > when + 1e-9:
                    continue
                candidate_key = (tooling.total_busy_time, tooling.id)
                if best_key is None or candidate_key < best_key:
                    best_key = candidate_key
                    best_tooling = tooling
            if best_tooling is None:
                return None
            selected_toolings.append(best_tooling)
            used_toolings.add(best_tooling.id)

        selected_people = []
        used_people: set[str] = set()
        for skill_id in op.required_personnel_skills:
            best_person = None
            best_key = None
            for person in self._personnel_candidates.get(op.id, {}).get(skill_id, []):
                if person.id in used_people:
                    continue
                if person.current_finish_time > when + 1e-9:
                    continue
                ready_at = self._resource_ready_at(person, when, ready_cache)
                if ready_at > when + 1e-9:
                    continue
                candidate_key = (person.total_busy_time, person.id)
                if best_key is None or candidate_key < best_key:
                    best_key = candidate_key
                    best_person = person
            if best_person is None:
                return None
            selected_people.append(best_person)
            used_people.add(best_person.id)

        return selected_toolings, selected_people

    def _next_requirement_ready_time(self, shop: ShopFloor, op: Operation, when: float, ready_cache: dict[tuple[str, float], float] | None = None) -> float:
        ready_cache = ready_cache or {}
        required_times = [when]
        used_toolings: set[str] = set()
        for tooling_type in op.required_tooling_types:
            selected = None
            selected_ready = float("inf")
            for tooling in self._tooling_candidates.get(op.id, {}).get(tooling_type, []):
                if tooling.id in used_toolings:
                    continue
                ready_at = self._resource_ready_at(tooling, when, ready_cache)
                candidate_key = (ready_at, tooling.id)
                if selected is None or candidate_key < (selected_ready, selected.id):
                    selected = tooling
                    selected_ready = ready_at
            if selected is None:
                return float("inf")
            used_toolings.add(selected.id)
            required_times.append(selected_ready)

        used_people: set[str] = set()
        for skill_id in op.required_personnel_skills:
            selected = None
            selected_ready = float("inf")
            for person in self._personnel_candidates.get(op.id, {}).get(skill_id, []):
                if person.id in used_people:
                    continue
                ready_at = self._resource_ready_at(person, when, ready_cache)
                candidate_key = (ready_at, person.id)
                if selected is None or candidate_key < (selected_ready, selected.id):
                    selected = person
                    selected_ready = ready_at
            if selected is None:
                return float("inf")
            used_people.add(selected.id)
            required_times.append(selected_ready)

        return max(required_times)

    def _earliest_feasible_start(self, shop: ShopFloor, machine, op: Operation, not_before: float):
        probe = max(not_before, self._release_time_cache.get(op.id, shop.get_operation_release_time(op)), machine.current_finish_time)
        ready_cache: dict[tuple[str, float], float] = {}
        for _ in range(1000):
            machine_ready = machine.next_available_time(probe)
            if machine_ready == float("inf"):
                return float("inf"), [], []
            assignment = self._select_aux_resources(shop, op, machine_ready, ready_cache)
            if assignment is not None:
                toolings, people = assignment
                joint_start = _joint_next_available_time([machine, *toolings, *people], machine_ready)
                if joint_start <= machine_ready + 1e-9:
                    return machine_ready, toolings, people
                probe = joint_start
                continue
            next_aux_ready = self._next_requirement_ready_time(shop, op, machine_ready, ready_cache)
            if next_aux_ready == float("inf"):
                return float("inf"), [], []
            if next_aux_ready <= probe + 1e-9:
                next_aux_ready = probe + 0.001
            probe = next_aux_ready
        return float("inf"), [], []

    def _features(self, op, task, order, machine, shop, now: float) -> dict:
        external_due = task.due_date if task else (order.due_date if order else 9999.0)
        due = op.derived_due_date if op and op.derived_due_date < float("inf") else external_due
        remaining = task.remaining_time if task else op.work_remaining
        slack = due - now - remaining
        progress = task.progress if task else 0.0
        priority = order.priority if order else 1
        release_time = self._release_time_cache.get(op.id, shop.get_operation_release_time(op))

        prereq_done = 0
        prereq_total = len(op.predecessor_tasks) + len(op.predecessor_ops)
        if prereq_total > 0:
            for predecessor_task_id in op.predecessor_tasks:
                if predecessor_task_id in self._completed_tasks:
                    prereq_done += 1
            for predecessor_op_id in op.predecessor_ops:
                if predecessor_op_id in self._completed_ops:
                    prereq_done += 1

        return {
            "slack": slack,
            "remaining": remaining,
            "processing_time": op.work_remaining,
            "due_date": due,
            "external_due_date": external_due,
            "task_due_date": task.due_date if task else external_due,
            "op_due_date": due,
            "urgency": max(0.0, -slack),
            "progress": progress,
            "priority": priority,
            "is_main": 1.0 if (task and task.is_main) else 0.0,
            "wait_time": max(0.0, now - release_time),
            "prereq_ratio": prereq_done / prereq_total if prereq_total > 0 else 1.0,
            "machine_busy_time": machine.total_busy_time,
            "tooling_demand": float(len(op.required_tooling_types)),
            "personnel_demand": float(len(op.required_personnel_skills)),
            "critical_slack": op.critical_slack if op else float("inf"),
        }

    def _compute_kpi(self, shop: ShopFloor, schedule: list, elapsed_seconds: float, event_count: int) -> SimResult:
        result = SimResult()
        result.schedule = schedule
        result.wall_time_ms = round(elapsed_seconds * 1000, 1)
        result.event_count = event_count
        result.total_jobs = len(shop.operations)

        tardiness_list: list[float] = []
        flowtime_list: list[float] = []
        wait_list: list[float] = []
        max_end = 0.0

        for task in shop.tasks.values():
            if task.completion_time is None:
                continue
            tardiness = max(0.0, task.completion_time - task.due_date)
            tardiness_list.append(tardiness)
            if tardiness > 0:
                result.tardy_job_count += 1
            flowtime = task.completion_time - task.release_time
            flowtime_list.append(flowtime)
            productive = sum(op.processing_time for op in task.operations)
            wait_list.append(max(0.0, flowtime - productive))
            max_end = max(max_end, task.completion_time)

        result.makespan = max_end
        result.total_tardiness = sum(tardiness_list)
        result.avg_tardiness = result.total_tardiness / len(tardiness_list) if tardiness_list else 0.0
        result.max_tardiness = max(tardiness_list) if tardiness_list else 0.0
        result.avg_flowtime = sum(flowtime_list) / len(flowtime_list) if flowtime_list else 0.0
        result.total_wait_time = sum(wait_list)
        result.avg_wait_time = result.total_wait_time / len(wait_list) if wait_list else 0.0

        for order in shop.orders.values():
            if not order.main_task_id or order.main_task_id not in shop.tasks:
                continue
            result.total_main_orders += 1
            main_task = shop.tasks[order.main_task_id]
            if main_task.completion_time is not None and main_task.completion_time > order.due_date:
                result.main_order_tardy_count += 1
                result.main_order_tardy_total_time += main_task.completion_time - order.due_date
        result.main_order_tardy_ratio = (
            result.main_order_tardy_count / result.total_main_orders
            if result.total_main_orders
            else 0.0
        )

        if max_end > 0:
            machine_utils = [machine.total_busy_time / max_end for machine in shop.machines.values()]
            result.avg_utilization = sum(machine_utils) / len(machine_utils) if machine_utils else 0.0
            critical = shop.get_critical_machines()
            if critical:
                result.critical_utilization = sum(machine.total_busy_time / max_end for machine in critical) / len(critical)
            else:
                result.critical_utilization = result.avg_utilization

        return result


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
    if end <= start:
        return 0.0

    productive = 0.0
    when = _joint_next_available_time(resources, start)
    for _ in range(10000):
        if when == float("inf") or when >= end - 1e-9:
            return productive
        unavailable = _joint_next_unavailable_time(resources, when)
        productive += max(0.0, min(unavailable, end) - when)
        if unavailable >= end - 1e-9:
            return productive
        when = _joint_next_available_time(resources, unavailable)
    return productive
