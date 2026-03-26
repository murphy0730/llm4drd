from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

from ..core.models import Downtime, OpStatus, ResourceState, ShopFloor, uid
from ..core.simulator import (
    Simulator,
    _joint_compute_effective_end,
    _joint_next_available_time,
    _joint_productive_time,
)
from ..core.rules import BUILTIN_RULES

logger = logging.getLogger(__name__)


@dataclass
class OnlineState:
    current_time: float = 0.0
    rule_name: str = "ATC"
    machine_states: dict = field(default_factory=dict)
    completed_ops: list = field(default_factory=list)
    is_running: bool = False


class OnlineSchedulerV3:
    def __init__(self, shop: ShopFloor, rule_name: str = "ATC"):
        self.base_shop = shop
        self.sim_shop = copy.deepcopy(shop)
        self.sim_shop.build_indexes()
        self.state = OnlineState(rule_name=rule_name, is_running=True)
        for machine in self.sim_shop.machines.values():
            self.state.machine_states[machine.id] = {
                "state": ResourceState.IDLE.name,
                "current_op": None,
                "finish_at": 0.0,
                "breakdown_until": None,
                "next_dispatch_at": 0.0,
            }
        self._promote_ready_ops(0.0)

    def _resource_available_now(self, resource, now: float) -> bool:
        return (
            resource.state == ResourceState.IDLE
            and resource.current_finish_time <= now + 1e-9
            and resource.next_available_time(now) <= now + 1e-9
        )

    def _select_aux_resources(self, op, when: float):
        selected_toolings = []
        for tooling_type in op.required_tooling_types:
            candidates = [
                tooling
                for tooling in self.sim_shop.get_toolings_for_type(tooling_type)
                if tooling.id not in {item.id for item in selected_toolings}
                and self._resource_available_now(tooling, when)
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda tooling: (tooling.total_busy_time, tooling.id))
            selected_toolings.append(candidates[0])

        selected_people = []
        for skill_id in op.required_personnel_skills:
            candidates = [
                person
                for person in self.sim_shop.get_personnel_for_skill(skill_id)
                if person.id not in {item.id for item in selected_people}
                and self._resource_available_now(person, when)
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda person: (person.total_busy_time, person.id))
            selected_people.append(candidates[0])

        return selected_toolings, selected_people

    def _next_resource_ready_time(self, resource, now: float) -> float:
        return resource.next_available_time(max(now, resource.current_finish_time))

    def _next_requirement_ready_time(self, op, when: float) -> float:
        required_times = [when]
        used_toolings: set[str] = set()
        for tooling_type in op.required_tooling_types:
            candidates = [
                tooling
                for tooling in self.sim_shop.get_toolings_for_type(tooling_type)
                if tooling.id not in used_toolings
            ]
            if not candidates:
                return float("inf")
            candidates.sort(key=lambda tooling: (self._next_resource_ready_time(tooling, when), tooling.id))
            selected = candidates[0]
            used_toolings.add(selected.id)
            required_times.append(self._next_resource_ready_time(selected, when))

        used_people: set[str] = set()
        for skill_id in op.required_personnel_skills:
            candidates = [
                person
                for person in self.sim_shop.get_personnel_for_skill(skill_id)
                if person.id not in used_people
            ]
            if not candidates:
                return float("inf")
            candidates.sort(key=lambda person: (self._next_resource_ready_time(person, when), person.id))
            selected = candidates[0]
            used_people.add(selected.id)
            required_times.append(self._next_resource_ready_time(selected, when))

        return max(required_times)

    def _earliest_feasible_start(self, machine, op, not_before: float):
        probe = max(not_before, self.sim_shop.get_operation_release_time(op), machine.current_finish_time)
        for _ in range(1000):
            machine_ready = machine.next_available_time(probe)
            if machine_ready == float("inf"):
                return float("inf"), [], []
            assignment = self._select_aux_resources(op, machine_ready)
            if assignment is not None:
                toolings, people = assignment
                joint_start = _joint_next_available_time([machine, *toolings, *people], machine_ready)
                if joint_start <= machine_ready + 1e-9:
                    return machine_ready, toolings, people
                probe = joint_start
                continue
            next_aux_ready = self._next_requirement_ready_time(op, machine_ready)
            if next_aux_ready == float("inf"):
                return float("inf"), [], []
            if next_aux_ready <= probe + 1e-9:
                next_aux_ready = probe + 0.001
            probe = next_aux_ready
        return float("inf"), [], []

    def _promote_ready_ops(self, now: float) -> None:
        for op in self.sim_shop.operations.values():
            if op.status != OpStatus.PENDING:
                continue
            if self.sim_shop.check_op_ready(op) and self.sim_shop.get_operation_release_time(op) <= now:
                op.status = OpStatus.READY

    def _dispatch_idle_machines(self, now: float, rule_fn) -> None:
        for machine in self.sim_shop.machines.values():
            machine_state = self.state.machine_states[machine.id]
            if machine.state != ResourceState.IDLE:
                machine_state["next_dispatch_at"] = None
                continue
            if machine_state.get("breakdown_until") and machine_state["breakdown_until"] > now:
                machine_state["next_dispatch_at"] = machine_state["breakdown_until"]
                continue

            machine_ready = machine.next_available_time(now)
            if machine_ready > now + 1e-9:
                machine_state["next_dispatch_at"] = machine_ready
                continue

            best = None
            next_dispatch_time = float("inf")
            for op in self.sim_shop.operations.values():
                if op.status != OpStatus.READY:
                    continue
                if machine.id not in {candidate.id for candidate in self.sim_shop.get_eligible_machines(op)}:
                    continue
                start_time, toolings, people = self._earliest_feasible_start(machine, op, now)
                if start_time == float("inf"):
                    continue
                if start_time > now + 1e-9:
                    next_dispatch_time = min(next_dispatch_time, start_time)
                    continue
                task = self.sim_shop.tasks.get(op.task_id)
                order = self.sim_shop.orders.get(task.order_id) if task else None
                due = task.due_date if task else (order.due_date if order else 9999.0)
                remaining = task.remaining_time if task else op.work_remaining
                slack = due - now - remaining
                features = {
                    "slack": slack,
                    "remaining": remaining,
                    "processing_time": op.work_remaining,
                    "due_date": due,
                    "urgency": max(0.0, -slack),
                    "progress": task.progress if task else 0.0,
                    "priority": order.priority if order else 1,
                    "is_main": 1.0 if (task and task.is_main) else 0.0,
                    "wait_time": max(0.0, now - self.sim_shop.get_operation_release_time(op)),
                    "prereq_ratio": 1.0,
                    "machine_busy_time": machine.total_busy_time,
                    "tooling_demand": float(len(op.required_tooling_types)),
                    "personnel_demand": float(len(op.required_personnel_skills)),
                }
                try:
                    score = rule_fn(op, machine, features, self.sim_shop)
                except Exception:
                    score = 0.0
                candidate = (score, -op.work_remaining, op.id, op, toolings, people)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate

            if best is None:
                machine_state["next_dispatch_at"] = next_dispatch_time if next_dispatch_time < float("inf") else None
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

            for resource in resources:
                resource.state = ResourceState.BUSY
                resource.current_op_id = op.id
                resource.current_finish_time = end_time

            machine_state["state"] = ResourceState.BUSY.name
            machine_state["current_op"] = op.id
            machine_state["finish_at"] = end_time
            machine_state["next_dispatch_at"] = None

            task = self.sim_shop.tasks.get(op.task_id)
            order = self.sim_shop.orders.get(task.order_id) if task else None
            self.state.completed_ops.append(
                {
                    "op_id": op.id,
                    "op_name": op.name,
                    "task_id": op.task_id,
                    "machine_id": machine.id,
                    "machine_name": machine.name,
                    "start": round(start_time, 3),
                    "end": round(end_time, 3),
                    "duration": round(productive_duration, 3),
                    "elapsed_duration": round(end_time - start_time, 3),
                    "order_id": order.id if order else "",
                    "tooling_ids": [tool.id for tool in toolings],
                    "personnel_ids": [person.id for person in people],
                    "status": "in_progress",
                }
            )

    def _complete_finished_ops(self, now: float) -> None:
        for machine in self.sim_shop.machines.values():
            if machine.state != ResourceState.BUSY or machine.current_finish_time > now + 1e-9:
                continue
            op = self.sim_shop.operations.get(machine.current_op_id) if machine.current_op_id else None
            if not op:
                continue

            productive_duration = op.remaining_processing_time if op.remaining_processing_time is not None else op.processing_time

            op.status = OpStatus.COMPLETED
            op.end_time = now
            op.remaining_processing_time = 0.0
            task = self.sim_shop.tasks.get(op.task_id)
            if task and task.is_completed:
                task.completion_time = now

            for tooling_id in op.assigned_tooling_ids:
                tooling = self.sim_shop.toolings.get(tooling_id)
                if tooling and tooling.current_op_id == op.id:
                    tooling.state = ResourceState.IDLE
                    tooling.current_op_id = None
                    tooling.current_finish_time = 0.0
                    tooling.total_busy_time += productive_duration
            for person_id in op.assigned_personnel_ids:
                person = self.sim_shop.personnel.get(person_id)
                if person and person.current_op_id == op.id:
                    person.state = ResourceState.IDLE
                    person.current_op_id = None
                    person.current_finish_time = 0.0
                    person.total_busy_time += productive_duration

            machine.state = ResourceState.IDLE
            machine.total_busy_time += productive_duration
            machine.current_op_id = None
            machine.current_finish_time = 0.0

            machine_state = self.state.machine_states[machine.id]
            machine_state["state"] = ResourceState.IDLE.name
            machine_state["current_op"] = None
            machine_state["finish_at"] = 0.0
            machine_state["next_dispatch_at"] = now

            for entry in reversed(self.state.completed_ops):
                if entry["op_id"] == op.id and entry["status"] == "in_progress":
                    entry["status"] = "completed"
                    entry["actual_end"] = round(now, 3)
                    break

        self._promote_ready_ops(now)

    def advance(self, delta: float) -> dict:
        target_time = self.state.current_time + delta
        rule_fn = BUILTIN_RULES.get(self.state.rule_name, BUILTIN_RULES["ATC"])

        while self.state.current_time < target_time - 1e-9:
            now = self.state.current_time
            self._promote_ready_ops(now)
            self._dispatch_idle_machines(now, rule_fn)

            next_times = [target_time]
            for machine_state in self.state.machine_states.values():
                if machine_state["state"] == ResourceState.BUSY.name and machine_state["finish_at"] > now + 1e-9:
                    next_times.append(machine_state["finish_at"])
                if machine_state.get("next_dispatch_at") and machine_state["next_dispatch_at"] > now + 1e-9:
                    next_times.append(machine_state["next_dispatch_at"])
                if machine_state.get("breakdown_until") and machine_state["breakdown_until"] > now + 1e-9:
                    next_times.append(machine_state["breakdown_until"])

            for op in self.sim_shop.operations.values():
                if op.status == OpStatus.PENDING and self.sim_shop.check_op_ready(op):
                    release_time = self.sim_shop.get_operation_release_time(op)
                    if release_time > now + 1e-9:
                        next_times.append(release_time)

            next_time = min(next_times)
            if next_time <= now + 1e-9:
                next_time = target_time

            self.state.current_time = min(next_time, target_time)
            now = self.state.current_time

            for machine in self.sim_shop.machines.values():
                machine_state = self.state.machine_states[machine.id]
                if machine_state.get("breakdown_until") and machine_state["breakdown_until"] <= now + 1e-9:
                    machine_state["breakdown_until"] = None
                    if machine.state == ResourceState.MAINTENANCE:
                        machine.state = ResourceState.IDLE
                        machine.current_finish_time = 0.0
                    machine_state["state"] = machine.state.name
                    machine_state["next_dispatch_at"] = now

            self._complete_finished_ops(now)

        self._promote_ready_ops(self.state.current_time)
        self._dispatch_idle_machines(self.state.current_time, rule_fn)
        return self.get_status()

    def on_breakdown(self, machine_id: str, repair_at: float) -> dict:
        now = self.state.current_time
        machine = self.sim_shop.machines.get(machine_id)
        machine_state = self.state.machine_states.get(machine_id)
        if not machine or not machine_state:
            return self.get_status()

        if repair_at <= now:
            repair_at = now + 0.1

        machine.downtimes.append(
            Downtime(
                id=uid("dt_"),
                machine_id=machine_id,
                downtime_type="unplanned",
                start_time=now,
                end_time=repair_at,
            )
        )
        machine.downtimes.sort(key=lambda dt: (dt.start_time, dt.end_time))

        if machine.current_op_id:
            op = self.sim_shop.operations.get(machine.current_op_id)
            if op and op.status == OpStatus.PROCESSING and op.start_time is not None:
                toolings = [self.sim_shop.toolings[tooling_id] for tooling_id in op.assigned_tooling_ids if tooling_id in self.sim_shop.toolings]
                people = [self.sim_shop.personnel[person_id] for person_id in op.assigned_personnel_ids if person_id in self.sim_shop.personnel]
                allocated_duration = op.remaining_processing_time if op.remaining_processing_time is not None else op.processing_time
                processed = _joint_productive_time([machine, *toolings, *people], op.start_time, now)
                processed = min(processed, allocated_duration)
                op.remaining_processing_time = max(0.001, allocated_duration - processed)
                op.status = OpStatus.READY
                machine.total_busy_time += processed
                for tooling_id in op.assigned_tooling_ids:
                    tooling = self.sim_shop.toolings.get(tooling_id)
                    if tooling and tooling.current_op_id == op.id:
                        tooling.state = ResourceState.IDLE
                        tooling.current_op_id = None
                        tooling.current_finish_time = 0.0
                        tooling.total_busy_time += processed
                for person_id in op.assigned_personnel_ids:
                    person = self.sim_shop.personnel.get(person_id)
                    if person and person.current_op_id == op.id:
                        person.state = ResourceState.IDLE
                        person.current_op_id = None
                        person.current_finish_time = 0.0
                        person.total_busy_time += processed
                op.assigned_machine_id = None
                op.assigned_tooling_ids = []
                op.assigned_personnel_ids = []
                op.start_time = None
                op.end_time = None
                for entry in reversed(self.state.completed_ops):
                    if entry["op_id"] == op.id and entry["status"] == "in_progress":
                        entry["status"] = "interrupted"
                        entry["actual_end"] = round(now, 3)
                        entry["processed_duration"] = round(processed, 3)
                        break

        machine.state = ResourceState.MAINTENANCE
        machine.current_op_id = None
        machine.current_finish_time = repair_at
        machine_state["state"] = ResourceState.MAINTENANCE.name
        machine_state["current_op"] = None
        machine_state["finish_at"] = repair_at
        machine_state["breakdown_until"] = repair_at
        machine_state["next_dispatch_at"] = repair_at

        return self.get_status()

    def on_repair(self, machine_id: str) -> dict:
        now = self.state.current_time
        machine = self.sim_shop.machines.get(machine_id)
        machine_state = self.state.machine_states.get(machine_id)
        if not machine or not machine_state:
            return self.get_status()

        for downtime in reversed(machine.downtimes):
            if downtime.downtime_type == "unplanned" and downtime.end_time > now:
                downtime.end_time = now
                break

        machine.state = ResourceState.IDLE
        machine.current_finish_time = 0.0
        machine_state["state"] = ResourceState.IDLE.name
        machine_state["finish_at"] = 0.0
        machine_state["breakdown_until"] = None
        machine_state["next_dispatch_at"] = now
        return self.get_status()

    def _build_remaining_shop(self) -> ShopFloor:
        remaining_shop = copy.deepcopy(self.sim_shop)
        snapshot = self.state.current_time

        completed_op_ids = {op.id for op in remaining_shop.operations.values() if op.status == OpStatus.COMPLETED}
        completed_task_ids = {task.id for task in remaining_shop.tasks.values() if task.is_completed}

        for op_id in completed_op_ids:
            remaining_shop.operations.pop(op_id, None)
        for task_id in list(remaining_shop.tasks.keys()):
            task = remaining_shop.tasks[task_id]
            task.operations = [op for op in task.operations if op.id not in completed_op_ids]
            if task_id in completed_task_ids:
                remaining_shop.tasks.pop(task_id, None)

        for order_id in list(remaining_shop.orders.keys()):
            order = remaining_shop.orders[order_id]
            order.task_ids = [task_id for task_id in order.task_ids if task_id in remaining_shop.tasks]
            if not order.task_ids:
                remaining_shop.orders.pop(order_id, None)

        for op in remaining_shop.operations.values():
            if op.status == OpStatus.PROCESSING and op.start_time is not None:
                machine = remaining_shop.machines.get(op.assigned_machine_id) if op.assigned_machine_id else None
                toolings = [remaining_shop.toolings[tooling_id] for tooling_id in op.assigned_tooling_ids if tooling_id in remaining_shop.toolings]
                people = [remaining_shop.personnel[person_id] for person_id in op.assigned_personnel_ids if person_id in remaining_shop.personnel]
                resources = [resource for resource in [machine, *toolings, *people] if resource is not None]
                allocated_duration = op.remaining_processing_time if op.remaining_processing_time is not None else op.processing_time
                processed = _joint_productive_time(resources, op.start_time, snapshot) if resources else 0.0
                processed = min(processed, allocated_duration)
                op.remaining_processing_time = max(0.001, allocated_duration - processed)
            op.status = OpStatus.PENDING
            op.assigned_machine_id = None
            op.assigned_tooling_ids = []
            op.assigned_personnel_ids = []
            op.start_time = None
            op.end_time = None
            op.predecessor_ops = [pred for pred in op.predecessor_ops if pred in remaining_shop.operations]
            op.predecessor_tasks = [pred for pred in op.predecessor_tasks if pred in remaining_shop.tasks]

        for task in remaining_shop.tasks.values():
            task.release_time = max(0.0, task.release_time - snapshot)
            task.due_date = max(0.0, task.due_date - snapshot)
            task.completion_time = None
            task.predecessor_task_ids = [pred for pred in task.predecessor_task_ids if pred in remaining_shop.tasks]
        for order in remaining_shop.orders.values():
            order.release_time = max(0.0, order.release_time - snapshot)
            order.due_date = max(0.0, order.due_date - snapshot)
        for resource in [*remaining_shop.machines.values(), *remaining_shop.toolings.values(), *remaining_shop.personnel.values()]:
            resource.state = ResourceState.IDLE
            resource.current_op_id = None
            resource.current_finish_time = 0.0
            resource.downtimes = [
                Downtime(
                    id=dt.id,
                    machine_id=dt.machine_id,
                    downtime_type=dt.downtime_type,
                    start_time=max(0.0, dt.start_time - snapshot),
                    end_time=max(0.0, dt.end_time - snapshot),
                )
                for dt in resource.downtimes
                if dt.end_time > snapshot
            ]
        remaining_shop.plan_start_at = self.sim_shop.offset_to_datetime(snapshot)
        remaining_shop.build_indexes()
        return remaining_shop

    def reschedule(self, rule_name: str = None) -> dict:
        new_rule_name = rule_name or self.state.rule_name
        remaining_shop = self._build_remaining_shop()
        old_rule_name = self.state.rule_name

        old_result = Simulator(remaining_shop, BUILTIN_RULES.get(old_rule_name, BUILTIN_RULES["ATC"])).run()
        new_result = Simulator(remaining_shop, BUILTIN_RULES.get(new_rule_name, BUILTIN_RULES["ATC"])).run()

        old_td = old_result.total_tardiness
        new_td = new_result.total_tardiness
        improvement = (old_td - new_td) / max(old_td, 0.01) * 100
        self.state.rule_name = new_rule_name

        return {
            "old_rule": old_rule_name,
            "new_rule": new_rule_name,
            "old_tardiness": round(old_td, 2),
            "new_tardiness": round(new_td, 2),
            "improvement_pct": round(improvement, 2),
            "current_time": round(self.state.current_time, 3),
            "current_time_at": self.sim_shop.time_label(self.state.current_time),
        }

    def get_status(self) -> dict:
        machine_info = []
        for machine in self.sim_shop.machines.values():
            machine_state = self.state.machine_states[machine.id]
            machine_info.append(
                {
                    "id": machine.id,
                    "name": machine.name,
                    "type": machine.type_id,
                    "state": machine_state["state"],
                    "current_op": machine_state["current_op"],
                    "finish_at": round(machine_state["finish_at"], 3) if machine_state["finish_at"] else 0.0,
                    "finish_at_label": self.sim_shop.time_label(machine_state["finish_at"]) if machine_state["finish_at"] else None,
                    "breakdown_until": round(machine_state["breakdown_until"], 3) if machine_state["breakdown_until"] else None,
                    "breakdown_until_label": self.sim_shop.time_label(machine_state["breakdown_until"]) if machine_state["breakdown_until"] else None,
                }
            )

        total_ops = len(self.sim_shop.operations)
        completed = sum(1 for op in self.sim_shop.operations.values() if op.status == OpStatus.COMPLETED)
        in_progress = sum(1 for op in self.sim_shop.operations.values() if op.status == OpStatus.PROCESSING)
        ready = sum(1 for op in self.sim_shop.operations.values() if op.status == OpStatus.READY)
        gantt = []
        for entry in self.state.completed_ops[-50:]:
            payload = dict(entry)
            payload["start_at"] = self.sim_shop.time_label(entry.get("start"))
            payload["end_at"] = self.sim_shop.time_label(entry.get("end"))
            if entry.get("actual_end") is not None:
                payload["actual_end_at"] = self.sim_shop.time_label(entry.get("actual_end"))
            gantt.append(payload)

        return {
            "current_time": round(self.state.current_time, 3),
            "current_time_at": self.sim_shop.time_label(self.state.current_time),
            "rule": self.state.rule_name,
            "machines": machine_info,
            "toolings": [
                {
                    "id": tooling.id,
                    "name": tooling.name,
                    "type": tooling.type_id,
                    "state": tooling.state.name,
                    "current_op": tooling.current_op_id,
                }
                for tooling in self.sim_shop.toolings.values()
            ],
            "personnel": [
                {
                    "id": person.id,
                    "name": person.name,
                    "skills": person.skills,
                    "state": person.state.name,
                    "current_op": person.current_op_id,
                }
                for person in self.sim_shop.personnel.values()
            ],
            "ops_total": total_ops,
            "ops_completed": completed,
            "ops_in_progress": in_progress,
            "ops_ready": ready,
            "gantt": gantt,
        }
