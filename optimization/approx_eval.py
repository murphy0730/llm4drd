from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from ..core.models import Operation, ShopFloor
from ..core.rules import BUILTIN_RULES
from ..core.simulator import SimResult
from .objectives import ScheduleAnalytics, build_schedule_analytics
from .solution_model import CandidateParameters, FEATURE_NAMES, OptimizationSolution


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


@dataclass
class ApproximateEvaluation:
    analytics: ScheduleAnalytics
    metrics: dict[str, float]
    schedule_signature: str
    stored_schedule: list[dict]


class ApproximateScheduleEvaluator:
    def __init__(
        self,
        shop: ShopFloor,
        graph_features: dict[str, dict[str, float]],
        time_scale: float,
        due_scale: float,
        priority_scale: float,
        keep_schedule_limit: int = 0,
    ):
        self.shop = shop
        self.graph_features = graph_features
        self.time_scale = max(1.0, time_scale)
        self.due_scale = max(1.0, due_scale)
        self.priority_scale = max(1.0, priority_scale)
        self.keep_schedule_limit = max(0, keep_schedule_limit)
        self._operation_release = {
            op_id: self.shop.get_operation_release_time(op)
            for op_id, op in self.shop.operations.items()
        }
        self._eligible_machines = {
            op_id: list(self.shop.get_eligible_machines(op))
            for op_id, op in self.shop.operations.items()
        }
        self._tooling_candidates = {
            op_id: {
                tooling_type: list(self.shop.get_toolings_for_type(tooling_type))
                for tooling_type in op.required_tooling_types
            }
            for op_id, op in self.shop.operations.items()
        }
        self._personnel_candidates = {
            op_id: {
                skill_id: list(self.shop.get_personnel_for_skill(skill_id))
                for skill_id in op.required_personnel_skills
            }
            for op_id, op in self.shop.operations.items()
        }
        self._remaining_task_work_base = {
            task_id: sum(op.processing_time for op in task.operations)
            for task_id, task in self.shop.tasks.items()
        }
        self._successors_by_op = {op_id: [] for op_id in self.shop.operations}
        self._successors_by_task = {task_id: [] for task_id in self.shop.tasks}
        self._combined_predecessors = {}
        for op_id, op in self.shop.operations.items():
            combined = list(op.predecessor_ops) + list(op.predecessor_tasks)
            self._combined_predecessors[op_id] = len(combined)
            for predecessor_op_id in op.predecessor_ops:
                self._successors_by_op.setdefault(predecessor_op_id, []).append(op_id)
            for predecessor_task_id in op.predecessor_tasks:
                self._successors_by_task.setdefault(predecessor_task_id, []).append(op_id)

    def _feature_map(
        self,
        op: Operation,
        base_ready: float,
        remaining_task_work: dict[str, float],
        predecessor_remaining: dict[str, int],
        machine_ready_time: dict[str, float],
    ) -> dict[str, float]:
        task = self.shop.tasks[op.task_id]
        order = self.shop.orders[task.order_id]
        total_preds = max(1, len(op.predecessor_ops) + len(op.predecessor_tasks))
        preds_done = total_preds - predecessor_remaining.get(op.id, 0)
        eligible = self._eligible_machines.get(op.id, [])
        avg_machine_ready = (
            sum(machine_ready_time.get(machine.id, 0.0) for machine in eligible) / len(eligible)
            if eligible
            else base_ready
        )
        remaining = max(0.0, remaining_task_work.get(task.id, op.processing_time))
        productive_remaining = max(0.0, remaining - op.processing_time)
        slack = task.due_date - base_ready - productive_remaining
        return {
            "urgency": max(0.0, order.due_date - base_ready),
            "slack": slack,
            "remaining": productive_remaining,
            "processing_time": op.processing_time,
            "priority": float(order.priority),
            "is_main": 1.0 if task.is_main else 0.0,
            "wait_time": max(0.0, base_ready - self._operation_release.get(op.id, 0.0)),
            "prereq_ratio": preds_done / total_preds,
            "machine_busy_time": avg_machine_ready,
            "tooling_demand": float(len(op.required_tooling_types)),
            "personnel_demand": float(len(op.required_personnel_skills)),
            "due_date": task.due_date,
        }

    def _priority_score(
        self,
        candidate: CandidateParameters,
        op: Operation,
        base_ready: float,
        remaining_task_work: dict[str, float],
        predecessor_remaining: dict[str, int],
        machine_ready_time: dict[str, float],
    ) -> float:
        features = self._feature_map(op, base_ready, remaining_task_work, predecessor_remaining, machine_ready_time)
        graph_values = self.graph_features.get(op.id, {})
        score_components = {
            "urgency": features["urgency"] / self.time_scale,
            "slack": -features["slack"] / self.time_scale,
            "remaining": -features["remaining"] / self.time_scale,
            "processing_time": -features["processing_time"] / self.time_scale,
            "priority": features["priority"] / self.priority_scale,
            "is_main": features["is_main"],
            "wait_time": features["wait_time"] / self.time_scale,
            "prereq_ratio": features["prereq_ratio"],
            "machine_load": -features["machine_busy_time"] / self.time_scale,
            "tooling_demand": -features["tooling_demand"],
            "personnel_demand": -features["personnel_demand"],
            "predecessor_depth": graph_values.get("predecessor_depth", 0.0),
            "assembly_criticality": graph_values.get("assembly_criticality", 0.0),
            "shared_resource_degree": -graph_values.get("shared_resource_degree", 0.0),
            "bottleneck_adjacency": graph_values.get("bottleneck_adjacency", 0.0),
            "due_date": -features["due_date"] / self.due_scale,
        }
        score = sum(candidate.feature_weights.get(name, 0.0) * score_components.get(name, 0.0) for name in FEATURE_NAMES)
        score += candidate.op_bias.get(op.id, 0.0)
        builtin = BUILTIN_RULES.get(candidate.seed_rule_name or "")
        if builtin is not None:
            try:
                eligible = self._eligible_machines.get(op.id) or [None]
                score += 0.12 * builtin(op, eligible[0], features, self.shop)
            except Exception:
                pass
        score -= 0.06 * (base_ready / self.time_scale)
        return score

    def _select_auxiliary_group(
        self,
        candidates_by_key: dict[str, list],
        keys: list[str],
        not_before: float,
        ready_times: dict[str, float],
    ) -> tuple[list, float] | None:
        selected = []
        used_ids: set[str] = set()
        probe = not_before
        for _ in range(16):
            selected.clear()
            used_ids.clear()
            shifted = False
            for key in keys:
                options = candidates_by_key.get(key, [])
                best_resource = None
                best_ready = float("inf")
                for resource in options:
                    if resource.id in used_ids:
                        continue
                    ready = resource.next_available_time(max(probe, ready_times.get(resource.id, 0.0)))
                    if ready < best_ready - 1e-9 or (
                        abs(ready - best_ready) <= 1e-9 and resource.id < getattr(best_resource, "id", resource.id)
                    ):
                        best_resource = resource
                        best_ready = ready
                if best_resource is None or best_ready == float("inf"):
                    return None
                selected.append(best_resource)
                used_ids.add(best_resource.id)
                if best_ready > probe + 1e-9:
                    probe = best_ready
                    shifted = True
            if not shifted:
                return list(selected), probe
        return None

    def _plan_operation(
        self,
        op: Operation,
        predecessor_completion: dict[str, float],
        task_completion: dict[str, float],
        machine_ready_time: dict[str, float],
        tooling_ready_time: dict[str, float],
        personnel_ready_time: dict[str, float],
    ) -> dict | None:
        base_ready = self._operation_release.get(op.id, 0.0)
        if op.predecessor_ops:
            base_ready = max(base_ready, max(predecessor_completion.get(pred_id, 0.0) for pred_id in op.predecessor_ops))
        if op.predecessor_tasks:
            task_ready = max(task_completion.get(task_id, float("inf")) for task_id in op.predecessor_tasks)
            if task_ready == float("inf"):
                return None
            base_ready = max(base_ready, task_ready)

        best_plan = None
        for machine in self._eligible_machines.get(op.id, []):
            machine_probe = max(base_ready, machine_ready_time.get(machine.id, 0.0))
            toolings = []
            people = []
            probe = machine_probe
            tooling_group = self._select_auxiliary_group(
                self._tooling_candidates.get(op.id, {}),
                op.required_tooling_types,
                probe,
                tooling_ready_time,
            )
            if tooling_group is None and op.required_tooling_types:
                continue
            if tooling_group is not None:
                toolings, probe = tooling_group
            personnel_group = self._select_auxiliary_group(
                self._personnel_candidates.get(op.id, {}),
                op.required_personnel_skills,
                probe,
                personnel_ready_time,
            )
            if personnel_group is None and op.required_personnel_skills:
                continue
            if personnel_group is not None:
                people, probe = personnel_group

            resources = [machine] + toolings + people
            start = _joint_next_available_time(resources, probe)
            if start == float("inf"):
                continue
            end = _joint_compute_effective_end(resources, start, op.processing_time)
            if end == float("inf"):
                continue
            plan = {
                "machine": machine,
                "toolings": toolings,
                "people": people,
                "start": start,
                "end": end,
            }
            if best_plan is None or end < best_plan["end"] - 1e-9 or (
                abs(end - best_plan["end"]) <= 1e-9 and start < best_plan["start"] - 1e-9
            ):
                best_plan = plan
        return best_plan

    def evaluate(
        self,
        candidate: CandidateParameters,
        source: str,
        generation: int,
    ) -> OptimizationSolution:
        predecessor_remaining = dict(self._combined_predecessors)
        task_remaining_ops = {
            task_id: len(task.operations)
            for task_id, task in self.shop.tasks.items()
        }
        remaining_task_work = dict(self._remaining_task_work_base)
        predecessor_completion: dict[str, float] = {}
        task_completion: dict[str, float] = {}
        machine_ready_time = {machine_id: 0.0 for machine_id in self.shop.machines}
        tooling_ready_time = {tooling_id: 0.0 for tooling_id in self.shop.toolings}
        personnel_ready_time = {person_id: 0.0 for person_id in self.shop.personnel}

        scheduled_ops: set[str] = set()
        ready_heap: list[tuple[float, float, str]] = []
        inserted_ops: set[str] = set()
        machine_busy: dict[str, float] = {}
        tooling_busy: dict[str, float] = {}
        personnel_busy: dict[str, float] = {}
        schedule: list[dict] = []

        def push_ready(op_id: str) -> None:
            if op_id in inserted_ops or op_id in scheduled_ops:
                return
            op = self.shop.operations[op_id]
            base_ready = self._operation_release.get(op_id, 0.0)
            if op.predecessor_ops:
                base_ready = max(base_ready, max(predecessor_completion.get(pred_id, 0.0) for pred_id in op.predecessor_ops))
            if op.predecessor_tasks:
                base_ready = max(base_ready, max(task_completion.get(task_id, 0.0) for task_id in op.predecessor_tasks))
            score = self._priority_score(candidate, op, base_ready, remaining_task_work, predecessor_remaining, machine_ready_time)
            heapq.heappush(ready_heap, (-score, base_ready, op_id))
            inserted_ops.add(op_id)

        for op_id, count in predecessor_remaining.items():
            if count <= 0:
                push_ready(op_id)

        beam_width = 5
        while ready_heap:
            beam: list[tuple[float, float, str]] = []
            while ready_heap and len(beam) < beam_width:
                entry = heapq.heappop(ready_heap)
                if entry[2] in scheduled_ops:
                    continue
                beam.append(entry)
            if not beam:
                break

            best_choice = None
            search_entries = list(beam)
            while True:
                for neg_score, _, op_id in search_entries:
                    op = self.shop.operations[op_id]
                    plan = self._plan_operation(
                        op,
                        predecessor_completion,
                        task_completion,
                        machine_ready_time,
                        tooling_ready_time,
                        personnel_ready_time,
                    )
                    if plan is None:
                        continue
                    score = -neg_score
                    value = score - 0.12 * (plan["start"] / self.time_scale) - 0.08 * (plan["end"] / self.time_scale)
                    choice = (value, -score, -plan["end"], -plan["start"], op_id, plan)
                    if best_choice is None or choice > best_choice:
                        best_choice = choice
                if best_choice is not None or not ready_heap or len(search_entries) >= 24:
                    break
                entry = heapq.heappop(ready_heap)
                if entry[2] not in scheduled_ops:
                    search_entries.append(entry)

            if best_choice is None:
                break

            chosen_op_id = best_choice[4]
            chosen_plan = best_choice[5]
            chosen_op = self.shop.operations[chosen_op_id]

            scheduled_ops.add(chosen_op_id)
            inserted_ops.discard(chosen_op_id)

            start = chosen_plan["start"]
            end = chosen_plan["end"]
            machine = chosen_plan["machine"]
            toolings = chosen_plan["toolings"]
            people = chosen_plan["people"]

            machine_ready_time[machine.id] = end
            machine_busy[machine.id] = machine_busy.get(machine.id, 0.0) + chosen_op.processing_time
            for tooling in toolings:
                tooling_ready_time[tooling.id] = end
                tooling_busy[tooling.id] = tooling_busy.get(tooling.id, 0.0) + max(0.0, end - start)
            for person in people:
                personnel_ready_time[person.id] = end
                personnel_busy[person.id] = personnel_busy.get(person.id, 0.0) + max(0.0, end - start)

            predecessor_completion[chosen_op_id] = end
            task_id = chosen_op.task_id
            task_completion[task_id] = max(task_completion.get(task_id, 0.0), end)
            remaining_task_work[task_id] = max(0.0, remaining_task_work.get(task_id, 0.0) - chosen_op.processing_time)
            task_remaining_ops[task_id] = max(0, task_remaining_ops.get(task_id, 0) - 1)

            schedule.append(
                {
                    "op_id": chosen_op_id,
                    "task_id": task_id,
                    "machine_id": machine.id,
                    "machine_name": machine.name,
                    "start": round(start, 6),
                    "end": round(end, 6),
                    "duration": round(chosen_op.processing_time, 6),
                    "elapsed_duration": round(max(0.0, end - start), 6),
                    "tooling_ids": [tool.id for tool in toolings],
                    "personnel_ids": [person.id for person in people],
                }
            )

            for successor_id in self._successors_by_op.get(chosen_op_id, []):
                predecessor_remaining[successor_id] = max(0, predecessor_remaining.get(successor_id, 0) - 1)
                if predecessor_remaining[successor_id] == 0:
                    push_ready(successor_id)

            if task_remaining_ops[task_id] == 0:
                for successor_id in self._successors_by_task.get(task_id, []):
                    predecessor_remaining[successor_id] = max(0, predecessor_remaining.get(successor_id, 0) - 1)
                    if predecessor_remaining[successor_id] == 0:
                        push_ready(successor_id)

            for entry in search_entries:
                if entry[2] != chosen_op_id:
                    heapq.heappush(ready_heap, entry)

        sim_result = SimResult(schedule=schedule)
        analytics = build_schedule_analytics(self.shop, sim_result)
        metrics = {
            "makespan": analytics.objective_values.get("makespan", 0.0),
            "total_tardiness": analytics.objective_values.get("total_tardiness", 0.0),
            "avg_tardiness": analytics.objective_values.get("avg_tardiness", 0.0),
            "max_tardiness": analytics.objective_values.get("max_tardiness", 0.0),
            "tardy_job_count": analytics.objective_values.get("tardy_job_count", 0.0),
            "total_jobs": len(self.shop.tasks),
            "avg_flowtime": analytics.objective_values.get("avg_flowtime", 0.0),
            "main_order_tardy_count": analytics.objective_values.get("main_order_tardy_count", 0.0),
            "main_order_tardy_total_time": analytics.objective_values.get("main_order_tardy_total_time", 0.0),
            "main_order_tardy_ratio": analytics.objective_values.get("main_order_tardy_ratio", 0.0),
            "total_main_orders": analytics.summary.get("total_main_orders", 0.0),
            "avg_utilization": analytics.objective_values.get("avg_utilization", 0.0),
            "critical_utilization": analytics.objective_values.get("critical_utilization", 0.0),
            "total_wait_time": analytics.objective_values.get("total_wait_time", 0.0),
            "avg_wait_time": analytics.summary.get("avg_wait_time", 0.0),
            "tooling_utilization": analytics.objective_values.get("tooling_utilization", 0.0),
            "personnel_utilization": analytics.objective_values.get("personnel_utilization", 0.0),
            "assembly_sync_penalty": analytics.objective_values.get("assembly_sync_penalty", 0.0),
            "wall_time_ms": 0.0,
            "event_count": len(schedule),
            "completed_operations": analytics.completed_operations,
            "total_operations": len(self.shop.operations),
            "feasible": analytics.feasible,
            "evaluation_mode": "approximate",
            "approximation_method": "graph_constrained_list_scheduling",
        }
        metrics.update({key: round(value, 6) for key, value in analytics.objective_values.items()})

        signature = f"approx::{candidate.signature()}"
        stored_schedule = schedule[: self.keep_schedule_limit] if self.keep_schedule_limit > 0 else []
        analytics_summary = {
            "tardy_order_ids": list(analytics.tardy_order_ids),
            "tardy_task_ids": list(analytics.tardy_task_ids),
            "bottleneck_machine_ids": list(analytics.bottleneck_machine_ids),
            "order_tardiness": dict(analytics.order_tardiness),
            "task_completion": dict(analytics.task_completion),
            "order_main_gap": dict(analytics.order_main_gap),
            "machine_utilization": dict(analytics.machine_utilization),
            "tooling_utilization": dict(analytics.tooling_utilization),
            "personnel_utilization": dict(analytics.personnel_utilization),
            "completed_operations": analytics.completed_operations,
            "total_operations": len(self.shop.operations),
            "evaluation_mode": "approximate",
        }
        objectives = {}
        for key, value in analytics.objective_values.items():
            objectives[key] = value

        return OptimizationSolution(
            solution_id=f"A-{candidate.signature()[:10]}",
            source=source,
            generation=generation,
            candidate=candidate.clone(),
            objectives=objectives,
            metrics=metrics,
            schedule=stored_schedule,
            feasible=analytics.feasible,
            schedule_signature=signature,
            analytics_summary=analytics_summary,
        )
