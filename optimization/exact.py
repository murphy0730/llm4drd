from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..core.models import OpStatus
from ..core.simulator import SimResult
from .objectives import build_schedule_analytics


@dataclass(frozen=True)
class ExactObjectiveSpec:
    key: str
    label: str
    direction: str
    support_mode: str
    solver_key: str
    description: str


EXACT_OBJECTIVES: dict[str, ExactObjectiveSpec] = {
    "makespan": ExactObjectiveSpec("makespan", "Makespan", "min", "direct", "makespan", "Minimize total completion span."),
    "total_tardiness": ExactObjectiveSpec("total_tardiness", "Total Tardiness", "min", "direct", "total_tardiness", "Minimize total task tardiness."),
    "total_completion_time": ExactObjectiveSpec("total_completion_time", "Total Completion Time", "min", "direct", "total_completion_time", "Minimize sum of task completion times."),
    "total_wait_time": ExactObjectiveSpec("total_wait_time", "Total Wait Time", "min", "proxy", "total_completion_time", "Proxy objective via minimizing total completion time."),
    "avg_flowtime": ExactObjectiveSpec("avg_flowtime", "Average Flowtime", "min", "proxy", "total_completion_time", "Proxy objective via minimizing total completion time."),
    "max_tardiness": ExactObjectiveSpec("max_tardiness", "Max Tardiness", "min", "direct", "max_tardiness", "Minimize worst tardiness."),
    "tardy_job_count": ExactObjectiveSpec("tardy_job_count", "Tardy Job Count", "min", "direct", "tardy_job_count", "Minimize count of tardy tasks."),
    "main_order_tardy_total_time": ExactObjectiveSpec("main_order_tardy_total_time", "Main Order Total Tardiness", "min", "direct", "main_order_tardy_total_time", "Minimize tardiness of main-order tasks."),
    "main_order_tardy_count": ExactObjectiveSpec("main_order_tardy_count", "Main Order Tardy Count", "min", "direct", "main_order_tardy_count", "Minimize count of tardy main orders."),
    "avg_utilization": ExactObjectiveSpec("avg_utilization", "Average Utilization", "max", "proxy", "makespan", "Proxy objective via minimizing makespan."),
    "critical_utilization": ExactObjectiveSpec("critical_utilization", "Critical Utilization", "max", "proxy", "makespan", "Proxy objective via minimizing makespan."),
    "avg_active_window_utilization": ExactObjectiveSpec("avg_active_window_utilization", "Average Active Window Utilization", "max", "proxy", "makespan", "Proxy objective via minimizing makespan."),
    "critical_active_window_utilization": ExactObjectiveSpec("critical_active_window_utilization", "Critical Active Window Utilization", "max", "proxy", "makespan", "Proxy objective via minimizing makespan."),
    "avg_net_available_utilization": ExactObjectiveSpec("avg_net_available_utilization", "Average Net Available Utilization", "max", "proxy", "makespan", "Proxy objective via minimizing makespan."),
    "critical_net_available_utilization": ExactObjectiveSpec("critical_net_available_utilization", "Critical Net Available Utilization", "max", "proxy", "makespan", "Proxy objective via minimizing makespan."),
    "tooling_utilization": ExactObjectiveSpec("tooling_utilization", "Tooling Utilization", "max", "proxy", "makespan", "Proxy objective via minimizing makespan."),
    "personnel_utilization": ExactObjectiveSpec("personnel_utilization", "Personnel Utilization", "max", "proxy", "makespan", "Proxy objective via minimizing makespan."),
}


def exact_objective_catalog_payload() -> list[dict]:
    return [
        {
            "key": spec.key,
            "label": spec.label,
            "direction": spec.direction,
            "support_mode": spec.support_mode,
            "solver_key": spec.solver_key,
            "description": spec.description,
        }
        for spec in EXACT_OBJECTIVES.values()
    ]


@dataclass
class ExactResult:
    status: str
    objectives: dict = field(default_factory=dict)
    schedule: list = field(default_factory=list)
    solve_time_s: float = 0.0
    bounds: dict = field(default_factory=dict)
    request: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "status": self.status,
            "objectives": {key: round(value, 3) for key, value in self.objectives.items()},
            "solve_time_s": round(self.solve_time_s, 2),
            "bounds": {
                key: round(value, 3) if isinstance(value, (int, float)) else value
                for key, value in self.bounds.items()
            },
            "request": self.request,
            "num_schedule_entries": len(self.schedule),
        }


class ExactSolver:
    def __init__(self, shop, objectives: list | None = None, time_limit_s: int = 60, objective_weights: dict[str, float] | None = None):
        self.shop = shop
        self.objectives = objectives or ["makespan"]
        self.time_limit_s = time_limit_s
        self.objective_weights = objective_weights or {}

    def _resolve_request(self) -> dict:
        positive_weights = {
            key: float(value)
            for key, value in self.objective_weights.items()
            if key in EXACT_OBJECTIVES and abs(float(value)) > 1e-9
        }
        if positive_weights:
            total = sum(abs(value) for value in positive_weights.values())
            normalized = {key: abs(value) / total for key, value in positive_weights.items()}
            return {
                "mode": "weighted",
                "objective_key": None,
                "objective_weights": normalized,
                "support": {key: EXACT_OBJECTIVES[key].support_mode for key in normalized},
            }

        objective_key = next((key for key in self.objectives if key in EXACT_OBJECTIVES), None)
        if not objective_key:
            objective_key = "makespan"
        spec = EXACT_OBJECTIVES[objective_key]
        return {
            "mode": "single",
            "objective_key": objective_key,
            "objective_weights": {objective_key: 1.0},
            "support": {objective_key: spec.support_mode},
        }

    def solve(self, warm_start_schedule=None) -> ExactResult:
        del warm_start_schedule
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            return ExactResult(status="ERROR", bounds={"error": "ortools not installed"})

        request = self._resolve_request()
        started_at = time.time()
        model = cp_model.CpModel()
        scale = 60

        operations = list(self.shop.operations.values())
        if not operations:
            return ExactResult(status="INFEASIBLE", request=request)

        self.shop.build_indexes()
        completed_ops, fixed_processing_ops, decision_ops = self._classify_operations(operations)

        latest_due = max((task.due_date for task in self.shop.tasks.values()), default=0.0)
        total_decision_work = sum(op.work_remaining for op in decision_ops)
        max_fixed_end = max(
            (self._fixed_end_hours(op, clamp_nonnegative=True) for op in [*completed_ops, *fixed_processing_ops]),
            default=0.0,
        )
        horizon_hours = max(latest_due, total_decision_work * 1.5, max_fixed_end + total_decision_work * 1.5) + 24.0
        horizon = max(1, int(round(horizon_hours * scale)))

        op_vars: dict[str, dict[str, object]] = {}
        op_end_exprs: dict[str, object] = {}
        fixed_schedule: list[dict] = []

        for op in completed_ops:
            op_end_exprs[op.id] = model.NewConstant(int(round(self._fixed_end_hours(op, clamp_nonnegative=True) * scale)))

        for op in fixed_processing_ops:
            op_end_exprs[op.id] = model.NewConstant(int(round(self._fixed_end_hours(op, clamp_nonnegative=True) * scale)))
            fixed_schedule.append(self._fixed_schedule_entry(op, status="in_progress"))

        for op in decision_ops:
            duration = max(1, int(round(op.work_remaining * scale)))
            start = model.NewIntVar(0, horizon, f"s_{op.id}")
            end = model.NewIntVar(0, horizon, f"e_{op.id}")
            interval = model.NewIntervalVar(start, duration, end, f"i_{op.id}")
            op_vars[op.id] = {"start": start, "end": end, "interval": interval, "duration": duration}
            op_end_exprs[op.id] = end
            model.Add(start >= max(0, int(round(self.shop.get_operation_release_time(op) * scale))))

        machine_intervals: dict[str, list] = {}
        tooling_intervals: dict[str, list] = {}
        personnel_intervals: dict[str, list] = {}
        machine_choices: dict[str, dict[str, object]] = {}
        tooling_choices: dict[tuple[str, int], dict[str, object]] = {}
        personnel_choices: dict[tuple[str, int], dict[str, object]] = {}

        for op in fixed_processing_ops:
            fixed_window = self._fixed_processing_window(op, scale)
            if fixed_window is None:
                return ExactResult(status="INFEASIBLE", bounds={"error": f"Invalid initial processing window for {op.id}"}, request=request)
            start_value, duration_value = fixed_window
            if not op.assigned_machine_id or op.assigned_machine_id not in self.shop.machines:
                return ExactResult(status="INFEASIBLE", bounds={"error": f"Initial processing op {op.id} missing assigned machine"}, request=request)
            machine_intervals.setdefault(op.assigned_machine_id, []).append(
                model.NewFixedSizeIntervalVar(start_value, duration_value, f"fixed_machine_{op.id}")
            )
            for tooling_id in op.assigned_tooling_ids:
                if tooling_id in self.shop.toolings:
                    tooling_intervals.setdefault(tooling_id, []).append(
                        model.NewFixedSizeIntervalVar(start_value, duration_value, f"fixed_tool_{op.id}_{tooling_id}")
                    )
            for person_id in op.assigned_personnel_ids:
                if person_id in self.shop.personnel:
                    personnel_intervals.setdefault(person_id, []).append(
                        model.NewFixedSizeIntervalVar(start_value, duration_value, f"fixed_person_{op.id}_{person_id}")
                    )

        for op in decision_ops:
            eligible_machines = self.shop.get_eligible_machines(op)
            if not eligible_machines:
                return ExactResult(status="INFEASIBLE", bounds={"error": f"No eligible machine for {op.id}"}, request=request)
            if len(eligible_machines) == 1:
                machine_id = eligible_machines[0].id
                machine_intervals.setdefault(machine_id, []).append(op_vars[op.id]["interval"])
                machine_choices[op.id] = {"selected": machine_id, "bools": {}}
            else:
                bools = {}
                for machine in eligible_machines:
                    selector = model.NewBoolVar(f"sel_machine_{op.id}_{machine.id}")
                    interval = model.NewOptionalIntervalVar(
                        op_vars[op.id]["start"],
                        op_vars[op.id]["duration"],
                        op_vars[op.id]["end"],
                        selector,
                        f"oi_machine_{op.id}_{machine.id}",
                    )
                    machine_intervals.setdefault(machine.id, []).append(interval)
                    bools[machine.id] = selector
                model.AddExactlyOne(bools.values())
                machine_choices[op.id] = {"selected": None, "bools": bools}

            for requirement_index, tooling_type in enumerate(op.required_tooling_types):
                candidates = self.shop.get_toolings_for_type(tooling_type)
                if not candidates:
                    return ExactResult(status="INFEASIBLE", bounds={"error": f"No tooling of type {tooling_type} for {op.id}"}, request=request)
                if len(candidates) == 1:
                    tooling_intervals.setdefault(candidates[0].id, []).append(op_vars[op.id]["interval"])
                    tooling_choices[(op.id, requirement_index)] = {"selected": candidates[0].id, "bools": {}}
                else:
                    bools = {}
                    for tooling in candidates:
                        selector = model.NewBoolVar(f"sel_tool_{op.id}_{requirement_index}_{tooling.id}")
                        interval = model.NewOptionalIntervalVar(
                            op_vars[op.id]["start"],
                            op_vars[op.id]["duration"],
                            op_vars[op.id]["end"],
                            selector,
                            f"oi_tool_{op.id}_{requirement_index}_{tooling.id}",
                        )
                        tooling_intervals.setdefault(tooling.id, []).append(interval)
                        bools[tooling.id] = selector
                    model.AddExactlyOne(bools.values())
                    tooling_choices[(op.id, requirement_index)] = {"selected": None, "bools": bools}

            for requirement_index, skill_id in enumerate(op.required_personnel_skills):
                candidates = self.shop.get_personnel_for_skill(skill_id)
                if not candidates:
                    return ExactResult(status="INFEASIBLE", bounds={"error": f"No personnel with skill {skill_id} for {op.id}"}, request=request)
                if len(candidates) == 1:
                    personnel_intervals.setdefault(candidates[0].id, []).append(op_vars[op.id]["interval"])
                    personnel_choices[(op.id, requirement_index)] = {"selected": candidates[0].id, "bools": {}}
                else:
                    bools = {}
                    for person in candidates:
                        selector = model.NewBoolVar(f"sel_person_{op.id}_{requirement_index}_{person.id}")
                        interval = model.NewOptionalIntervalVar(
                            op_vars[op.id]["start"],
                            op_vars[op.id]["duration"],
                            op_vars[op.id]["end"],
                            selector,
                            f"oi_person_{op.id}_{requirement_index}_{person.id}",
                        )
                        personnel_intervals.setdefault(person.id, []).append(interval)
                        bools[person.id] = selector
                    model.AddExactlyOne(bools.values())
                    personnel_choices[(op.id, requirement_index)] = {"selected": None, "bools": bools}

        for resource_id, resource in self.shop.machines.items():
            intervals = machine_intervals.setdefault(resource_id, [])
            intervals.extend(self._fixed_unavailability_intervals(model, resource.unavailable_windows(horizon_hours), scale, resource_id))
            if intervals:
                model.AddNoOverlap(intervals)
        for resource_id, resource in self.shop.toolings.items():
            intervals = tooling_intervals.setdefault(resource_id, [])
            intervals.extend(self._fixed_unavailability_intervals(model, resource.unavailable_windows(horizon_hours), scale, resource_id))
            if intervals:
                model.AddNoOverlap(intervals)
        for resource_id, resource in self.shop.personnel.items():
            intervals = personnel_intervals.setdefault(resource_id, [])
            intervals.extend(self._fixed_unavailability_intervals(model, resource.unavailable_windows(horizon_hours), scale, resource_id))
            if intervals:
                model.AddNoOverlap(intervals)

        for op in decision_ops:
            start_var = op_vars[op.id]["start"]
            for predecessor_id in op.predecessor_ops:
                predecessor_end = op_end_exprs.get(predecessor_id)
                if predecessor_end is not None:
                    model.Add(start_var >= predecessor_end)
            for predecessor_task_id in op.predecessor_tasks:
                predecessor_task = self.shop.tasks.get(predecessor_task_id)
                if not predecessor_task:
                    continue
                for predecessor_op in predecessor_task.operations:
                    predecessor_end = op_end_exprs.get(predecessor_op.id)
                    if predecessor_end is not None:
                        model.Add(start_var >= predecessor_end)

        makespan_var = model.NewIntVar(0, horizon, "makespan")
        all_end_exprs = list(op_end_exprs.values())
        if all_end_exprs:
            model.AddMaxEquality(makespan_var, all_end_exprs)
        else:
            model.Add(makespan_var == 0)

        tardiness_vars = {}
        tardy_flags = {}
        main_tardiness_vars = {}
        main_tardy_flags = {}
        completion_terms = []

        for task in self.shop.tasks.values():
            relevant_end_exprs = [op_end_exprs[op.id] for op in task.operations if op.id in op_end_exprs]
            if not relevant_end_exprs:
                continue
            latest_end = model.NewIntVar(0, horizon, f"task_end_{task.id}")
            model.AddMaxEquality(latest_end, relevant_end_exprs)
            completion_terms.append(latest_end)

            due = int(round(task.due_date * scale))
            tardiness_raw = model.NewIntVar(-horizon, horizon, f"task_td_raw_{task.id}")
            tardiness = model.NewIntVar(0, horizon, f"task_td_{task.id}")
            model.Add(tardiness_raw == latest_end - due)
            model.AddMaxEquality(tardiness, [tardiness_raw, model.NewConstant(0)])
            tardiness_vars[task.id] = tardiness

            tardy_flag = model.NewBoolVar(f"task_tardy_flag_{task.id}")
            model.Add(tardiness >= 1).OnlyEnforceIf(tardy_flag)
            model.Add(tardiness <= 0).OnlyEnforceIf(tardy_flag.Not())
            tardy_flags[task.id] = tardy_flag

            if task.is_main:
                main_tardiness_vars[task.id] = tardiness
                main_tardy_flags[task.id] = tardy_flag

        total_completion_time = model.NewIntVar(0, horizon * max(1, len(completion_terms)), "total_completion_time")
        if completion_terms:
            model.Add(total_completion_time == sum(completion_terms))
        else:
            model.Add(total_completion_time == 0)

        total_tardiness = model.NewIntVar(0, horizon * max(1, len(tardiness_vars)), "total_tardiness")
        if tardiness_vars:
            model.Add(total_tardiness == sum(tardiness_vars.values()))
        else:
            model.Add(total_tardiness == 0)

        max_tardiness = model.NewIntVar(0, horizon, "max_tardiness")
        if tardiness_vars:
            model.AddMaxEquality(max_tardiness, list(tardiness_vars.values()))
        else:
            model.Add(max_tardiness == 0)

        tardy_job_count = model.NewIntVar(0, len(tardy_flags), "tardy_job_count")
        if tardy_flags:
            model.Add(tardy_job_count == sum(tardy_flags.values()))
        else:
            model.Add(tardy_job_count == 0)

        main_order_tardy_total_time = model.NewIntVar(0, horizon * max(1, len(main_tardiness_vars)), "main_order_tardy_total_time")
        if main_tardiness_vars:
            model.Add(main_order_tardy_total_time == sum(main_tardiness_vars.values()))
        else:
            model.Add(main_order_tardy_total_time == 0)

        main_order_tardy_count = model.NewIntVar(0, len(main_tardy_flags), "main_order_tardy_count")
        if main_tardy_flags:
            model.Add(main_order_tardy_count == sum(main_tardy_flags.values()))
        else:
            model.Add(main_order_tardy_count == 0)

        objective_terms = {
            "makespan": makespan_var,
            "total_tardiness": total_tardiness,
            "total_completion_time": total_completion_time,
            "max_tardiness": max_tardiness,
            "tardy_job_count": tardy_job_count,
            "main_order_tardy_total_time": main_order_tardy_total_time,
            "main_order_tardy_count": main_order_tardy_count,
        }
        objective_scales = {
            "makespan": max(1, horizon),
            "total_tardiness": max(1, horizon * max(1, len(tardiness_vars))),
            "total_completion_time": max(1, horizon * max(1, len(completion_terms))),
            "max_tardiness": max(1, horizon),
            "tardy_job_count": max(1, len(tardy_flags)),
            "main_order_tardy_total_time": max(1, horizon * max(1, len(main_tardiness_vars))),
            "main_order_tardy_count": max(1, len(main_tardy_flags)),
        }

        if request["mode"] == "weighted":
            weighted_terms = []
            for objective_key, weight in request["objective_weights"].items():
                spec = EXACT_OBJECTIVES[objective_key]
                expr = objective_terms[spec.solver_key]
                coefficient = max(1, int(round(weight * 100000 / objective_scales[spec.solver_key])))
                weighted_terms.append(expr * coefficient)
            model.Minimize(sum(weighted_terms) if weighted_terms else makespan_var)
        else:
            objective_key = request["objective_key"]
            spec = EXACT_OBJECTIVES[objective_key]
            model.Minimize(objective_terms[spec.solver_key])

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_s
        solver.parameters.num_search_workers = 4
        status = solver.Solve(model)
        solve_time = time.time() - started_at

        status_map = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.UNKNOWN: "UNKNOWN",
        }
        status_label = status_map.get(status, "UNKNOWN")
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return ExactResult(status=status_label, solve_time_s=solve_time, request=request)

        schedule = list(fixed_schedule)
        for op in decision_ops:
            machine_id = self._selected_resource(solver, machine_choices[op.id])
            tooling_ids = [self._selected_resource(solver, tooling_choices[(op.id, idx)]) for idx, _ in enumerate(op.required_tooling_types)]
            personnel_ids = [self._selected_resource(solver, personnel_choices[(op.id, idx)]) for idx, _ in enumerate(op.required_personnel_skills)]
            machine = self.shop.machines.get(machine_id)
            schedule.append(
                {
                    "op_id": op.id,
                    "op_name": op.name,
                    "task_id": op.task_id,
                    "machine_id": machine_id,
                    "machine_name": machine.name if machine else machine_id,
                    "tooling_ids": tooling_ids,
                    "personnel_ids": personnel_ids,
                    "start": round(solver.Value(op_vars[op.id]["start"]) / scale, 3),
                    "end": round(solver.Value(op_vars[op.id]["end"]) / scale, 3),
                    "duration": round(op.work_remaining, 3),
                    "status": "scheduled",
                }
            )
        schedule.sort(key=lambda entry: (entry.get("start", 0.0), entry.get("machine_id") or "", entry.get("op_id") or ""))

        analytics = build_schedule_analytics(self.shop, SimResult(schedule=schedule))
        objectives = dict(analytics.objective_values)
        if request["mode"] == "weighted":
            weighted_score = 0.0
            for objective_key, weight in request["objective_weights"].items():
                spec = EXACT_OBJECTIVES[objective_key]
                raw_value = float(objectives.get(objective_key, objectives.get(spec.solver_key, 0.0)))
                reference = max(1.0, objective_scales[spec.solver_key] / scale)
                weighted_score += weight * (raw_value / reference)
            objectives["weighted_score"] = round(weighted_score, 6)

        bounds = {}
        if request["mode"] == "single" and request["objective_key"]:
            spec = EXACT_OBJECTIVES[request["objective_key"]]
            bounds[f"{spec.solver_key}_lb"] = solver.BestObjectiveBound() / scale

        return ExactResult(
            status=status_label,
            objectives=objectives,
            schedule=schedule,
            solve_time_s=solve_time,
            bounds=bounds,
            request=request,
        )

    def solve_pareto_front(self, num_points: int = 10) -> list:
        del num_points
        return [self.solve()]

    def _fixed_unavailability_intervals(self, model, windows, scale: int, resource_id: str):
        fixed_intervals = []
        for index, (start, end) in enumerate(windows):
            duration = int(round((end - start) * scale))
            if duration <= 0:
                continue
            fixed_intervals.append(
                model.NewFixedSizeIntervalVar(
                    int(round(start * scale)),
                    duration,
                    f"unavail_{resource_id}_{index}",
                )
            )
        return fixed_intervals

    def _selected_resource(self, solver, choice_info):
        if choice_info["selected"] is not None:
            return choice_info["selected"]
        for resource_id, selector in choice_info["bools"].items():
            if solver.Value(selector):
                return resource_id
        return None

    def _classify_operations(self, operations):
        completed_ops = []
        fixed_processing_ops = []
        decision_ops = []
        for op in operations:
            if op.status == OpStatus.COMPLETED:
                completed_ops.append(op)
            elif op.status == OpStatus.PROCESSING and op.assigned_machine_id and op.end_time is not None:
                fixed_processing_ops.append(op)
            else:
                decision_ops.append(op)
        return completed_ops, fixed_processing_ops, decision_ops

    def _fixed_end_hours(self, op, clamp_nonnegative: bool = False) -> float:
        if op.end_time is not None:
            end_time = float(op.end_time)
        elif op.status == OpStatus.COMPLETED:
            end_time = 0.0
        else:
            productive = max(0.001, float(op.work_remaining))
            start_time = float(op.start_time or 0.0)
            end_time = start_time + productive
        return max(0.0, end_time) if clamp_nonnegative else end_time

    def _fixed_processing_window(self, op, scale: int):
        if op.end_time is None:
            return None
        productive = max(0.001, float(op.work_remaining))
        end_time = float(op.end_time)
        start_time = float(op.start_time if op.start_time is not None else max(0.0, end_time - productive))
        window_start = max(0.0, min(start_time, end_time))
        window_end = max(0.0, end_time)
        if window_end <= window_start:
            window_end = window_start + productive
        start_value = int(round(window_start * scale))
        duration_value = max(1, int(round((window_end - window_start) * scale)))
        return start_value, duration_value

    def _fixed_schedule_entry(self, op, status: str) -> dict:
        machine = self.shop.machines.get(op.assigned_machine_id) if op.assigned_machine_id else None
        start_time = float(op.start_time if op.start_time is not None else 0.0)
        end_time = float(op.end_time if op.end_time is not None else start_time + max(0.001, float(op.work_remaining)))
        productive_duration = max(0.001, float(op.work_remaining))
        return {
            "op_id": op.id,
            "op_name": op.name,
            "task_id": op.task_id,
            "machine_id": op.assigned_machine_id,
            "machine_name": machine.name if machine else op.assigned_machine_id,
            "tooling_ids": list(op.assigned_tooling_ids or []),
            "personnel_ids": list(op.assigned_personnel_ids or []),
            "start": round(start_time, 3),
            "end": round(end_time, 3),
            "duration": round(productive_duration, 3),
            "elapsed_duration": round(max(0.0, end_time - start_time), 3),
            "status": status,
        }
