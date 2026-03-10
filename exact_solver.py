"""
精确求解器 — OR-Tools CP-SAT
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExactResult:
    status: str  # OPTIMAL | FEASIBLE | INFEASIBLE | UNKNOWN | ERROR
    objectives: dict = field(default_factory=dict)
    schedule: list = field(default_factory=list)
    solve_time_s: float = 0.0
    bounds: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "status": self.status,
            "objectives": {k: round(v, 3) for k, v in self.objectives.items()},
            "solve_time_s": round(self.solve_time_s, 2),
            "bounds": {k: round(v, 3) for k, v in self.bounds.items()},
            "num_schedule_entries": len(self.schedule),
        }


class ExactSolver:
    def __init__(self, shop, objectives: list = None, time_limit_s: int = 60):
        self.shop = shop
        self.objectives = objectives or ["makespan", "total_tardiness"]
        self.time_limit_s = time_limit_s

    def solve(self, warm_start_schedule=None) -> ExactResult:
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            return ExactResult(status="ERROR", objectives={}, solve_time_s=0,
                               bounds={"error": "ortools not installed"})

        t0 = time.time()
        model = cp_model.CpModel()

        # Scale factor: convert hours to integer minutes
        SCALE = 60

        ops = list(self.shop.operations.values())
        if not ops:
            return ExactResult(status="INFEASIBLE")

        # Estimate horizon
        total_pt = sum(op.processing_time for op in ops)
        horizon = int(total_pt * SCALE * 2) + 1

        # Create interval variables for each operation
        op_vars = {}  # op_id -> (start_var, end_var, interval_var, dur)
        for op in ops:
            dur = max(1, int(op.processing_time * SCALE))
            start = model.NewIntVar(0, horizon, f"s_{op.id}")
            end = model.NewIntVar(0, horizon, f"e_{op.id}")
            interval = model.NewIntervalVar(start, dur, end, f"i_{op.id}")
            op_vars[op.id] = (start, end, interval, dur)

        # No-overlap constraints per machine
        machine_intervals = {}
        for op in ops:
            machines = self.shop.get_eligible_machines(op)
            if not machines:
                continue
            if len(machines) == 1:
                mid = machines[0].id
                machine_intervals.setdefault(mid, []).append(op_vars[op.id][2])
            else:
                # Optional intervals for machine selection
                machine_bools = []
                s, e, _, dur = op_vars[op.id]
                for m in machines:
                    b = model.NewBoolVar(f"sel_{op.id}_{m.id}")
                    opt_interval = model.NewOptionalIntervalVar(s, dur, e, b, f"oi_{op.id}_{m.id}")
                    machine_intervals.setdefault(m.id, []).append(opt_interval)
                    machine_bools.append(b)
                model.AddExactlyOne(machine_bools)

        for mid, intervals in machine_intervals.items():
            model.AddNoOverlap(intervals)

        # Precedence constraints
        for op in ops:
            s_op = op_vars[op.id][0]
            for pred_id in op.predecessor_ops:
                if pred_id in op_vars:
                    e_pred = op_vars[pred_id][1]
                    model.Add(s_op >= e_pred)
            for pred_task_id in op.predecessor_tasks:
                if pred_task_id in self.shop.tasks:
                    pred_task = self.shop.tasks[pred_task_id]
                    for pred_op in pred_task.operations:
                        if pred_op.id in op_vars:
                            e_pred = op_vars[pred_op.id][1]
                            model.Add(s_op >= e_pred)

        # Objective: minimize makespan
        makespan_var = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(makespan_var, [op_vars[op.id][1] for op in ops])

        primary_obj = self.objectives[0] if self.objectives else "makespan"
        if primary_obj == "makespan":
            model.Minimize(makespan_var)
        elif primary_obj == "total_tardiness":
            tardi_vars = []
            for task in self.shop.tasks.values():
                due_scaled = int(task.due_date * SCALE)
                last_op_end = None
                for op in task.operations:
                    if op.id in op_vars:
                        last_op_end = op_vars[op.id][1]
                if last_op_end is not None:
                    td = model.NewIntVar(0, horizon, f"td_{task.id}")
                    tardiness_raw = model.NewIntVar(-horizon, horizon, f"tdr_{task.id}")
                    zero_var = model.NewConstant(0)
                    model.Add(tardiness_raw == last_op_end - due_scaled)
                    model.AddMaxEquality(td, [tardiness_raw, zero_var])
                    tardi_vars.append(td)
            if tardi_vars:
                total_td = model.NewIntVar(0, horizon * len(tardi_vars), "total_td")
                model.Add(total_td == sum(tardi_vars))
                model.Minimize(total_td)
            else:
                model.Minimize(makespan_var)
        else:
            model.Minimize(makespan_var)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_s
        solver.parameters.num_search_workers = 4

        status = solver.Solve(model)

        solve_time = time.time() - t0

        status_map = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.UNKNOWN: "UNKNOWN",
        }
        status_str = status_map.get(status, "UNKNOWN")

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            schedule = []
            for op in ops:
                s_val = solver.Value(op_vars[op.id][0]) / SCALE
                e_val = solver.Value(op_vars[op.id][1]) / SCALE
                machines = self.shop.get_eligible_machines(op)
                mid = machines[0].id if machines else "unknown"
                m = self.shop.machines.get(mid)
                schedule.append({
                    "op_id": op.id, "op_name": op.name,
                    "task_id": op.task_id,
                    "machine_id": mid,
                    "machine_name": m.name if m else mid,
                    "start": round(s_val, 3),
                    "end": round(e_val, 3),
                    "duration": round(e_val - s_val, 3),
                })

            makespan_val = solver.Value(makespan_var) / SCALE

            # Compute tardiness
            total_tard = 0.0
            for task in self.shop.tasks.values():
                last_end = 0.0
                for op in task.operations:
                    if op.id in op_vars:
                        e_val = solver.Value(op_vars[op.id][1]) / SCALE
                        last_end = max(last_end, e_val)
                total_tard += max(0, last_end - task.due_date)

            objectives = {
                "makespan": round(makespan_val, 3),
                "total_tardiness": round(total_tard, 3),
            }

            bounds = {}
            if primary_obj == "makespan":
                bounds["makespan_lb"] = round(solver.BestObjectiveBound() / SCALE, 3)

            return ExactResult(
                status=status_str,
                objectives=objectives,
                schedule=schedule,
                solve_time_s=round(solve_time, 2),
                bounds=bounds
            )
        else:
            return ExactResult(
                status=status_str,
                objectives={},
                schedule=[],
                solve_time_s=round(solve_time, 2),
            )

    def solve_pareto_front(self, num_points: int = 10) -> list:
        """Epsilon-constraint method: vary epsilon for secondary objective."""
        results = []
        r0 = self.solve()
        if r0.status not in ("OPTIMAL", "FEASIBLE"):
            return [r0]
        results.append(r0)
        return results  # Simplified: just return one solution for now
