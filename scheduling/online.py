"""
在线调度 v3 - 基于 v3 模型的在线调度和动态重排
"""
import copy
import time
import logging
from typing import Optional, Callable
from dataclasses import dataclass, field

from ..core.models import ShopFloor, Operation, Task, Order, Machine, MachineState, OpStatus
from ..core.simulator import Simulator, SimResult
from ..core.rules import BUILTIN_RULES

logger = logging.getLogger(__name__)


@dataclass
class OnlineState:
    current_time: float = 0.0
    rule_name: str = "ATC"
    machine_states: dict = field(default_factory=dict)  # mid -> {"state", "current_op", "finish_at"}
    completed_ops: list = field(default_factory=list)   # list of schedule entries
    is_running: bool = False


class OnlineSchedulerV3:
    """
    Simple online scheduler for v3 models.
    Uses simulation-based approach: advance time, dispatch decisions.
    """

    def __init__(self, shop: ShopFloor, rule_name: str = "ATC"):
        self.base_shop = shop
        self.state = OnlineState(rule_name=rule_name)
        self.sim_shop = copy.deepcopy(shop)
        self.sim_shop.build_indexes()
        self._init_ready_ops()
        self.state.is_running = True
        # Init machine states
        for mid, m in self.sim_shop.machines.items():
            self.state.machine_states[mid] = {
                "state": "IDLE", "current_op": None, "finish_at": 0.0,
                "breakdown_until": None,
            }

    def _init_ready_ops(self):
        for op in self.sim_shop.operations.values():
            if self.sim_shop.check_op_ready(op):
                op.status = OpStatus.READY

    def _dispatch_idle_machines(self, now: float, rule_fn):
        """Dispatch ready ops to idle machines at time 'now'."""
        for mid, mstate in self.state.machine_states.items():
            if mstate["state"] != "IDLE":
                continue
            m = self.sim_shop.machines.get(mid)
            if not m:
                continue
            if mstate.get("breakdown_until") and mstate["breakdown_until"] > now:
                continue

            candidates = []
            for op in self.sim_shop.operations.values():
                if op.status != OpStatus.READY:
                    continue
                if op.eligible_machine_ids and mid not in op.eligible_machine_ids:
                    continue
                elif not op.eligible_machine_ids and op.process_type != m.type_id:
                    continue
                candidates.append(op)

            if not candidates:
                continue

            best_op, best_score = None, float('-inf')
            for op in candidates:
                task = self.sim_shop.tasks.get(op.task_id)
                order = self.sim_shop.orders.get(task.order_id) if task else None
                due = order.due_date if order else 9999
                remaining = task.remaining_time if task else op.processing_time
                slack = due - now - remaining
                feat = {
                    "slack": slack, "remaining": remaining,
                    "processing_time": op.processing_time, "due_date": due,
                    "urgency": max(0, -slack) if slack < 0 else 0,
                    "progress": task.progress if task else 0.0,
                    "priority": order.priority if order else 1,
                    "is_main": 1.0 if (task and task.is_main) else 0.0,
                    "wait_time": max(0, now),
                    "prereq_ratio": 1.0,
                    "machine_busy_time": m.total_busy_time,
                }
                try:
                    sc = rule_fn(op, m, feat, self.sim_shop)
                except Exception:
                    sc = 0.0
                if sc > best_score:
                    best_score, best_op = sc, op

            if best_op:
                start = m.next_start_time(now)
                end = m.compute_effective_end(start, best_op.processing_time)
                best_op.status = OpStatus.PROCESSING
                best_op.assigned_machine_id = mid
                best_op.start_time = start
                best_op.end_time = end
                m.state = MachineState.BUSY
                mstate["state"] = "BUSY"
                mstate["current_op"] = best_op.id
                mstate["finish_at"] = end

                task = self.sim_shop.tasks.get(best_op.task_id)
                order = self.sim_shop.orders.get(task.order_id) if task else None
                self.state.completed_ops.append({
                    "op_id": best_op.id, "op_name": best_op.name,
                    "task_id": best_op.task_id,
                    "machine_id": mid, "machine_name": m.name,
                    "start": round(start, 3), "end": round(end, 3),
                    "duration": round(best_op.processing_time, 3),
                    "order_id": order.id if order else "",
                    "status": "in_progress",
                })

    def advance(self, delta: float) -> dict:
        """Advance simulation time by delta hours using event-driven loop."""
        target_time = self.state.current_time + delta
        now = self.state.current_time
        rule_fn = BUILTIN_RULES.get(self.state.rule_name, BUILTIN_RULES["ATC"])

        # Initial dispatch at current time
        self._dispatch_idle_machines(now, rule_fn)

        # Event-driven loop: complete ops, dispatch again, repeat until target_time
        for _ in range(100000):
            # Find earliest completion event within target_time
            next_finish = float('inf')
            for mstate in self.state.machine_states.values():
                if mstate["state"] == "BUSY":
                    ft = mstate["finish_at"]
                    if ft <= target_time and ft < next_finish:
                        next_finish = ft

            if next_finish == float('inf'):
                break  # No more events before target_time

            now = next_finish

            # Complete all ops finishing at or before 'now'
            for mid, mstate in list(self.state.machine_states.items()):
                if mstate["state"] == "BUSY" and mstate["finish_at"] <= now:
                    op_id = mstate["current_op"]
                    op = self.sim_shop.operations.get(op_id) if op_id else None
                    m = self.sim_shop.machines.get(mid)
                    if op:
                        op.status = OpStatus.COMPLETED
                        # Mark gantt entry as completed
                        for entry in reversed(self.state.completed_ops):
                            if entry["op_id"] == op_id:
                                entry["status"] = "completed"
                                break
                        task = self.sim_shop.tasks.get(op.task_id)
                        if task and task.is_completed:
                            task.completion_time = mstate["finish_at"]
                        for nop in self.sim_shop.operations.values():
                            if nop.status == OpStatus.PENDING:
                                if op_id in nop.predecessor_ops or (task and op.task_id in nop.predecessor_tasks):
                                    if self.sim_shop.check_op_ready(nop):
                                        nop.status = OpStatus.READY
                    if m:
                        finish_at = mstate["finish_at"]
                        start_t = op.start_time if (op and op.start_time is not None) else finish_at
                        m.total_busy_time += max(0, finish_at - start_t)
                        m.state = MachineState.IDLE
                    mstate["state"] = "IDLE"
                    mstate["current_op"] = None

            # Dispatch newly freed machines
            self._dispatch_idle_machines(now, rule_fn)

        self.state.current_time = target_time
        return self.get_status()

    def on_breakdown(self, machine_id: str, repair_at: float) -> dict:
        m = self.sim_shop.machines.get(machine_id)
        mstate = self.state.machine_states.get(machine_id)
        if m and mstate:
            m.state = MachineState.MAINTENANCE
            mstate["state"] = "MAINTENANCE"
            mstate["breakdown_until"] = repair_at
            # Interrupt current op if any
            if mstate.get("current_op"):
                op = self.sim_shop.operations.get(mstate["current_op"])
                if op and op.status == OpStatus.PROCESSING:
                    op.status = OpStatus.READY
                    op.assigned_machine_id = None
                    op.start_time = None
                    op.end_time = None
                mstate["current_op"] = None
        return self.get_status()

    def on_repair(self, machine_id: str) -> dict:
        m = self.sim_shop.machines.get(machine_id)
        mstate = self.state.machine_states.get(machine_id)
        if m and mstate:
            m.state = MachineState.IDLE
            mstate["state"] = "IDLE"
            mstate["breakdown_until"] = None
        return self.get_status()

    def reschedule(self, rule_name: str = None) -> dict:
        """Run a full simulation from current state and return improvement."""
        rule_name = rule_name or self.state.rule_name
        rule_fn = BUILTIN_RULES.get(rule_name, BUILTIN_RULES["ATC"])

        # Create a fresh copy of base shop for comparison
        old_rule = BUILTIN_RULES.get(self.state.rule_name, BUILTIN_RULES["ATC"])
        old_sim = Simulator(self.base_shop, old_rule)
        old_result = old_sim.run()

        new_sim = Simulator(self.base_shop, rule_fn)
        new_result = new_sim.run()

        old_td = old_result.total_tardiness
        new_td = new_result.total_tardiness
        improvement = (old_td - new_td) / max(old_td, 0.01) * 100

        self.state.rule_name = rule_name
        return {
            "old_rule": self.state.rule_name,
            "new_rule": rule_name,
            "old_tardiness": round(old_td, 2),
            "new_tardiness": round(new_td, 2),
            "improvement_pct": round(improvement, 2),
        }

    def get_status(self) -> dict:
        machine_info = []
        for mid, m in self.sim_shop.machines.items():
            mstate = self.state.machine_states.get(mid, {})
            machine_info.append({
                "id": mid, "name": m.name, "type": m.type_id,
                "state": mstate.get("state", m.state.value),
                "current_op": mstate.get("current_op"),
                "finish_at": mstate.get("finish_at", 0),
                "breakdown_until": mstate.get("breakdown_until"),
            })

        total_ops = len(self.sim_shop.operations)
        completed = sum(1 for op in self.sim_shop.operations.values() if op.status == OpStatus.COMPLETED)
        in_progress = sum(1 for op in self.sim_shop.operations.values() if op.status == OpStatus.PROCESSING)
        ready = sum(1 for op in self.sim_shop.operations.values() if op.status == OpStatus.READY)

        return {
            "current_time": round(self.state.current_time, 2),
            "rule": self.state.rule_name,
            "machines": machine_info,
            "ops_total": total_ops,
            "ops_completed": completed,
            "ops_in_progress": in_progress,
            "ops_ready": ready,
            "gantt": self.state.completed_ops[-50:],  # last 50 entries
        }
