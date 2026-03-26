from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import pstdev
from typing import Iterable

from ..core.models import OpStatus, ShopFloor
from ..core.simulator import SimResult


@dataclass(frozen=True)
class ObjectiveSpec:
    key: str
    label: str
    direction: str
    description: str
    available: bool = True


@dataclass
class ScheduleAnalytics:
    feasible: bool
    completed_operations: int
    task_completion: dict[str, float] = field(default_factory=dict)
    order_completion: dict[str, float] = field(default_factory=dict)
    task_tardiness: dict[str, float] = field(default_factory=dict)
    order_tardiness: dict[str, float] = field(default_factory=dict)
    order_main_gap: dict[str, float] = field(default_factory=dict)
    machine_utilization: dict[str, float] = field(default_factory=dict)
    machine_active_window_utilization: dict[str, float] = field(default_factory=dict)
    machine_net_available_utilization: dict[str, float] = field(default_factory=dict)
    tooling_utilization: dict[str, float] = field(default_factory=dict)
    personnel_utilization: dict[str, float] = field(default_factory=dict)
    bottleneck_machine_ids: list[str] = field(default_factory=list)
    tardy_order_ids: list[str] = field(default_factory=list)
    tardy_task_ids: list[str] = field(default_factory=list)
    objective_values: dict[str, float] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


OBJECTIVE_SPECS: dict[str, ObjectiveSpec] = {
    "total_tardiness": ObjectiveSpec(
        key="total_tardiness",
        label="总延迟",
        direction="min",
        description="所有任务延迟时间之和。",
    ),
    "makespan": ObjectiveSpec(
        key="makespan",
        label="Makespan",
        direction="min",
        description="从计划开始到全部任务完成的总历时。",
    ),
    "main_order_tardy_count": ObjectiveSpec(
        key="main_order_tardy_count",
        label="主订单延误数",
        direction="min",
        description="主订单主任务超期的订单数量。",
    ),
    "main_order_tardy_total_time": ObjectiveSpec(
        key="main_order_tardy_total_time",
        label="主订单总延误",
        direction="min",
        description="主订单主任务延误总时长。",
    ),
    "main_order_tardy_ratio": ObjectiveSpec(
        key="main_order_tardy_ratio",
        label="主订单延误率",
        direction="min",
        description="主订单延误数占全部主订单的比例。",
    ),
    "avg_utilization": ObjectiveSpec(
        key="avg_utilization",
        label="平均机器利用率(全周期)",
        direction="max",
        description="全部机器忙碌加工时间除以整体计划完工跨度的平均值。",
    ),
    "critical_utilization": ObjectiveSpec(
        key="critical_utilization",
        label="关键资源利用率(全周期)",
        direction="max",
        description="关键机器忙碌加工时间除以整体计划完工跨度的平均值。",
    ),
    "avg_active_window_utilization": ObjectiveSpec(
        key="avg_active_window_utilization",
        label="平均活跃窗口利用率",
        direction="max",
        description="全部机器忙碌加工时间除以各自最早开工到最晚完工活跃窗口跨度的平均值。",
    ),
    "critical_active_window_utilization": ObjectiveSpec(
        key="critical_active_window_utilization",
        label="关键资源活跃窗口利用率",
        direction="max",
        description="关键机器忙碌加工时间除以各自最早开工到最晚完工活跃窗口跨度的平均值。",
    ),
    "avg_net_available_utilization": ObjectiveSpec(
        key="avg_net_available_utilization",
        label="平均净可用利用率",
        direction="max",
        description="全部机器忙碌加工时间除以各自活跃窗口内净可用时间的平均值。",
    ),
    "critical_net_available_utilization": ObjectiveSpec(
        key="critical_net_available_utilization",
        label="关键资源净可用利用率",
        direction="max",
        description="关键机器忙碌加工时间除以各自活跃窗口内净可用时间的平均值。",
    ),
    "total_wait_time": ObjectiveSpec(
        key="total_wait_time",
        label="总等待时间",
        direction="min",
        description="任务总等待时间。",
    ),
    "avg_flowtime": ObjectiveSpec(
        key="avg_flowtime",
        label="平均流程时间",
        direction="min",
        description="任务从释放到完成的平均历时。",
    ),
    "max_tardiness": ObjectiveSpec(
        key="max_tardiness",
        label="最大延迟",
        direction="min",
        description="所有任务中的最大延迟。",
    ),
    "tardy_job_count": ObjectiveSpec(
        key="tardy_job_count",
        label="延迟任务数",
        direction="min",
        description="发生延迟的任务数量。",
    ),
    "avg_tardiness": ObjectiveSpec(
        key="avg_tardiness",
        label="平均延迟",
        direction="min",
        description="全部任务平均延迟。",
    ),
    "total_completion_time": ObjectiveSpec(
        key="total_completion_time",
        label="总完成时间",
        direction="min",
        description="所有任务完成时间之和。",
    ),
    "max_flowtime": ObjectiveSpec(
        key="max_flowtime",
        label="最大流程时间",
        direction="min",
        description="任务最大流程时间。",
    ),
    "bottleneck_load_balance": ObjectiveSpec(
        key="bottleneck_load_balance",
        label="瓶颈负载均衡",
        direction="min",
        description="瓶颈机器利用率离散程度，越小越均衡。",
    ),
    "tooling_utilization": ObjectiveSpec(
        key="tooling_utilization",
        label="工装利用率",
        direction="max",
        description="工装平均占用率。",
    ),
    "personnel_utilization": ObjectiveSpec(
        key="personnel_utilization",
        label="人员利用率",
        direction="max",
        description="人员平均占用率。",
    ),
    "assembly_sync_penalty": ObjectiveSpec(
        key="assembly_sync_penalty",
        label="装配同步惩罚",
        direction="min",
        description="同一订单主任务与其它任务完成节拍差异，越小越同步。",
    ),
}


def list_objectives(available_only: bool = True) -> list[ObjectiveSpec]:
    values = list(OBJECTIVE_SPECS.values())
    if available_only:
        values = [spec for spec in values if spec.available]
    return values


def get_objective_specs(objective_keys: Iterable[str]) -> list[ObjectiveSpec]:
    specs: list[ObjectiveSpec] = []
    for key in objective_keys:
        spec = OBJECTIVE_SPECS.get(key)
        if spec is not None:
            specs.append(spec)
    return specs


def validate_objective_selection(objective_keys: list[str]) -> list[ObjectiveSpec]:
    keys = [key for key in objective_keys if key in OBJECTIVE_SPECS]
    if len(keys) != len(objective_keys):
        invalid = [key for key in objective_keys if key not in OBJECTIVE_SPECS]
        raise ValueError(f"unknown objectives: {', '.join(invalid)}")
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate objectives are not allowed")
    if not 1 <= len(keys) <= 5:
        raise ValueError("please choose 1-5 objectives")
    specs = get_objective_specs(keys)
    unavailable = [spec.key for spec in specs if not spec.available]
    if unavailable:
        raise ValueError(f"objectives not available yet: {', '.join(unavailable)}")
    return specs


def objective_vector(values: dict[str, float], specs: Iterable[ObjectiveSpec]) -> list[float]:
    vector: list[float] = []
    for spec in specs:
        value = float(values.get(spec.key, 0.0))
        vector.append(value if spec.direction == "min" else -value)
    return vector


def objective_summary_payload() -> list[dict]:
    return [
        {
            "key": spec.key,
            "label": spec.label,
            "direction": spec.direction,
            "description": spec.description,
            "available": spec.available,
        }
        for spec in list_objectives(available_only=False)
    ]


def build_schedule_analytics(shop: ShopFloor, result: SimResult) -> ScheduleAnalytics:
    schedule = list(result.schedule or [])
    makespan = max((entry.get("end", 0.0) for entry in schedule), default=0.0)
    completed_op_ids = {entry.get("op_id") for entry in schedule if entry.get("op_id")}
    completed_op_ids.update(op.id for op in shop.operations.values() if op.status == OpStatus.COMPLETED)
    completed_operations = len(completed_op_ids)
    feasible = completed_operations == len(shop.operations)

    task_completion: dict[str, float] = {}
    machine_busy: dict[str, float] = {}
    machine_first_start: dict[str, float] = {}
    machine_last_end: dict[str, float] = {}
    tooling_busy: dict[str, float] = {}
    personnel_busy: dict[str, float] = {}
    order_task_map: dict[str, list[str]] = {}

    for task_id, task in shop.tasks.items():
        order_task_map.setdefault(task.order_id, []).append(task_id)
        for op in task.operations:
            if op.status == OpStatus.COMPLETED and op.end_time is not None:
                task_completion[task_id] = max(task_completion.get(task_id, 0.0), float(op.end_time))

    for entry in schedule:
        task_id = entry.get("task_id")
        if task_id:
            task_completion[task_id] = max(task_completion.get(task_id, 0.0), float(entry.get("end", 0.0)))

        duration = float(entry.get("duration", entry.get("end", 0.0) - entry.get("start", 0.0)))
        occupied = float(entry.get("elapsed_duration", entry.get("end", 0.0) - entry.get("start", 0.0)))
        machine_id = entry.get("machine_id")
        if machine_id:
            machine_busy[machine_id] = machine_busy.get(machine_id, 0.0) + duration
            start = float(entry.get("start", 0.0))
            end = float(entry.get("end", 0.0))
            machine_first_start[machine_id] = min(machine_first_start.get(machine_id, start), start)
            machine_last_end[machine_id] = max(machine_last_end.get(machine_id, end), end)
        for tooling_id in entry.get("tooling_ids", []) or []:
            tooling_busy[tooling_id] = tooling_busy.get(tooling_id, 0.0) + occupied
        for personnel_id in entry.get("personnel_ids", []) or []:
            personnel_busy[personnel_id] = personnel_busy.get(personnel_id, 0.0) + occupied

    order_completion: dict[str, float] = {}
    task_tardiness: dict[str, float] = {}
    tardy_task_ids: list[str] = []
    total_completion_time = 0.0
    total_wait_time = 0.0
    max_flowtime = 0.0
    total_tardiness = 0.0

    for task_id, task in shop.tasks.items():
        completion = task_completion.get(task_id)
        if completion is None:
            if makespan <= 0:
                completion = task.due_date if math.isfinite(task.due_date) else 0.0
            else:
                completion = makespan + max(task.remaining_time, 0.0)
            feasible = False
            task_completion[task_id] = completion
        order_completion[task.order_id] = max(order_completion.get(task.order_id, 0.0), completion)

        tardiness = max(0.0, completion - task.due_date)
        flowtime = completion - task.release_time
        productive = sum(op.processing_time for op in task.operations)
        wait_time = max(0.0, flowtime - productive)

        task_tardiness[task_id] = tardiness
        total_tardiness += tardiness
        total_completion_time += completion
        total_wait_time += wait_time
        max_flowtime = max(max_flowtime, flowtime)
        if tardiness > 1e-9:
            tardy_task_ids.append(task_id)

    tardy_job_count = len(tardy_task_ids)
    avg_tardiness = total_tardiness / len(shop.tasks) if shop.tasks else 0.0
    max_tardiness = max(task_tardiness.values(), default=0.0)
    avg_flowtime = (
        sum(task_completion[task_id] - task.release_time for task_id, task in shop.tasks.items()) / len(shop.tasks)
        if shop.tasks
        else 0.0
    )
    avg_wait_time = total_wait_time / len(shop.tasks) if shop.tasks else 0.0

    order_tardiness: dict[str, float] = {}
    tardy_order_ids: list[str] = []
    main_order_tardy_count = 0
    main_order_tardy_total_time = 0.0
    total_main_orders = 0
    assembly_sync_penalty = 0.0
    order_main_gap: dict[str, float] = {}

    for order_id, order in shop.orders.items():
        completion = order_completion.get(order_id, 0.0)
        tardiness = max(0.0, completion - order.due_date)
        order_tardiness[order_id] = tardiness
        if tardiness > 1e-9:
            tardy_order_ids.append(order_id)

        task_ids = order_task_map.get(order_id, [])
        main_task = shop.tasks.get(order.main_task_id) if order.main_task_id else None
        if main_task:
            total_main_orders += 1
            main_completion = task_completion.get(main_task.id, 0.0)
            main_tardiness = max(0.0, main_completion - order.due_date)
            if main_tardiness > 1e-9:
                main_order_tardy_count += 1
                main_order_tardy_total_time += main_tardiness

            order_gap = 0.0
            for task_id in task_ids:
                if task_id == main_task.id:
                    continue
                order_gap += abs(task_completion.get(task_id, main_completion) - main_completion)
            order_main_gap[order_id] = order_gap
            assembly_sync_penalty += order_gap
        elif task_ids:
            finishes = [task_completion.get(task_id, 0.0) for task_id in task_ids]
            gap = max(finishes) - min(finishes)
            order_main_gap[order_id] = gap
            assembly_sync_penalty += gap

    main_order_tardy_ratio = main_order_tardy_count / total_main_orders if total_main_orders else 0.0

    max_end = max(order_completion.values(), default=makespan)
    if max_end <= 0.0:
        max_end = makespan

    machine_utilization = {
        machine_id: busy / max_end if max_end > 0 else 0.0
        for machine_id, busy in machine_busy.items()
    }
    machine_active_window_utilization = {}
    machine_net_available_utilization = {}
    for machine_id, busy in machine_busy.items():
        first_start = machine_first_start.get(machine_id)
        last_end = machine_last_end.get(machine_id)
        active_span = max(0.0, (last_end - first_start)) if first_start is not None and last_end is not None else 0.0
        active_ratio = busy / active_span if active_span > 1e-9 else 0.0
        machine_active_window_utilization[machine_id] = min(1.0, max(0.0, active_ratio))
        machine = shop.machines.get(machine_id)
        if machine is not None and active_span > 1e-9:
            available_span = machine.available_time_between(first_start, last_end)
            available_ratio = busy / available_span if available_span > 1e-9 else 0.0
            machine_net_available_utilization[machine_id] = min(1.0, max(0.0, available_ratio))
        else:
            machine_net_available_utilization[machine_id] = 0.0
    for machine_id in shop.machines:
        machine_utilization.setdefault(machine_id, 0.0)
        machine_active_window_utilization.setdefault(machine_id, 0.0)
        machine_net_available_utilization.setdefault(machine_id, 0.0)

    tooling_utilization = {
        tooling_id: busy / max_end if max_end > 0 else 0.0
        for tooling_id, busy in tooling_busy.items()
    }
    for tooling_id in shop.toolings:
        tooling_utilization.setdefault(tooling_id, 0.0)

    personnel_utilization = {
        personnel_id: busy / max_end if max_end > 0 else 0.0
        for personnel_id, busy in personnel_busy.items()
    }
    for personnel_id in shop.personnel:
        personnel_utilization.setdefault(personnel_id, 0.0)

    critical_machine_ids = {machine.id for machine in shop.get_critical_machines()}
    critical_utils = [machine_utilization[machine_id] for machine_id in critical_machine_ids if machine_id in machine_utilization]
    critical_active_utils = [machine_active_window_utilization[machine_id] for machine_id in critical_machine_ids if machine_id in machine_active_window_utilization]
    critical_net_utils = [machine_net_available_utilization[machine_id] for machine_id in critical_machine_ids if machine_id in machine_net_available_utilization]
    if not critical_utils:
        critical_utils = list(machine_utilization.values())
    if not critical_active_utils:
        critical_active_utils = list(machine_active_window_utilization.values())
    if not critical_net_utils:
        critical_net_utils = list(machine_net_available_utilization.values())

    bottleneck_load_balance = pstdev(critical_utils) if len(critical_utils) > 1 else 0.0
    bottleneck_machine_ids = [
        machine_id
        for machine_id, _ in sorted(machine_utilization.items(), key=lambda item: item[1], reverse=True)[:5]
    ]

    avg_utilization = result.avg_utilization if result.avg_utilization else (
        sum(machine_utilization.values()) / len(machine_utilization) if machine_utilization else 0.0
    )
    critical_utilization = result.critical_utilization if result.critical_utilization else (
        sum(critical_utils) / len(critical_utils) if critical_utils else avg_utilization
    )
    avg_active_window_utilization = (
        sum(machine_active_window_utilization.values()) / len(machine_active_window_utilization)
        if machine_active_window_utilization else 0.0
    )
    critical_active_window_utilization = (
        sum(critical_active_utils) / len(critical_active_utils)
        if critical_active_utils else avg_active_window_utilization
    )
    avg_net_available_utilization = (
        sum(machine_net_available_utilization.values()) / len(machine_net_available_utilization)
        if machine_net_available_utilization else 0.0
    )
    critical_net_available_utilization = (
        sum(critical_net_utils) / len(critical_net_utils)
        if critical_net_utils else avg_net_available_utilization
    )

    objective_values = {
        "total_tardiness": total_tardiness,
        "makespan": max_end,
        "main_order_tardy_count": float(main_order_tardy_count),
        "main_order_tardy_total_time": main_order_tardy_total_time,
        "main_order_tardy_ratio": main_order_tardy_ratio,
        "avg_utilization": avg_utilization,
        "critical_utilization": critical_utilization,
        "avg_active_window_utilization": avg_active_window_utilization,
        "critical_active_window_utilization": critical_active_window_utilization,
        "avg_net_available_utilization": avg_net_available_utilization,
        "critical_net_available_utilization": critical_net_available_utilization,
        "total_wait_time": total_wait_time,
        "avg_flowtime": avg_flowtime,
        "max_tardiness": max_tardiness,
        "tardy_job_count": float(tardy_job_count),
        "avg_tardiness": avg_tardiness,
        "total_completion_time": total_completion_time,
        "max_flowtime": max_flowtime,
        "bottleneck_load_balance": bottleneck_load_balance,
        "tooling_utilization": (
            sum(tooling_utilization.values()) / len(tooling_utilization) if tooling_utilization else 0.0
        ),
        "personnel_utilization": (
            sum(personnel_utilization.values()) / len(personnel_utilization) if personnel_utilization else 0.0
        ),
        "assembly_sync_penalty": assembly_sync_penalty,
    }

    summary = {
        "completed_operations": completed_operations,
        "total_operations": len(shop.operations),
        "tardy_task_ids": tardy_task_ids,
        "tardy_order_ids": tardy_order_ids,
        "bottleneck_machine_ids": bottleneck_machine_ids,
        "avg_wait_time": avg_wait_time,
        "avg_utilization": avg_utilization,
        "critical_utilization": critical_utilization,
        "avg_active_window_utilization": avg_active_window_utilization,
        "critical_active_window_utilization": critical_active_window_utilization,
        "avg_net_available_utilization": avg_net_available_utilization,
        "critical_net_available_utilization": critical_net_available_utilization,
        "total_main_orders": total_main_orders,
    }

    return ScheduleAnalytics(
        feasible=feasible,
        completed_operations=completed_operations,
        task_completion=task_completion,
        order_completion=order_completion,
        task_tardiness=task_tardiness,
        order_tardiness=order_tardiness,
        order_main_gap=order_main_gap,
        machine_utilization=machine_utilization,
        machine_active_window_utilization=machine_active_window_utilization,
        machine_net_available_utilization=machine_net_available_utilization,
        tooling_utilization=tooling_utilization,
        personnel_utilization=personnel_utilization,
        bottleneck_machine_ids=bottleneck_machine_ids,
        tardy_order_ids=tardy_order_ids,
        tardy_task_ids=tardy_task_ids,
        objective_values=objective_values,
        summary=summary,
    )
