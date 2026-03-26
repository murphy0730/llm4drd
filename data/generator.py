from __future__ import annotations

import random
from collections import defaultdict
from datetime import datetime

from ..core.models import (
    Machine,
    MachineType,
    Operation,
    Order,
    Personnel,
    Shift,
    ShopFloor,
    Task,
    Tooling,
    ToolingType,
)
from ..core.time_utils import default_plan_start, ensure_aware


DEFAULT_PROCESS_TYPES = [
    ("turning", "Turning", True),
    ("milling", "Milling", True),
    ("grinding", "Grinding", False),
    ("drilling", "Drilling", False),
    ("boring", "Boring", False),
    ("coating", "Coating", False),
    ("assembly", "Assembly", True),
    ("testing", "Testing", False),
]


class InstanceGenerator:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

    @staticmethod
    def _resource_parallel_units(resources: list, schedule_days: int) -> float:
        horizon_hours = max(24.0 * max(1, schedule_days), 1.0)
        total_hours = 0.0
        for resource in resources:
            total_hours += sum(max(0.0, shift.hours) for shift in getattr(resource, "shifts", []))
        if total_hours <= 0:
            return float(len(resources))
        return max(total_hours / horizon_hours, 0.1)

    def _effective_capacity_by_process(self, shop: ShopFloor, process_catalog: list, schedule_days: int) -> dict[str, float]:
        capacity: dict[str, float] = {}
        for process_type_id, _, _ in process_catalog:
            machines = [machine for machine in shop.machines.values() if machine.type_id == process_type_id]
            machine_units = self._resource_parallel_units(machines, schedule_days)
            tooling_units = self._resource_parallel_units(
                [tool for tool in shop.toolings.values() if tool.type_id == f"tool_{process_type_id}"],
                schedule_days,
            )
            personnel_units = self._resource_parallel_units(
                [person for person in shop.personnel.values() if f"skill_{process_type_id}" in person.skills],
                schedule_days,
            )
            capacity[process_type_id] = max(0.2, min(machine_units, tooling_units, personnel_units))
        return capacity

    def _assign_generated_due_dates(
        self,
        shop: ShopFloor,
        order_profiles: dict[str, dict],
        process_catalog: list,
        due_date_factor: float,
        schedule_days: int,
    ) -> None:
        process_capacity = self._effective_capacity_by_process(shop, process_catalog, schedule_days)
        backlog_work = {process_type_id: 0.0 for process_type_id, _, _ in process_catalog}

        ordered_ids = sorted(
            order_profiles,
            key=lambda order_id: (
                order_profiles[order_id]["release_time"],
                -shop.orders[order_id].priority,
                order_id,
            ),
        )
        previous_release = 0.0
        slack_multiplier_base = max(1.15, 0.95 + 0.58 * due_date_factor)

        for order_id in ordered_ids:
            profile = order_profiles[order_id]
            order = shop.orders[order_id]
            release_time = profile["release_time"]
            delta_release = max(0.0, release_time - previous_release)
            if delta_release > 0:
                for process_type_id, capacity in process_capacity.items():
                    backlog_work[process_type_id] = max(0.0, backlog_work[process_type_id] - delta_release * capacity)
            previous_release = release_time

            process_work = profile["process_work"]
            serial_lead = 0.0
            queue_delay = 0.0
            weighted_queue = 0.0
            total_work = max(profile["order_workload"], 1e-6)
            active_types = [process_type_id for process_type_id, work in process_work.items() if work > 1e-9]

            for process_type_id in active_types:
                capacity = process_capacity.get(process_type_id, 1.0)
                work = process_work[process_type_id]
                serial_lead += work / capacity
                local_queue = backlog_work.get(process_type_id, 0.0) / capacity
                queue_delay = max(queue_delay, local_queue)
                weighted_queue += local_queue * (work / total_work)

            structure_factor = 1.0 + 0.035 * max(0, profile["task_count"] - 1) + 0.012 * max(0, profile["op_count"] - 4)
            priority_factor = max(0.82, 1.1 - 0.045 * max(1, order.priority))
            buffer_hours = max(2.0, 0.08 * total_work + 0.35 * profile["task_count"])
            due_offset = (
                queue_delay
                + 0.45 * weighted_queue
                + serial_lead * slack_multiplier_base * structure_factor * priority_factor
                + buffer_hours
            )
            order_due = round(release_time + due_offset, 3)
            order.due_date = order_due

            for task_id in profile["task_ids"]:
                task = shop.tasks.get(task_id)
                if task is not None:
                    task.due_date = order_due

            for process_type_id in active_types:
                backlog_work[process_type_id] += process_work[process_type_id]

    def generate(
        self,
        num_orders: int = 10,
        tasks_per_order: tuple = (2, 5),
        ops_per_task: tuple = (2, 5),
        machines_per_type: int = 3,
        process_types: list | None = None,
        processing_time_range: tuple = (1, 12),
        due_date_factor: float = 1.5,
        arrival_spread: float = 0.0,
        day_shift_hours: float = 10,
        night_shift_hours: float = 8,
        schedule_days: int | None = 0,
        maintenance_prob: float = 0.05,
        toolings_per_type: int | None = None,
        personnel_per_skill: int | None = None,
        plan_start_at: datetime | None = None,
    ) -> ShopFloor:
        shop = ShopFloor(plan_start_at=ensure_aware(plan_start_at or default_plan_start()))
        process_catalog = process_types or DEFAULT_PROCESS_TYPES
        initial_schedule_days = max(7, int(schedule_days or 0)) if (schedule_days or 0) > 0 else 21

        tooling_count = toolings_per_type if toolings_per_type is not None else max(1, machines_per_type - 1)
        personnel_count = personnel_per_skill if personnel_per_skill is not None else max(1, machines_per_type)

        for process_type_id, process_type_name, is_critical in process_catalog:
            shop.machine_types[process_type_id] = MachineType(
                id=process_type_id,
                name=process_type_name,
                is_critical=is_critical,
            )
            shop.tooling_types[f"tool_{process_type_id}"] = ToolingType(
                id=f"tool_{process_type_id}",
                name=f"{process_type_name} tooling",
            )

            machine_total = machines_per_type if is_critical else max(1, machines_per_type - 1)
            for index in range(machine_total):
                shifts = []
                for day in range(initial_schedule_days):
                    if random.random() < maintenance_prob:
                        shifts.append(Shift(day=day, start_hour=0, hours=0))
                    else:
                        shifts.append(Shift(day=day, start_hour=8, hours=day_shift_hours))
                        if night_shift_hours > 0:
                            shifts.append(Shift(day=day, start_hour=20, hours=night_shift_hours))
                machine = Machine(
                    id=f"{process_type_id}_{index + 1}",
                    name=f"{process_type_name}-{index + 1}",
                    type_id=process_type_id,
                    shifts=shifts,
                )
                shop.machines[machine.id] = machine

            for index in range(tooling_count):
                tool = Tooling(
                    id=f"TL-{process_type_id}-{index + 1:02d}",
                    name=f"{process_type_name} tooling-{index + 1}",
                    type_id=f"tool_{process_type_id}",
                    shifts=[Shift(day=day, start_hour=8, hours=day_shift_hours) for day in range(initial_schedule_days)],
                )
                shop.toolings[tool.id] = tool

            for index in range(personnel_count):
                person = Personnel(
                    id=f"PS-{process_type_id}-{index + 1:02d}",
                    name=f"{process_type_name} operator-{index + 1}",
                    skills=[f"skill_{process_type_id}"],
                    shifts=[Shift(day=day, start_hour=8, hours=day_shift_hours) for day in range(initial_schedule_days)],
                )
                if night_shift_hours > 0 and index % 2 == 0:
                    for day in range(initial_schedule_days):
                        person.shifts.append(Shift(day=day, start_hour=20, hours=night_shift_hours))
                shop.personnel[person.id] = person

        pure_process_types = [item[0] for item in process_catalog if item[0] not in {"assembly", "testing"}]
        order_profiles: dict[str, dict] = {}

        for order_index in range(num_orders):
            order_id = f"ORD-{order_index + 1:04d}"
            release_offset = random.uniform(0, arrival_spread * num_orders * 5) if arrival_spread > 0 else 0.0
            order = Order(
                id=order_id,
                name=f"Order-{order_index + 1}",
                release_time=round(release_offset, 3),
                priority=random.randint(1, 5),
            )

            task_total = random.randint(*tasks_per_order)
            sub_task_ids = []
            order_workload = 0.0
            process_work = defaultdict(float)
            order_task_ids: list[str] = []
            op_count = 0

            for task_index in range(max(1, task_total - 1)):
                task_id = f"T-{order_index + 1:04d}-{task_index + 1:02d}"
                task = Task(
                    id=task_id,
                    order_id=order_id,
                    name=f"Part-{task_index + 1}",
                    release_time=release_offset,
                )
                op_total = random.randint(*ops_per_task)
                predecessor_op_id = None
                for op_index in range(op_total):
                    process_type = random.choice(pure_process_types)
                    duration = round(random.uniform(*processing_time_range), 3)
                    order_workload += duration
                    process_work[process_type] += duration
                    op_count += 1
                    operation = Operation(
                        id=f"OP-{order_index + 1:04d}-{task_index + 1:02d}-{op_index + 1:02d}",
                        task_id=task_id,
                        name=f"{shop.machine_types[process_type].name} step",
                        process_type=process_type,
                        processing_time=duration,
                        predecessor_ops=[predecessor_op_id] if predecessor_op_id else [],
                        required_tooling_types=[f"tool_{process_type}"],
                        required_personnel_skills=[f"skill_{process_type}"],
                    )
                    task.operations.append(operation)
                    shop.operations[operation.id] = operation
                    predecessor_op_id = operation.id

                task.due_date = float("inf")
                shop.tasks[task_id] = task
                order.task_ids.append(task_id)
                sub_task_ids.append(task_id)
                order_task_ids.append(task_id)

            main_task_id = f"T-{order_index + 1:04d}-MAIN"
            main_task = Task(
                id=main_task_id,
                order_id=order_id,
                name="Assembly and test",
                is_main=True,
                predecessor_task_ids=list(sub_task_ids),
                release_time=release_offset,
            )

            assembly_duration = round(random.uniform(2, 15), 3)
            order_workload += assembly_duration
            process_work["assembly"] += assembly_duration
            op_count += 1
            assembly_op = Operation(
                id=f"OP-{order_index + 1:04d}-ASM",
                task_id=main_task_id,
                name="Assembly",
                process_type="assembly",
                processing_time=assembly_duration,
                predecessor_tasks=list(sub_task_ids),
                required_tooling_types=["tool_assembly"],
                required_personnel_skills=["skill_assembly"],
            )
            main_task.operations.append(assembly_op)
            shop.operations[assembly_op.id] = assembly_op

            testing_duration = round(random.uniform(1, 5), 3)
            order_workload += testing_duration
            process_work["testing"] += testing_duration
            op_count += 1
            testing_op = Operation(
                id=f"OP-{order_index + 1:04d}-TEST",
                task_id=main_task_id,
                name="Testing",
                process_type="testing",
                processing_time=testing_duration,
                predecessor_ops=[assembly_op.id],
                required_tooling_types=["tool_testing"],
                required_personnel_skills=["skill_testing"],
            )
            main_task.operations.append(testing_op)
            shop.operations[testing_op.id] = testing_op

            order.due_date = float("inf")
            main_task.due_date = float("inf")

            shop.tasks[main_task_id] = main_task
            order.task_ids.append(main_task_id)
            order.main_task_id = main_task_id
            shop.orders[order_id] = order
            order_task_ids.append(main_task_id)
            order_profiles[order_id] = {
                "release_time": round(release_offset, 3),
                "task_ids": list(order_task_ids),
                "task_count": len(order_task_ids),
                "op_count": op_count,
                "order_workload": round(order_workload, 6),
                "process_work": dict(process_work),
            }

        self._assign_generated_due_dates(
            shop=shop,
            order_profiles=order_profiles,
            process_catalog=process_catalog,
            due_date_factor=due_date_factor,
            schedule_days=initial_schedule_days,
        )

        shop.build_indexes()
        shop.ensure_calendar_capacity(
            min_days=max(initial_schedule_days, 14),
            safety_factor=1.45,
            max_days=max(720, max(initial_schedule_days, 14)),
        )
        return shop

    def generate_training_set(self, n=5, **kwargs) -> list[ShopFloor]:
        instances = []
        for _ in range(n):
            params = dict(kwargs)
            params.setdefault("num_orders", random.randint(5, 15))
            params.setdefault("due_date_factor", random.uniform(1.0, 2.0))
            instances.append(self.generate(**params))
        return instances
