"""
问题实例生成器 v3 — 面向业务, 规模可配置, CSV导出
=================================================
生成结构: 订单 → 任务(子任务+主任务) → 工序(前置关系) → 机器(类别+班次)
机器按工艺分类: 车/铣/磨/钻/镗/装配/检测 等
"""
import random
from ..core.models import (
    ShopFloor, MachineType, Machine, Shift, Order, Task, Operation,
    OpStatus, uid
)


# 默认工艺类别定义
DEFAULT_PROCESS_TYPES = [
    ("turning", "车床", True),
    ("milling", "铣床", True),
    ("grinding", "磨床", False),
    ("drilling", "钻床", False),
    ("boring", "镗床", False),
    ("coating", "涂装线", False),
    ("assembly", "装配站", True),
    ("testing", "检测站", False),
]


class InstanceGenerator:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

    def generate(
        self,
        num_orders: int = 10,
        tasks_per_order: tuple = (2, 5),
        ops_per_task: tuple = (2, 5),
        machines_per_type: int = 3,
        process_types: list = None,
        processing_time_range: tuple = (1, 12),  # 小时
        due_date_factor: float = 1.5,
        arrival_spread: float = 0.0,
        day_shift_hours: float = 10,
        night_shift_hours: float = 8,
        schedule_days: int = 30,
        maintenance_prob: float = 0.05,
    ) -> ShopFloor:
        shop = ShopFloor()
        ptypes = process_types or DEFAULT_PROCESS_TYPES

        # 1. 机器类别和机器
        for pt_id, pt_name, is_crit in ptypes:
            mt = MachineType(id=pt_id, name=pt_name, is_critical=is_crit)
            shop.machine_types[pt_id] = mt
            n_machines = machines_per_type if is_crit else max(1, machines_per_type - 1)
            for i in range(n_machines):
                mid = f"{pt_id}_{i+1}"
                shifts = []
                for d in range(schedule_days):
                    if random.random() < maintenance_prob:
                        shifts.append(Shift(day=d, start_hour=0, hours=0))  # 维修
                    else:
                        shifts.append(Shift(day=d, start_hour=8, hours=day_shift_hours))
                        shifts.append(Shift(day=d, start_hour=20, hours=night_shift_hours))
                m = Machine(id=mid, name=f"{pt_name}-{i+1}", type_id=pt_id, shifts=shifts)
                shop.machines[mid] = m

        proc_type_ids = [p[0] for p in ptypes if p[0] != "assembly" and p[0] != "testing"]
        asm_type = "assembly"
        test_type = "testing"

        # 2. 订单
        total_proc_time_sum = 0
        for oi in range(num_orders):
            order_id = f"ORD-{oi+1:04d}"
            release = random.uniform(0, arrival_spread * num_orders * 5) if arrival_spread > 0 else 0
            order = Order(
                id=order_id,
                name=f"订单-{oi+1}",
                release_time=round(release, 1),
                priority=random.randint(1, 5),
            )

            n_tasks = random.randint(*tasks_per_order)
            sub_task_ids = []
            order_total_pt = 0

            # 子任务 (零件/半成品加工)
            for ti in range(n_tasks - 1):
                task_id = f"T-{oi+1:04d}-{ti+1:02d}"
                task = Task(
                    id=task_id, order_id=order_id,
                    name=f"子件-{ti+1}", is_main=False,
                    release_time=release,
                )
                n_ops = random.randint(*ops_per_task)
                prev_op_id = None
                for opi in range(n_ops):
                    op_id = f"OP-{oi+1:04d}-{ti+1:02d}-{opi+1:02d}"
                    pt = random.choice(proc_type_ids)
                    proc_time = round(random.uniform(*processing_time_range), 1)
                    order_total_pt += proc_time
                    op = Operation(
                        id=op_id, task_id=task_id,
                        name=f"{shop.machine_types[pt].name}工序",
                        process_type=pt,
                        processing_time=proc_time,
                        predecessor_ops=[prev_op_id] if prev_op_id else [],
                    )
                    task.operations.append(op)
                    shop.operations[op_id] = op
                    prev_op_id = op_id

                task.due_date = round(release + order_total_pt * due_date_factor * 1.2, 1)
                shop.tasks[task_id] = task
                order.task_ids.append(task_id)
                sub_task_ids.append(task_id)

            # 主任务 (装配 + 检测)
            main_task_id = f"T-{oi+1:04d}-MAIN"
            main_task = Task(
                id=main_task_id, order_id=order_id,
                name=f"总装检测", is_main=True,
                predecessor_task_ids=list(sub_task_ids),
                release_time=release,
            )

            # 装配工序
            asm_time = round(random.uniform(2, 15), 1)
            order_total_pt += asm_time
            asm_op = Operation(
                id=f"OP-{oi+1:04d}-ASM",
                task_id=main_task_id,
                name="总装",
                process_type=asm_type,
                processing_time=asm_time,
                predecessor_tasks=list(sub_task_ids),
            )
            main_task.operations.append(asm_op)
            shop.operations[asm_op.id] = asm_op

            # 检测工序
            test_time = round(random.uniform(1, 5), 1)
            order_total_pt += test_time
            test_op = Operation(
                id=f"OP-{oi+1:04d}-TEST",
                task_id=main_task_id,
                name="检测",
                process_type=test_type,
                processing_time=test_time,
                predecessor_ops=[asm_op.id],
            )
            main_task.operations.append(test_op)
            shop.operations[test_op.id] = test_op

            # 设置交期
            n_machines_total = len(shop.machines)
            est = release + order_total_pt / max(n_machines_total * 0.3, 1) * due_date_factor
            order.due_date = round(est, 1)
            main_task.due_date = order.due_date

            shop.tasks[main_task_id] = main_task
            order.task_ids.append(main_task_id)
            order.main_task_id = main_task_id
            shop.orders[order_id] = order
            total_proc_time_sum += order_total_pt

        shop.build_indexes()
        return shop

    def generate_training_set(self, n=5, **kw) -> list[ShopFloor]:
        instances = []
        for i in range(n):
            kw2 = dict(kw)
            kw2.setdefault("num_orders", random.randint(5, 15))
            kw2.setdefault("due_date_factor", random.uniform(1.0, 2.0))
            instances.append(self.generate(**kw2))
        return instances
