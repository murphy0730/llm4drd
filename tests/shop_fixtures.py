from __future__ import annotations

from copy import deepcopy

from llm4drd.core.models import (
    Machine, MachineType, Operation, Order, Personnel, Shift,
    ShopFloor, Task, Tooling, ToolingType,
)


def make_graph_context_shop() -> ShopFloor:
    shop = ShopFloor()
    shop.machine_types["cut"] = MachineType("cut", "Cut", is_critical=True)
    shop.machine_types["asm"] = MachineType("asm", "Assembly", is_critical=False)
    shifts = [Shift(day=day, start_hour=0.0, hours=24.0) for day in range(14)]
    shop.machines["M-C1"] = Machine("M-C1", "Cutter 1", "cut", shifts=list(shifts))
    shop.machines["M-C2"] = Machine("M-C2", "Cutter 2", "cut", shifts=list(shifts))
    shop.machines["M-A1"] = Machine("M-A1", "Assembly 1", "asm", shifts=list(shifts))
    shop.tooling_types["TL-CUT"] = ToolingType("TL-CUT", "Cut fixture")
    shop.toolings["TL-1"] = Tooling("TL-1", "Fixture 1", "TL-CUT", shifts=list(shifts))
    shop.personnel["P-1"] = Personnel("P-1", "Assembler", ["SK-ASM"], shifts=list(shifts))

    shop.orders["O-1"] = Order("O-1", "Order 1", release_time=0.0, due_date=40.0, priority=3)
    shop.orders["O-2"] = Order("O-2", "Order 2", release_time=1.0, due_date=48.0, priority=1)
    t11 = Task("T-11", "O-1", "Part", False, [], release_time=0.0, due_date=24.0)
    t12 = Task("T-12", "O-1", "Main", True, ["T-11"], release_time=0.0, due_date=40.0)
    t21 = Task("T-21", "O-2", "Second", True, [], release_time=1.0, due_date=48.0)
    shop.tasks.update({task.id: task for task in (t11, t12, t21)})

    op11 = Operation("OP-11", "T-11", "Cut", "cut", 4.0, eligible_machine_ids=["M-C1"])
    op12 = Operation("OP-12", "T-11", "Finish", "cut", 2.0, predecessor_ops=["OP-11"])
    op13 = Operation(
        "OP-13", "T-12", "Assemble", "asm", 3.0,
        predecessor_tasks=["T-11"], required_tooling_types=["TL-CUT"],
        required_personnel_skills=["SK-ASM"],
    )
    op21 = Operation("OP-21", "T-21", "Other cut", "cut", 5.0)
    for task, operations in ((t11, [op11, op12]), (t12, [op13]), (t21, [op21])):
        task.operations.extend(operations)
        shop.orders[task.order_id].task_ids.append(task.id)
        if task.is_main:
            shop.orders[task.order_id].main_task_id = task.id
        for operation in operations:
            shop.operations[operation.id] = operation
    shop.build_indexes()
    return shop


def canonical_graph_signature(graph) -> tuple:
    nodes = tuple(sorted((node_id, tuple(sorted(attrs.items()))) for node_id, attrs in graph.nodes(data=True)))
    edges = tuple(sorted((source, target, tuple(sorted(attrs.items()))) for source, target, attrs in graph.edges(data=True)))
    return nodes, edges


def hybrid_result_signature(result) -> dict:
    payload = deepcopy(result.to_export_dict())
    payload["baseline"]["metrics"]["wall_time_ms"] = 0.0
    for solution in payload["solutions"]:
        solution["metrics"]["wall_time_ms"] = 0.0
    return {
        "baseline": payload["baseline"],
        "solutions": payload["solutions"],
        "archive_size": payload["archive_size"],
        "found_solution_count": payload["found_solution_count"],
        "generations_completed": payload["generations_completed"],
        "total_evaluations": payload["total_evaluations"],
        "approximate_evaluations": payload["approximate_evaluations"],
        "exact_evaluations": payload["exact_evaluations"],
    }
