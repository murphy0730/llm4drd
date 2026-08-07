import asyncio
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import BackgroundTasks, HTTPException
from pydantic import ValidationError

from llm4drd.api import server
from llm4drd.core.models import Machine, Operation, Order, Shift, ShopFloor, Task
from llm4drd.data.db import InstanceStore, init_db
from llm4drd.optimization.exact import ExactSolver
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class ExactWindowScopeTests(unittest.TestCase):
    def test_window_uses_selected_schedule_start_not_release_time(self):
        shop = make_graph_context_shop()
        shop.tasks["T-12"].release_time = 500.0
        schedule = [
            {"op_id": "OP-13", "task_id": "T-12", "start": 2.0, "end": 5.0},
            {"op_id": "OP-21", "task_id": "T-21", "start": 24.0, "end": 29.0},
        ]

        order_ids, starts = server._exact_window_order_ids(shop, schedule, 1)

        self.assertEqual(order_ids, ["O-1"])
        self.assertEqual(starts, {"O-1": 2.0})

    def test_window_end_is_exclusive(self):
        shop = make_graph_context_shop()
        schedule = [
            {"op_id": "OP-13", "task_id": "T-12", "start": 23.999, "end": 24.0},
            {"op_id": "OP-21", "task_id": "T-21", "start": 24.0, "end": 29.0},
        ]

        order_ids, _ = server._exact_window_order_ids(shop, schedule, 1)

        self.assertEqual(order_ids, ["O-1"])

    def test_scope_keeps_same_order_tasks_and_external_dependency_floor(self):
        shop = make_graph_context_shop()
        shop.orders["O-X"] = Order("O-X", "External", task_ids=["T-X"])
        external_task = Task("T-X", "O-X", "External task", False)
        external_op = Operation(
            "OP-X", "T-X", "External op", "cut", 2.0, turnover_time=3.0,
            eligible_machine_ids=["M-C1"],
        )
        external_task.operations.append(external_op)
        shop.tasks[external_task.id] = external_task
        shop.operations[external_op.id] = external_op
        shop.operations["OP-13"].predecessor_tasks.append("T-X")
        shop.build_indexes()
        seed = [
            {"op_id": "OP-X", "task_id": "T-X", "start": 2.0, "end": 5.0, "machine_id": "M-C1"},
            {"op_id": "OP-11", "task_id": "T-11", "start": 0.0, "end": 4.0, "machine_id": "M-C1"},
            {"op_id": "OP-12", "task_id": "T-11", "start": 4.0, "end": 6.0, "machine_id": "M-C2"},
            {"op_id": "OP-13", "task_id": "T-12", "start": 8.0, "end": 11.0, "machine_id": "M-A1"},
        ]

        scope, hints, boundary_count = server._build_exact_window_shop(shop, ["O-1"], seed)

        self.assertEqual(set(scope.tasks), {"T-11", "T-12"})
        self.assertEqual(set(scope.operations), {"OP-11", "OP-12", "OP-13"})
        self.assertNotIn("T-X", scope.operations["OP-13"].predecessor_tasks)
        self.assertAlmostEqual(scope.operations["OP-13"].flow_release_floor, 8.0)
        self.assertEqual(boundary_count, 1)
        self.assertEqual({entry["op_id"] for entry in hints}, set(scope.operations))

    def test_weight_validation(self):
        self.assertEqual(
            server._validate_exact_window_weights({"makespan": 0.4, "total_tardiness": 0.6}),
            {"makespan": 0.4, "total_tardiness": 0.6},
        )
        with self.assertRaises(HTTPException):
            server._validate_exact_window_weights({"makespan": 0.7, "total_tardiness": 0.4})
        with self.assertRaises(HTTPException):
            server._validate_exact_window_weights({"makespan": float("nan")})
        with self.assertRaises(HTTPException):
            server._validate_exact_window_weights({"assembly_sync_penalty": 1.0})

    def test_days_are_limited_to_one_through_ten(self):
        with self.assertRaises(ValidationError):
            server.ExactWindowSolveReq(window_days=0, objective_weights={"makespan": 1.0})
        with self.assertRaises(ValidationError):
            server.ExactWindowSolveReq(window_days=11, objective_weights={"makespan": 1.0})


class ExactWindowSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import ortools  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("ortools not installed") from exc

    def test_exact_solver_consumes_schedule_hints_without_locking(self):
        shop = make_graph_context_shop()
        seed, _ = server._exact_window_seed_plan(
            shop,
            server.ExactWindowSolveReq(objective_weights={"makespan": 1.0}),
        )

        result = ExactSolver(
            shop,
            objectives=["makespan"],
            objective_weights={"makespan": 1.0},
            time_limit_s=5,
        ).solve(warm_start_schedule=seed.schedule)

        self.assertIn(result.status, {"OPTIMAL", "FEASIBLE"})
        self.assertEqual(result.request["hinted_operation_count"], len(shop.operations))
        self.assertTrue(math.isfinite(result.objectives["makespan"]))

    def test_exact_solver_continues_long_operation_across_shared_shifts(self):
        shop = ShopFloor()
        shifts = [
            Shift(day=0, start_hour=8.0, hours=8.0),
            Shift(day=1, start_hour=8.0, hours=8.0),
            Shift(day=2, start_hour=8.0, hours=8.0),
        ]
        shop.machines["M1"] = Machine("M1", "Machine 1", "cut", shifts=shifts)
        order = Order("O1", "Order 1", task_ids=["T1"], due_date=72.0)
        task = Task("T1", "O1", "Main task", True, due_date=72.0)
        operation = Operation(
            "OP1",
            "T1",
            "Long operation",
            "cut",
            12.0,
            eligible_machine_ids=["M1"],
        )
        task.operations.append(operation)
        shop.orders[order.id] = order
        shop.tasks[task.id] = task
        shop.operations[operation.id] = operation
        shop.build_indexes()

        result = ExactSolver(shop, time_limit_s=5).solve(
            warm_start_schedule=[
                {"op_id": "OP1", "machine_id": "M1", "start": 0.0, "end": 28.0}
            ]
        )

        self.assertIn(result.status, {"OPTIMAL", "FEASIBLE"})
        self.assertEqual(result.request["calendar_mode"], "resource_calendar_preemptive")
        self.assertAlmostEqual(result.schedule[0]["start"], 0.0)
        self.assertAlmostEqual(result.schedule[0]["end"], 28.0)

    def test_async_endpoint_runs_without_hybrid_task(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        db_path = str(Path(temp.name) / "exact-window.db")
        init_db(db_path)
        store = InstanceStore(db_path)
        store.save_from_shopfloor(make_graph_context_shop())
        originals = (
            server.inst_store,
            server.shop,
            server._active_shop_cache,
            server._hybrid_tasks,
            server._exact_tasks,
        )

        def restore():
            (
                server.inst_store,
                server.shop,
                server._active_shop_cache,
                server._hybrid_tasks,
                server._exact_tasks,
            ) = originals

        self.addCleanup(restore)
        server.inst_store = store
        server.shop = None
        server._active_shop_cache = None
        server._hybrid_tasks = {}
        server._exact_tasks = {}

        async def run():
            background = BackgroundTasks()
            response = await server.exact_window_solve(
                server.ExactWindowSolveReq(
                    window_days=1,
                    objective_weights={"makespan": 1.0},
                    time_limit_s=5,
                ),
                background,
            )
            await background()
            return response, await server.exact_window_status(response["task_id"])

        response, status = asyncio.run(run())

        self.assertEqual(status["status"], "done", status.get("error"))
        self.assertEqual(status["result"]["scope"]["kind"], "window")
        self.assertEqual(status["result"]["schedule_total"], 4)
        self.assertFalse(server._hybrid_tasks)
        self.assertEqual(response["status"], "running")


if __name__ == "__main__":
    unittest.main()
