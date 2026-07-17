from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

from llm4drd.ai import evolution as evolution_module
from llm4drd.core.models import (
    Machine,
    MachineType,
    Operation,
    OpStatus,
    Order,
    Shift,
    ShopFloor,
    Task,
)
from llm4drd.core.rules import BUILTIN_RULES
from llm4drd.optimization import pareto as pareto_module
from llm4drd.scheduling.online import OnlineSchedulerV3
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class _CountingDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.values_calls = 0

    def values(self):
        self.values_calls += 1
        return super().values()


def _many_machine_shop(machine_count: int = 24) -> ShopFloor:
    shifts = [Shift(day=day, start_hour=0.0, hours=24.0) for day in range(3)]
    shop = ShopFloor()
    for index in range(machine_count):
        type_id = f"P{index}"
        machine_id = f"M{index}"
        shop.machine_types[type_id] = MachineType(type_id, type_id)
        shop.machines[machine_id] = Machine(machine_id, machine_id, type_id, shifts=list(shifts))

    task = Task("T1", "O1", "T1", due_date=100.0)
    for index in range(machine_count):
        op = Operation(
            id=f"OP{index}",
            task_id=task.id,
            name=f"OP{index}",
            process_type=f"P{index}",
            processing_time=1.0,
        )
        task.operations.append(op)
        shop.operations[op.id] = op
    shop.tasks[task.id] = task
    shop.orders["O1"] = Order(
        "O1",
        "O1",
        due_date=100.0,
        task_ids=[task.id],
        main_task_id=task.id,
    )
    shop.build_indexes()
    return shop


class OptimizerRuntimeReuseTests(unittest.TestCase):
    def test_pareto_evaluation_passes_one_shared_runtime_to_all_simulators(self):
        shop = make_graph_context_shop()
        original_runtime = getattr(pareto_module, "SimulationRuntime", None)
        original_simulator = pareto_module.Simulator
        runtimes = []

        class FakeRuntime:
            def __init__(self, runtime_shop):
                self.shop = runtime_shop

        class FakeResult:
            total_tardiness = 0.0
            makespan = 1.0
            schedule = []

        class FakeSimulator:
            def __init__(self, sim_shop, rule, runtime=None):
                del sim_shop, rule
                runtimes.append(runtime)

            def run(self):
                return FakeResult()

        try:
            pareto_module.SimulationRuntime = FakeRuntime
            pareto_module.Simulator = FakeSimulator
            pareto_module.ParetoOptimizer(shop, ["makespan"]).evaluate(
                {"A": BUILTIN_RULES["ATC"], "B": BUILTIN_RULES["EDD"]}
            )
        finally:
            pareto_module.Simulator = original_simulator
            if original_runtime is None:
                delattr(pareto_module, "SimulationRuntime")
            else:
                pareto_module.SimulationRuntime = original_runtime

        self.assertEqual(len(runtimes), 2)
        self.assertIsNotNone(runtimes[0])
        self.assertIs(runtimes[0], runtimes[1])

    def test_nsga2_reuses_runtime_between_candidate_evaluations(self):
        shop = make_graph_context_shop()
        original_runtime = getattr(pareto_module, "SimulationRuntime", None)
        original_simulator = pareto_module.Simulator
        runtimes = []

        class FakeRuntime:
            def __init__(self, runtime_shop):
                self.shop = runtime_shop

        class FakeResult:
            total_tardiness = 0.0
            makespan = 1.0

        class FakeSimulator:
            def __init__(self, sim_shop, rule, runtime=None):
                del sim_shop, rule
                runtimes.append(runtime)

            def run(self):
                return FakeResult()

        try:
            pareto_module.SimulationRuntime = FakeRuntime
            pareto_module.Simulator = FakeSimulator
            optimizer = pareto_module.NSGA2Optimizer(
                shop,
                ["makespan"],
                pop_size=2,
                generations=0,
            )
            optimizer._evaluate([1.0] * optimizer.K)
            optimizer._evaluate([0.5] * optimizer.K)
        finally:
            pareto_module.Simulator = original_simulator
            if original_runtime is None:
                delattr(pareto_module, "SimulationRuntime")
            else:
                pareto_module.SimulationRuntime = original_runtime

        self.assertEqual(len(runtimes), 2)
        self.assertIsNotNone(runtimes[0])
        self.assertIs(runtimes[0], runtimes[1])

    def test_evolution_reuses_one_runtime_per_training_instance(self):
        shop = make_graph_context_shop()
        original_runtime = evolution_module.SimulationRuntime
        original_simulator = evolution_module.Simulator
        runtimes = []

        class FakeRuntime:
            def __init__(self, runtime_shop):
                self.shop = runtime_shop

        class FakeResult:
            total_tardiness = 0.0

        class FakeSimulator:
            def __init__(self, sim_shop, rule, runtime=None):
                del sim_shop, rule
                runtimes.append(runtime)

            def run(self):
                return FakeResult()

        engine = evolution_module.EvolutionEngine(
            evolution_module.EvolutionConfig(
                population_size=2,
                elite_size=2,
                max_generations=1,
            ),
            evolution_module.LLMInterface(api_key=""),
        )
        engine.population = [
            evolution_module.RuleIndividual(
                id=f"r{index}",
                name=f"r{index}",
                code="def evolved_rule(op, machine, f, shop): return 0.0",
            )
            for index in range(2)
        ]
        try:
            evolution_module.SimulationRuntime = FakeRuntime
            evolution_module.Simulator = FakeSimulator
            engine.evolve([shop])
        finally:
            evolution_module.SimulationRuntime = original_runtime
            evolution_module.Simulator = original_simulator

        self.assertEqual(len(runtimes), 2)
        self.assertIsNotNone(runtimes[0])
        self.assertIs(runtimes[0], runtimes[1])


class OnlineDispatchIndexTests(unittest.TestCase):
    def test_dispatch_does_not_rescan_all_operations_for_each_machine(self):
        scheduler = OnlineSchedulerV3(_many_machine_shop(), rule_name="ATC")
        counted = _CountingDict(scheduler.sim_shop.operations)
        scheduler.sim_shop.operations = counted

        scheduler._dispatch_idle_machines(0.0, BUILTIN_RULES["ATC"])

        self.assertEqual(
            counted.values_calls,
            0,
            "派工应从工艺类型 READY 桶取候选，不能对每台机器全扫描 operations",
        )

    def test_initial_ready_operations_are_seeded_into_dispatch_buckets(self):
        shop = _many_machine_shop(machine_count=2)
        shop.operations["OP0"].status = OpStatus.READY
        scheduler = OnlineSchedulerV3(shop, rule_name="ATC")

        status = scheduler.advance(2.0)

        self.assertEqual(status["ops_completed"], status["ops_total"])


class DemoPackageBootstrapTests(unittest.TestCase):
    def test_demo_imports_with_documented_pythonpath(self):
        repo = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo.parent)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy; runpy.run_path('demo.py', run_name='demo_import_test')",
            ],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
