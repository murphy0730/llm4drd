import unittest

from llm4drd.core.models import (
    Machine, MachineType, Personnel, Shift, ShopFloor, Tooling, ToolingType,
)
from llm4drd.core.rules import BUILTIN_RULES
from llm4drd.core.sim_runtime import SimulationRuntime, SimulationRuntimePool
from llm4drd.core.simulator import Simulator
from llm4drd.tests.shop_fixtures import make_graph_context_shop


def _shop_with_colliding_resource_ids() -> ShopFloor:
    """机器/工装/人员共用同一个 ID —— 模型未禁止跨类型重名。"""
    shop = ShopFloor()
    shifts = [Shift(day=day, start_hour=0.0, hours=24.0) for day in range(3)]
    shop.machine_types["cut"] = MachineType("cut", "Cut")
    shop.tooling_types["TL"] = ToolingType("TL", "Fixture")
    shop.machines["R1"] = Machine("R1", "Machine R1", "cut", shifts=list(shifts))
    shop.toolings["R1"] = Tooling("R1", "Tooling R1", "TL", shifts=list(shifts))
    shop.personnel["R1"] = Personnel("R1", "Person R1", ["SK"], shifts=list(shifts))
    shop.build_indexes()
    return shop


def _comparable(result):
    payload = result.to_dict()
    payload["wall_time_ms"] = 0.0
    return payload


class SimulationRuntimeTests(unittest.TestCase):
    def test_reuse_produces_identical_results(self):
        shop = make_graph_context_shop()
        runtime = SimulationRuntime(shop)
        rule = BUILTIN_RULES["ATC"]
        first = Simulator(shop, rule, runtime=runtime).run()
        second = Simulator(shop, rule, runtime=runtime).run()
        self.assertEqual(_comparable(first), _comparable(second))
        self.assertEqual(first.schedule, second.schedule)

    def test_interleaved_rules_do_not_leak_state(self):
        shop = make_graph_context_shop()
        runtime = SimulationRuntime(shop)
        names = sorted(BUILTIN_RULES)
        self.assertGreaterEqual(len(names), 2)
        first = Simulator(shop, BUILTIN_RULES[names[0]], runtime=runtime).run()
        Simulator(shop, BUILTIN_RULES[names[1]], runtime=runtime).run()
        third = Simulator(shop, BUILTIN_RULES[names[0]], runtime=runtime).run()
        self.assertEqual(first.schedule, third.schedule)
        self.assertEqual(_comparable(first), _comparable(third))

    def test_matches_standalone_simulator(self):
        shop = make_graph_context_shop()
        runtime = SimulationRuntime(shop)
        pooled = Simulator(shop, BUILTIN_RULES["ATC"], runtime=runtime).run()
        standalone = Simulator(shop, BUILTIN_RULES["ATC"]).run()
        self.assertEqual(pooled.schedule, standalone.schedule)
        self.assertEqual(_comparable(pooled), _comparable(standalone))

    def test_original_shop_not_mutated(self):
        shop = make_graph_context_shop()
        before = {op_id: (op.status, op.start_time, op.end_time)
                  for op_id, op in shop.operations.items()}
        runtime = SimulationRuntime(shop)
        Simulator(shop, BUILTIN_RULES["ATC"], runtime=runtime).run()
        after = {op_id: (op.status, op.start_time, op.end_time)
                 for op_id, op in shop.operations.items()}
        self.assertEqual(before, after)

    def test_reset_keeps_same_id_resources_of_different_kinds_apart(self):
        # 快照若只按 resource.id 索引，三类资源会互相覆盖 —— 且是静默的，
        # 会在第一次仿真前就把初始状态篡改掉。
        shop = _shop_with_colliding_resource_ids()
        shop.machines["R1"].total_busy_time = 11.0
        shop.toolings["R1"].total_busy_time = 3.0
        shop.personnel["R1"].total_busy_time = 7.0

        runtime = SimulationRuntime(shop)
        runtime.reset()

        self.assertEqual(runtime.shop.machines["R1"].total_busy_time, 11.0)
        self.assertEqual(runtime.shop.toolings["R1"].total_busy_time, 3.0)
        self.assertEqual(runtime.shop.personnel["R1"].total_busy_time, 7.0)

    def test_reset_restores_every_resource_kind_after_mutation(self):
        shop = _shop_with_colliding_resource_ids()
        runtime = SimulationRuntime(shop)
        for bucket in (runtime.shop.machines, runtime.shop.toolings, runtime.shop.personnel):
            bucket["R1"].total_busy_time = 99.0
            bucket["R1"].current_op_id = "dirty"
        runtime.reset()
        for bucket in (runtime.shop.machines, runtime.shop.toolings, runtime.shop.personnel):
            self.assertEqual(bucket["R1"].total_busy_time, 0.0)
            self.assertIsNone(bucket["R1"].current_op_id)

    def test_implicit_runtime_does_not_freeze_the_shop(self):
        # 不传 runtime 时，语义必须与重构前一致：每次 run() 反映调用方当前的 shop。
        # 若把自建 runtime 永久挂在实例上，同一个 Simulator 会一直用旧快照。
        shop = make_graph_context_shop()
        simulator = Simulator(shop, BUILTIN_RULES["ATC"])
        first = simulator.run()

        for op in shop.operations.values():
            op.processing_time *= 5
        shop.build_indexes()

        second = simulator.run()
        fresh = Simulator(shop, BUILTIN_RULES["ATC"]).run()
        self.assertEqual(second.makespan, fresh.makespan)
        self.assertNotEqual(second.makespan, first.makespan)

    def test_explicit_runtime_is_still_reused(self):
        # 显式传入的 runtime 才是"我知道 shop 不变，请复用"的契约。
        shop = make_graph_context_shop()
        runtime = SimulationRuntime(shop)
        simulator = Simulator(shop, BUILTIN_RULES["ATC"], runtime=runtime)
        simulator.run()
        self.assertIs(simulator._runtime, runtime)

    def test_pool_lazily_creates_up_to_max(self):
        shop = make_graph_context_shop()
        pool = SimulationRuntimePool(shop, max_size=2)
        first = pool.acquire()
        second = pool.acquire()
        self.assertIsNot(first, second)
        pool.release(first)
        third = pool.acquire()
        self.assertIs(first, third)


if __name__ == "__main__":
    unittest.main()
