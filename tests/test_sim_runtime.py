import unittest

from llm4drd.core.rules import BUILTIN_RULES
from llm4drd.core.sim_runtime import SimulationRuntime, SimulationRuntimePool
from llm4drd.core.simulator import Simulator
from llm4drd.tests.shop_fixtures import make_graph_context_shop


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
