import unittest
from collections import Counter

from llm4drd.core.models import ShopFloor
from llm4drd.core.rules import BUILTIN_RULES
from llm4drd.core.simulator import Simulator
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class FlowGateCacheTests(unittest.TestCase):
    def test_flow_gate_computed_at_most_once_per_operation(self):
        shop = make_graph_context_shop()
        calls: Counter = Counter()
        original = ShopFloor.get_operation_flow_ready_time

        def counting(self, op, release_time=None):
            calls[op.id] += 1
            return original(self, op, release_time=release_time)

        ShopFloor.get_operation_flow_ready_time = counting
        try:
            result = Simulator(shop, BUILTIN_RULES["ATC"]).run()
        finally:
            ShopFloor.get_operation_flow_ready_time = original

        self.assertTrue(result.feasible)
        self.assertTrue(calls, "simulation should query the flow gate")
        worst = max(calls.values())
        self.assertLessEqual(
            worst, 1,
            f"flow gate recomputed {worst} times for one operation: {calls}",
        )

    def test_gate_cache_does_not_change_schedule(self):
        shop = make_graph_context_shop()
        shop.operations["OP-11"].turnover_time = 4.0
        shop.build_indexes()
        result = Simulator(shop, BUILTIN_RULES["ATC"]).run()
        entries = {entry["op_id"]: entry for entry in result.schedule}
        # OP-12 前驱 OP-11：开工不得早于 end + turnover
        self.assertGreaterEqual(
            entries["OP-12"]["start"] + 1e-9,
            entries["OP-11"]["end"] + 4.0,
        )


if __name__ == "__main__":
    unittest.main()
