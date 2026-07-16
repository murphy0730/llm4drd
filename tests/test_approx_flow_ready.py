import unittest

from llm4drd.optimization.approx_eval import ApproximateScheduleEvaluator
from llm4drd.optimization.solution_model import CandidateParameters
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class ApproxFlowReadyTests(unittest.TestCase):
    def test_successor_waits_for_slowest_flow_of_predecessor_task(self):
        shop = make_graph_context_shop()
        # OP-13 依赖整个 T-11（含 OP-11、OP-12）；给两道前驱不同的转库时间
        shop.operations["OP-11"].turnover_time = 4.0
        shop.operations["OP-12"].turnover_time = 1.0
        shop.build_indexes()
        evaluator = ApproximateScheduleEvaluator(
            shop, {}, 1.0, 1.0, 1.0, keep_schedule_limit=16,
        )
        candidate = CandidateParameters(
            feature_weights={}, destroy_weights={}, repair_weights={},
        )
        solution = evaluator.evaluate(candidate, "test", 0)
        entries = {entry["op_id"]: entry for entry in solution.schedule}
        self.assertIn("OP-13", entries)
        gate = max(
            entries["OP-11"]["end"] + 4.0,
            entries["OP-12"]["end"] + 1.0,
        )
        self.assertGreaterEqual(entries["OP-13"]["start"] + 1e-9, gate)


if __name__ == "__main__":
    unittest.main()
