import unittest

from llm4drd.optimization.solution_model import CandidateParameters, OptimizationSolution


def _make_solution() -> OptimizationSolution:
    return OptimizationSolution(
        solution_id="S-1",
        source="test",
        generation=0,
        candidate=CandidateParameters(
            feature_weights={}, destroy_weights={}, repair_weights={},
        ),
        objectives={"makespan": 10.0},
        metrics={"makespan": 10.0, "feasible": True},
        schedule=[{"op_id": "OP-1", "start": 0.0, "end": 1.0}],
        feasible=True,
        schedule_signature="sig-1",
        analytics_summary={"completed_operations": 1},
    )


class SolutionCloneTests(unittest.TestCase):
    def test_clone_shares_immutable_payload(self):
        solution = _make_solution()
        cloned = solution.clone()
        self.assertIs(cloned.schedule, solution.schedule)
        self.assertIs(cloned.analytics_summary, solution.analytics_summary)

    def test_clone_still_isolates_mutable_fields(self):
        solution = _make_solution()
        cloned = solution.clone()
        cloned.metrics["makespan"] = 99.0
        cloned.objectives["makespan"] = 99.0
        cloned.candidate.feature_weights["urgency"] = 1.0
        self.assertEqual(solution.metrics["makespan"], 10.0)
        self.assertEqual(solution.objectives["makespan"], 10.0)
        self.assertEqual(solution.candidate.feature_weights, {})

    def test_clone_preserves_payload_contents(self):
        solution = _make_solution()
        cloned = solution.clone()
        self.assertEqual(cloned.schedule, [{"op_id": "OP-1", "start": 0.0, "end": 1.0}])
        self.assertEqual(cloned.analytics_summary, {"completed_operations": 1})
        self.assertEqual(cloned.schedule_signature, "sig-1")
        self.assertTrue(cloned.feasible)


if __name__ == "__main__":
    unittest.main()
