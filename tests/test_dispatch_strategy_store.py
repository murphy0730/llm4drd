import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm4drd.data.db import DispatchStrategyStore, get_instance_version, init_db
from llm4drd.optimization.solution_model import CandidateParameters, FEATURE_NAMES


class DispatchStrategyStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = str(Path(self.tmp.name) / "strategies.db")
        init_db(self.db_path)
        self.store = DispatchStrategyStore(self.db_path)

    def _candidate(self) -> CandidateParameters:
        return CandidateParameters(
            feature_weights={name: index / 10 for index, name in enumerate(FEATURE_NAMES)},
            destroy_weights={"tardy_order_destroy": 1.25},
            repair_weights={"due_date_repair": 0.8},
            op_bias={"OP-1": 0.123456789},
            destroy_fraction=0.27,
            seed_rule_name="ATC",
            graph_profile="balanced",
        )

    def test_candidate_roundtrip_is_lossless(self):
        candidate = self._candidate()
        restored = CandidateParameters.from_dict(candidate.to_dict())
        self.assertEqual(restored.signature(), candidate.signature())
        self.assertEqual(restored.op_bias, candidate.op_bias)

    def test_named_set_roundtrip_and_no_instance_version_bump(self):
        before = get_instance_version(self.db_path)
        saved = self.store.save_set(
            {
                "name": "三目标冷启动",
                "source_task_id": "task-1",
                "source_mode": "cold",
                "objective_keys": ["total_tardiness", "makespan"],
                "instance_version": 3,
            },
            [{
                "name": "方案一｜均衡型",
                "source_solution_id": "S-1",
                "candidate_json": self._candidate().to_dict(),
                "objective_values": {"total_tardiness": 12.5},
                "feature_names": list(FEATURE_NAMES),
                "scale_json": {"time_scale": 24.0},
            }],
        )
        self.assertEqual(saved["name"], "三目标冷启动")
        self.assertEqual(saved["strategies"][0]["name"], "方案一｜均衡型")
        strategy = self.store.get_strategy(saved["strategies"][0]["id"])
        self.assertEqual(strategy["candidate_json"]["op_bias"], {"OP-1": 0.123456789})
        self.assertEqual(self.store.list_sets()[0]["id"], saved["id"])
        self.assertEqual(get_instance_version(self.db_path), before)


if __name__ == "__main__":
    unittest.main()
