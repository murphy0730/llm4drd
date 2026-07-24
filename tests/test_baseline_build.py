"""夜间基准方案库构建管线端到端（短预算）。

跑一次真实优化 → 抽取 → 门禁 → 落库，验证摘要结构、落库内容与批次轮转。
用小 time_limit_s 保证测试快速；断言不依赖"必然有方案过门禁"（小实例上 ATC 可能已很强），
只在 passed>0 时校验落库内容，避免 flaky。
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from llm4drd.data.db import BaselineSolutionStore, init_db
from llm4drd.optimization.baseline_build import DEFAULT_OBJECTIVE_KEYS, build_baseline_library
from llm4drd.optimization.solution_model import FEATURE_NAMES
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class BaselineBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = str(Path(self.tmp.name) / "build.db")
        init_db(self.db_path)
        self.store = BaselineSolutionStore(self.db_path)

    def _build(self, **overrides):
        params = dict(
            time_limit_s=15,
            generations=3,
            stagnation_generations=2,
            store=self.store,
            db_path=self.db_path,
        )
        params.update(overrides)
        return build_baseline_library(make_graph_context_shop(), **params)

    def test_pipeline_produces_summary_and_persists(self):
        summary = self._build()
        self.assertEqual(summary["objective_keys"], DEFAULT_OBJECTIVE_KEYS)
        self.assertGreaterEqual(summary["extracted"], 1)
        self.assertEqual(summary["passed"], len(summary["rows"]))
        self.assertTrue(summary["batch_id"])

        active = self.store.load_active()
        self.assertEqual(len(active), summary["passed"])
        for row in active:
            self.assertEqual(row["feature_names"], list(FEATURE_NAMES))
            self.assertEqual(set(row["reached_objectives"].keys()), set(DEFAULT_OBJECTIVE_KEYS))
            self.assertIn("objectives", row["baseline_compare"])
            self.assertIn("verdict", row["baseline_compare"])
            self.assertEqual(row["batch_id"], summary["batch_id"])

    def test_real_candidate_persists_and_roundtrips_when_gate_passes(self):
        # 微实例上 ATC 常打平、门禁 0 通过，前面的落库断言会被跳过（空跑）。这里强制门禁
        # 通过，确保"真实优化器候选 → 组装行 → 落库 → 读回"整条写入路径被真正执行，且真实
        # 候选的 feature_weights / op_bias / scale 无损往返。
        with patch(
            "llm4drd.optimization.baseline_build.passes_quality_gate",
            return_value=(True, {"objectives": {}, "verdict": {"tier": "forced"}}),
        ):
            summary = self._build()
        self.assertGreaterEqual(summary["passed"], 1)
        active = self.store.load_active()
        self.assertEqual(len(active), summary["passed"])
        row = active[0]
        # 真实候选权重必须是完整 16 维且为有限浮点。
        self.assertEqual(set(row["feature_weights"].keys()), set(FEATURE_NAMES))
        self.assertTrue(all(isinstance(v, float) for v in row["feature_weights"].values()))
        self.assertIn("time_scale", row["scale_json"])
        self.assertGreater(row["scale_json"]["time_scale"], 0)
        # 与内存里组装的行对比，确认 JSON 往返未丢字段。
        mem = {r["id"]: r for r in summary["rows"]}
        self.assertEqual(row["feature_weights"], mem[row["id"]]["feature_weights"])
        self.assertEqual(row["op_bias"], mem[row["id"]]["op_bias"])

    def test_second_build_rotates_history(self):
        first = self._build(seed=1)
        second = self._build(seed=2)
        # 无论每批过门禁多少条，库里 batch_id 最多两个（最新 active + 上一批 previous）。
        active = self.store.load_active()
        if second["passed"] > 0:
            self.assertTrue(all(row["batch_id"] == second["batch_id"] for row in active))
        distinct_batches = {row["batch_id"] for row in active}
        self.assertLessEqual(len(distinct_batches), 1)  # active 只属于一个批次


if __name__ == "__main__":
    unittest.main()
