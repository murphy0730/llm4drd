"""基准方案库的存取与批次轮转契约。

夜间冷搜索蒸馏出的派工参数方案落库、白天热启动读回；连续构建时始终只保留
"最新 active + 上一批 previous"两批。这张表不是实例数据，写入不得触发实例版本递增。
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm4drd.data.db import (
    BaselineSolutionStore,
    get_db,
    get_instance_version,
    init_db,
)


def _make_row(rid: str, emphasis: str = "balanced") -> dict:
    return {
        "id": rid,
        "emphasis": emphasis,
        "objective_keys": ["main_order_tardy_ratio", "critical_active_window_utilization", "avg_flowtime"],
        "feature_names": ["urgency", "slack", "due_date"],
        "feature_weights": {"urgency": 1.5, "slack": -0.3, "due_date": 0.8},
        "scale_json": {"time_scale": 120.0, "due_scale": 5000.0, "priority_scale": 5.0},
        "op_bias": {"op-1": 0.42},
        "destroy_weights": {"tardy_order_destroy": 1.0},
        "repair_weights": {},
        "destroy_fraction": 0.25,
        "reached_objectives": {"main_order_tardy_ratio": 0.1, "avg_flowtime": 300.0},
        "baseline_compare": {"main_order_tardy_ratio": {"atc": 0.2, "value": 0.1, "improved": True}},
        "snapshot_id": "snap-1",
        "snapshot_version": 7,
        "created_at": "2026-07-24T02:00:00",
        "source": "learned",
    }


class BaselineSolutionStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = str(Path(self.tmp.name) / "baseline.db")
        init_db(self.db_path)
        self.store = BaselineSolutionStore(self.db_path)

    def test_init_db_idempotent(self):
        # 二次建表不报错、不清空已写入数据。
        self.store.save_batch("b1", [_make_row("s1")])
        init_db(self.db_path)
        self.assertEqual(len(self.store.load_active()), 1)

    def test_roundtrip_preserves_json_fields(self):
        row = _make_row("s1", "min_main_order_tardy_ratio")
        self.store.save_batch("b1", [row])
        loaded = self.store.load_active()
        self.assertEqual(len(loaded), 1)
        got = loaded[0]
        self.assertEqual(got["emphasis"], "min_main_order_tardy_ratio")
        self.assertEqual(got["feature_weights"], {"urgency": 1.5, "slack": -0.3, "due_date": 0.8})
        self.assertEqual(got["feature_names"], ["urgency", "slack", "due_date"])
        self.assertEqual(got["scale_json"]["due_scale"], 5000.0)
        self.assertEqual(got["op_bias"], {"op-1": 0.42})
        self.assertEqual(got["snapshot_version"], 7)
        self.assertAlmostEqual(got["destroy_fraction"], 0.25)
        self.assertEqual(got["batch_id"], "b1")
        self.assertEqual(got["status"], "active")

    def test_has_active(self):
        self.assertFalse(self.store.has_active())
        self.store.save_batch("b1", [_make_row("s1")])
        self.assertTrue(self.store.has_active())

    def test_rotation_keeps_latest_two_batches(self):
        self.store.save_batch("b1", [_make_row("s1")])
        self.store.save_batch("b2", [_make_row("s2"), _make_row("s3", "max_critical_active_window_utilization")])
        self.store.save_batch("b3", [_make_row("s4")])

        active = self.store.load_active()
        self.assertEqual({r["id"] for r in active}, {"s4"})

        # 库里应只剩最新(active) + 上一批(previous)；b1 已被删除。
        with get_db(self.db_path) as conn:
            batches = {
                row["batch_id"]: row["status"]
                for row in conn.execute("SELECT batch_id, status FROM baseline_solutions").fetchall()
            }
        self.assertEqual(batches.get("b3"), "active")
        self.assertEqual(batches.get("b2"), "previous")
        self.assertNotIn("b1", batches)

    def test_write_does_not_bump_instance_version(self):
        before = get_instance_version(self.db_path)
        self.store.save_batch("b1", [_make_row("s1")])
        self.store.save_batch("b2", [_make_row("s2")])
        self.assertEqual(get_instance_version(self.db_path), before)


if __name__ == "__main__":
    unittest.main()
