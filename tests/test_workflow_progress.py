"""流程进度快照的存取与失效契约。

校验/仿真/优化/评审的结果都按实例版本号存库，供进程重启后恢复；实例一改动，
这些快照必须自动失效，否则界面会拿着旧排程当成当前实例的结论。
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm4drd.data.db import (
    DowntimeStore,
    InstanceStore,
    WorkflowProgressStore,
    init_db,
)
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class WorkflowProgressStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = str(Path(self.tmp.name) / "progress.db")
        init_db(self.db_path)
        self.instances = InstanceStore(self.db_path)
        self.progress = WorkflowProgressStore(self.db_path)
        self.instances.save_from_shopfloor(make_graph_context_shop())

    def test_roundtrip_preserves_payload(self):
        payload = {"status": "passed", "error_count": 0, "errors": [], "中文": "值"}
        self.progress.save("validation", payload)
        self.assertEqual(self.progress.load("validation"), payload)

    def test_save_overwrites_previous_snapshot(self):
        self.progress.save("simulation", {"rule": "ATC"})
        self.progress.save("simulation", {"rule": "EDD"})
        self.assertEqual(self.progress.load("simulation"), {"rule": "EDD"})

    def test_missing_step_returns_none(self):
        self.assertIsNone(self.progress.load("optimization"))

    def test_instance_edit_invalidates_snapshot(self):
        self.progress.save("simulation", {"rule": "ATC", "makespan": 12.0})
        self.instances.update_order(
            "O-1", {"order_name": "Renamed", "release_time": 0.0, "due_date": 40.0, "priority": 3})
        self.assertIsNone(
            self.progress.load("simulation"),
            "实例改动后旧排程不再成立，必须失效而不是返回过期结果",
        )

    def test_downtime_edit_invalidates_snapshot(self):
        self.progress.save("simulation", {"rule": "ATC"})
        DowntimeStore(self.db_path).save("M-C1", "maintenance", 1.0, 2.0)
        self.assertIsNone(self.progress.load("simulation"), "停机参与排程，改动后快照必须失效")

    def test_reimport_invalidates_snapshot(self):
        self.progress.save("validation", {"status": "passed"})
        self.instances.save_from_shopfloor(make_graph_context_shop())
        self.assertIsNone(self.progress.load("validation"), "重新导入实例后校验结论必须重算")

    def test_save_does_not_invalidate_other_steps(self):
        self.progress.save("validation", {"status": "passed"})
        self.progress.save("simulation", {"rule": "ATC"})
        self.assertEqual(
            self.progress.load("validation"),
            {"status": "passed"},
            "写进度表不得递增实例版本号，否则会连带作废已存的其他步骤",
        )

    def test_load_all_returns_only_valid_steps(self):
        self.progress.save("validation", {"status": "passed"})
        self.progress.save("simulation", {"rule": "ATC"})
        self.assertEqual(
            self.progress.load_all(),
            {"validation": {"status": "passed"}, "simulation": {"rule": "ATC"}},
        )

    def test_load_all_empty_after_instance_change(self):
        self.progress.save("validation", {"status": "passed"})
        self.instances.clear_all()
        self.assertEqual(self.progress.load_all(), {})

    def test_clear_removes_single_step(self):
        self.progress.save("validation", {"status": "passed"})
        self.progress.save("simulation", {"rule": "ATC"})
        self.progress.clear("simulation")
        self.assertIsNone(self.progress.load("simulation"))
        self.assertIsNotNone(self.progress.load("validation"))

    def test_unknown_step_is_rejected(self):
        with self.assertRaises(ValueError):
            self.progress.save("nonsense", {})


if __name__ == "__main__":
    unittest.main()
