"""实例版本号与 _active_shop() 缓存的失效契约。

_active_shop() 缓存整个 ShopFloor（大实例重建一次 ~2s），键是库里的实例版本号：
任何写库操作都必须让缓存失效，否则会静默返回过期数据——这里逐个写入点钉死。
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm4drd.api import server
from llm4drd.data.db import (
    DowntimeStore,
    InstanceStore,
    get_instance_version,
    init_db,
)
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class InstanceVersionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = str(Path(self.tmp.name) / "inst.db")
        init_db(self.db_path)
        self.store = InstanceStore(self.db_path)
        self.downtimes = DowntimeStore(self.db_path)
        self.store.save_from_shopfloor(make_graph_context_shop())

    def _version(self) -> int:
        return get_instance_version(self.db_path)

    def assertBumps(self, label, action):
        before = self._version()
        action()
        after = self._version()
        self.assertGreater(after, before, f"{label} 写库后版本号必须递增，否则 _active_shop() 会返回过期实例")

    def test_save_from_shopfloor_bumps_version(self):
        self.assertBumps("save_from_shopfloor", lambda: self.store.save_from_shopfloor(make_graph_context_shop()))

    def test_clear_all_bumps_version(self):
        self.assertBumps("clear_all", self.store.clear_all)

    def test_update_order_bumps_version(self):
        self.assertBumps("update_order", lambda: self.store.update_order(
            "O-1", {"order_name": "Renamed", "release_time": 0.0, "due_date": 40.0, "priority": 3}))

    def test_update_task_bumps_version(self):
        self.assertBumps("update_task", lambda: self.store.update_task(
            "T-11", {"order_id": "O-1", "task_name": "Renamed", "is_main": False,
                     "predecessor_task_ids": "", "release_time": 0.0, "due_date": 24.0}))

    def test_update_machine_bumps_version(self):
        self.assertBumps("update_machine", lambda: self.store.update_machine(
            "M-C1", {"machine_name": "Renamed", "type_id": "cut", "shifts": ""}))

    def test_downtime_save_bumps_version(self):
        self.assertBumps("downtime.save", lambda: self.downtimes.save("M-C1", "maintenance", 1.0, 2.0))

    def test_downtime_replace_all_bumps_version(self):
        self.assertBumps("downtime.replace_all", lambda: self.downtimes.replace_all(
            [{"machine_id": "M-C1", "downtime_type": "maintenance", "start_time": 1.0, "end_time": 2.0}]))

    def test_downtime_delete_bumps_version(self):
        downtime_id = self.downtimes.save("M-C1", "maintenance", 1.0, 2.0)
        self.assertBumps("downtime.delete", lambda: self.downtimes.delete(downtime_id))

    def test_downtime_clear_all_bumps_version(self):
        self.downtimes.save("M-C1", "maintenance", 1.0, 2.0)
        self.assertBumps("downtime.clear_all", self.downtimes.clear_all)


class ActiveShopCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = str(Path(self.tmp.name) / "inst.db")
        init_db(self.db_path)

        originals = (server.inst_store, server.shop, server._active_shop_cache)
        self.addCleanup(lambda: setattr(server, "inst_store", originals[0]))
        self.addCleanup(lambda: setattr(server, "shop", originals[1]))
        self.addCleanup(lambda: setattr(server, "_active_shop_cache", originals[2]))

        server.inst_store = InstanceStore(self.db_path)
        server.shop = None
        server._active_shop_cache = None
        server.inst_store.save_from_shopfloor(make_graph_context_shop())

    def test_repeated_calls_reuse_the_same_instance(self):
        first = server._active_shop()
        second = server._active_shop()
        self.assertIsNotNone(first)
        self.assertIs(first, second, "版本号未变时应直接复用缓存实例，而不是重建")

    def test_write_invalidates_cache_and_serves_fresh_data(self):
        first = server._active_shop()
        self.assertEqual(first.orders["O-1"].name, "Order 1")

        server.inst_store.update_order(
            "O-1", {"order_name": "Renamed", "release_time": 0.0, "due_date": 40.0, "priority": 3})

        second = server._active_shop()
        self.assertIsNot(first, second, "写库后必须重建，不能返回旧对象")
        self.assertEqual(second.orders["O-1"].name, "Renamed", "写库后必须读到新数据")

    def test_downtime_write_invalidates_cache(self):
        first = server._active_shop()
        self.assertEqual(first.machines["M-C1"].downtimes, [])

        DowntimeStore(self.db_path).save("M-C1", "maintenance", 1.0, 2.0)

        second = server._active_shop()
        self.assertIsNot(first, second, "停机写库后必须重建")
        self.assertEqual(len(second.machines["M-C1"].downtimes), 1, "停机数据参与 build_shopfloor，必须反映到新实例")


if __name__ == "__main__":
    unittest.main()
