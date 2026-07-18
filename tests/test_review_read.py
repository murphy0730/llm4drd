import unittest

from llm4drd.api.review_read import (
    ReviewReadCache,
    build_review_solution_index,
    search_order_facets,
)
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class CountingSchedule(list):
    def __init__(self, values):
        super().__init__(values)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()


class ReviewReadIndexTests(unittest.TestCase):
    def test_build_scans_once_and_keeps_entry_references(self):
        entry = {
            "order_id": "O-100",
            "order_name": "订单100",
            "machine_id": "M-C1",
            "start": 0.0,
            "end": 10.0,
        }
        schedule = CountingSchedule([entry])
        index = build_review_solution_index(make_graph_context_shop(), "S-1", schedule)
        self.assertEqual(schedule.iterations, 1)
        self.assertIs(index.entries_by_order["O-100"][0], entry)
        self.assertEqual(index.machine_type_utilization["cut"]["utilization"], 1.0)

    def test_cache_reuses_value_and_evicts_lru(self):
        cache = ReviewReadCache(max_entries=2)
        builds = []

        def build(label):
            builds.append(label)
            return label

        self.assertEqual(cache.get_or_build((1, "t", "a"), lambda: build("a")), "a")
        self.assertEqual(cache.get_or_build((1, "t", "a"), lambda: build("again")), "a")
        cache.get_or_build((1, "t", "b"), lambda: build("b"))
        cache.get_or_build((1, "t", "c"), lambda: build("c"))
        self.assertEqual(builds, ["a", "b", "c"])
        self.assertNotIn((1, "t", "a"), cache.keys())

    def test_search_ranks_id_before_name_and_limits(self):
        shop = make_graph_context_shop()
        first = build_review_solution_index(shop, "S-1", [
            {"order_id": "X-001", "order_name": "普通订单", "machine_id": "M-C1", "start": 0, "end": 1},
            {"order_id": "001-X", "order_name": "普通订单", "machine_id": "M-C1", "start": 1, "end": 2},
            {"order_id": "X-900", "order_name": "名称001", "machine_id": "M-C1", "start": 2, "end": 3},
        ])
        result = search_order_facets([first], "001", 2)
        self.assertEqual([item["order_id"] for item in result], ["001-X", "X-001"])
