import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

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


class TrackingEvent:
    def __init__(self):
        self._event = threading.Event()
        self.wait_started = threading.Event()

    def wait(self, timeout=None):
        self.wait_started.set()
        return self._event.wait(timeout)

    def set(self):
        self._event.set()


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

    def test_cache_same_key_concurrent_reads_build_once(self):
        cache = ReviewReadCache(max_entries=2)
        key = (1, "t", "a")
        build_started = threading.Event()
        release_build = threading.Event()
        build_count = 0
        build_count_lock = threading.Lock()
        value = object()

        def build():
            nonlocal build_count
            with build_count_lock:
                build_count += 1
            build_started.set()
            release_build.wait()
            return value

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(cache.get_or_build, key, build)
            self.assertTrue(build_started.wait(timeout=1))
            tracked_event = TrackingEvent()
            with cache._lock:
                cache._in_flight[key].event = tracked_event
            second = executor.submit(cache.get_or_build, key, build)
            self.assertTrue(tracked_event.wait_started.wait(timeout=1))
            release_build.set()
            futures = [first, second]
            results = [future.result(timeout=1) for future in futures]

        self.assertEqual(build_count, 1)
        self.assertIs(results[0], value)
        self.assertIs(results[1], value)

    def test_cache_different_keys_build_concurrently(self):
        cache = ReviewReadCache(max_entries=2)
        build_barrier = threading.Barrier(2)

        def build(value):
            build_barrier.wait(timeout=1)
            return value

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(cache.get_or_build, (1, "t", "a"), lambda: build("a"))
            second = executor.submit(cache.get_or_build, (1, "t", "b"), lambda: build("b"))
            self.assertEqual(first.result(timeout=2), "a")
            self.assertEqual(second.result(timeout=2), "b")

    def test_cache_failed_build_wakes_waiters_and_allows_retry(self):
        cache = ReviewReadCache(max_entries=2)
        key = (1, "t", "a")
        build_started = threading.Event()
        release_build = threading.Event()
        build_count = 0
        build_count_lock = threading.Lock()
        failure = ValueError("cannot build")

        def fail():
            nonlocal build_count
            with build_count_lock:
                build_count += 1
            build_started.set()
            release_build.wait()
            raise failure

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(cache.get_or_build, key, fail)
            self.assertTrue(build_started.wait(timeout=1))
            tracked_event = TrackingEvent()
            with cache._lock:
                cache._in_flight[key].event = tracked_event
            second = executor.submit(cache.get_or_build, key, fail)
            self.assertTrue(tracked_event.wait_started.wait(timeout=1))
            release_build.set()
            futures = [first, second]
            errors = []
            for future in futures:
                with self.assertRaises(ValueError) as raised:
                    future.result(timeout=1)
                errors.append(raised.exception)

        self.assertEqual(build_count, 1)
        self.assertIs(errors[0], failure)
        self.assertIs(errors[1], failure)
        self.assertEqual(
            cache.get_or_build(key, lambda: "recovered"),
            "recovered",
        )

    def test_cache_reentrant_same_key_fails_without_deadlock_and_can_retry(self):
        cache = ReviewReadCache(max_entries=2)
        key = (1, "t", "a")

        with self.assertRaisesRegex(RuntimeError, "reentrant"):
            cache.get_or_build(
                key,
                lambda: cache.get_or_build(key, lambda: "nested"),
            )

        self.assertEqual(cache.get_or_build(key, lambda: "recovered"), "recovered")

    def test_search_ranks_id_before_name_and_limits(self):
        shop = make_graph_context_shop()
        first = build_review_solution_index(shop, "S-1", [
            {"order_id": "X-001", "order_name": "普通订单", "machine_id": "M-C1", "start": 0, "end": 1},
            {"order_id": "001-X", "order_name": "普通订单", "machine_id": "M-C1", "start": 1, "end": 2},
            {"order_id": "X-900", "order_name": "名称001", "machine_id": "M-C1", "start": 2, "end": 3},
        ])
        result = search_order_facets([first], "001", 2)
        self.assertEqual([item["order_id"] for item in result], ["001-X", "X-001"])
