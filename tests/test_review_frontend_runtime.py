import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReviewFrontendRuntimeTests(unittest.TestCase):
    def _run_node(self, body: str):
        script = textwrap.dedent(
            f"""
            const assert = require("assert");
            const runtime = require("./frontend/review_runtime.js");
            (async () => {{
              {body}
            }})().catch((error) => {{
              console.error(error);
              process.exitCode = 1;
            }});
            """
        )
        subprocess.run(
            ["node", "-e", script],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_initial_window_is_four_days_and_clamped_to_schedule(self):
        self._run_node(
            """
            const fourDays = runtime.computeInitialWindow(
              "2026-01-01T00:00:00.000Z",
              "2026-01-11T00:00:00.000Z",
              "2026-01-05T00:00:00.000Z"
            );
            assert.strictEqual(
              new Date(fourDays.end) - new Date(fourDays.start),
              96 * 3600000
            );
            assert.strictEqual(fourDays.start, "2026-01-04T12:00:00.000Z");
            assert.deepStrictEqual(
              runtime.computeInitialWindow(
                "2026-01-01T00:00:00.000Z",
                "2026-01-03T00:00:00.000Z",
                "2026-01-02T00:00:00.000Z"
              ),
              {
                start: "2026-01-01T00:00:00.000Z",
                end: "2026-01-03T00:00:00.000Z"
              }
            );
            """
        )

    def test_keys_normalize_selection_ids(self):
        self._run_node(
            """
            assert.deepStrictEqual(
              runtime.normalizeIds(["S-3", "S-1", "", "S-2", "S-1", "S-5"]),
              ["S-1", "S-2", "S-3", "S-5"]
            );
            assert.strictEqual(
              runtime.selectionKey("t1", ["S-2", "S-1"]),
              "t1::S-1,S-2"
            );
            assert.strictEqual(
              runtime.scheduleKey("t1", ["S-2", "S-1"], "O-7"),
              "t1::S-1,S-2::O-7"
            );
            """
        )

    def test_order_ranking_is_deterministic_and_capped(self):
        self._run_node(
            """
            const ranked = runtime.rankOrders([
              {order_id: "X-001", order_name: "普通"},
              {order_id: "001-X", order_name: "普通"},
              {order_id: "X-900", order_name: "名称001"},
              {order_id: "001", order_name: "普通"}
            ], "001", 2);
            assert.deepStrictEqual(
              ranked.map((item) => item.order_id),
              ["001", "001-X"]
            );
            assert.strictEqual(
              runtime.rankOrders(
                [{order_id: "X-001", order_name: "普通"},
                 {order_id: "001-X", order_name: "普通"},
                 {order_id: "X-900", order_name: "名称001"}],
                "001",
                2
              ).map((item) => item.order_id).join(","),
              "001-X,X-001"
            );
            assert.strictEqual(
              runtime.rankOrders(
                Array.from({length: 80}, (_, index) => ({
                  order_id: `A-${index}`,
                  order_name: "匹配"
                })),
                "匹配",
                80
              ).length,
              50
            );
            """
        )

    def test_data_requests_cancel_stale_work_and_cache_completed_payloads(self):
        self._run_node(
            """
            let calls = 0;
            const client = runtime.createClient({
              fetchReviewData: (_args, signal) => new Promise((resolve, reject) => {
                calls += 1;
                const timer = setTimeout(
                  () => resolve({order_id: "O-1", schemes: {}}),
                  calls === 1 ? 30 : 1
                );
                signal.addEventListener("abort", () => {
                  clearTimeout(timer);
                  reject(new DOMException("aborted", "AbortError"));
                });
              }),
              fetchOrders: async () => ({orders: []})
            });
            const first = client.loadData({
              taskId: "t1",
              ids: ["S-1"],
              orderId: "O-1",
              includeUtilization: true
            });
            const second = client.loadData({
              taskId: "t1",
              ids: ["S-1"],
              orderId: "O-2",
              includeUtilization: false
            });
            assert.deepStrictEqual(await first, {cancelled: true});
            assert.strictEqual((await second).payload.order_id, "O-1");
            const cached = await client.loadData({
              taskId: "t1",
              ids: ["S-1"],
              orderId: "O-2",
              includeUtilization: false
            });
            assert.strictEqual(cached.fromCache, true);
            assert.strictEqual(calls, 2);
            """
        )

    def test_order_requests_have_independent_cancellation_and_cache(self):
        self._run_node(
            """
            let dataCalls = 0;
            let orderCalls = 0;
            const client = runtime.createClient({
              fetchReviewData: async () => {
                dataCalls += 1;
                return {schemes: {}};
              },
              fetchOrders: (_args, signal) => new Promise((resolve, reject) => {
                orderCalls += 1;
                const timer = setTimeout(
                  () => resolve({orders: [{order_id: "O-2"}]}),
                  orderCalls === 1 ? 30 : 1
                );
                signal.addEventListener("abort", () => {
                  clearTimeout(timer);
                  reject(new DOMException("aborted", "AbortError"));
                });
              })
            });
            const data = client.loadData({
              taskId: "t1", ids: ["S-1"], orderId: "O-1"
            });
            const first = client.searchOrders({
              taskId: "t1", ids: ["S-1"], query: "O"
            });
            const second = client.searchOrders({
              taskId: "t1", ids: ["S-1"], query: "O-2"
            });
            assert.deepStrictEqual(await first, {cancelled: true});
            assert.strictEqual((await second).orders[0].order_id, "O-2");
            assert.strictEqual((await data).payload.schemes instanceof Object, true);
            const cached = await client.searchOrders({
              taskId: "t1", ids: ["S-1"], query: " O-2 "
            });
            assert.strictEqual(cached.fromCache, true);
            assert.strictEqual(orderCalls, 2);
            assert.strictEqual(dataCalls, 1);
            """
        )

    def test_cached_read_invalidates_a_different_inflight_read(self):
        self._run_node(
            """
            let calls = 0;
            let finishFirstB;
            const client = runtime.createClient({
              fetchReviewData: (args) => {
                calls += 1;
                if (args.orderId === "B" && !finishFirstB) {
                  return new Promise((resolve) => { finishFirstB = resolve; });
                }
                return Promise.resolve({order_id: args.orderId, schemes: {}});
              },
              fetchOrders: async () => ({orders: []})
            });
            await client.loadData({
              taskId: "t1", ids: ["S-1"], orderId: "A", includeUtilization: false
            });
            const pendingB = client.loadData({
              taskId: "t1", ids: ["S-1"], orderId: "B", includeUtilization: false
            });
            const cachedA = await client.loadData({
              taskId: "t1", ids: ["S-1"], orderId: "A", includeUtilization: false
            });
            assert.strictEqual(cachedA.fromCache, true);
            finishFirstB({order_id: "B", schemes: {}});
            assert.deepStrictEqual(await pendingB, {cancelled: true});
            const freshB = await client.loadData({
              taskId: "t1", ids: ["S-1"], orderId: "B", includeUtilization: false
            });
            assert.strictEqual(freshB.payload.order_id, "B");
            assert.strictEqual(calls, 3);
            """
        )

    def test_late_same_selection_order_read_cannot_overwrite_newer_order(self):
        self._run_node(
            """
            const finishes = {};
            const client = runtime.createClient({
              fetchReviewData: (args) => new Promise((resolve) => {
                finishes[args.orderId] = resolve;
              }),
              fetchOrders: async () => ({orders: []})
            });
            const first = client.loadData({
              taskId: "t1", ids: ["S-1"], orderId: "O-1", includeUtilization: false
            });
            const second = client.loadData({
              taskId: "t1", ids: ["S-1"], orderId: "O-2", includeUtilization: false
            });
            finishes["O-2"]({order_id: "O-2", schemes: {}});
            assert.strictEqual((await second).payload.order_id, "O-2");
            finishes["O-1"]({order_id: "O-1", schemes: {}});
            assert.deepStrictEqual(await first, {cancelled: true});
            """
        )

    def test_reset_invalidates_late_data_completion_and_its_cache(self):
        self._run_node(
            """
            let calls = 0;
            let finishFirst;
            const client = runtime.createClient({
              fetchReviewData: () => {
                calls += 1;
                if (calls === 1) {
                  return new Promise((resolve) => { finishFirst = resolve; });
                }
                return Promise.resolve({order_id: "O-1", schemes: {}});
              },
              fetchOrders: async () => ({orders: []})
            });
            const pending = client.loadData({
              taskId: "t1", ids: ["S-1"], orderId: "O-1", includeUtilization: false
            });
            client.reset();
            finishFirst({order_id: "O-1", schemes: {}});
            assert.deepStrictEqual(await pending, {cancelled: true});
            const afterReset = await client.loadData({
              taskId: "t1", ids: ["S-1"], orderId: "O-1", includeUtilization: false
            });
            assert.strictEqual(afterReset.fromCache, false);
            assert.strictEqual(calls, 2);
            """
        )


if __name__ == "__main__":
    unittest.main()
