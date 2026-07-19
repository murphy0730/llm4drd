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

    def test_empty_order_query_prioritizes_current_then_recent_without_duplicates(self):
        self._run_node(
            """
            const current = {order_id: "O-3", order_name: "当前"};
            const recent = [
              {order_id: "O-2", order_name: "最近"},
              {order_id: "O-3", order_name: "重复当前"},
              {order_id: "O-1", order_name: "较早"}
            ];
            const ranked = runtime.rankOrders([
              {order_id: "O-4", order_name: "第四"},
              {order_id: "O-2", order_name: "重复最近"},
              {order_id: "O-1", order_name: "重复较早"},
              {order_id: "O-5", order_name: "第五"}
            ], "", 50, {current, recent});
            assert.deepStrictEqual(
              ranked.map((item) => item.order_id),
              ["O-3", "O-2", "O-1", "O-4", "O-5"]
            );
            assert.strictEqual(
              new Set(ranked.map((item) => item.order_id)).size,
              ranked.length
            );
            assert.strictEqual(ranked[1].order_name, "重复最近");
            assert.strictEqual(ranked[2].order_name, "重复较早");
            """
        )

    def test_text_order_query_uses_authoritative_records_not_stale_pins(self):
        self._run_node(
            """
            const ranked = runtime.rankOrders([
              {order_id: "O-1", order_name: "Alpha"},
              {order_id: "O-2", order_name: "Alphanumeric"}
            ], "alpha", 50, {
              current: {order_id: "O-1", order_name: ""},
              recent: [{order_id: "O-2", order_name: "旧名称"}]
            });
            assert.deepStrictEqual(
              ranked.map((item) => [item.order_id, item.order_name]),
              [["O-1", "Alpha"], ["O-2", "Alphanumeric"]]
            );
            """
        )

    def test_recent_order_store_is_bounded_lru_deduped_and_resettable(self):
        self._run_node(
            """
            const store = runtime.createRecentOrderStore({
              contextLimit: 2,
              itemLimit: 3
            });
            assert.deepStrictEqual(store.read("missing"), []);
            assert.deepStrictEqual(store.stats(), {
              size: 0,
              keys: [],
              itemSizes: []
            });
            store.record("ignored", null);
            assert.strictEqual(store.stats().size, 0);

            store.record("ctx-a", {order_id: "A-1", order_name: "旧"});
            store.record("ctx-a", {order_id: "A-2", order_name: "二"});
            store.record("ctx-a", {order_id: "A-1", order_name: "新"});
            store.record("ctx-a", {order_id: "A-3", order_name: "三"});
            store.record("ctx-a", {order_id: "A-4", order_name: "四"});
            assert.deepStrictEqual(
              store.read("ctx-a").map((item) => [item.order_id, item.order_name]),
              [["A-4", "四"], ["A-3", "三"], ["A-1", "新"]]
            );

            store.record("ctx-b", {order_id: "B-1"});
            store.read("ctx-a"); // Refresh ctx-a so ctx-b becomes oldest.
            store.record("ctx-c", {order_id: "C-1"});
            assert.deepStrictEqual(store.stats(), {
              size: 2,
              keys: ["ctx-a", "ctx-c"],
              itemSizes: [3, 1]
            });
            assert.deepStrictEqual(store.read("ctx-b"), []);
            assert.strictEqual(store.stats().size, 2);

            store.reset();
            assert.deepStrictEqual(store.stats(), {
              size: 0,
              keys: [],
              itemSizes: []
            });
            """
        )

    def test_real_focus_and_click_events_use_shared_binding_without_double_search(self):
        self._run_node(
            """
            const timers = [];
            const target = new EventTarget();
            const queries = [];
            const controller = runtime.createOrderComboboxController({
              schedule(fn) {
                const handle = {fn, cancelled: false};
                timers.push(handle);
                return handle;
              },
              cancelSchedule(handle) {
                if (handle) handle.cancelled = true;
              },
              search: async (query) => {
                queries.push(query);
                return [{order_id: "O-1", order_name: "Alpha"}];
              },
              select: async () => {}
            });
            const unbind = runtime.bindOrderComboboxOpen(target, controller);

            target.dispatchEvent(new Event("focus"));
            target.dispatchEvent(new Event("click"));
            assert.strictEqual(timers.length, 1);
            timers.shift().fn();
            await Promise.resolve();
            await Promise.resolve();
            assert.deepStrictEqual(queries, [""]);
            assert.strictEqual(controller.getState().open, true);

            unbind();
            controller.close();
            target.dispatchEvent(new Event("focus"));
            assert.strictEqual(timers.length, 0);
            """
        )

    def test_combobox_focus_and_click_share_one_empty_query_and_mouse_selects(self):
        self._run_node(
            """
            const timers = [];
            const scheduler = {
              set(fn) {
                const handle = {fn, cancelled: false};
                timers.push(handle);
                return handle;
              },
              clear(handle) {
                if (handle) handle.cancelled = true;
              },
              flushAll() {
                while (timers.length) {
                  const handle = timers.shift();
                  if (!handle.cancelled) handle.fn();
                }
              }
            };
            const queries = [];
            const selections = [];
            const controller = runtime.createOrderComboboxController({
              schedule: scheduler.set,
              cancelSchedule: scheduler.clear,
              current: {order_id: "O-2", order_name: "当前"},
              recent: [{order_id: "O-1", order_name: "最近"}],
              search: async (query, signal) => {
                assert.strictEqual(signal.aborted, false);
                queries.push(query);
                return [
                  {order_id: "O-3", order_name: "第三"},
                  {order_id: "O-2", order_name: "重复当前"}
                ];
              },
              select: async (order) => selections.push(order.order_id)
            });

            // Browser focus followed by click must not schedule duplicate searches.
            assert.strictEqual(controller.open(), true);
            assert.strictEqual(controller.open(), false);
            assert.strictEqual(timers.length, 1);
            scheduler.flushAll();
            await Promise.resolve();
            await Promise.resolve();
            assert.deepStrictEqual(queries, [""]);
            assert.deepStrictEqual(
              controller.getState().results.map((item) => item.order_id),
              ["O-2", "O-1", "O-3"]
            );

            const mouseOrder = controller.getState().results[2];
            assert.strictEqual(await controller.choose(mouseOrder), true);
            assert.deepStrictEqual(selections, ["O-3"]);
            """
        )

    def test_combobox_new_input_invalidates_old_response_before_debounce_runs(self):
        self._run_node(
            """
            const timers = [];
            const scheduler = {
              set(fn) {
                const handle = {fn, cancelled: false};
                timers.push(handle);
                return handle;
              },
              clear(handle) {
                if (handle) handle.cancelled = true;
              },
              flushNext() {
                const handle = timers.shift();
                if (handle && !handle.cancelled) handle.fn();
              }
            };
            const finishes = {};
            const signals = {};
            const controller = runtime.createOrderComboboxController({
              delay: 200,
              schedule: scheduler.set,
              cancelSchedule: scheduler.clear,
              search: (query, signal) => new Promise((resolve) => {
                finishes[query] = resolve;
                signals[query] = signal;
              }),
              select: async () => {}
            });

            controller.input("old");
            scheduler.flushNext();
            controller.input("new");
            assert.strictEqual(signals.old.aborted, true);
            finishes.old([{order_id: "OLD"}]);
            await Promise.resolve();
            await Promise.resolve();
            assert.strictEqual(controller.getState().open, false);
            assert.deepStrictEqual(controller.getState().results, []);

            scheduler.flushNext();
            finishes.new([{order_id: "NEW"}]);
            await Promise.resolve();
            await Promise.resolve();
            assert.strictEqual(controller.getState().open, true);
            assert.deepStrictEqual(
              controller.getState().results.map((item) => item.order_id),
              ["NEW"]
            );
            """
        )

    def test_combobox_escape_cancels_pending_and_inflight_searches(self):
        self._run_node(
            """
            const timers = [];
            const scheduler = {
              set(fn) {
                const handle = {fn, cancelled: false};
                timers.push(handle);
                return handle;
              },
              clear(handle) {
                if (handle) handle.cancelled = true;
              },
              flushAll() {
                while (timers.length) {
                  const handle = timers.shift();
                  if (!handle.cancelled) handle.fn();
                }
              }
            };
            let calls = 0;
            let finish;
            let signal;
            let selections = 0;
            const controller = runtime.createOrderComboboxController({
              schedule: scheduler.set,
              cancelSchedule: scheduler.clear,
              search: (_query, currentSignal) => {
                calls += 1;
                signal = currentSignal;
                return new Promise((resolve) => { finish = resolve; });
              },
              select: async () => { selections += 1; }
            });

            controller.input("pending");
            controller.close();
            scheduler.flushAll();
            assert.strictEqual(calls, 0);

            controller.input("running");
            scheduler.flushAll();
            assert.strictEqual(calls, 1);
            controller.close();
            assert.strictEqual(signal.aborted, true);
            finish([{order_id: "STALE"}]);
            await Promise.resolve();
            await Promise.resolve();
            assert.strictEqual(controller.getState().open, false);
            assert.strictEqual(controller.getState().activeIndex, -1);
            assert.deepStrictEqual(controller.getState().results, []);
            assert.strictEqual(await controller.chooseActive(), false);
            assert.strictEqual(selections, 0);

            controller.input("detached");
            scheduler.flushAll();
            assert.strictEqual(calls, 2);
            controller.dispose();
            assert.strictEqual(signal.aborted, true);
            """
        )

    def test_combobox_rapid_queries_and_selection_guard_are_deterministic(self):
        self._run_node(
            """
            const timers = [];
            const scheduler = {
              set(fn) {
                const handle = {fn, cancelled: false};
                timers.push(handle);
                return handle;
              },
              clear(handle) {
                if (handle) handle.cancelled = true;
              },
              flushAll() {
                while (timers.length) {
                  const handle = timers.shift();
                  if (!handle.cancelled) handle.fn();
                }
              }
            };
            const queries = [];
            let finishSelection;
            let selections = 0;
            const controller = runtime.createOrderComboboxController({
              schedule: scheduler.set,
              cancelSchedule: scheduler.clear,
              search: async (query) => {
                queries.push(query);
                return [{order_id: query.toUpperCase()}];
              },
              select: () => {
                selections += 1;
                return new Promise((resolve) => { finishSelection = resolve; });
              }
            });

            controller.input("a");
            controller.input("ab");
            controller.input("abc");
            scheduler.flushAll();
            await Promise.resolve();
            await Promise.resolve();
            assert.deepStrictEqual(queries, ["abc"]);
            assert.strictEqual(controller.getState().results[0].order_id, "ABC");

            const order = controller.getState().results[0];
            const keyboardChoice = controller.chooseActive();
            const mouseChoice = controller.choose(order);
            assert.strictEqual(await mouseChoice, false);
            assert.strictEqual(selections, 1);
            finishSelection();
            assert.strictEqual(await keyboardChoice, true);
            assert.strictEqual(await controller.chooseActive(), false);
            assert.strictEqual(selections, 1);
            """
        )

    def test_combobox_escape_key_prevents_browser_default_and_stays_closed(self):
        self._run_node(
            """
            const timers = [];
            const scheduler = {
              set(fn) {
                const handle = {fn, cancelled: false};
                timers.push(handle);
                return handle;
              },
              clear(handle) {
                if (handle) handle.cancelled = true;
              },
              flushAll() {
                while (timers.length) {
                  const handle = timers.shift();
                  if (!handle.cancelled) handle.fn();
                }
              }
            };
            let calls = 0;
            let finish;
            let signal;
            const controller = runtime.createOrderComboboxController({
              schedule: scheduler.set,
              cancelSchedule: scheduler.clear,
              search: (_query, currentSignal) => {
                calls += 1;
                signal = currentSignal;
                return new Promise((resolve) => { finish = resolve; });
              },
              select: async () => {}
            });
            controller.input("active");
            scheduler.flushAll();

            const event = {
              key: "Escape",
              defaultPrevented: false,
              preventDefault() { this.defaultPrevented = true; }
            };
            assert.strictEqual(
              await runtime.handleOrderComboboxKey(controller, event),
              true
            );
            assert.strictEqual(event.defaultPrevented, true);
            assert.strictEqual(signal.aborted, true);

            // Simulate the search-input browser default: it would clear the input
            // and emit a new input event only when preventDefault was omitted.
            if (!event.defaultPrevented) controller.input("");
            scheduler.flushAll();
            finish([{order_id: "STALE"}]);
            await Promise.resolve();
            await Promise.resolve();
            assert.strictEqual(calls, 1);
            assert.strictEqual(controller.getState().open, false);
            assert.deepStrictEqual(controller.getState().results, []);
            """
        )

    def test_review_failure_notes_are_precise_generic_and_escaped(self):
        self._run_node(
            """
            const partial = runtime.renderReviewFailureNotes({
              failedIds: ["RULE:OLD", "RULE:MISSING"],
              failureMessages: {
                "RULE:OLD": '请先计算 <script>alert("x")</script>'
              },
              selected: [
                {id: "RULE:OLD", name: "旧规则 <b>"},
                {id: "RULE:MISSING", name: "无消息规则"}
              ],
              hasData: true
            });
            assert(partial.includes("请先计算 &lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;"));
            assert(partial.includes("旧规则 &lt;b&gt;"));
            assert(partial.includes("无消息规则 在该订单下无可回放"));
            assert(!partial.includes("所选方案在当前订单下暂无可展示"));
            assert(!partial.includes("<script>"));

            const noData = runtime.renderReviewFailureNotes({
              failedIds: [],
              failureMessages: {},
              selected: [],
              hasData: false
            });
            assert(noData.includes("所选方案在当前订单下暂无可展示"));
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

    def test_order_search_external_abort_reaches_fetch_and_preserves_later_cache(self):
        self._run_node(
            """
            let calls = 0;
            let injectedSignal;
            const client = runtime.createClient({
              fetchReviewData: async () => ({schemes: {}}),
              fetchOrders: (_args, signal) => {
                calls += 1;
                injectedSignal = signal;
                if (calls > 1) {
                  return Promise.resolve({orders: [{order_id: "O-NEW"}]});
                }
                return new Promise((_resolve, reject) => {
                  signal.addEventListener("abort", () => {
                    reject(new DOMException("aborted", "AbortError"));
                  });
                });
              }
            });
            const external = new AbortController();
            const pending = client.searchOrders(
              {taskId: "t1", ids: ["S-1"], query: "old"},
              external.signal
            );
            assert.strictEqual(injectedSignal.aborted, false);
            external.abort();
            assert.strictEqual(injectedSignal.aborted, true);
            assert.deepStrictEqual(await pending, {cancelled: true});

            const fresh = await client.searchOrders({
              taskId: "t1", ids: ["S-1"], query: "new"
            });
            assert.strictEqual(fresh.orders[0].order_id, "O-NEW");
            assert.strictEqual(fresh.fromCache, false);
            const cached = await client.searchOrders({
              taskId: "t1", ids: ["S-1"], query: " new "
            });
            assert.strictEqual(cached.fromCache, true);
            assert.strictEqual(calls, 2);
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

    def test_client_caches_are_bounded_lru_and_resettable(self):
        self._run_node(
            """
            let dataCalls = 0;
            let orderCalls = 0;
            const client = runtime.createClient({
              dataCacheLimit: 2,
              orderCacheLimit: 2,
              fetchReviewData: async (args) => {
                dataCalls += 1;
                return {order_id: args.orderId, schemes: {}};
              },
              fetchOrders: async (args) => {
                orderCalls += 1;
                return {orders: [{order_id: args.query}]};
              }
            });
            const dataArgs = (orderId) => ({
              taskId: "t1", ids: ["S-1"], orderId, includeUtilization: false
            });
            const orderArgs = (query) => ({
              taskId: "t1", ids: ["S-1"], query
            });

            await client.loadData(dataArgs("A"));
            await client.loadData(dataArgs("B"));
            assert.strictEqual((await client.loadData(dataArgs("A"))).fromCache, true);
            await client.loadData(dataArgs("C"));
            assert.deepStrictEqual(client.cacheStats().dataKeys.map((key) => key.split("::")[2]), ["A", "C"]);
            assert.strictEqual((await client.loadData(dataArgs("B"))).fromCache, false);
            assert.strictEqual(dataCalls, 4);
            assert.strictEqual(client.cacheStats().dataSize, 2);

            await client.searchOrders(orderArgs("a"));
            await client.searchOrders(orderArgs("b"));
            assert.strictEqual((await client.searchOrders(orderArgs("a"))).fromCache, true);
            await client.searchOrders(orderArgs("c"));
            assert.deepStrictEqual(client.cacheStats().orderKeys.map((key) => key.split("::")[2]), ["a", "c"]);
            assert.strictEqual((await client.searchOrders(orderArgs("b"))).fromCache, false);
            assert.strictEqual(orderCalls, 4);
            assert.strictEqual(client.cacheStats().orderSize, 2);

            client.reset();
            assert.deepStrictEqual(client.cacheStats(), {
              dataSize: 0,
              orderSize: 0,
              dataKeys: [],
              orderKeys: [],
              limits: {data: 2, orders: 2}
            });
            """
        )

    def test_invalid_and_fractional_cache_limits_are_finite_positive_integers(self):
        self._run_node(
            """
            const makeClient = (dataCacheLimit, orderCacheLimit) =>
              runtime.createClient({
                dataCacheLimit,
                orderCacheLimit,
                fetchReviewData: async () => ({schemes: {}}),
                fetchOrders: async () => ({orders: []})
              });
            assert.deepStrictEqual(
              makeClient(Infinity, NaN).cacheStats().limits,
              {data: 12, orders: 100}
            );
            assert.deepStrictEqual(
              makeClient(0, -4).cacheStats().limits,
              {data: 12, orders: 100}
            );
            assert.deepStrictEqual(
              makeClient(2.9, 3.8).cacheStats().limits,
              {data: 2, orders: 3}
            );
            assert.deepStrictEqual(
              makeClient(1000000, 1000000).cacheStats().limits,
              {data: 1000, orders: 1000}
            );
            """
        )


if __name__ == "__main__":
    unittest.main()
