import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "frontend" / "app_v2.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend" / "app_v2.css").read_text(encoding="utf-8")
RUNTIME_JS = (ROOT / "frontend" / "review_runtime.js").read_text(encoding="utf-8")


class ReviewFrontendContractTests(unittest.TestCase):
    def test_all_gantts_use_shared_initial_window(self):
        self.assertIn("function ganttWindowPayload(", JS)
        self.assertGreaterEqual(JS.count("ganttWindowPayload("), 3)
        self.assertIn("ReviewRuntime.computeInitialWindow", JS)
        self.assertIn("fullWindow", JS)
        self.assertIn("initialWindow", JS)
        self.assertIn("data.initialWindow", JS)
        self.assertNotIn("data.window ?", JS)

    def test_gantt_ranges_are_bound_to_logical_views(self):
        self.assertIn("ganttViewWindows: {}", JS)
        self.assertIn("stored?.viewKey === data.viewKey", JS)
        self.assertIn('timeline.on("rangechanged"', JS)
        self.assertIn("min: data.fullWindow", JS)
        self.assertIn("max: data.fullWindow", JS)
        self.assertIn("app.ganttViewWindows = {}", JS)

    def test_view_keys_include_logical_dimensions(self):
        self.assertIn("options.selectedOrder", JS)
        self.assertIn("options.solutionIds", JS)
        self.assertIn("groupMode", JS)
        self.assertIn("selected.map((item) => item.id)", JS)

    def test_review_api_accepts_abort_signals(self):
        self.assertIn("getReviewData(taskId, solutionIds, orderId, includeUtilization, signal)", JS)
        self.assertIn("searchReviewOrders(taskId, solutionIds, query, signal)", JS)
        self.assertGreaterEqual(JS.count("{ signal }"), 2)

    def test_review_linkage_is_batched_and_locally_rendered(self):
        self.assertIn("getReviewData(", JS)
        self.assertNotIn("Promise.all(ids.map((id) =>", JS)
        self.assertIn('id="review-comparison-region"', JS)
        self.assertIn('id="review-utilization-region"', JS)
        self.assertIn('id="review-gantt-region"', JS)
        self.assertIn("refreshReviewDynamicRegions()", JS)
        self.assertIn(".items.clear()", JS)
        self.assertIn(".groups.clear()", JS)

    def test_review_loads_do_not_render_the_whole_page(self):
        start = JS.index("async function loadReviewData(")
        end = JS.index("\nfunction ", start)
        self.assertNotIn("renderCurrentPage()", JS[start:end])

    def test_review_selection_drops_stale_pending_canvas_without_full_render(self):
        self.assertIn('app.pendingGantts.delete("gantt-review-compare")', JS)
        start = JS.index('if (action === "toggle-candidate")')
        end = JS.index('if (action === "retry-type-utilization")', start)
        self.assertNotIn("renderCurrentPage()", JS[start:end])
        self.assertIn("ensureReviewData(getSelectedReviewCandidates())", JS[start:end])

    def test_order_selectors_use_accessible_fuzzy_combobox(self):
        for token in (
            'role="combobox"',
            'role="listbox"',
            "aria-expanded",
            "ORDER_SEARCH_DEBOUNCE_MS = 200",
            "ORDER_SEARCH_LIMIT = 50",
        ):
            self.assertIn(token, JS)
        for key in ('"ArrowDown"', '"ArrowUp"', '"Enter"', '"Escape"'):
            self.assertIn(key, RUNTIME_JS)
        self.assertNotIn("data-review-gantt-order", JS)
        self.assertNotIn("data-gantt-order-select", JS)

    def test_combobox_selection_paths_preserve_gantt_loading_contracts(self):
        self.assertIn("ReviewRuntime.rankOrders(", JS)
        self.assertIn("ReviewRuntime.createOrderComboboxController(", JS)
        self.assertIn("ReviewRuntime.handleOrderComboboxKey(controller, event)", JS)
        self.assertIn("reviewDataClient.searchOrders({ taskId, ids, query }, signal)", JS)
        self.assertIn("api.searchReviewOrders(taskId, [solutionId], query, signal)", JS)
        self.assertIn("loadReviewData(getSelectedReviewCandidates(), order.order_id, false)", JS)
        self.assertIn("loadPlanGantt(taskId, solutionId, order.order_id)", JS)

    def test_comboboxes_mount_after_timeline_and_review_region_rendering(self):
        gantt_start = JS.index("function mountGantts()")
        gantt_end = JS.index("\nasync function renderCurrentPage(", gantt_start)
        self.assertIn("mountOrderComboboxes()", JS[gantt_start:gantt_end])
        review_start = JS.index("function refreshReviewDynamicRegions()")
        review_end = JS.index("\nfunction renderReview()", review_start)
        self.assertIn("mountOrderComboboxes()", JS[review_start:review_end])

    def test_combobox_focus_click_and_recent_order_contract(self):
        mount_start = JS.index("function mountOrderComboboxes()")
        mount_end = JS.index("\nfunction formatNumber", mount_start)
        mount_source = JS[mount_start:mount_end]
        self.assertIn('input.addEventListener("focus"', mount_source)
        self.assertIn('input.addEventListener("click"', mount_source)
        self.assertGreaterEqual(mount_source.count("controller.open()"), 2)
        self.assertIn("orderComboboxRecent", JS)
        reset_start = JS.index("function resetInstanceDerivedState()")
        reset_end = JS.index("\nfunction ", reset_start)
        self.assertIn("app.orderComboboxRecent.clear()", JS[reset_start:reset_end])

    def test_review_gantt_preserves_escaped_backend_failure_messages(self):
        start = JS.index("function reviewGanttStatusHtml(")
        end = JS.index("\nfunction renderReviewGantt(", start)
        source = JS[start:end]
        self.assertIn("state.failureMessages", source)
        self.assertIn("failedId", source)
        self.assertIn("escapeHtml(failureMessage)", source)
        self.assertIn("在该订单下无可回放", source)
        self.assertNotIn("escapeHtml(failedNames.join", source)
        self.assertIn("if (data) return failedNote;", source)

    def test_review_commits_are_guarded_by_request_generation_and_schedule(self):
        self.assertIn("let reviewReadRequestGeneration = 0", JS)
        self.assertIn("function isCurrentReviewReadRequest(", JS)
        start = JS.index("async function loadReviewData(")
        end = JS.index("\nfunction ensureReviewData(", start)
        self.assertGreaterEqual(JS[start:end].count("isCurrentReviewReadRequest("), 2)

    def test_pending_order_does_not_relabel_or_upsert_committed_schemes(self):
        start = JS.index("function currentReviewGanttData(")
        end = JS.index("\nfunction reviewGanttStatusHtml(", start)
        self.assertIn("if (app.reviewRead.loading) return null;", JS[start:end])
        load_start = JS.index("async function loadReviewData(")
        load_end = JS.index("\nfunction ensureReviewData(", load_start)
        self.assertIn("orderId: selectionChanged ? null : previous.orderId", JS[load_start:load_end])
        self.assertIn("scheduleKey: selectionChanged ? null : previous.scheduleKey", JS[load_start:load_end])

    def test_review_tables_use_compact_approved_content(self):
        comparison_start = JS.index("function renderReviewCandidateComparison")
        comparison_end = JS.index("function renderReviewTypeUtilization")
        comparison_source = JS[comparison_start:comparison_end]
        self.assertIn(
            '<th class="compare-check"><span class="sr-only">选择方案</span></th>',
            comparison_source,
        )
        self.assertNotIn('const headers = ["选"', comparison_source)
        self.assertIn('aria-label="勾选 ${escapeHtml(item.name)}"', comparison_source)

        utilization_start = JS.index("function renderReviewTypeUtilization")
        utilization_end = JS.index("function buildReviewGanttData")
        utilization_source = JS[utilization_start:utilization_end]
        for token in (
            "used_machines",
            "machines_total",
            "type_id",
            "is_critical",
            "util-bar",
            "cell-sub",
        ):
            self.assertNotIn(token, utilization_source)
        self.assertIn(
            "<td><strong>${escapeHtml(type.type_name)}</strong></td>",
            utilization_source,
        )
        self.assertIn("<strong>${formatPercent(entry.utilization)}</strong>", utilization_source)
        self.assertIn('class="${isBest ? "is-best" : ""}"', utilization_source)

        self.assertIn(".util-col-type { width: 132px; }", CSS)
        util_css_start = CSS.index(".util-table")
        util_css = CSS[util_css_start:CSS.index(".table-shell", util_css_start)]
        self.assertIn("white-space: normal", util_css)
        self.assertIn("overflow: visible", util_css)
        self.assertIn("word-break: break-word", util_css)
        self.assertNotIn("text-overflow: ellipsis", util_css)
        self.assertNotIn("max-width", util_css)
        self.assertNotIn(".util-bar", util_css)

        sr_only_start = CSS.index(".sr-only")
        sr_only_css = CSS[sr_only_start:CSS.index("}", sr_only_start)]
        self.assertIn("position: absolute", sr_only_css)
        self.assertIn("clip:", sr_only_css)


if __name__ == "__main__":
    unittest.main()
