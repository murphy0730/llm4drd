import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "frontend" / "app_v2.js").read_text(encoding="utf-8")


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

    def test_review_order_change_reuses_utilization(self):
        start = JS.index('if (target.matches("[data-review-gantt-order]"))')
        end = JS.index("\n    }", start)
        self.assertIn("loadReviewData(getSelectedReviewCandidates(), target.value, false)", JS[start:end])

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


if __name__ == "__main__":
    unittest.main()
