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


if __name__ == "__main__":
    unittest.main()
