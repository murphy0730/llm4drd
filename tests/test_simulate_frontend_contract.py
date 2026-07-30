import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "frontend" / "app_v2.js").read_text(encoding="utf-8")


class SimulateFrontendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        start = JS.index("function renderSimulatePage()")
        end = JS.index("function renderObjectiveSelectors()", start)
        cls.source = JS[start:end]

    def test_rule_simulation_uses_requested_kpis(self):
        expected = {
            "主订单延误率": "simMetrics.main_order_tardy_ratio",
            "关键资源活跃窗口利用率": "simMetrics.critical_active_window_utilization",
            "平均流程时间": "simMetrics.avg_flowtime",
            "完成工序": "simMetrics.completed_operations",
        }
        for label, metric in expected.items():
            with self.subTest(label=label):
                self.assertIn(label, self.source)
                self.assertIn(metric, self.source)

    def test_main_order_tardy_card_shows_count_breakdown(self):
        self.assertIn("simMetrics.main_order_tardy_count", self.source)
        self.assertIn("simMetrics.total_main_orders", self.source)
        self.assertIn("<br>主订单延误数 / 核心订单数（is_main=Y）", self.source)

    def test_replaced_kpis_are_not_rendered_on_rule_simulation_page(self):
        for label in ("总延误</span>", "总周期 (Makespan)</span>", "净可用利用率</span>"):
            with self.subTest(label=label):
                self.assertNotIn(label, self.source)


if __name__ == "__main__":
    unittest.main()
