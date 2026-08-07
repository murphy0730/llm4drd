import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "frontend" / "app_v2.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend" / "app_v2.css").read_text(encoding="utf-8")


class InsertionFrontendContractTests(unittest.TestCase):
    def test_review_tab_is_immediately_after_library(self):
        library = 'data-review-tab="library">方案库</button>'
        insertion = 'data-review-tab="insertion">插单模拟</button>'
        self.assertIn(library, JS)
        self.assertIn(insertion, JS)
        self.assertLess(JS.index(library), JS.index(insertion))
        between = JS[JS.index(library) + len(library):JS.index(insertion)]
        self.assertNotIn("data-review-tab", between)

    def test_three_input_modes_and_two_policies_are_wired(self):
        for token in (
            'data-action="add-insertion-order"',
            'data-action="trigger-insertion-import"',
            'data-action="apply-insertion-paste"',
            'value="frozen"',
            'value="due_protected"',
            'data-action="run-insertion-evaluation"',
        ):
            self.assertIn(token, JS)

    def test_demo_data_uses_current_instance_resources(self):
        self.assertIn('data-action="fill-insertion-demo"', JS)
        self.assertIn("function buildInsertionDemoData()", JS)
        self.assertIn("app.instanceDetails?.machine_types", JS)
        self.assertIn("app.instanceDetails?.tooling_types", JS)
        self.assertIn("app.instanceDetails?.personnel", JS)
        self.assertIn("app.insertion.basePlan = `saved:${firstSet.strategies[0].id}`", JS)
        self.assertIn("URG-DEMO-001-PART", JS)
        self.assertIn('predecessor_ops: "URG-D2-OP-01"', JS)
        self.assertIn('predecessor_tasks: "URG-DEMO-001-PART"', JS)

    def test_published_strategy_is_the_only_base(self):
        function_start = JS.index("function ensureInsertionBaseSelection()")
        function_end = JS.index("function insertionHasBase()", function_start)
        body = JS[function_start:function_end]
        # 基准只能是已发布方案（方案集 + 预排方案）：basePlan 失效时回退到第一个可用方案
        self.assertIn("saved:${firstSet.strategies[0].id}", body)
        # 「使用当前规则仿真结果作为基准」勾选项已移除，其残留会让方案集下拉被禁用
        self.assertNotIn("useSimulationBase", JS)
        self.assertNotIn("insertion-use-sim", JS)
        self.assertIn('base_source: "strategy"', JS)

    def test_result_contains_all_required_views(self):
        for label in (
            "新订单交期结论",
            "新工序插入明细",
            "插单后合并甘特图",
            "KPI 对比",
            "现有订单保护校验",
        ):
            self.assertIn(label, JS)
        self.assertIn('data-action="export-insertion-result"', JS)

    def test_insertion_api_contract_and_visual_distinction_exist(self):
        for endpoint in (
            '"/insertion/evaluate"',
            '"/insertion/template"',
            '"/insertion/import"',
            "getInsertionSchedule(runId)",
            "exportInsertion(runId)",
        ):
            self.assertIn(endpoint, JS)
        self.assertIn("inserted-order", JS)
        self.assertIn(".gantt-canvas .vis-item.inserted-order", CSS)


if __name__ == "__main__":
    unittest.main()
