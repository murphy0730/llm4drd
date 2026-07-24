"""活跃窗口利用率口径：分子用墙钟占用时长（含跨班次间隔），与甘特图条块一致。

对照：machine_utilization（全周期）与 machine_net_available_utilization（净可用）
仍用净加工时长 duration，不受本次口径修正影响。
"""
import unittest

from llm4drd.core.models import Machine, Shift
from llm4drd.core.simulator import SimResult
from llm4drd.optimization.objectives import build_schedule_analytics
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class ActiveWindowUtilizationTests(unittest.TestCase):
    def test_cross_shift_entry_counts_wall_clock_occupancy(self):
        """8h/天班次机器，一条跨班次条目占满两天班次窗口 → 活跃窗口利用率 1.0。"""
        shop = make_graph_context_shop()
        shifts = [Shift(day=day, start_hour=8.0, hours=8.0) for day in range(14)]
        shop.machines["M-C1"] = Machine("M-C1", "Cutter 1", "cut", shifts=shifts)
        shop.build_indexes()

        # start=8(D0班次开始) → end=40(D1班次结束)：墙钟占用 32h，净加工 16h（两天各 8h）
        result = SimResult()
        result.schedule = [{
            "op_id": "OP-11", "task_id": "T-11", "machine_id": "M-C1",
            "start": 8.0, "end": 40.0, "duration": 16.0, "elapsed_duration": 32.0,
        }]

        analytics = build_schedule_analytics(shop, result)
        # 旧口径：16 / 32 = 0.5；新口径：32 / 32 = 1.0
        self.assertAlmostEqual(analytics.machine_active_window_utilization["M-C1"], 1.0)
        # 净可用口径不变：净加工 16 / 窗口 [8,40] 内可用工时 16
        self.assertAlmostEqual(analytics.machine_net_available_utilization["M-C1"], 1.0)

        # 另两个指标仍用净加工 duration：抹掉 elapsed_duration 后数值不变
        result_no_elapsed = SimResult()
        result_no_elapsed.schedule = [
            {k: v for k, v in entry.items() if k != "elapsed_duration"} for entry in result.schedule
        ]
        baseline = build_schedule_analytics(shop, result_no_elapsed)
        self.assertAlmostEqual(analytics.machine_utilization["M-C1"], baseline.machine_utilization["M-C1"])
        self.assertAlmostEqual(
            analytics.machine_net_available_utilization["M-C1"],
            baseline.machine_net_available_utilization["M-C1"],
        )

    def test_without_elapsed_duration_falls_back_to_span(self):
        """条目缺 elapsed_duration 时回退 end-start，仍为墙钟口径。"""
        shop = make_graph_context_shop()
        result = SimResult()
        result.schedule = [
            {"op_id": "OP-11", "machine_id": "M-C1", "start": 0.0, "end": 10.0, "duration": 6.0},
            {"op_id": "OP-12", "machine_id": "M-C1", "start": 10.0, "end": 20.0, "duration": 6.0},
        ]
        analytics = build_schedule_analytics(shop, result)
        self.assertAlmostEqual(analytics.machine_active_window_utilization["M-C1"], 1.0)


if __name__ == "__main__":
    unittest.main()
