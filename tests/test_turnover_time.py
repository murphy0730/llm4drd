"""工序流转等待时间（turnover_time）语义测试。

对应设计：docs/superpowers/specs/2026-07-16-operation-turnover-time-design.md
"""
import io
import unittest

import openpyxl

from llm4drd.core.models import Operation
from llm4drd.data.template_builder import build_instance_template_bytes


class TestTurnoverField(unittest.TestCase):
    def test_turnover_time_defaults_to_zero(self):
        """既有构造点不传 turnover_time 时必须落 0，这是零回归的锚点。"""
        op = Operation(id="OP1", task_id="T1", name="OP1", process_type="turning", processing_time=5.0)
        self.assertEqual(op.turnover_time, 0.0)

    def test_turnover_time_is_settable(self):
        op = Operation(
            id="OP1", task_id="T1", name="OP1", process_type="turning",
            processing_time=5.0, turnover_time=3.5,
        )
        self.assertEqual(op.turnover_time, 3.5)


class TestTemplateColumn(unittest.TestCase):
    def _operations_headers(self) -> list[str]:
        wb = openpyxl.load_workbook(io.BytesIO(build_instance_template_bytes()))
        ws = wb["operations"]
        return [cell.value for cell in ws[1]]

    def test_turnover_column_follows_processing_time(self):
        headers = self._operations_headers()
        self.assertIn("turnover_time_hrs", headers)
        self.assertEqual(
            headers.index("turnover_time_hrs"),
            headers.index("processing_time_hrs") + 1,
            "turnover_time_hrs 必须紧跟在 processing_time_hrs 之后",
        )

    def test_template_demonstrates_nonzero_turnover(self):
        wb = openpyxl.load_workbook(io.BytesIO(build_instance_template_bytes()))
        ws = wb["operations"]
        headers = [cell.value for cell in ws[1]]
        column = headers.index("turnover_time_hrs")
        values = [row[column].value for row in ws.iter_rows(min_row=2)]
        self.assertTrue(any(v for v in values), "模板应至少有一行非零 turnover 以演示语义")


if __name__ == "__main__":
    unittest.main()
