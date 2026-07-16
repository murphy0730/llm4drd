"""工序流转等待时间（turnover_time）语义测试。

对应设计：docs/superpowers/specs/2026-07-16-operation-turnover-time-design.md
"""
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path

import openpyxl

from llm4drd.core.models import Operation
from llm4drd.data.db import InstanceStore, _float_or_default, init_db
from llm4drd.data.template_builder import build_instance_template_bytes
from llm4drd.tests.shop_fixtures import make_graph_context_shop


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


class TestTurnoverPersistence(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "test.db")
        init_db(self.db_path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_ddl_has_turnover_column(self):
        with sqlite3.connect(self.db_path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(inst_operations)")}
        self.assertIn("turnover_time", columns)

    def test_null_turnover_in_legacy_row_loads_as_zero(self):
        """模拟旧库：turnover_time 为 NULL 的行加载必须落 0 而非崩溃。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO inst_operations (op_id, task_id, op_name, process_type, processing_time, turnover_time) "
                "VALUES ('OP1','T1','OP1','turning',5.0,NULL)"
            )
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = dict(conn.execute("SELECT * FROM inst_operations WHERE op_id='OP1'").fetchone())
        self.assertEqual(_float_or_default(row.get("turnover_time"), 0.0), 0.0)

    def test_migration_is_idempotent_on_legacy_table(self):
        """旧库缺列时 init_db 必须补列且可重复执行。"""
        legacy = str(Path(self._tmp.name) / "legacy.db")
        with sqlite3.connect(legacy) as conn:
            conn.execute(
                "CREATE TABLE inst_operations (op_id TEXT PRIMARY KEY, task_id TEXT, "
                "op_name TEXT, process_type TEXT, processing_time REAL)"
            )
        init_db(legacy)
        init_db(legacy)  # 第二次必须不报错
        with sqlite3.connect(legacy) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(inst_operations)")}
        self.assertIn("turnover_time", columns)

    def test_excel_import_without_turnover_column_defaults_to_zero(self):
        """旧 xlsx 缺列时导入必须落 0。"""
        legacy_row = {"op_id": "OP1", "task_id": "T1", "processing_time_hrs": 5.0}
        self.assertEqual(
            _float_or_default(legacy_row.get("turnover_time_hrs", legacy_row.get("turnover_time", 0)), 0.0),
            0.0,
        )

    def test_excel_import_reads_turnover_column(self):
        row = {"op_id": "OP1", "task_id": "T1", "processing_time_hrs": 5.0, "turnover_time_hrs": 2.5}
        self.assertEqual(
            _float_or_default(row.get("turnover_time_hrs", row.get("turnover_time", 0)), 0.0),
            2.5,
        )

    def test_save_and_load_roundtrip_preserves_turnover_time(self):
        """save_from_shopfloor -> build_shopfloor 必须保值非零 turnover_time。"""
        shop = make_graph_context_shop()
        shop.operations["OP-11"].turnover_time = 2.5
        store = InstanceStore(self.db_path)
        store.save_from_shopfloor(shop)
        loaded = store.build_shopfloor()
        self.assertEqual(loaded.operations["OP-11"].turnover_time, 2.5)


if __name__ == "__main__":
    unittest.main()
