import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import openpyxl
import pandas as pd

from llm4drd.convert.convert_to_instance import (
    apply_os_pred_turnover,
    finalize_split_report,
    merge_continuous_ops,
    split_large_ops,
    write_merge_report,
)
from llm4drd.convert.restore_schedule import (
    build_merge_map,
    build_source_lookup,
    build_split_map,
    enrich,
    restore_operations,
)
from llm4drd.core.rules import BUILTIN_RULES
from llm4drd.core.simulator import Simulator
from llm4drd.data.db import InstanceStore, init_db


def _operation(op_id, hours, predecessor="", *, process_type="CNC", quantity=10,
               machines="M1,M2,M3", work_num=1, op_name=None):
    return {
        "op_id": op_id,
        "task_id": "T1",
        "op_name": op_name or f"{process_type}1-E800",
        "process_type": process_type,
        "processing_time_hrs": hours,
        "predecessor_ops": predecessor,
        "predecessor_tasks": "",
        "eligible_machine_ids": machines,
        "turnover_time_hrs": 2,
        "_work_num": work_num,
        "_part_qty": quantity,
        "_part_qty_issue": "" if quantity is not None else "零件数量无效",
    }


def _initial_rows(operations):
    return [
        {
            "op_id": row["op_id"],
            "initial_start_time": 0,
            "initial_end_time": 100,
            "initial_status": "PENDING",
            "initial_remaining_processing_time": row["processing_time_hrs"],
            "initial_assigned_machine_id": "",
        }
        for row in operations
    ]


class ConvertOperationSplitTests(unittest.TestCase):
    def test_split_runs_after_merge_and_uses_merged_total_hours(self):
        operations = [
            _operation("A", 30, work_num=1, op_name="CNC1-E800", quantity=11),
            _operation("B", 30, "A", work_num=2, op_name="CNC2-E800", quantity=11),
            _operation("N", 1, "B", process_type="QC", machines="Q1", work_num=3),
        ]

        merged_ops, merged_initial = merge_continuous_ops(
            operations, _initial_rows(operations), [],
        )
        rows, _ = split_large_ops(
            merged_ops, merged_initial, {"M1", "M2", "M3", "Q1"}, [],
        )
        children = [row for row in rows if row["op_id"].startswith("A__S")]
        successor = next(row for row in rows if row["op_id"] == "N")

        self.assertEqual(len(children), 3)
        self.assertEqual(sum(row["_part_qty"] for row in children), 11)
        self.assertEqual(sum(row["processing_time_hrs"] for row in children), 60)
        self.assertEqual(successor["predecessor_ops"], "A__S01;A__S02;A__S03")

    def test_merged_quantity_mismatch_is_kept_unsplit(self):
        operations = [
            _operation("A", 30, work_num=1, op_name="CNC1-E800", quantity=10),
            _operation("B", 30, "A", work_num=2, op_name="CNC2-E800", quantity=11),
        ]
        split_report = []

        merged_ops, merged_initial = merge_continuous_ops(
            operations, _initial_rows(operations), [],
        )
        rows, _ = split_large_ops(
            merged_ops, merged_initial, {"M1", "M2", "M3"}, split_report,
        )

        self.assertEqual([row["op_id"] for row in rows], ["A"])
        self.assertEqual(split_report[0]["类型"], "未拆分")
        self.assertIn("不一致", split_report[0]["未拆分原因"])

    def test_machine_count_limits_split_and_rewires_successor(self):
        operations = [
            _operation("P", 1, process_type="DP", machines="P1", work_num=1),
            _operation("O", 60, "P", machines="M1,M2", work_num=2),
            _operation("N", 1, "O", process_type="QC-OS", machines="OS_1",
                       work_num=3, op_name="QC-OS"),
        ]
        report = []

        split_ops, split_initial = split_large_ops(
            operations, _initial_rows(operations), {"P1", "M1", "M2", "OS_1"}, report,
        )
        apply_os_pred_turnover(split_ops)
        finalize_split_report(report, split_ops)

        children = [row for row in split_ops if row["op_id"].startswith("O__S")]
        successor = next(row for row in split_ops if row["op_id"] == "N")
        self.assertEqual([row["op_id"] for row in children], ["O__S01", "O__S02"])
        self.assertEqual([row["_part_qty"] for row in children], [5, 5])
        self.assertEqual(sum(row["processing_time_hrs"] for row in children), 60)
        self.assertEqual(successor["predecessor_ops"], "O__S01;O__S02")
        self.assertTrue(all(row["turnover_time_hrs"] == 24 for row in children))
        self.assertEqual({row["op_id"] for row in split_ops},
                         {row["op_id"] for row in split_initial})
        self.assertTrue(all(row["最终turnover_time_hrs"] == 24 for row in report))

    def test_remainder_and_hours_are_deterministic_and_conserved(self):
        def run_once():
            operations = [_operation("O", 51, quantity=11)]
            report = []
            rows, _ = split_large_ops(
                operations, _initial_rows(operations), {"M1", "M2", "M3"}, report,
            )
            return [
                (row["op_id"], row["_part_qty"], row["processing_time_hrs"])
                for row in rows
            ]

        first = run_once()
        second = run_once()
        self.assertEqual(first, second)
        self.assertEqual(sorted(quantity for _, quantity, _ in first), [3, 4, 4])
        self.assertEqual(sum(quantity for _, quantity, _ in first), 11)
        self.assertAlmostEqual(sum(hours for _, _, hours in first), 51.0, places=6)

    def test_time_boundaries_process_prefix_and_three_task_cap(self):
        operations = [
            _operation("AT25", 25, work_num=1),
            _operation("ABOVE25", 25.01, work_num=2),
            _operation("AT50", 50, work_num=3),
            _operation("ABOVE50", 50.01, work_num=4),
            _operation("WEDM100", 100, process_type="WEDM_C", work_num=5),
            _operation("EDM100", 100, process_type="EDM", work_num=6),
        ]

        rows, _ = split_large_ops(
            operations, _initial_rows(operations), {"M1", "M2", "M3"}, [],
        )
        counts = {}
        for row in rows:
            parent_id = row["op_id"].split("__S", 1)[0]
            counts[parent_id] = counts.get(parent_id, 0) + 1

        self.assertEqual(counts, {
            "AT25": 1,
            "ABOVE25": 2,
            "AT50": 2,
            "ABOVE50": 3,
            "WEDM100": 3,
            "EDM100": 1,
        })

    def test_invalid_quantity_and_insufficient_machine_do_not_split(self):
        invalid = _operation("INVALID", 60, quantity=None)
        one_machine = _operation("ONE", 60, quantity=10, machines="M1", work_num=2)
        operations = [invalid, one_machine]
        report = []

        rows, states = split_large_ops(
            operations, _initial_rows(operations), {"M1"}, report,
        )

        self.assertEqual([row["op_id"] for row in rows], ["INVALID", "ONE"])
        self.assertEqual([row["op_id"] for row in states], ["INVALID", "ONE"])
        self.assertEqual([row["类型"] for row in report], ["未拆分", "未拆分"])
        self.assertIn("零件数量", report[0]["未拆分原因"])
        self.assertIn("机台", report[1]["未拆分原因"])

    def test_consecutive_splits_form_full_barrier(self):
        operations = [
            _operation("A", 51, quantity=11, work_num=1),
            _operation("B", 75, "A", quantity=11, work_num=2),
            _operation("C", 1, "B", process_type="QC", machines="Q1", work_num=3),
        ]

        rows, _ = split_large_ops(
            operations, _initial_rows(operations), {"M1", "M2", "M3", "Q1"}, [],
        )
        by_id = {row["op_id"]: row for row in rows}
        first_group = {"A__S01", "A__S02", "A__S03"}
        for index in range(1, 4):
            self.assertEqual(
                set(by_id[f"B__S{index:02d}"]["predecessor_ops"].split(";")),
                first_group,
            )
        self.assertEqual(by_id["C"]["predecessor_ops"], "B__S01;B__S02;B__S03")

    def test_report_preserves_merge_sheet_and_adds_split_sheet(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "merge_report.xlsx"
            write_merge_report([], path, [])
            workbook = openpyxl.load_workbook(path, read_only=True)
            self.assertEqual(workbook.sheetnames, ["合并清单", "拆分清单"])
            workbook.close()


class RestoreSplitScheduleTests(unittest.TestCase):
    def test_legacy_merge_only_report_remains_compatible(self):
        merge_df = pd.DataFrame([
            {"类型": "已合并", "合并后op_id": "O", "原op_id": "X",
             "原op_name": "CNC1-E800", "WORK编号": 10,
             "原processing_time_hrs": 2, "合并后turnover_time_hrs": 0},
            {"类型": "已合并", "合并后op_id": "O", "原op_id": "Y",
             "原op_name": "CNC2-E800", "WORK编号": 11,
             "原processing_time_hrs": 3, "合并后turnover_time_hrs": 2},
        ])
        anchor = pd.Timestamp("2026-01-01 08:00:00")
        plan_df = pd.DataFrame([
            self._plan_row("P", "", 0, 1, anchor),
            self._plan_row("O", "P", 3, 8, anchor),
            self._plan_row("N", "O", 10, 11, anchor),
        ])

        seg_map, tail_map, _ = build_merge_map(merge_df)
        restored = restore_operations(plan_df, seg_map, tail_map)
        by_id = {row["工序ID"]: row for _, row in restored.iterrows()}

        self.assertEqual(list(restored["工序ID"]), ["P", "X", "Y", "N"])
        self.assertEqual(by_id["X"]["前工序ID"], "P")
        self.assertEqual(by_id["Y"]["前工序ID"], "X")
        self.assertEqual(by_id["N"]["前工序ID"], "Y")
        self.assertEqual(by_id["X"]["时长(小时)"], 2)
        self.assertEqual(by_id["Y"]["时长(小时)"], 3)

    def test_merged_split_children_restore_without_duplicate_quantity_or_time(self):
        merge_df = pd.DataFrame([
            {"类型": "已合并", "合并后op_id": "O", "原op_id": "X",
             "原op_name": "CNC1-E800", "WORK编号": 10,
             "原processing_time_hrs": 30, "合并后turnover_time_hrs": 0},
            {"类型": "已合并", "合并后op_id": "O", "原op_id": "Y",
             "原op_name": "CNC2-E800", "WORK编号": 11,
             "原processing_time_hrs": 30, "合并后turnover_time_hrs": 2},
        ])
        split_df = pd.DataFrame([
            {"类型": "已拆分", "拆分前op_id": "O", "子op_id": "O__S01",
             "WORK编号": 10, "拆分序号": 1, "拆分数量": 2,
             "原零件数量": 10, "子批零件数量": 5,
             "子processing_time_hrs": 30, "最终turnover_time_hrs": 24},
            {"类型": "已拆分", "拆分前op_id": "O", "子op_id": "O__S02",
             "WORK编号": 10, "拆分序号": 2, "拆分数量": 2,
             "原零件数量": 10, "子批零件数量": 5,
             "子processing_time_hrs": 30, "最终turnover_time_hrs": 24},
        ])
        anchor = pd.Timestamp("2026-01-01 08:00:00")
        plan_df = pd.DataFrame([
            self._plan_row("P", "", 0, 1, anchor),
            self._plan_row("O__S01", "P", 3, 33, anchor),
            self._plan_row("O__S02", "P", 3, 33, anchor),
            self._plan_row("N", "O__S01;O__S02", 57, 58, anchor),
        ])

        seg_map, tail_map, turnover_map = build_merge_map(merge_df)
        restored = restore_operations(
            plan_df, seg_map, tail_map, build_split_map(split_df),
        )
        source = pd.DataFrame([
            {"任务令": "T1", "WORK": f"WORK{work}", "零件数量": 10,
             "工艺排配时间": anchor}
            for work in (10, 11)
        ])
        restored = enrich(
            restored, build_source_lookup(source), turnover_map, set(source.columns),
        )
        by_id = {row["工序ID"]: row for _, row in restored.iterrows()}

        self.assertEqual(by_id["N"]["前工序ID"], "Y__S01;Y__S02")
        self.assertEqual(by_id["X__S01"]["零件数量"], 5)
        self.assertEqual(by_id["Y__S02"]["零件数量"], 5)
        self.assertEqual(by_id["N"]["预计齐套时间"], anchor + pd.Timedelta(hours=57))
        self.assertEqual(
            sum(by_id[op_id]["时长(小时)"] for op_id in ("X__S01", "Y__S01")),
            30,
        )

    @staticmethod
    def _plan_row(op_id, predecessor, start, end, anchor):
        return {
            "任务令": "T1", "工序ID": op_id, "工序": op_id,
            "前工序ID": predecessor,
            "开始(小时)": start, "结束(小时)": end,
            "计划开工时间": anchor + pd.Timedelta(hours=start),
            "计划完工时间": anchor + pd.Timedelta(hours=end),
            "时长(小时)": end - start, "占用时长(小时)": end - start,
        }


class AlgorithmMultiPredecessorIntegrationTests(unittest.TestCase):
    def test_semicolon_predecessors_are_imported_and_enforced_as_and(self):
        shifts = ";".join(f"{day}/0/24" for day in range(30))
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "instance.db")
            init_db(db_path)
            store = InstanceStore(db_path)
            store.save_from_csv(
                [{"order_id": "O1", "order_name": "O1", "release_time": 0,
                  "due_date": 100, "priority": 1}],
                [{"task_id": "T1", "order_id": "O1", "task_name": "T1",
                  "is_main": "Y", "predecessor_task_ids": "", "release_time": 0,
                  "due_date": 100}],
                [
                    self._db_op("P", "PRE", 1, "", "P1", 2),
                    self._db_op("O-1", "CNC", 10, "P", "C1,C2,C3", 2),
                    self._db_op("O-2", "CNC", 8, "P", "C1,C2,C3", 2),
                    self._db_op("O-3", "CNC", 7, "P", "C1,C2,C3", 2),
                    self._db_op("N", "NEXT", 1, "O-1;O-2;O-3", "N1", 0),
                ],
                [
                    {"type_id": "PRE", "type_name": "PRE", "is_critical": "N"},
                    {"type_id": "CNC", "type_name": "CNC", "is_critical": "Y"},
                    {"type_id": "NEXT", "type_name": "NEXT", "is_critical": "N"},
                ],
                [
                    {"type_id": "PRE", "machine_id": "P1", "machine_name": "P1", "shifts": shifts},
                    *[
                        {"type_id": "CNC", "machine_id": f"C{i}",
                         "machine_name": f"C{i}", "shifts": shifts}
                        for i in range(1, 4)
                    ],
                    {"type_id": "NEXT", "machine_id": "N1", "machine_name": "N1", "shifts": shifts},
                ],
                plan_start_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            shop = store.build_shopfloor()

            self.assertEqual(shop.operations["N"].predecessor_ops, ["O-1", "O-2", "O-3"])
            from llm4drd.optimization.exact import ExactSolver
            exact_result = ExactSolver(shop).solve()
            exact_by_id = {row["op_id"]: row for row in exact_result.schedule}
            exact_join_floor = max(
                exact_by_id[op_id]["end"] + 2 for op_id in ("O-1", "O-2", "O-3")
            )
            self.assertGreaterEqual(exact_by_id["N"]["start"], exact_join_floor)

            result = Simulator(shop, BUILTIN_RULES["EDD"]).run()
            by_id = {row["op_id"]: row for row in result.schedule}
            join_floor = max(by_id[op_id]["end"] + 2 for op_id in ("O-1", "O-2", "O-3"))
            self.assertTrue(result.feasible)
            self.assertGreaterEqual(by_id["N"]["start"], join_floor)
            self.assertEqual(len({by_id[op_id]["machine_id"] for op_id in ("O-1", "O-2", "O-3")}), 3)

    @staticmethod
    def _db_op(op_id, process_type, hours, predecessors, machines, turnover):
        return {
            "op_id": op_id, "task_id": "T1", "op_name": op_id,
            "process_type": process_type, "processing_time_hrs": hours,
            "turnover_time_hrs": turnover, "predecessor_ops": predecessors,
            "predecessor_tasks": "", "eligible_machine_ids": machines,
        }


if __name__ == "__main__":
    unittest.main()
