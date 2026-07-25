import unittest

from fastapi import HTTPException

from llm4drd.api import server
from llm4drd.optimization.solution_model import schedule_payload_fields


def _entries(count: int) -> list[dict]:
    return [{"op_id": f"OP-{index}", "start": float(index), "end": float(index) + 1.0} for index in range(count)]


class SchedulePayloadFieldsTest(unittest.TestCase):
    def test_marks_truncation_and_keeps_real_total(self):
        payload = schedule_payload_fields(_entries(300), 120)
        self.assertTrue(payload["schedule_truncated"])
        self.assertEqual(payload["schedule_total"], 300)
        self.assertEqual(len(payload["schedule"]), 120)

    def test_no_limit_is_complete(self):
        payload = schedule_payload_fields(_entries(300), None)
        self.assertFalse(payload["schedule_truncated"])
        self.assertEqual(payload["schedule_total"], 300)
        self.assertEqual(len(payload["schedule"]), 300)

    def test_under_limit_is_complete(self):
        payload = schedule_payload_fields(_entries(10), 120)
        self.assertFalse(payload["schedule_truncated"])
        self.assertEqual(payload["schedule_total"], 10)


class SerializedScheduleFieldsTest(unittest.TestCase):
    def test_matches_solution_model_semantics(self):
        truncated = server._serialized_schedule_fields(_entries(120), 300, 120)
        self.assertTrue(truncated["schedule_truncated"])
        self.assertEqual(truncated["schedule_total"], 300)

        complete = server._serialized_schedule_fields(_entries(300), 300, None)
        self.assertFalse(complete["schedule_truncated"])


class RejectTruncatedScheduleTest(unittest.TestCase):
    def test_rejects_truncated_solution(self):
        solution = {"solution_id": "S-1", **schedule_payload_fields(_entries(300), 120)}
        with self.assertRaises(HTTPException) as ctx:
            server._reject_truncated_schedule(solution)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("300", ctx.exception.detail)

    def test_passes_complete_solution(self):
        solution = {"solution_id": "S-1", **schedule_payload_fields(_entries(300), None)}
        self.assertIs(server._reject_truncated_schedule(solution), solution)

    def test_export_falls_back_to_truncated_result_and_is_rejected(self):
        """没有 export_result 时 _resolve_export_solution 会回落到截断的 result，
        导出端点必须拒绝，而不是默默产出 120 条的残缺文件。"""
        truncated = {"solution_id": "S-1", **schedule_payload_fields(_entries(300), 120)}
        task = {"result": {"solutions": [truncated]}}
        resolved = server._resolve_export_solution(None, task, "S-1")
        self.assertTrue(resolved["schedule_truncated"])
        with self.assertRaises(HTTPException) as ctx:
            server._reject_truncated_schedule(resolved)
        self.assertEqual(ctx.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
