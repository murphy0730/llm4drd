import unittest

from fastapi.testclient import TestClient
from fastapi import HTTPException

from llm4drd.api import server
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class InsertionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(server.app)

    def test_template_can_be_downloaded_and_imported(self):
        template = self.client.get("/api/insertion/template")
        self.assertEqual(template.status_code, 200, template.text)
        self.assertIn("spreadsheetml", template.headers.get("content-type", ""))

        parsed = self.client.post(
            "/api/insertion/import",
            files={
                "file": (
                    "insertion_order_template.xlsx",
                    template.content,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
        self.assertEqual(parsed.status_code, 200, parsed.text)
        payload = parsed.json()
        self.assertGreaterEqual(len(payload["orders"]), 1)
        self.assertGreaterEqual(len(payload["operations"]), 1)

    def test_import_rejects_unknown_file_type(self):
        response = self.client.post(
            "/api/insertion/import",
            files={"file": ("orders.txt", b"order_id\nA", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("仅支持", response.json()["detail"])

    def test_stale_or_cross_instance_base_schedule_is_rejected(self):
        shop = make_graph_context_shop()
        with self.assertRaisesRegex(HTTPException, "与当前实例不一致"):
            server._validate_insertion_base_coverage(
                shop,
                [{"op_id": "OP-11"}, {"op_id": "OTHER-INSTANCE-OP"}],
            )

    def test_strategy_base_validation(self):
        shop = make_graph_context_shop()
        with self.assertRaisesRegex(HTTPException, "请选择一个已发布方案"):
            server._insertion_base_schedule(
                server.InsertionEvaluateReq(base_source="strategy", orders=[], operations=[]),
                shop,
            )
        with self.assertRaisesRegex(HTTPException, "不存在或已下线"):
            server._insertion_base_schedule(
                server.InsertionEvaluateReq(base_source="strategy", strategy_id="DSP-NOPE", orders=[], operations=[]),
                shop,
            )
        with self.assertRaisesRegex(HTTPException, "base_source 仅支持"):
            server._insertion_base_schedule(
                server.InsertionEvaluateReq(base_source="bogus", orders=[], operations=[]),
                shop,
            )


if __name__ == "__main__":
    unittest.main()
