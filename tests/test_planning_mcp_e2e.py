from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _PlanningHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/query/planning/overview"):
            body = json.dumps({
                "ok": True,
                "data": {
                    "task_id": "task-e2e",
                    "candidate_count": 2,
                    "solutions": [{
                        "solution_id": "S-6cf6190f25",
                        "solution_name": "方案二",
                    }],
                },
                "meta": {"time_unit": "hour"},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):
        if self.path == "/api/command/planning/run-rule":
            size = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(size))
            body = json.dumps({
                "ok": True,
                "data": {
                    "rule_name": request["rule_name"],
                    "metrics": {"total_tardiness": 3.5},
                    "operation_count": 4,
                    "planning_preview": [],
                    "planning_truncated": False,
                },
                "meta": {"time_unit": "hour"},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def log_message(self, _format, *_args):
        return


class PlanningMCPEndToEndTests(unittest.TestCase):
    def test_stdio_handshake_discovery_and_http_tool_call(self) -> None:
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _PlanningHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)

        requests = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"},
            },
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "get_planning_overview", "arguments": {}},
            },
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "run_rule_planning",
                    "arguments": {"rule_name": "atc"},
                },
            },
        ]
        project_parent = Path(__file__).resolve().parents[2]
        env = {
            **os.environ,
            "PLANNING_API_BASE_URL": f"http://127.0.0.1:{httpd.server_port}",
        }
        completed = subprocess.run(
            [
                sys.executable,
                str(project_parent / "llm4drd" / "mcp_server" / "server.py"),
            ],
            input="".join(json.dumps(item) + "\n" for item in requests),
            text=True,
            capture_output=True,
            cwd=project_parent,
            env=env,
            timeout=10,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        responses = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual([item["id"] for item in responses], [1, 2, 3, 4])
        self.assertEqual(responses[0]["result"]["protocolVersion"], "2024-11-05")
        published = {item["name"] for item in responses[1]["result"]["tools"]}
        self.assertEqual(len(published), len(responses[1]["result"]["tools"]))
        self.assertIn("run_rule_planning", published)
        self.assertIn("run_whatif_planning", published)
        tool_result = responses[2]["result"]
        self.assertFalse(tool_result["isError"])
        self.assertEqual(tool_result["structuredContent"]["data"]["candidate_count"], 2)
        self.assertEqual(
            tool_result["structuredContent"]["data"]["solutions"][0]["solution_name"],
            "方案二",
        )
        run_result = responses[3]["result"]
        self.assertFalse(run_result["isError"])
        self.assertEqual(run_result["structuredContent"]["data"]["rule_name"], "ATC")


if __name__ == "__main__":
    unittest.main()
