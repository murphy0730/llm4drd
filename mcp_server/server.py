from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if __package__:
    from .errors import PlanningAPIError
    from .planning_client import PlanningAPIClient
    from .schemas import TOOL_DEFINITIONS
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from llm4drd.mcp_server.errors import PlanningAPIError
    from llm4drd.mcp_server.planning_client import PlanningAPIClient
    from llm4drd.mcp_server.schemas import TOOL_DEFINITIONS


PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "llm4drd-planning", "version": "1"}


def handle_tool_call(name: str, arguments: dict, client: PlanningAPIClient) -> dict:
    try:
        validation = _validate_common(arguments)
        if validation is not None:
            return _tool_result(validation)
        if name == "list_planning_rules":
            payload = client.list_rules()
        elif name == "run_rule_planning":
            rule_name = _required_text(arguments, "rule_name").upper()
            payload = client.run_rule_planning(rule_name)
        elif name == "get_planning_overview":
            payload = client.get_overview(arguments.get("task_id"))
        elif name == "compare_planning_solutions":
            payload = client.compare_solutions(
                arguments.get("task_id"),
                arguments.get("solution_ids"),
                arguments.get("metric_keys"),
            )
        elif name == "search_planning_entities":
            payload = _search_entities(arguments, client)
        elif name == "get_order_planning":
            payload = _get_order_planning(arguments, client)
        elif name == "get_operation_planning":
            payload = _get_operation_planning(arguments, client)
        else:
            payload = _business_error("INVALID_ARGUMENT", f"未知 MCP 工具: {name}")
        return _tool_result(payload)
    except PlanningAPIError as error:
        return _tool_result(error.to_dict(), is_error=True)
    except (TypeError, ValueError) as error:
        return _tool_result(_business_error("INVALID_ARGUMENT", str(error)))


def _search_entities(arguments: dict, client: PlanningAPIClient) -> dict:
    entity_type = str(arguments.get("entity_type") or "")
    query = _required_text(arguments, "query")
    limit = _bounded_limit(arguments.get("limit", 20))
    if entity_type == "order":
        return client.search_orders(query, limit)
    if entity_type == "operation":
        return client.search_operations(query, arguments.get("order_id"), limit)
    return _business_error(
        "INVALID_ARGUMENT",
        "entity_type 必须是 order 或 operation",
    )


def _get_order_planning(arguments: dict, client: PlanningAPIClient) -> dict:
    query = _required_text(arguments, "order_query")
    search = client.search_orders(query, 10)
    if not search.get("ok"):
        return search
    matches = list(search.get("data") or [])
    match = _unique_match(matches, "order_id")
    if match is None:
        if not matches:
            return _business_error("ORDER_NOT_FOUND", f"未找到订单: {query}")
        return _business_error("AMBIGUOUS_ORDER", "匹配到多个订单", matches)
    return client.get_order_planning(
        str(match["order_id"]),
        arguments.get("task_id"),
        arguments.get("solution_ids"),
    )


def _get_operation_planning(arguments: dict, client: PlanningAPIClient) -> dict:
    query = _required_text(arguments, "operation_query")
    search = client.search_operations(query, limit=10)
    if not search.get("ok"):
        return search
    matches = list(search.get("data") or [])
    match = _unique_match(matches, "operation_id")
    if match is None:
        if not matches:
            return _business_error("OPERATION_NOT_FOUND", f"未找到工序: {query}")
        return _business_error("AMBIGUOUS_OPERATION", "匹配到多个工序", matches)
    return client.get_operation_planning(
        str(match["operation_id"]),
        arguments.get("task_id"),
        arguments.get("solution_ids"),
    )


def _validate_common(arguments: dict) -> dict | None:
    if not isinstance(arguments, dict):
        return _business_error("INVALID_ARGUMENT", "工具参数必须是 JSON 对象")
    solution_ids = arguments.get("solution_ids")
    if solution_ids is not None:
        if not isinstance(solution_ids, list) or not all(
            isinstance(item, str) and item for item in solution_ids
        ):
            return _business_error("INVALID_ARGUMENT", "solution_ids 必须是字符串数组")
        if len(dict.fromkeys(solution_ids)) > 4:
            return _business_error("INVALID_ARGUMENT", "单次最多比较 4 个方案")
    return None


def _required_text(arguments: dict, key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} 不能为空")
    return value.strip()


def _bounded_limit(value: object) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("limit 必须是整数") from error
    if not 1 <= result <= 50:
        raise ValueError("limit 必须在 1 到 50 之间")
    return result


def _unique_match(matches: list[dict], id_key: str) -> dict | None:
    exact = [item for item in matches if item.get("match_type") == "exact_id"]
    if len(exact) == 1:
        return exact[0]
    if len(matches) == 1 and id_key in matches[0]:
        return matches[0]
    return None


def _business_error(code: str, message: str, suggestions: list[dict] | None = None) -> dict:
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "suggestions": list(suggestions or []),
        },
    }


def _tool_result(payload: dict, *, is_error: bool = False) -> dict:
    summary = _summary(payload)
    return {
        "content": [{"type": "text", "text": summary}],
        "structuredContent": payload,
        "isError": is_error,
    }


def _summary(payload: dict) -> str:
    if not payload.get("ok"):
        error = payload.get("error") or {}
        return str(error.get("message") or "排产查询失败")[:1000]
    data = payload.get("data") or {}
    if isinstance(data, dict) and "rule_count" in data:
        return f"查询到 {data['rule_count']} 条内置排产规则"
    if isinstance(data, dict) and "rule_name" in data and "operation_count" in data:
        return f"已使用 {data['rule_name']} 完成排产，共排入 {data['operation_count']} 道工序"
    if isinstance(data, dict) and "candidate_count" in data:
        return f"查询到 {data['candidate_count']} 个候选排产方案"
    if isinstance(data, list):
        return f"查询到 {len(data)} 条结果"
    return "排产查询完成"


def _response(request_id: object, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error_response(request_id: object, code: int, message: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message[:1000]},
    }


def _send(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    base_url = os.environ.get("PLANNING_API_BASE_URL", "http://127.0.0.1:8888")
    timeout = os.environ.get("PLANNING_API_TIMEOUT_SECONDS", "10")
    client = PlanningAPIClient(base_url, float(timeout))
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(request, dict):
            continue
        method = request.get("method")
        request_id = request.get("id")
        if method == "notifications/initialized":
            continue
        try:
            if method == "initialize":
                requested_version = (
                    (request.get("params") or {}).get("protocolVersion")
                    or PROTOCOL_VERSION
                )
                result = {
                    "protocolVersion": (
                        requested_version
                        if requested_version == PROTOCOL_VERSION
                        else PROTOCOL_VERSION
                    ),
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO,
                }
                _send(_response(request_id, result))
            elif method == "tools/list":
                _send(_response(request_id, {"tools": TOOL_DEFINITIONS}))
            elif method == "tools/call":
                params = request.get("params") or {}
                result = handle_tool_call(
                    str(params.get("name") or ""),
                    params.get("arguments") or {},
                    client,
                )
                _send(_response(request_id, result))
            elif request_id is not None:
                _send(_error_response(request_id, -32601, f"unknown method {method}"))
        except Exception as error:  # protocol boundary: always answer the request
            if request_id is not None:
                _send(_error_response(request_id, -32603, f"{type(error).__name__}: {error}"))


if __name__ == "__main__":
    main()


__all__ = ["TOOL_DEFINITIONS", "handle_tool_call", "main"]
