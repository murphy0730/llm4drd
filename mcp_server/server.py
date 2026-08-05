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
                _optional_machine_limit(arguments),
            )
        elif name == "search_planning_entities":
            payload = _search_entities(arguments, client)
        elif name == "get_order_planning":
            payload = _get_order_planning(arguments, client)
        elif name == "get_operation_planning":
            payload = _get_operation_planning(arguments, client)
        elif name == "diagnose_bottleneck":
            # solution_id 故意不在本端做必填校验：漏传时后端会连可选方案一起回给
            # Agent，比本端干巴巴一句"不能为空"更容易自我修正。
            payload = client.diagnose_bottleneck(
                _optional_text(arguments, "solution_id"),
                arguments.get("task_id"),
                _optional_machine_limit(arguments),
            )
        elif name == "explain_order_delay":
            payload = client.explain_order_delay(
                _required_text(arguments, "order_id"),
                _optional_text(arguments, "solution_id"),
                arguments.get("task_id"),
            )
        elif name == "create_whatif_scenario":
            payload = client.create_whatif_scenario(arguments.get("name"))
        elif name == "apply_whatif_patch":
            payload = client.apply_whatif_patch(
                _required_text(arguments, "scenario_id"),
                _required_patches(arguments),
            )
        elif name == "describe_whatif_scenario":
            scenario_id = arguments.get("scenario_id")
            payload = (
                client.describe_whatif_scenario(str(scenario_id).strip())
                if isinstance(scenario_id, str) and scenario_id.strip()
                else client.list_whatif_scenarios()
            )
        elif name == "revert_whatif_patch":
            payload = client.revert_whatif_patch(
                _required_text(arguments, "scenario_id"),
                _bounded_count(arguments.get("count", 1)),
            )
        elif name == "run_whatif_planning":
            payload = client.run_whatif_planning(
                _required_text(arguments, "scenario_id"),
                _rule_names(arguments.get("rule_names")),
                bool(arguments.get("include_baseline", True)),
            )
        elif name == "get_whatif_run":
            payload = client.get_whatif_run(
                _required_text(arguments, "run_id"),
                _optional_machine_limit(arguments),
            )
        elif name == "compare_whatif_runs":
            payload = client.compare_whatif_runs(
                _required_id_list(arguments, "run_ids"),
                arguments.get("metric_keys"),
                _optional_machine_limit(arguments),
            )
        elif name == "apply_whatif_to_instance":
            payload = client.apply_whatif_to_instance(
                _required_text(arguments, "scenario_id"),
                _required_text(arguments, "confirm_token"),
            )
        else:
            payload = _business_error("INVALID_ARGUMENT", f"未知 MCP 工具: {name}")
        return _tool_result(payload)
    except PlanningAPIError as error:
        return _tool_result(error.to_dict(), is_error=True)
    except (TypeError, ValueError) as error:
        return _tool_result(_business_error("INVALID_ARGUMENT", str(error)))


RESOURCE_ENTITY_TYPES = ("machine", "machine_type", "tooling", "personnel")


def _search_entities(arguments: dict, client: PlanningAPIClient) -> dict:
    entity_type = str(arguments.get("entity_type") or "")
    limit = _bounded_limit(arguments.get("limit", 20))
    # 资源检索允许空 query（列出全部），订单/工序检索仍要求关键字。
    if entity_type in RESOURCE_ENTITY_TYPES:
        query = str(arguments.get("query") or "").strip()
        scenario_id = str(arguments.get("scenario_id") or "").strip() or None
        return client.search_resources(entity_type, query, limit, scenario_id)
    if arguments.get("scenario_id"):
        return _business_error(
            "INVALID_ARGUMENT",
            "scenario_id 只对资源类型（machine / machine_type / tooling / personnel）有效",
        )
    query = _required_text(arguments, "query")
    if entity_type == "order":
        return client.search_orders(query, limit)
    if entity_type == "operation":
        return client.search_operations(query, arguments.get("order_id"), limit)
    return _business_error(
        "INVALID_ARGUMENT",
        "entity_type 必须是 order、operation、machine、machine_type、tooling 或 personnel",
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


def _optional_text(arguments: dict, key: str) -> str | None:
    value = arguments.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _bounded_limit(value: object) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("limit 必须是整数") from error
    if not 1 <= result <= 50:
        raise ValueError("limit 必须在 1 到 50 之间")
    return result


def _optional_machine_limit(arguments: dict) -> int | None:
    """省略时返回 None，让后端用它自己的默认值（20），别在两处各写一份默认。"""
    value = arguments.get("machine_limit")
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("machine_limit 必须是整数") from error
    if not 1 <= result <= 50:
        raise ValueError("machine_limit 必须在 1 到 50 之间")
    return result


def _bounded_count(value: object) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("count 必须是整数") from error
    if not 1 <= result <= 200:
        raise ValueError("count 必须在 1 到 200 之间")
    return result


def _required_patches(arguments: dict) -> list[dict]:
    patches = arguments.get("patches")
    if not isinstance(patches, list) or not patches:
        raise ValueError("patches 必须是非空数组")
    if len(patches) > 200:
        raise ValueError("单次最多提交 200 条改动")
    for index, patch in enumerate(patches):
        if not isinstance(patch, dict):
            raise ValueError(f"第 {index + 1} 条 patch 必须是对象")
    return patches


def _required_id_list(arguments: dict, key: str) -> list[str]:
    values = arguments.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{key} 必须是非空数组")
    items = [str(item).strip() for item in values if str(item).strip()]
    if not items:
        raise ValueError(f"{key} 必须是非空数组")
    return list(dict.fromkeys(items))


def _rule_names(values: object) -> list[str] | None:
    if values is None:
        return None
    if not isinstance(values, list):
        raise ValueError("rule_names 必须是字符串数组")
    names = [str(item).strip().upper() for item in values if str(item).strip()]
    return list(dict.fromkeys(names)) or None


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
        # 总数放在最前：分项计数容易被当成总数，Case「只报了候选方案」就是这么来的。
        total = data.get("solution_count", data["candidate_count"])
        names = "、".join(
            f"{item.get('solution_name')}（{item.get('category_label')}）"
            for item in (data.get("solutions") or [])
            if isinstance(item, dict)
        )
        head = (
            f"当前共有 {total} 个排产方案，逐个是：{names}"
            if names else f"当前共有 {total} 个排产方案"
        )
        return (
            f"{head}。按来源拆分：基线 {data.get('baseline_count', 0)} 个、"
            f"优化候选 {data['candidate_count']} 个、参考 {data.get('reference_count', 0)} 个"
        )[:1000]
    if isinstance(data, dict) and "metric_keys" in data and "solutions" in data:
        rows = data.get("solutions") or []
        parts = [f"已对比 {len(rows)} 个方案"]
        top = _top_utilization(rows[0] if rows else None)
        if top:
            parts.append(f"首个方案{top}")
        if data.get("ranking_source") == "legacy_top5":
            parts.append("（该任务是旧数据，负荷榜只有 top5 且无利用率，需重跑优化才完整）")
        return "，".join(parts)[:1000]
    summary = _attribution_summary(data) or _whatif_summary(data)
    if summary is not None:
        return summary[:1000]
    if isinstance(data, list):
        return f"查询到 {len(data)} 条结果"
    return "排产查询完成"


WAIT_CAUSE_LABELS = {
    "capacity_bound": "产能不足（同类机器同期都占满，加机台有用）",
    "dispatch_bound": "选机/派工问题（同期有可用机器空着，加机台没用）",
    "off_shift": "该时段没排班（工时不足）",
    "downtime": "设备停机",
    "idle": "机器空着但没排上（工装/人员/前置资源）",
}


def _coverage_warning(data: dict) -> str | None:
    """未排入的订单必须先说——它比"延误"更严重，却最容易被当成没问题。"""
    unscheduled = int(data.get("unscheduled_order_count") or 0)
    partial = int(data.get("partially_scheduled_order_count") or 0)
    if not unscheduled and not partial:
        return None
    parts = []
    if unscheduled:
        parts.append(f"{unscheduled} 个订单完全未排入该方案")
    if partial:
        parts.append(f"{partial} 个订单只排了一部分")
    return (
        "⚠ " + "、".join(parts)
        + "，它们的延误未知且不在下面的归因里，比已知延误更需要先处理"
    )


def _attribution_summary(data: object) -> str | None:
    """延误归因的一句话结论——直接点名主因，别让 Agent 自己从五个数里挑。"""
    if not isinstance(data, dict):
        return None
    if "inevitable_tardiness_hours" in data:
        return _order_delay_summary(data)
    if "wait_breakdown" not in data:
        return None
    breakdown = data.get("wait_breakdown") or {}
    total = sum(value for value in breakdown.values() if isinstance(value, (int, float)))
    parts = [item for item in (_coverage_warning(data),) if item]
    if total <= 0:
        parts.append(
            f"方案「{data.get('solution_name')}」下没有可归因的等待"
            f"（延误订单 {data.get('tardy_order_count', 0)} 个）"
        )
        return "；".join(parts)
    cause, hours = max(breakdown.items(), key=lambda item: item[1])
    parts += [
        f"方案「{data.get('solution_name')}」有 {data.get('tardy_order_count')} 个订单延误，"
        f"合计 {data.get('total_order_tardiness_hours')} 小时",
        f"等待主因是{WAIT_CAUSE_LABELS.get(cause, cause)}（{hours}h / 共 {round(total, 1)}h）",
    ]
    machines = data.get("machines") or []
    if machines and isinstance(machines[0], dict):
        top = machines[0]
        name = top.get("machine_name") or top.get("machine_id")
        capacity = top.get("capacity_wait_hours") or 0.0
        parts.append(
            f"产能上卡得最狠的是 {name}（致等 {capacity}h，饱和度 {top.get('saturation')}）"
            if capacity > 0
            else "没有机器因产能不足卡住延误订单——加机台解决不了这次延误"
        )
    return "；".join(parts)


def _order_delay_summary(data: dict) -> str:
    name = data.get("order_name")
    if not data.get("planned", True):
        scheduled = data.get("scheduled_operation_count", 0)
        total = data.get("total_operation_count", 0)
        return (
            f"⚠ 订单「{name}」未完整排入该方案（{scheduled}/{total} 道工序），"
            "延误无法计算——这比已知延误更严重，应先查为什么排不进去"
        )
    inevitable = data.get("inevitable_tardiness_hours") or 0.0
    tardiness = data.get("tardiness_hours") or 0.0
    if tardiness <= 0:
        return f"订单「{name}」未延误（完工 {data.get('completion_at')}）"
    parts = [f"订单「{name}」延误 {tardiness} 小时"]
    if inevitable > 0:
        parts.append(
            f"其中 {inevitable}h 是工艺链本身就来不及（即使资源无限也躲不掉），"
            "加机器加班次无效，只能改交期或改工艺"
        )
    attribution = data.get("attribution") or {}
    if any(attribution.values()):
        cause, hours = max(attribution.items(), key=lambda item: item[1])
        parts.append(f"资源侧等待主因是{WAIT_CAUSE_LABELS.get(cause, cause)}（{hours}h）")
    return "；".join(parts)


def _top_utilization(entry: object) -> str | None:
    """把一行结果里负荷最高的机器压成一句话，省得 Agent 展开 structuredContent。

    刻意说"负荷最高"而不是"瓶颈"：这只是全周期利用率排名，瓶颈判定见 diagnose_bottleneck。
    """
    if not isinstance(entry, dict):
        return None
    machines = entry.get("machine_utilization_ranking") or []
    if not machines or not isinstance(machines[0], dict):
        return None
    top = machines[0]
    name = top.get("machine_name") or top.get("machine_id") or "未知机器"
    utilization = top.get("full_horizon_utilization")
    if isinstance(utilization, (int, float)):
        return f"全周期负荷最高的是 {name}（{utilization * 100:.0f}%，非瓶颈判定）"
    return f"全周期负荷最高的是 {name}（非瓶颈判定）"


def _whatif_summary(data: object) -> str | None:
    """what-if 各 payload 形状的中文摘要（结构化全文仍在 structuredContent 里）。"""
    if not isinstance(data, dict):
        return None
    if data.get("applied") is True and "instance_version" in data:
        return (
            f"改动已写入正式实例，实例版本号变为 {data['instance_version']}；"
            "四步流程历史快照已作废，需要重跑。备份：" + str(data.get("backup_path") or "无")
        )
    if "status" in data and "rule_names" in data:
        status = data.get("status")
        if status == "done":
            results = data.get("results") or [{}]
            best = results[0]
            metrics = best.get("metrics") or {}
            note = "（已含现状对照）" if data.get("include_baseline") else ""
            top = _top_utilization(best)
            return (
                f"场景「{data.get('scenario_name')}」跑出 {len(results)} 组结果{note}，"
                f"最优：{best.get('label') or data.get('scenario_name')} + {best.get('rule_name')}"
                f"，总延迟 {metrics.get('total_tardiness')}、Makespan {metrics.get('makespan')}"
                + (f"；{top}" if top else "")
            )
        if status == "failed":
            return f"推演失败：{(data.get('error') or {}).get('message') or '未知原因'}"
        return f"推演进行中（run_id={data.get('run_id')}），请用 get_whatif_run 轮询"
    if "entries" in data and "baseline" in data:
        baseline = data.get("baseline") or {}
        entries = data.get("entries") or []
        top = _top_utilization(entries[0] if entries else None)
        return (
            f"已对比 {len(entries)} 组结果，"
            f"基准为「{baseline.get('scenario_name')}」+ {baseline.get('rule_name')}"
            + (f"；首条结果{top}" if top else "")
        )
    if "scenario_count" in data:
        return f"当前有 {data['scenario_count']} 个 what-if 场景"
    if "applied" in data and "scale" in data:
        effects = data.get("applied") or []
        return "；".join(str(item.get("message")) for item in effects) or "改动已记录"
    if "scenario_id" in data and "patch_count" in data:
        validation = data.get("validation") or {}
        errors = len(validation.get("errors") or [])
        warnings = len(validation.get("warnings") or [])
        parts = [f"场景「{data.get('name')}」当前有 {data['patch_count']} 处改动"]
        if data.get("scale_delta"):
            parts.append("规模变化 " + str(data["scale_delta"]))
        if errors:
            parts.append(f"⚠ 校验有 {errors} 个错误级问题，不能直接跑")
        elif warnings:
            parts.append(f"校验有 {warnings} 个提醒")
        return "，".join(parts)
    if "entity_type" in data and "items" in data:
        return f"查询到 {data.get('total')} 个{data['entity_type']}"
    if "deleted" in data:
        return "场景已删除" if data["deleted"] else "场景不存在或已被淘汰"
    return None


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
