from __future__ import annotations


_TASK_ID = {"type": "string", "description": "优化任务 ID；省略时使用最近完成任务"}
_SOLUTION_IDS = {
    "type": "array",
    "items": {"type": "string"},
    "maxItems": 4,
    "uniqueItems": True,
    "description": "要比较的方案 ID；省略时使用候选方案，最多 4 个",
}
_BUILTIN_RULE_NAMES = [
    "EDD", "SPT", "LPT", "CR", "ATC", "FIFO", "MST", "PRIORITY",
    "KIT_AWARE", "BOTTLENECK", "COMPOSITE",
]


TOOL_DEFINITIONS = [
    {
        "name": "list_planning_rules",
        "description": "只读查询系统支持的内置排产规则、规则说明及默认规则。",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "run_rule_planning",
        "description": (
            "使用指定内置规则执行一次排产，并更新系统最近一次仿真结果。"
            "这是有副作用的执行操作，调用前应获得用户确认。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "rule_name": {
                    "type": "string",
                    "enum": _BUILTIN_RULE_NAMES,
                    "description": "要使用的内置排产规则名称",
                },
            },
            "required": ["rule_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_planning_overview",
        "description": (
            "查询排产概览、候选方案数量、Pareto Archive 规模及各方案的总延误和完工跨度。"
            "适用于：有多少候选方案、各方案总延误分别是多少。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"task_id": _TASK_ID},
            "additionalProperties": False,
        },
    },
    {
        "name": "compare_planning_solutions",
        "description": "比较一个或多个候选排产方案的指定指标，最多比较 4 个方案。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": _TASK_ID,
                "solution_ids": _SOLUTION_IDS,
                "metric_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "search_planning_entities",
        "description": "按编号、名称、任务或订单搜索排产中的订单或工序。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity_type": {"type": "string", "enum": ["order", "operation"]},
                "query": {"type": "string", "minLength": 1},
                "order_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
            },
            "required": ["entity_type", "query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_order_planning",
        "description": (
            "查询指定订单在一个或多个候选排产方案中的工序安排、完工时间和订单延误。"
            "适用于：某订单怎么排、何时完工、不同方案有什么差异。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "order_query": {"type": "string", "minLength": 1},
                "task_id": _TASK_ID,
                "solution_ids": _SOLUTION_IDS,
            },
            "required": ["order_query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_operation_planning",
        "description": (
            "查询指定工序在一个或多个候选排产方案中的计划开始、结束时间和资源。"
            "适用于：某工序排在什么时候、在哪台机器加工。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "operation_query": {"type": "string", "minLength": 1},
                "task_id": _TASK_ID,
                "solution_ids": _SOLUTION_IDS,
            },
            "required": ["operation_query"],
            "additionalProperties": False,
        },
    },
]
