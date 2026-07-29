from __future__ import annotations


_TASK_ID = {"type": "string", "description": "优化任务 ID；省略时使用最近完成任务"}
_SOLUTION_IDS = {
    "type": "array",
    "items": {"type": "string"},
    "maxItems": 4,
    "uniqueItems": True,
    "description": "要比较的方案 ID；省略时按评审页顺序取前 4 个（含基线方案）",
}
_BOTTLENECK_LIMIT = {
    "type": "integer",
    "minimum": 1,
    "maximum": 50,
    "description": (
        "返回几台瓶颈机器（按利用率降序）；省略时取前 20 台。"
        "返回的 bottleneck_machines 每项含机器名、所属类型、是否关键类型和利用率。"
    ),
}
_BUILTIN_RULE_NAMES = [
    "EDD", "SPT", "LPT", "CR", "ATC", "FIFO", "MST", "PRIORITY",
    "KIT_AWARE", "BOTTLENECK", "COMPOSITE",
]
_WHATIF_ENTITIES = [
    "order", "task", "operation", "machine", "machine_type",
    "tooling", "tooling_type", "personnel", "downtime", "planning_context",
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
            "查询排产概览与全部可选方案：基线方案、优化求解出的候选方案、以及参考方案"
            "（精确求解 / 启发式规则），并给出各方案的总延误和完工跨度。"
            "每个方案的 category 标明它属于哪一类；返回的 solution_name 与方案评审界面一致，"
            "solution_id 仅作为后续查询标识。"
            "适用于：现在有哪些方案、有多少候选方案、各方案总延误分别是多少。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"task_id": _TASK_ID},
            "additionalProperties": False,
        },
    },
    {
        "name": "compare_planning_solutions",
        "description": (
            "比较一个或多个排产方案的指定指标，最多比较 4 个方案。"
            "省略 solution_ids 时按评审页顺序取前 4 个（基线方案在最前，作为对照）。"
            "面向用户展示时使用返回的 solution_name，不使用内部 solution_id 作为方案名。"
            "每个方案同时返回 bottleneck_machines（按利用率降序的瓶颈机器），"
            "回答「哪台机器是瓶颈」用它，不要用 metric_keys。"
        ),
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
                "bottleneck_limit": _BOTTLENECK_LIMIT,
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "search_planning_entities",
        "description": (
            "按编号、名称、任务或订单搜索排产中的订单、工序，或车间资源（机器 / 机器类型 / 工装 / 人员）。"
            "资源类型会返回 type_id 与现有班次串，是写 what-if 改动前必须先拿到的现状信息——"
            "不要凭空编造机器编号、机器类型或班次格式。"
            "查资源时 query 可留空表示全量列出。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity_type": {
                    "type": "string",
                    "enum": ["order", "operation", "machine", "machine_type", "tooling", "personnel"],
                },
                "query": {"type": "string"},
                "order_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
            },
            "required": ["entity_type"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_order_planning",
        "description": (
            "查询指定订单在一个或多个候选排产方案中的工序安排、完工时间和订单延误。"
            "面向用户展示时使用返回的 solution_name。"
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
            "面向用户展示时使用返回的 solution_name。"
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
    # === What-if 场景推演：改内存里的原始数据 → 跑规则 → 比 KPI ===
    {
        "name": "create_whatif_scenario",
        "description": (
            "新建一个 what-if 推演场景，返回 scenario_id。场景会快照当前实例版本，"
            "后续所有改动都只活在服务进程内存里，不会写数据库、不影响正式排产数据。"
            "场景在服务重启后消失，且最多同时保留 8 个（超出后最旧的被淘汰）。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "maxLength": 80,
                    "description": "场景名，用业务语言描述这次假设，如「车床加一台」「全员只上白班」",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "apply_whatif_patch",
        "description": (
            "向场景追加对原始数据的改动，这是唯一的改数据入口。改动立即校验，不合法会整批拒绝。"
            "\n每条 patch 形如 {op, entity, values} —— op 取 add/update/remove；"
            "entity 取 order/task/operation/machine/machine_type/tooling/tooling_type/personnel/downtime/planning_context；"
            "values 的字段名与实例导入模板的列名一致。"
            "\nupdate 与 remove 必须提供 id、ids 或 where 三者之一作为选择器（只能给一个）："
            "id 定位单个实体，ids 定位一组，where 按字段等值批量匹配（如 {\"type_id\":\"turning\"} 命中全部车床）。"
            "\n常见写法：加机器 {op:add, entity:machine, values:{machine_id, machine_name, type_id, shifts}}；"
            "删机器 {op:remove, entity:machine, id}；改班次 {op:update, entity:machine, where:{type_id}, values:{shifts}}；"
            "改工时 {op:update, entity:operation, id, values:{processing_time}}；"
            "改交期 {op:update, entity:order, id, values:{due_date}}；"
            "加停机 {op:add, entity:downtime, values:{machine_id, downtime_type, start_time, end_time}}。"
            "\nshifts 是 \"day/start_hour/hours;...\" 串（如 \"0/8/10;0/20/8\" 表示白班+夜班）；"
            "时间类字段一律是相对计划基准时刻的偏移小时数。"
            "\n调用前应先用 search_planning_entities 查清现有实体的编号与格式。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "scenario_id": {"type": "string", "minLength": 1},
                "patches": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 200,
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "enum": ["add", "update", "remove"]},
                            "entity": {"type": "string", "enum": _WHATIF_ENTITIES},
                            "id": {"type": "string"},
                            "ids": {"type": "array", "items": {"type": "string"}},
                            "where": {"type": "object"},
                            "values": {"type": "object"},
                        },
                        "required": ["op", "entity"],
                    },
                },
            },
            "required": ["scenario_id", "patches"],
            "additionalProperties": False,
        },
    },
    {
        "name": "describe_whatif_scenario",
        "description": (
            "查看场景当前状态：改了哪些东西（人话清单）、各类实体的数量变化、改动后的实例校验结果，"
            "以及落库所需的 apply_token。省略 scenario_id 则列出全部场景。"
            "\n跑规则之前应该先调它，把改动复述给用户确认；"
            "若 validation.errors 非空说明改坏了（例如某工序失去全部可用机器），要先报告用户而不是硬跑。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"scenario_id": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "revert_whatif_patch",
        "description": "撤销场景中最近 N 条改动（默认 1 条），用于「刚才那步不算」这类修正。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scenario_id": {"type": "string", "minLength": 1},
                "count": {"type": "integer", "minimum": 1, "maximum": 200, "default": 1},
            },
            "required": ["scenario_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_whatif_planning",
        "description": (
            "在指定场景的数据上跑一条或多条内置排产规则，返回每条规则的 KPI。"
            "结果只存在内存里，不会覆盖系统的最近一次仿真结果。"
            "\ninclude_baseline 默认开启：同一次调用里会把**未改动的现状**也用同样的规则跑一遍，"
            "results 里每条带 variant（scenario / baseline）。所以问「改了之后比现在怎么样」"
            "**只需要建一个场景、打完 patch、调这一次**，不必再建基线场景重复跑。"
            "\n返回里 status 为 done 时 results 即为结果；为 running 说明实例较大还在跑，"
            "请拿 run_id 调 get_whatif_run 轮询；为 failed 时看 error（校验不通过会在这里拦住）。"
            "\n把这次的 run_id 交给 compare_whatif_runs 即可得到带方向判定的对比表。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "scenario_id": {"type": "string", "minLength": 1},
                "rule_names": {
                    "type": "array",
                    "items": {"type": "string", "enum": _BUILTIN_RULE_NAMES},
                    "minItems": 1,
                    "maxItems": 11,
                    "uniqueItems": True,
                    "description": "要跑的内置规则；省略时用 ATC",
                },
                "include_baseline": {
                    "type": "boolean",
                    "default": True,
                    "description": "是否同时跑未改动的现状作为对照；只有在场景本身就是现状时才设 false",
                },
            },
            "required": ["scenario_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_whatif_run",
        "description": (
            "查询一次 what-if 推演的状态与结果，用于 run_whatif_planning 返回 running 时轮询。"
            "每条结果带 bottleneck_machines（按利用率降序的瓶颈机器，含改动后新增的机器）。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "minLength": 1},
                "bottleneck_limit": _BOTTLENECK_LIMIT,
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compare_whatif_runs",
        "description": (
            "对比多次 what-if 推演的 KPI，可跨场景、跨规则。第一个 run 作为基准，"
            "其余给出绝对差、百分比差，以及 better 字段（已按该指标是越小越好还是越大越好判定）。"
            "面向用户叙述时请用 better 而不是自己猜方向。"
            "每条结果还带 bottleneck_machines，用它看改动后瓶颈转移到了哪台机器。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 8,
                    "uniqueItems": True,
                },
                "metric_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                    "description": "要比的指标；省略时用 总延迟 / Makespan / 主订单延误时长",
                },
                "bottleneck_limit": _BOTTLENECK_LIMIT,
            },
            "required": ["run_ids"],
            "additionalProperties": False,
        },
    },
    {
        "name": "apply_whatif_to_instance",
        "description": (
            "【高危·有破坏性】把场景的改动真正写入正式实例数据，覆盖数据库里的原始数据。"
            "\n必须先调 describe_whatif_scenario 拿到 apply_token，把完整改动清单展示给用户，"
            "并取得用户明确、直接的确认后才能调用。绝不允许在用户没有明确要求落库时主动调用它。"
            "\n副作用：实例版本号递增，校验 / 仿真 / 优化 / 方案评审四步流程的历史快照全部作废，需要重跑。"
            "调用前会自动做一次整库备份，返回值里给出备份路径。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "scenario_id": {"type": "string", "minLength": 1},
                "confirm_token": {
                    "type": "string",
                    "minLength": 1,
                    "description": "describe_whatif_scenario 返回的 apply_token，改动变化后会失效",
                },
            },
            "required": ["scenario_id", "confirm_token"],
            "additionalProperties": False,
        },
    },
]
