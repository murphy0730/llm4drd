# 排产项目 MCP Server 实现方案

## 1. 背景与目标

本方案面向 `llm4drd` 排产项目。项目负责持有车间实例、执行优化、保存候选方案及全量排产明细，并通过本地 stdio MCP Server 向 Manufacturing Agent 提供排产查询与按规则排产能力。

术语约定：

- `planning` 表示排产，包括候选方案、订单排产、工序排产和排产指标。
- `dispatching` 表示调度，例如现场派工、动态事件响应和在线重调度。
- 本次新增的类、文件、REST 路径、MCP 工具及测试名称统一使用 `planning`。
- 现有历史代码和持久化字段不在本次范围内做全局重命名，避免破坏兼容性；新查询层负责把既有数据映射为 `planning` 语义契约。

本期目标：

1. 查询最近一次或指定优化任务的候选排产方案。
2. 查询候选数量、Pareto Archive 规模及各方案关键指标。
3. 查询指定订单在一个或多个方案中的排产结果。
4. 查询指定工序在一个或多个方案中的计划时间和资源。
5. 通过 stdio MCP 向同机部署的 Manufacturing Agent 暴露以上能力。
6. 返回有边界、可解释、可测试的结构化数据，避免把完整大结果直接塞入模型上下文。
7. 查询系统内置排产规则，并按用户指定的内置规则执行一次排产。

本期不包含：

- 通过 MCP 发起多目标优化、人工调整排产、修改订单或写入评审结果。
- 自定义规则代码上传与执行；本期执行入口只接受系统内置规则白名单。
- 跨机器 HTTP MCP 传输。
- 历史上所有优化任务的长期归档查询；首版保证当前进程内任务和数据库中最近一次有效快照可查询。

## 2. 现有能力与缺口

现有后端已经具备：

- `workflow_progress` 持久化及服务重启恢复。
- Hybrid 优化任务、候选方案、基线方案和参考方案。
- `export_result` 中的全量排产明细。
- 订单评审、方案解析和资源利用率查询。
- 统一时间标签与排产指标计算。

主要缺口：

- 缺少面向 Agent 的紧凑、稳定、语义明确的查询契约。
- 现有部分接口面向前端页面，返回字段较多或存在展示截断。
- 当前新增的订单查询端点会重新运行 ATC 仿真，不代表指定优化候选方案。
- 缺少跨方案工序查询。
- 缺少 MCP Server 进程与端到端协议测试。

## 3. 总体架构

```text
Manufacturing Agent
  -> stdio JSON-RPC
llm4drd MCP Server
  -> HTTP http://127.0.0.1:8888
Planning Query REST API
  -> PlanningQueryService
  -> 当前 ShopFloor + Hybrid task export_result
  -> workflow_progress 持久化快照

Planning Command REST API
  -> 内置规则白名单校验
  -> Simulator
  -> 最近一次仿真结果 + workflow_progress 快照
```

采用“REST 领域查询层 + MCP 适配层”，不让 MCP Server 直接读取 SQLite，也不在 MCP Server 中重新实现指标计算。

## 4. 文件结构

建议新增：

```text
api/
  planning_query_service.py
mcp_server/
  __init__.py
  server.py
  planning_client.py
  schemas.py
  errors.py
tests/
  test_planning_query_service.py
  test_planning_query_api.py
  test_planning_mcp_server.py
  test_planning_mcp_e2e.py
```

现有 `api/server.py` 只保留薄路由和依赖装配，业务聚合进入 `PlanningQueryService`。

## 5. PlanningQueryService

### 5.1 职责

`PlanningQueryService` 提供纯查询方法：

```python
class PlanningQueryService:
    def get_overview(...): ...
    def compare_solutions(...): ...
    def search_orders(...): ...
    def get_order_planning(...): ...
    def search_operations(...): ...
    def get_operation_planning(...): ...
```

服务层不直接依赖 FastAPI，请求错误使用领域异常表达，路由层再转换为 HTTP 状态码。

### 5.2 任务解析

- `task_id` 未提供时，优先使用最近一次完成且包含结果的 Hybrid 任务。
- 进程重启后，使用 `workflow_progress` 恢复的最近一次优化快照。
- 实例版本与快照不一致时返回 `PLANNING_SNAPSHOT_STALE`。
- 任务运行中返回状态和进度，不把未完成结果伪装成候选方案。
- 指定任务不存在时返回 `PLANNING_TASK_NOT_FOUND`。

### 5.3 方案解析

查询全量明细时必须优先读取：

```text
task["export_result"]
```

不得使用可能为前端展示而截断的候选方案明细。方案分类为：

- `candidate`：优化器最终提供给用户选择的候选方案。
- `baseline`：基线规则方案。
- `reference`：启发式或精确参考方案。

计数定义：

- `candidate_count = len(result.solutions)`。
- `archive_size = result.archive_size`。
- 两者分别返回，不混用。
- `solution_count`：概览 `solutions` 列出的方案总数（基线 + 候选 + 参考），与评审页候选列表一致。

### 5.4 指标口径

新接口使用带单位的字段名：

- `total_tardiness_hours`：所有任务延误时长之和。
- `main_order_tardiness_hours`：主订单主任务延误时长之和。
- `order_tardiness_hours`：单个订单完工时间超过订单交期的时长。
- `makespan_hours`：整个方案的完工跨度。

单订单延误按以下公式计算：

```text
max(0, order_completion_hours - order_due_hours)
```

响应同时提供相对小时数和 ISO 8601 时间，时区沿用实例计划起点。

## 6. REST API 契约

统一前缀：

```text
/api/query/planning
```

### 6.1 排产概览

```text
GET /api/query/planning/overview?task_id=<optional>
```

返回：

```json
{
  "ok": true,
  "data": {
    "instance_version": 4,
    "task_id": "d7c3e797",
    "status": "done",
    "solution_count": 2,
    "candidate_count": 1,
    "archive_size": 1,
    "baseline_count": 1,
    "reference_count": 0,
    "solutions": [
      {
        "solution_id": "S-2f0b1c9a44",
        "solution_name": "方案一",
        "category": "baseline",
        "category_label": "基线方案",
        "rule_name": "ATC",
        "source": "baseline",
        "feasible": true,
        "total_tardiness_hours": 18.25,
        "main_order_tardiness_hours": 6.0,
        "makespan_hours": 160.02
      },
      {
        "solution_id": "S-6cf6190f25",
        "solution_name": "方案二",
        "category": "pareto",
        "category_label": "优化候选方案",
        "rule_name": null,
        "source": "hybrid",
        "feasible": true,
        "total_tardiness_hours": 0.0,
        "main_order_tardiness_hours": 0.0,
        "makespan_hours": 152.515
      }
    ]
  },
  "meta": {
    "time_unit": "hour",
    "task_defaulted": true
  }
}
```

### 6.2 方案比较

```text
GET /api/query/planning/solutions
```

参数：

- `task_id`：可选。
- `solution_ids`：可选，逗号分隔，最多 4 个。
- `metric_keys`：可选，逗号分隔。

未指定 `solution_ids` 时默认返回候选方案，不自动混入基线和参考方案。

所有含方案的查询结果同时返回 `solution_name` 与 `solution_id`：前者与方案评审界面的
“方案一 / 方案二 / ……”一致，用于面向用户展示；后者仅用于后续精确查询。

### 6.3 订单搜索

```text
GET /api/query/planning/orders/search?q=<query>&limit=20
```

匹配顺序：订单号精确、订单号前缀、订单号包含、订单名称包含。返回结果包含订单 ID、名称和匹配类型。

### 6.4 订单排产

```text
GET /api/query/planning/order/{order_id}
```

参数：

- `task_id`：可选。
- `solution_ids`：可选，最多 4 个。

每个方案返回订单完工、交期、订单延误、方案全局指标，以及该订单的完整工序列表。工序字段至少包括：

```text
operation_id, operation_name, task_id, machine_id,
start_hours, end_hours, start_at, end_at,
tooling_ids, personnel_ids
```

### 6.5 工序搜索

```text
GET /api/query/planning/operations/search?q=<query>&order_id=<optional>&limit=20
```

搜索工序 ID、名称、任务 ID和订单 ID。

### 6.6 工序排产

```text
GET /api/query/planning/operation/{operation_id}
```

每个方案必须返回一条 placement。未排入方案时返回：

```json
{
  "solution_id": "S-xxx",
  "planned": false,
  "reason": "operation_not_present_in_solution"
}
```

不能静默省略未排工序。

### 6.7 内置排产规则查询

```text
GET /api/query/planning/rules
```

返回所有内置规则的稳定名称、中文说明以及默认规则标记。该接口为只读操作。

### 6.8 按规则执行排产

```text
POST /api/command/planning/run-rule
```

请求：

```json
{"rule_name": "ATC"}
```

规则名忽略首尾空白并统一转换为大写。只有 `BUILTIN_RULES` 白名单中的规则可以执行；未知规则返回 `PLANNING_RULE_NOT_FOUND`，不能静默回退到默认规则。

该接口会执行一次真实排产，更新最近一次仿真结果并尝试写入 `workflow_progress`，因此属于有副作用的命令接口。响应返回规则、指标、排入工序数、最多 20 条排产预览及 `planning_truncated` 标记，不把整套甘特结果直接送入模型上下文。

## 7. MCP Server 契约

MCP Server 使用 stdio，只实现当前流程需要的：

```text
initialize
notifications/initialized
tools/list
tools/call
```

协议实现优先采用官方 Python MCP SDK，并锁定依赖版本；测试必须验证与 Manufacturing Agent 当前 `2024-11-05` Client 兼容。

### 7.1 MCP 工具

#### `get_planning_overview`

适用于“有多少候选方案”“各方案总延误是多少”。

#### `compare_planning_solutions`

输入：

```json
{
  "task_id": "optional",
  "solution_ids": ["optional"],
  "metric_keys": ["total_tardiness", "makespan"]
}
```

#### `search_planning_entities`

输入：

```json
{
  "entity_type": "order|operation",
  "query": "ORD-0004",
  "limit": 20
}
```

#### `get_order_planning`

输入：

```json
{
  "order_query": "ORD-0004",
  "task_id": "optional",
  "solution_ids": ["optional"]
}
```

工具内部可处理唯一的精确或模糊匹配；多条匹配时返回 `AMBIGUOUS_ORDER` 及候选列表，不自动猜测。

#### `get_operation_planning`

输入：

```json
{
  "operation_query": "OP-0004-02-01",
  "task_id": "optional",
  "solution_ids": ["optional"]
}
```

#### `list_planning_rules`

无输入参数。只读返回内置排产规则、说明和默认规则，适用于“有哪些规则”“ATC 是什么规则”。

#### `run_rule_planning`

输入：

```json
{"rule_name": "ATC"}
```

使用指定内置规则执行一次排产并更新最近一次仿真结果。该工具属于有副作用的执行工具，Agent 侧必须保留审批门禁，不能加入只读工具白名单。

### 7.2 返回大小控制

- 单次最多比较 4 个方案。
- 搜索默认 20 条、最大 50 条。
- 订单工序默认返回完整订单，但设置合理的最大工序数并在超限时返回 `truncated=true`。
- 工具结果返回简短文本摘要和结构化内容，不返回整套所有订单的全量排产。

## 8. 错误处理

普通业务错误作为结构化数据返回，让 Agent 可以澄清或修正：

```json
{
  "ok": false,
  "error": {
    "code": "AMBIGUOUS_OPERATION",
    "message": "匹配到多个工序",
    "suggestions": []
  }
}
```

以下情况使用 MCP `isError=true`：

- 排产 HTTP 服务不可连接。
- HTTP 调用超时。
- 返回内容无法解析或不符合契约。
- MCP Server 内部异常。

建议错误码：

```text
PLANNING_API_UNAVAILABLE
PLANNING_API_TIMEOUT
PLANNING_TASK_NOT_FOUND
PLANNING_TASK_NOT_READY
PLANNING_SNAPSHOT_STALE
PLANNING_RULE_NOT_FOUND
SOLUTION_NOT_FOUND
ORDER_NOT_FOUND
AMBIGUOUS_ORDER
OPERATION_NOT_FOUND
AMBIGUOUS_OPERATION
INVALID_ARGUMENT
```

## 9. 配置与运行

MCP Server 读取：

```text
PLANNING_API_BASE_URL=http://127.0.0.1:8888
PLANNING_API_TIMEOUT_SECONDS=10
```

Base URL 只能来自进程配置，不能由模型工具参数覆盖。首版只允许 loopback 地址。

启动顺序：

1. 启动 `llm4drd` FastAPI 服务。
2. Manufacturing Agent 启动 MCP Server 子进程。
3. MCP Server 完成握手和工具发布。
4. 每次工具调用再检查排产 API 是否可用。

## 10. 测试方案

### 10.1 服务层测试

- 正确解析最近完成任务。
- 指定任务不存在、运行中、失败和快照过期。
- 候选数量与 Archive 规模分别统计。
- 全量查询使用 `export_result`，不使用展示截断数据。
- 订单延误和全局总延误口径正确。
- 工序在不同方案中的 placement 正确。
- 未排工序显式返回 `planned=false`。
- 模糊匹配的唯一、无结果和歧义分支。

### 10.2 REST 测试

- 所有参数边界和错误码。
- 最多 4 个方案、最多 50 个搜索结果。
- 响应中无 NaN、Infinity 或不可序列化值。

### 10.3 MCP 测试

- 握手和工具发现。
- 七个工具的 JSON Schema。
- 工具调用到 REST 的参数映射。
- 规则名规范化、未知规则拒绝及按规则排产 POST 调用。
- 业务错误与 `isError` 的区分。
- HTTP 服务不可用和超时。
- 输出大小限制。

### 10.4 端到端测试

使用真实 stdio 子进程完成：

```text
MCP Client -> tools/list -> tools/call -> Planning REST API -> structured result
```

## 11. 实施步骤

1. 固定本文 REST 与 MCP 契约。
2. 编写失败测试，覆盖任务、方案、订单和工序查询。
3. 实现 `PlanningQueryService`。
4. 在 `api/server.py` 注册薄 REST 路由。
5. 实现 `planning_client.py` 和 MCP Server。
6. 增加协议、错误和输出边界测试。
7. 增加内置规则查询和按规则排产命令，并将执行工具标为有副作用能力。
8. 与 Manufacturing Agent 的 MCP Client 做真实子进程集成测试。
9. 验证服务重启后的最近优化快照仍可查询。

## 12. 验收标准

- “现在有多少候选方案，各自总延误是多少”通过一次 MCP 调用得到答案。
- “ORD-0004 的排产结果”能比较默认候选方案中的订单工序和完工时间。
- “OP-0004-02-01 排在什么时候”能返回各候选方案 placement。
- 歧义查询不会自动猜测。
- 排产服务停止后，MCP 返回明确的基础设施错误。
- 除 `run_rule_planning` 外的查询工具不触发任何排产数据写入。
- “有哪些内置排产规则”能返回规则名与说明；“使用 ATC 跑一遍排产”能执行并返回紧凑结果。
- 未知规则不会执行，也不会回退到 ATC。
- 返回结果使用 `planning` 命名并明确小时与绝对时间。
