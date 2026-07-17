# 端到端测试结论与实现计划（真实数据：516 订单 / 16,536 工序 / 1,149 机器）

> 本文档由「分析/诊断」阶段产出：只含测试结论、根因定位与精确到文件/行号/代码块的实现计划，
> 未对任何源代码、配置、前端文件做修改（`git diff` 应为空，仅新增本文档）。
> 实施时请按「总体实施路线」的顺序逐条落地；每条的「验证方式」必须执行。

---

## 1. 环境与测试方法

### 1.1 实际运行环境（与原始任务书的路径差异）

| 项 | 任务书假设 | 本机实际 |
|---|---|---|
| 项目路径 | `/Users/zhouwentao/Desktop/llm4drd` | `D:/github/llm4drd`（Windows + Git Bash） |
| 端口 | 8000 | 8888（`run_server.py:14` 硬编码）；本机 8888 已被一个运行中的实例占用，测试改用独立端口 8890 |
| 包名/PYTHONPATH | `llm4drd_platform` + `PYTHONPATH=Desktop` | `run_server.py:18-42` 在模块顶层自注册 `llm4drd_platform` 别名，无需 PYTHONPATH；且本机存在 sibling 目录 `D:/github/llm4drd_platform`，设置 PYTHONPATH 反而会造成包名遮蔽（AGENTS.md §4 已警告） |
| Python | `.venv/bin/python3` | `.venv/Scripts/python.exe` |

测试启动方式（未触碰 8888 上用户正在运行的服务与生产 `llm4drd.db`）：

```bash
# 临时启动器 /tmp/llm4drd_e2e_out/launcher.py（位于系统临时目录，不在仓库内）：
#   import sys; sys.path.insert(0, r"D:\github\llm4drd")
#   import run_server          # 模块顶层 _register_package() 注册 llm4drd_platform 别名
#   if __name__ == "__main__": # spawn 子进程重跑本文件时 __name__=="__mp_main__"，不会重复起服务
#       import uvicorn; uvicorn.run("llm4drd_platform.api.server:app", host="127.0.0.1", port=8890)
cd /d/github/llm4drd
export LLM_API_KEY=                                # 置空以覆盖 config.json 明文 Key（config.py:31 空串即生效）
export LLM4DRD_DB="$TEMP/llm4drd_e2e.db"           # 临时数据库，不污染生产 llm4drd.db
.venv/Scripts/python.exe /tmp/llm4drd_e2e_out/launcher.py
```

### 1.2 数据规模（`instance_output.xlsx`，sheet 名匹配导入要求）

orders 516 / tasks 4,363 / operations 16,536 / machine_types 53 / machines 1,149 /
toolings 0 / personnel 0 / downtimes 0（仅表头）/ initial_state 16,536（全部 pending）。
`plan_start_at = 2026-07-10T08:00:00+08:00`，日历 1001 天。

### 1.3 四步主流程结论矩阵

| 步骤 | 结论 | 证据 |
|---|---|---|
| ① 导入 | 可用（但响应 394MB，属严重隐患，见 P0-1） | `POST /api/instance/import-excel` → HTTP 200，墙钟 165.7s；`status=="ok"`；`validation.status=="passed"`、`error_count==0`、`warning_count==0`；`summary` 与 Excel 完全吻合（516/4363/16536/1149/53）；响应体 394,054,098 字节 |
| ② 仿真（ATC） | 可用但极慢（56.6 分钟） | `POST /api/simulate {"rule_name":"ATC"}` → HTTP 200，墙钟 3396.6s（服务端 `wall_time_ms=3,332,693`）；`feasible=true`，16,536/16,536 全部排入；`makespan=2072.82h`、`total_tardiness=849,968.3`、`avg_net_available_utilization=0.3157`；`event_count=1,717,612`（约 1.9ms/事件，印证派工扫描热点）；gantt 16,536 条字段齐全；可行故 `diagnosis=null`；仅 445/1,149 台机器分到工序 |
| ③ 优化（hybrid 短时长） | 可用但异常：管道跑通，短预算下实际未发生"优化" | task `7946548e`，总时长 3144s（52.4min）；轮询：coarse 阶段 approx 1→9（单近似评估 ~40–60s，`coarse_deadline=54s` 形同虚设——已提交的初始批次必须完成）；exact_promotion 阶段 exact 0→1 耗时 ~48min（基线 ATC 完整仿真）；最终 `approximate_evaluations=9 / exact_evaluations=1 / coarse_pool_size=9 / feasible_ratio=1.0`；result 仅 1 个解（baseline，exact，`feasible=true`，tardiness 849,968.313 / makespan 2072.82，与 /api/simulate 完全一致，交叉验证通过）；晋升 0 个、精修 0 轮（预算被基线吃光，§5.5 预测实锤）；进程池后端正常（无 `process pool unavailable` 降级日志，6 worker） |
| ④ 评审 | 可用（精确冠军参考除外，不可用） | `PUT /api/workflow/review` → 200 saved（0.04s）；`POST /api/optimize/hybrid/export-solution` → 200，20MB 有效 xlsx，但耗时 241s 且全程阻塞事件循环（P0-4）；`POST /api/ai/pareto/recommend` → 200，`used_model="heuristic-fallback"`（无 Key 静默回退正常；注意 `requirement` 必填，缺省 422）；`GET /api/workflow/progress` 四步快照齐全；候选/矩阵代码级确认（`getReviewCandidates` 五类来源 → `renderCandidateMetricMatrix` 消费 metrics/objectives/summary，`app_v2.js:853-888, 923-946`）；精确冠军参考：请求发出 15s 后 `/api/health` 超时（HTTP 000），事件循环被同步 CP-SAT 建模阻塞，服务器 RSS 899MB→2.88GB 仍攀升——P0-2 实锤 |
| 前端联动 | 代码级核对完成 + 接口实测（无 Playwright，环境未安装） | 导入后前端只读 `result.validation`（`frontend/app_v2.js:4782`），随后 `syncCurrentScene→loadInstanceBundle` 重新全量拉取 `/api/instance/details`（实测 133s/394MB）与 `/api/instance/db`（内嵌同量 details 且前端零消费）——导入后摘要更新实际被 ~1.2GB 下载拖死，修复见 P0-1/P0-3；甘特经 `renderTimeline→buildGanttData→mountGantts`（vis-timeline）；优化进度 1.5s 轮询（`app_v2.js:4689-4694`） |

### 1.4 性能基准（Before，本机实测墙钟）

| 步骤 | Before | 说明 |
|---|---|---|
| 导入 Excel | 165.7 s（响应 394 MB） | 其中解析+校验+序列化；响应体积本身是前端风险（P0-1） |
| ATC 单规则仿真 | 3396.6 s（其中仿真 3332.7s + 序列化/传输 ~64s，响应 7.7MB） | 事件 1,717,612 个；优化后目标见 P3-1/P3-2/P3-3（预期降一个数量级） |
| `GET /api/instance/details` | 133.0 s / 394 MB | 前端导入后必拉；精简方案见 P0-3 |
| hybrid 优化（短时长配置） | 3144 s（120s 预算配置下实际墙钟） | coarse ~6min（9 次近似评估）+ 基线精确仿真 ~48min；exact_evaluations=1，晋升/精修 0（配置见 §5.5） |

---

## 2. 跑通性问题（优先级 P0，先于 A/B/性能落地）

### P0-1 导入响应携带 394MB `details`，前端从不使用却必须下载并解析

现象：导入 16,536 工序实例，后端返回 394 MB JSON；浏览器端 `XMLHttpRequest` 下载后
`JSON.parse(xhr.responseText)`（`frontend/app_v2.js:245`）会长时间冻结主线程，低配机器直接 OOM 崩溃。
用户体感"导入卡死/浏览器无响应"。

根因：`api/server.py:1637` 把完整实例详情塞进导入响应：

```python
# api/server.py:1637（逐字）
        return {"status": "ok", "summary": shop.summary(), "details": _instance_details(shop), "validation": validation}
```

而前端导入成功路径只消费 `validation` 字段：

```js
// frontend/app_v2.js:4781-4784（逐字）
    setImportProgress({ busy: true, percent: 95, label: "正在刷新实例数据…", note: "即将完成" });
    app.validation = result?.validation || null;
    await syncCurrentScene(true);
    resetInstanceDerivedState();
```

`syncCurrentScene` 会另行调用 `GET /api/instance/details`（`app_v2.js:1085-1087 → 1072-1081`），
即 394MB 的 `details` 在导入响应里完全是死重。

改动方案（替换，`api/server.py` 函数 `import_excel`，锚定 1637 行）：

替换前（Before）：

```python
        return {"status": "ok", "summary": shop.summary(), "details": _instance_details(shop), "validation": validation}
```

替换后（After）：

```python
        # 大实例 details 可达数百 MB：前端导入流程只读 validation/summary，
        # 实例详情由前端随后走 GET /api/instance/details 按需拉取（见 app_v2.js loadInstanceBundle）。
        # 此处不回传 details，避免浏览器下载/解析巨量 JSON 冻结主线程。
        return {"status": "ok", "summary": shop.summary(), "validation": validation}
```

为何这样改：`details` 是导入响应中唯一的大字段且前端不消费，删除后响应从 394MB 降到 KB 级，
导入链路总耗时（165.7s 中序列化+传输+浏览器解析占比可观）同步下降；不改变任何后端语义。

验证方式：
1. `curl -F "file=@instance_output.xlsx" .../api/instance/import-excel` 返回 JSON 无 `details` 键，
   `status/summary/validation` 与原一致；响应体积 < 1MB。
2. 前端导入后"实例摘要/订单规模"正常刷新（数据来自 `syncCurrentScene` 的 details 接口，不受影响）。
3. `grep -n "result?.details\|result.details" frontend/app_v2.js` 确认无消费方（本次诊断已确认无）。

实施顺序与依赖：无依赖，第一批落地。注意 P0-1 只解决"导入响应"，`GET /api/instance/details`
本身的体积问题见「遗留风险 R-2」（前端确实需要其中的 machines/summary/orders 用于甘特与数据表，
不能一刀切，需单独的精简设计）。

---

### P0-2 精确冠军参考（CP-SAT）在大实例下无规模保护，且同步阻塞事件循环

现象：评审页点"精确冠军参考"后，在 16,536 工序规模下请求长时间无响应；
由于端点是 `async def` 内直接跑同步 CP-SAT 建模，整个后端事件循环被卡死，
期间所有其他请求（含前端轮询）全部挂起。

根因 1 — 无规模保护：`optimization/exact.py` 对每个（工序 × 候选机器）建一个
`sel_machine_{op}_{machine}` 布尔变量（`exact.py:286`），16,536 工序 × 数十~数百候选机器，
建模本身可达数十分钟+数 GB 内存；且时间预算只约束搜索阶段、不约束建模
（`exact.py:497-503` 逐字注释"time_limit_s 是整个 solve() 的预算……搜索阶段只使用扣除建模耗时后的余额"）。

根因 2 — 同步阻塞 + 双倍建模：`api/server.py:2863-2881`：

```python
# api/server.py:2863-2881（逐字）
@app.post("/api/optimize/exact-reference")
async def optimize_exact_reference(req: ExactReferenceReq):
    task_id, task = _resolve_hybrid_task(req.task_id)
    current_shop = _active_shop()
    if current_shop is None:
        raise HTTPException(400, "当前没有可用实例")
    objective_keys = _requested_exact_objective_keys((task.get("result") or {}).get("objective_keys", []), req)
    solution = _build_exact_reference_solution(current_shop, task, objective_keys, req, schedule_limit=120)
    existing = _task_reference_solution_index(task)
    existing[solution["solution_id"]] = solution
    task["reference_solutions"] = list(existing.values())
    export_result = task.get("export_result")
    if export_result is not None:
        export_refs = {item.get("solution_id"): item for item in export_result.get("reference_solutions", []) or [] if item.get("solution_id")}
        full_solution = _build_exact_reference_solution(current_shop, task, objective_keys, req, schedule_limit=None)
        export_refs[full_solution["solution_id"]] = full_solution
        export_result["reference_solutions"] = list(export_refs.values())
    _save_workflow_progress("optimization", {"task_id": task_id, "task": task})
    return {"task_id": task_id, "solution": solution, "reference_solution_count": len(task.get("reference_solutions", []))}
```

注意 2870 与 2877 在存在 `export_result` 时把同一 CP-SAT 求解跑两遍（仅 schedule 截断不同）。

改动方案（替换，`api/server.py`，锚定 2863-2870 行；2871-2881 其余行不变）：

替换前（Before）：见上 2863-2870 六行（`@app.post` 到 `solution = ...`）。

替换后（After）：

```python
# 大实例保护：CP-SAT 建模随 工序数×候选机台 超线性增长，建模阶段不受 time_limit_s 约束
# （exact.py:497-503），超过阈值直接拒绝，避免同步端点卡死整个事件循环。
EXACT_REFERENCE_MAX_OPERATIONS = 2000


@app.post("/api/optimize/exact-reference")
def optimize_exact_reference(req: ExactReferenceReq):
    task_id, task = _resolve_hybrid_task(req.task_id)
    current_shop = _active_shop()
    if current_shop is None:
        raise HTTPException(400, "当前没有可用实例")
    if len(current_shop.operations) > EXACT_REFERENCE_MAX_OPERATIONS:
        raise HTTPException(
            413,
            f"精确冠军参考仅支持不超过 {EXACT_REFERENCE_MAX_OPERATIONS} 道工序的实例"
            f"（当前 {len(current_shop.operations)} 道）。大实例请直接使用混合优化产出的帕累托方案。",
        )
    objective_keys = _requested_exact_objective_keys((task.get("result") or {}).get("objective_keys", []), req)
    solution = _build_exact_reference_solution(current_shop, task, objective_keys, req, schedule_limit=120)
```

为何这样改：① `async def` → `def` 让 FastAPI 把该端点丢进线程池（与 `/api/simulate` 的
`api/server.py:2750-2751` 注释同一模式），CP-SAT 再慢也不阻塞事件循环；② 工序数阈值把"必然不可用"
的大实例在前端可读的 413 错误下快速失败；③ 2877 行的二次求解不变（小实例下成本可接受，
本规模已被阈值拦截）。

验证方式：
1. 16,536 工序实例下 `POST /api/optimize/exact-reference {}` → HTTP 413，响应 < 1s，
   且期间 `GET /api/health` 正常响应（事件循环未阻塞）。
2. 小实例（≤2000 工序）下功能回归：返回 `solution.schedule` 非空、HTTP 200。
3. 前端评审页在该实例下展示 413 文案而非无限转圈（前端 `handleGenerateExact` 已有 error→toast 路径）。

实施顺序与依赖：无依赖，第一批落地。同类隐患 `POST /api/exact/solve`（`api/server.py:3588-3697`，
后台线程跑 CP-SAT，不阻塞事件循环但仍会耗资源）建议加同款阈值，见「遗留风险 R-3」。

---

### P0-3 前端导入后被迫下载 ~1.2GB JSON（details 394MB + db 内嵌 details 394MB + 原始表）

现象（实测）：`GET /api/instance/details` 墙钟 133.0s、响应 394,053,490 字节；
`GET /api/instance/db` 内含 `load_all()` 原始表 + 再次内嵌同一份 394MB details（`api/server.py:1489`）。
前端导入成功后 `loadInstanceBundle`（`app_v2.js:1072-1081`）把这两个接口都拉一遍——
叠加 P0-1 的导入响应，一次导入共 ~1.2GB JSON 下载 + 三次全量 `JSON.parse`，
浏览器主线程冻结数十秒甚至崩溃。这正是"导入后实例摘要/订单规模迟迟不更新"的直接原因。

根因：
① `_instance_details`（`api/server.py:1659-1772`）把 516 订单 × 任务 × 16,536 工序
（每工序 23 个字段）全量嵌套序列化，且 `orders` 部分占体积的绝对大头；
② 前端对 `app.instanceDetails` 的全部活消费只有 `summary / plan_start_at / machines` 三处
（`app_v2.js:724/1092、446/715/1290/1388、783`）——`orders` 仅被
`flattenTaskRecords/flattenOperationRecords`（`app_v2.js:1018-1055`）引用，
而这两个函数在全仓库无任何调用点（死代码）；
③ `app.instanceDb` 同样零消费方（仅 104 声明、1078 赋值、1100 置空），
`api.getInstanceDb()` 仅 1074 一处调用——整次下载是死重。

改动方案：

改动 1（替换，`_instance_details` 增加 lite 形参）：`api/server.py`，锚定 1659-1661。

Before（逐字）：

```python
def _instance_details(s: ShopFloor):
    orders = []
    for order_id, order in s.orders.items():
```

After：

```python
def _instance_details(s: ShopFloor, lite: bool = False):
    # lite=True 跳过 orders 明细构造：大实例下该部分可达数百 MB（16,536 工序 × 23 字段），
    # 而摘要/甘特/机器筛选等调用方只需要 machines + summary + plan_start_at。
    orders = []
    for order_id, order in (s.orders.items() if not lite else ()):
```

（函数其余部分不动；`return` 中 `"orders": orders` 在 lite 下为空数组。）

改动 2（替换，`/api/instance/details` 支持 lite 查询参数）：`api/server.py`，锚定 1458-1465。

Before（逐字）：

```python
@app.get("/api/instance/details")
async def inst_details():
    global shop
    if not shop and inst_store.has_data():
        shop = inst_store.build_shopfloor()
    if not shop:
        raise HTTPException(400, "请先生成实例或导入CSV")
    return _instance_details(shop)
```

After：

```python
@app.get("/api/instance/details")
async def inst_details(lite: bool = False):
    global shop
    if not shop and inst_store.has_data():
        shop = inst_store.build_shopfloor()
    if not shop:
        raise HTTPException(400, "请先生成实例或导入CSV")
    return _instance_details(shop, lite=lite)
```

改动 3（替换，`/api/instance/db` 内嵌 details 改为 lite）：`api/server.py`，锚定 1489。

Before（逐字）：

```python
    data["details"] = _instance_details(shop)
```

After：

```python
    # 前端已无 instanceDb.details 消费方；保留字段但只给 lite，避免内嵌数百 MB。
    data["details"] = _instance_details(shop, lite=True)
```

改动 4（替换，前端 details 走 lite）：`frontend/app_v2.js`，锚定 228。

Before（逐字）：

```js
  getInstanceDetails() { return this.json("/instance/details"); },
```

After：

```js
  getInstanceDetails(lite = false) { return this.json(`/instance/details${lite ? "?lite=1" : ""}`); },
```

改动 5（替换，前端 `loadInstanceBundle` 用 lite 并移除死亡的 db 拉取）：`frontend/app_v2.js`，锚定 1072-1081。

Before（逐字）：

```js
  const [details, db, downtimes] = await Promise.all([
    api.getInstanceDetails(),
    api.getInstanceDb(),
    api.getDowntimes().catch(() => []),
  ]);
  app.instanceDetails = details;
  app.instanceDb = db;
  app.downtimes = Array.isArray(downtimes?.downtimes)
    ? downtimes.downtimes
    : (Array.isArray(downtimes) ? downtimes : []);
```

After：

```js
  // details 走 lite（跳过 orders 明细，大实例 394MB → MB 级）；
  // instanceDb 在全仓库无消费方（死状态），不再每次导入后全量下载原始表。
  const [details, downtimes] = await Promise.all([
    api.getInstanceDetails(true),
    api.getDowntimes().catch(() => []),
  ]);
  app.instanceDetails = details;
  app.instanceDb = null;
  app.downtimes = Array.isArray(downtimes?.downtimes)
    ? downtimes.downtimes
    : (Array.isArray(downtimes) ? downtimes : []);
```

为何这样改：活消费方只用 `summary/plan_start_at/machines`，lite 全部保留 ⇒ 前端行为零变化；
默认 `lite=False` 保持 API 向后兼容（外部调用方不受影响）；
预计导入后刷新从"133s 下载 + 数十秒解析冻结"降到 1–3s。
（`flattenTaskRecords/flattenOperationRecords` 是死代码，不删——非本次最小必要范围。）

验证方式：
1. `curl ".../api/instance/details?lite=1"` 响应 < 20MB、< 5s，含 `summary/machines/plan_start_at`，`orders==[]`；
   不带参数时响应与原一致（回归）。
2. 前端导入后：实例摘要/订单规模秒级刷新；甘特机器分组、类型下拉、停机遮罩数据源（machines）完整。
3. `GET /api/instance/db` 响应不再含巨型 details（体积下降一个数量级以上）。

实施顺序与依赖：与 P0-1 同批（都动 `_instance_details` 的调用方，建议同一提交避免冲突）；
前端改动 4/5 与后端改动 1/2/3 需同时上线（前端先上线会因 `?lite=1` 被旧后端忽略而无害，后端先上线无影响）。

---

### P0-4 导出方案（export-solution）在 `async def` 内同步构建 xlsx，阻塞事件循环 241s

现象（实测）：`POST /api/optimize/hybrid/export-solution` 导出本实例 baseline 方案，
HTTP 200、产出 20MB 有效 xlsx，但墙钟 241s——期间整个后端（含优化进度轮询）失去响应。

根因：`api/server.py:3262-3263`（逐字）：

```python
@app.post("/api/optimize/hybrid/export-solution")
async def optimize_hybrid_export_solution(req: ExportSolutionReq):
```

`async def` 端点内调用同步重活 `_build_solution_export_bytes`（`api/server.py:4154-4233`，
openpyxl 逐单元格写 5 个 sheet：schedule 16,536 行 × 19 列 + 1,149 台机器的
`machine_calendar` 班次/停机窗口），241s 的 CPU 工作全程占住事件循环。与 P0-2 同一缺陷模式。

改动方案（替换，`api/server.py`，锚定 3263 行签名一行）：

替换前（Before）：

```python
async def optimize_hybrid_export_solution(req: ExportSolutionReq):
```

替换后（After）：

```python
def optimize_hybrid_export_solution(req: ExportSolutionReq):
```

为何这样改：与 `/api/simulate` 同款模式（`api/server.py:2750-2751` 注释）——
`def` 端点由 FastAPI 放入线程池，xlsx 构建再慢也不阻塞事件循环；一行改动，零语义影响。
（241s 本身的构建提速——write_only workbook / machine_calendar 裁剪——列为后续可选优化，
不在本最小计划内。）

验证方式：
1. 导出期间并发 `GET /api/health` 应立即 200（不再卡死）。
2. 导出产物 xlsx 可打开、5 个 sheet 数据与改前一致。
3. 前端"导出 Excel"按钮（`handleExportSolution`，`app_v2.js:5230-5247`）行为不变。

实施顺序与依赖：与 P0-2 同批（同一缺陷模式）。顺带排查：凡 `async def` 且内部有
同步重计算的端点都应改 `def`——本次诊断已确认的还有 `_simulate_locked` 调用方
`/api/simulate/compare`（`api/server.py:2827-2828` 已是 `def`，正确）与
`/api/simulate/reference-solutions`（`api/server.py:2848`，见 R-1）。

---

## 3. 问题 A：甘特图资源呈现（资源不可用 + 排产时间段 + 资源全貌）

### 现状与根因总览

甘特为前端客户端渲染：`renderTimeline`（`frontend/app_v2.js:1473-1546`）生成 HTML 字符串 →
`mountGantts`（`app_v2.js:4360-4406`）用 vis-timeline 实例化；数据构造在
`buildGanttData`（`app_v2.js:1387-1471`）；停机/班次遮罩在 `buildMachineOverlays`（`app_v2.js:1316-1379`）。
已确认以下事实（全部逐字核对过行号）：

• A1 资源被静默截断：`CONFIG.GANTT_MAX_GROUPS = 40`（`app_v2.js:11`），

  `buildGanttData:1412-1426` 在分组超过 40 时只保留"工序最多的前 40 台机器"。
  本实例 1,149 台机器 → 96.5% 资源不可见，含停机的机器若不在前 40 同样消失。
• A2 遮罩无悬停信息：`buildMachineOverlays` 为每段停机/班次外生成了

  `title: "计划停机 · 起 ~ 止"`（`app_v2.js:1334, 1360, 1374`），但 `buildGanttData:1452-1460`
  转成 vis background 条目时丢弃了 title（逐字见下），用户悬停无提示。
  CSS 斜纹带本身已存在且正确（`app_v2.css:2235-2237` 的
  `.vis-item.vis-background.offshift/.planned/.unplanned`），图例与色带视觉一致。
• A3 数据构造重复执行：`buildGanttData` 在 `renderTimeline:1495` 与 `mountGantts:4387`

  各算一遍；本规模下每遍要归一化 16,536 条 + 逐台机器生成遮罩，白白翻倍。
• 排产条块本身是正确的：条目 `start/end` 经 `ganttOffsetToISO`（`app_v2.js:1383-1385`）

  从 `plan_start_at` 基准换算，与后端 `gantt` 的 `start/end`（偏移小时）一致；
  三态 className `status-completed/processing/future` 由后端 `status` 字段归一而来
  （`normalizeScheduleStatus`，`app_v2.js:459-464`）；缩放/平移为 vis 原生能力（`mountGantts:4394-4397`），
  时刻换算与缩放无关，无需改动。验证点而非修复点（见 A-验证）。

### A1+A2：机器维度筛选/分页/聚合 + 遮罩悬停（核心改动）

根因定位（逐字摘录）：

```js
// frontend/app_v2.js:1412-1426 —— 静默截断（Before）
  // 大实例保护：分组超限时只保留工序条目最多的前 N 台机器，避免 vis-timeline 锁死主线程
  const totalGroups = groups.length;
  const totalOps = normalized.length;
  let visible = normalized;
  if (groups.length > CONFIG.GANTT_MAX_GROUPS) {
    const countByMachine = new Map();
    normalized.forEach((item) => countByMachine.set(item.machineId, (countByMachine.get(item.machineId) || 0) + 1));
    const keep = new Set(
      Array.from(countByMachine.keys())
        .sort((a, b) => countByMachine.get(b) - countByMachine.get(a))
        .slice(0, CONFIG.GANTT_MAX_GROUPS)
    );
    visible = normalized.filter((item) => keep.has(item.machineId));
    groups = groups.filter((g) => keep.has(g.id));
  }
```

```js
// frontend/app_v2.js:1450-1461 —— 遮罩条目丢弃 title（Before）
  // 遮罩：班次外 / 停机 -> background 项
  groups.forEach((g) => {
    const overlays = buildMachineOverlays(machineMap.get(g.id), horizonStart, horizonEnd);
    overlays.forEach((ov, i) => {
      const cls = ov.className.includes("unplanned") ? "unplanned" : ov.className.includes("planned") ? "planned" : "offshift";
      items.push({
        id: `bg-${g.id}-${i}`,
        group: g.id,
        start: ganttOffsetToISO(ov.startOffset, base),
        end: ganttOffsetToISO(ov.endOffset, base),
        type: "background",
        className: cls,
      });
    });
  });
```

设计决策：不砍掉 96% 资源，也不上虚拟滚动（vis-timeline 不支持分组虚拟滚动，自绘成本/风险高）。
改为「机器维度筛选 + 分页」：每页只画 `GANTT_PAGE_SIZE`（默认 40，沿用旧安全值）台机器，
vis 单页负载与现状持平、不会锁死主线程；用户可按机器类型筛选、只看含停机机器、按机器号搜索、
翻页遍历全部 1,149 台。默认排序改为"含停机优先 + 工序数降序"，保证异常资源最先可见。
选中订单时，机器集合天然限于"该订单涉及的机器"（分组来自条目），分页在其上叠加。

#### 改动 1（新增）：`CONFIG` 增加页大小常量

文件 `frontend/app_v2.js`，锚定第 11 行 `GANTT_MAX_GROUPS: 40,` 之后新增一行：

```js
  // 甘特每页机器行数（筛选/分页后单页渲染上限，沿用原 40 台的 vis 安全值）
  GANTT_PAGE_SIZE: 40,
```

#### 改动 2（新增）：`app` 状态增加两个筛选表

文件 `frontend/app_v2.js`，锚定 166-170 行（`pendingGantts`/`ganttInstances`/`ganttOrderFilter` 块）。

替换前（Before）：

```js
  pendingGantts: new Map(),
  ganttInstances: [],
  // 甘特图订单筛选：canvasId -> 选中的订单 id（"__all__" 表示全部，仅小实例提供）
  ganttOrderFilter: {},
```

替换后（After）：

```js
  pendingGantts: new Map(),
  ganttInstances: [],
  // 甘特图订单筛选：canvasId -> 选中的订单 id（"__all__" 表示全部，仅小实例提供）
  ganttOrderFilter: {},
  // 甘特图机器筛选与分页：canvasId -> { type, downtimeOnly, query, page }
  ganttMachineFilter: {},
  // 甘特图订单内二级筛选：canvasId -> { status, query, from, to }
  ganttEntryFilter: {},
```

#### 改动 3（新增）：筛选/分页辅助函数

文件 `frontend/app_v2.js`，插入位置：`function buildGanttData`（第 1387 行）之前整段新增：

```js
const GANTT_MACHINE_FILTER_DEFAULT = Object.freeze({ type: "__all__", downtimeOnly: false, query: "", page: 1 });
const GANTT_ENTRY_FILTER_DEFAULT = Object.freeze({ status: "__all__", query: "", from: "", to: "" });

function getGanttMachineFilter(canvasId) {
  return { ...GANTT_MACHINE_FILTER_DEFAULT, ...(app.ganttMachineFilter[canvasId] || {}) };
}

function getGanttEntryFilter(canvasId) {
  return { ...GANTT_ENTRY_FILTER_DEFAULT, ...(app.ganttEntryFilter[canvasId] || {}) };
}

// 机器维度信息：类型名 / 是否含停机 / 工序数，供筛选、排序与分页使用
function describeGanttMachine(machineId, machineName, machineMap, opCount) {
  const machine = machineMap.get(machineId);
  const typeName = machine?.type_name || machine?.type || machine?.machine_type || "未分组";
  // machineDowntimeRows 双数据源（machine.downtimes 优先，回退 /api/downtime 的 app.downtimes）
  const hasDowntime = machineDowntimeRows(machine || { machine_id: machineId }).length > 0;
  return { id: machineId, name: machineName, typeName, hasDowntime, opCount };
}

function filterGanttMachineRows(rows, filter) {
  const query = String(filter.query || "").trim().toLowerCase();
  return rows.filter((row) => {
    if (filter.type !== "__all__" && row.typeName !== filter.type) return false;
    if (filter.downtimeOnly && !row.hasDowntime) return false;
    if (query && !`${row.id} ${row.name}`.toLowerCase().includes(query)) return false;
    return true;
  });
}

// 含停机的机器优先，其次工序数降序——资源异常先被看到，而不是“最忙的前 N 台”
function sortGanttMachineRows(rows) {
  return rows.slice().sort((a, b) =>
    (Number(b.hasDowntime) - Number(a.hasDowntime))
    || (b.opCount - a.opCount)
    || String(a.id).localeCompare(String(b.id), "zh-CN", { numeric: true }));
}

// 订单内二级筛选：状态 / 关键字（工序号·任务·机器） / 时间窗（偏移小时）
function filterGanttEntries(entries, filter) {
  const query = String(filter.query || "").trim().toLowerCase();
  const from = filter.from === "" || filter.from === null || filter.from === undefined ? null : Number(filter.from);
  const to = filter.to === "" || filter.to === null || filter.to === undefined ? null : Number(filter.to);
  return entries.filter((item) => {
    if (filter.status !== "__all__" && normalizeScheduleStatus(item.status) !== filter.status) return false;
    if (query) {
      const haystack = `${item.op_id || item.operation_id || item.id || ""} ${item.task_id || ""} ${item.machine_id || ""} ${item.machine_name || ""}`.toLowerCase();
      if (!haystack.includes(query)) return false;
    }
    const start = Number(item.start ?? item.start_time ?? 0);
    const end = Number(item.end ?? item.end_time ?? 0);
    if (from !== null && !Number.isNaN(from) && end < from) return false;
    if (to !== null && !Number.isNaN(to) && start > to) return false;
    return true;
  });
}
```

为何这样改：筛选/排序/分页逻辑独立成纯函数，renderTimeline/buildGanttData 只编排；
含停机判定复用已有 `machineDowntimeRows`（`app_v2.js:989-999`，双数据源），停机数据真正来自后端存储。

#### 改动 4（替换）：`buildGanttData` 用"筛选+分页"替换"静默截断"，遮罩补 title

文件 `frontend/app_v2.js`，锚定 1387-1471 整函数。

替换前（Before，逐字全量）：

```js
function buildGanttData(entries, options = {}) {
  const planStartAt = tryParseDate(app.instanceDetails?.plan_start_at);
  const hasRealBase = Boolean(planStartAt);
  const base = hasRealBase ? planStartAt.toISOString() : GANTT_FALLBACK_BASE;

  const normalized = asArray(entries)
    .map((item) => ({
      machineId: item.machine_id || item.machine_name || item.resource_id || "unknown",
      machineName: item.machine_name || item.machine_id || item.resource_name || "未知资源",
      opId: item.op_id || item.operation_id || item.id || "-",
      orderId: item.order_id || "-",
      taskId: item.task_id || "-",
      start: Number(item.start ?? item.start_time ?? 0),
      end: Number(item.end ?? item.end_time ?? 0),
      status: normalizeScheduleStatus(item.status),
      statusLabel: item.status_label || (normalizeScheduleStatus(item.status) === "completed" ? "已完成" : normalizeScheduleStatus(item.status) === "processing" ? "进行中" : "未来排产"),
    }))
    .filter((item) => !Number.isNaN(item.start) && !Number.isNaN(item.end) && item.end > item.start);

  if (!normalized.length) return null;

  const groupsMap = new Map();
  normalized.forEach((item) => { if (!groupsMap.has(item.machineId)) groupsMap.set(item.machineId, item.machineName); });
  let groups = Array.from(groupsMap, ([id, content]) => ({ id, content: escapeHtml(content) }));

  // 大实例保护：分组超限时只保留工序条目最多的前 N 台机器，避免 vis-timeline 锁死主线程
  const totalGroups = groups.length;
  const totalOps = normalized.length;
  let visible = normalized;
  if (groups.length > CONFIG.GANTT_MAX_GROUPS) {
    const countByMachine = new Map();
    normalized.forEach((item) => countByMachine.set(item.machineId, (countByMachine.get(item.machineId) || 0) + 1));
    const keep = new Set(
      Array.from(countByMachine.keys())
        .sort((a, b) => countByMachine.get(b) - countByMachine.get(a))
        .slice(0, CONFIG.GANTT_MAX_GROUPS)
    );
    visible = normalized.filter((item) => keep.has(item.machineId));
    groups = groups.filter((g) => keep.has(g.id));
  }

  const machineMap = getMachineMap();
  const horizonStart = Math.min(...visible.map((i) => i.start));
  const horizonEnd = Math.max(...visible.map((i) => i.end));

  const items = [];
  visible.forEach((item, index) => {
    items.push({
      id: `op-${index}`,
      group: item.machineId,
      start: ganttOffsetToISO(item.start, base),
      end: ganttOffsetToISO(item.end, base),
      content: escapeHtml(item.opId),
      className: `status-${item.status}`,
      title: `${item.statusLabel} · ${item.opId}\n订单:${item.orderId} 任务:${item.taskId}\n${hasRealBase ? `${formatDateTime(ganttOffsetToISO(item.start, base))} ~ ${formatDateTime(ganttOffsetToISO(item.end, base))}` : `相对 ${item.start}h ~ ${item.end}h`}`,
    });
  });

  // 遮罩：班次外 / 停机 -> background 项
  groups.forEach((g) => {
    const overlays = buildMachineOverlays(machineMap.get(g.id), horizonStart, horizonEnd);
    overlays.forEach((ov, i) => {
      const cls = ov.className.includes("unplanned") ? "unplanned" : ov.className.includes("planned") ? "planned" : "offshift";
      items.push({
        id: `bg-${g.id}-${i}`,
        group: g.id,
        start: ganttOffsetToISO(ov.startOffset, base),
        end: ganttOffsetToISO(ov.endOffset, base),
        type: "background",
        className: cls,
      });
    });
  });

  const padH = Math.max((horizonEnd - horizonStart) * 0.02, 1);
  return {
    groups,
    items,
    hasRealBase,
    truncation: totalGroups > groups.length
      ? { totalGroups, shownGroups: groups.length, totalOps, shownOps: visible.length }
      : null,
    window: { start: ganttOffsetToISO(horizonStart - padH, base), end: ganttOffsetToISO(horizonEnd + padH, base) },
  };
}
```

替换后（After，全量）：

```js
function buildGanttData(entries, options = {}) {
  const planStartAt = tryParseDate(app.instanceDetails?.plan_start_at);
  const hasRealBase = Boolean(planStartAt);
  const base = hasRealBase ? planStartAt.toISOString() : GANTT_FALLBACK_BASE;

  const normalized = asArray(entries)
    .map((item) => ({
      machineId: item.machine_id || item.machine_name || item.resource_id || "unknown",
      machineName: item.machine_name || item.machine_id || item.resource_name || "未知资源",
      opId: item.op_id || item.operation_id || item.id || "-",
      orderId: item.order_id || "-",
      taskId: item.task_id || "-",
      start: Number(item.start ?? item.start_time ?? 0),
      end: Number(item.end ?? item.end_time ?? 0),
      status: normalizeScheduleStatus(item.status),
      statusLabel: item.status_label || (normalizeScheduleStatus(item.status) === "completed" ? "已完成" : normalizeScheduleStatus(item.status) === "processing" ? "进行中" : "未来排产"),
    }))
    .filter((item) => !Number.isNaN(item.start) && !Number.isNaN(item.end) && item.end > item.start);

  if (!normalized.length) return null;

  const machineMap = getMachineMap();
  const countByMachine = new Map();
  const nameByMachine = new Map();
  normalized.forEach((item) => {
    countByMachine.set(item.machineId, (countByMachine.get(item.machineId) || 0) + 1);
    if (!nameByMachine.has(item.machineId)) nameByMachine.set(item.machineId, item.machineName);
  });

  // 资源维度：类型 / 仅含停机 / 关键字 筛选 + 分页（替代旧的"只画最忙前 N 台"硬截断，
  // 大实例下全部机器都能翻页看到；单页行数仍受 GANTT_PAGE_SIZE 限制，vis 不会锁死）
  const machineFilter = getGanttMachineFilter(options.canvasId);
  const allRows = sortGanttMachineRows(
    Array.from(countByMachine.keys(), (machineId) =>
      describeGanttMachine(machineId, nameByMachine.get(machineId), machineMap, countByMachine.get(machineId)))
  );
  const typeOptions = Array.from(new Set(allRows.map((row) => row.typeName)))
    .sort((a, b) => String(a).localeCompare(String(b), "zh-CN", { numeric: true }));
  const filteredRows = filterGanttMachineRows(allRows, machineFilter);
  const pageSize = CONFIG.GANTT_PAGE_SIZE || CONFIG.GANTT_MAX_GROUPS;
  const pageCount = Math.max(1, Math.ceil(filteredRows.length / pageSize));
  const page = Math.min(Math.max(1, Number(machineFilter.page) || 1), pageCount);
  const pageRows = filteredRows.slice((page - 1) * pageSize, page * pageSize);
  const keep = new Set(pageRows.map((row) => row.id));

  const visible = normalized.filter((item) => keep.has(item.machineId));
  const groups = pageRows.map((row) => ({
    id: row.id,
    content: `${escapeHtml(row.name)}${row.hasDowntime ? ' <span class="gantt-group-downtime" title="该机台含停机时段">⚠</span>' : ""}`,
  }));
  const machineFacet = {
    totalGroups: allRows.length,
    filteredGroups: filteredRows.length,
    downtimeGroups: allRows.filter((row) => row.hasDowntime).length,
    page, pageCount, pageSize, typeOptions, filter: machineFilter,
  };

  if (!visible.length) {
    // 筛选/翻页后本页无条目（如关键字无命中）：保留 facet 供工具条渲染
    return { groups, items: [], hasRealBase, machineFacet, window: null };
  }

  const horizonStart = Math.min(...visible.map((i) => i.start));
  const horizonEnd = Math.max(...visible.map((i) => i.end));

  const items = [];
  visible.forEach((item, index) => {
    items.push({
      id: `op-${index}`,
      group: item.machineId,
      start: ganttOffsetToISO(item.start, base),
      end: ganttOffsetToISO(item.end, base),
      content: escapeHtml(item.opId),
      className: `status-${item.status}`,
      title: `${item.statusLabel} · ${item.opId}\n订单:${item.orderId} 任务:${item.taskId}\n${hasRealBase ? `${formatDateTime(ganttOffsetToISO(item.start, base))} ~ ${formatDateTime(ganttOffsetToISO(item.end, base))}` : `相对 ${item.start}h ~ ${item.end}h`}`,
    });
  });

  // 遮罩：班次外 / 停机 -> background 项；title 供悬停显示起止时刻与类型（vis 对 background 项同样出 tooltip）
  groups.forEach((g) => {
    const overlays = buildMachineOverlays(machineMap.get(g.id), horizonStart, horizonEnd);
    overlays.forEach((ov, i) => {
      const cls = ov.className.includes("unplanned") ? "unplanned" : ov.className.includes("planned") ? "planned" : "offshift";
      items.push({
        id: `bg-${g.id}-${i}`,
        group: g.id,
        start: ganttOffsetToISO(ov.startOffset, base),
        end: ganttOffsetToISO(ov.endOffset, base),
        type: "background",
        className: cls,
        title: ov.title,
      });
    });
  });

  const padH = Math.max((horizonEnd - horizonStart) * 0.02, 1);
  return {
    groups,
    items,
    hasRealBase,
    machineFacet,
    window: { start: ganttOffsetToISO(horizonStart - padH, base), end: ganttOffsetToISO(horizonEnd + padH, base) },
  };
}
```

为何这样改：① 截断改为"先筛选、再分页"，任何机器都能经翻页/筛选到达，资源全貌不再被砍；
② 默认排序含停机优先，异常资源第一页可见；③ `title: ov.title` 一行补回遮罩悬停信息；
④ 返回 `machineFacet` 供 renderTimeline 画工具条与分页器（替代旧 `truncation` 提示）；
⑤ 页大小沿用 40 的 vis 安全值，单页渲染负载与原截断方案相同，不引入主线程风险。

#### 改动 5（替换）：`renderTimeline` 增加机器筛选条与分页器

文件 `frontend/app_v2.js`，锚定 1473-1546 整函数。

替换前（Before，逐字全量）：

```js
function renderTimeline(entries, options = {}) {
  const id = options.canvasId || `gantt-${(options.title || "t").replace(/[^a-zA-Z0-9]/g, "").slice(0, 24)}`;
  const allEntries = asArray(entries);

  // 订单筛选（同图谱页的按订单聚焦逻辑）：大实例整图渲染会锁死主线程，
  // 一次只展示一个订单及其工序、涉及的机器；小实例才提供"全部订单"。
  const orderNames = new Map();
  allEntries.forEach((item) => {
    const key = item.order_id || "-";
    if (!orderNames.has(key)) orderNames.set(key, item.order_name || "");
  });
  const orderOptions = Array.from(orderNames.keys())
    .sort((a, b) => String(a).localeCompare(String(b), "zh-CN", { numeric: true }));
  const allowAll = allEntries.length <= CONFIG.GANTT_ALL_ORDERS_MAX_OPS;
  let selectedOrder = app.ganttOrderFilter[id];
  if (selectedOrder === "__all__" && !allowAll) selectedOrder = null;
  if (selectedOrder !== "__all__" && !orderOptions.includes(selectedOrder)) selectedOrder = null;
  if (!selectedOrder) selectedOrder = allowAll ? "__all__" : orderOptions[0];
  const visibleEntries = selectedOrder === "__all__"
    ? allEntries
    : allEntries.filter((item) => (item.order_id || "-") === selectedOrder);

  const data = buildGanttData(visibleEntries, options);
  if (!data) {
    return renderEmptyState("暂无甘特数据", "当前方案还没有可显示的资源排程。");
  }
  app.pendingGantts.set(id, { entries: visibleEntries, options });

  const orderSelector = orderOptions.length > 1 || !allowAll ? `
    <div class="field-inline">
      <span>订单</span>
      <select data-gantt-order-select data-canvas="${escapeHtml(id)}">
        ${allowAll ? `<option value="__all__" ${selectedOrder === "__all__" ? "selected" : ""}>全部订单（${formatInt(allEntries.length)} 道工序）</option>` : ""}
        ${orderOptions.map((orderId) => {
          const name = orderNames.get(orderId);
          return `<option value="${escapeHtml(orderId)}" ${orderId === selectedOrder ? "selected" : ""}>${escapeHtml(name && name !== orderId ? `${orderId} · ${name}` : orderId)}</option>`;
        }).join("")}
      </select>
    </div>
  ` : "";

  const statusCounts = data.items.reduce((acc, it) => {
    if (it.type === "background") return acc;
    const key = (it.className || "").replace("status-", "");
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, { completed: 0, processing: 0, future: 0 });

  return `
    <div class="surface-card">
      <div class="card-head">
        <h3>${escapeHtml(options.title || "资源甘特图")}</h3>
        <p>可滚轮缩放、左右拖拽平移查看全程；条块颜色区分已完成 / 进行中 / 未来排产，斜纹遮罩显示班次外与停机占用。</p>
      </div>
      ${orderSelector}
      <div class="timeline-summary-strip">
        <div class="timeline-summary-card"><span>当前展示</span><strong>${selectedOrder === "__all__" ? "全部订单" : escapeHtml(selectedOrder)}（共 ${formatInt(orderOptions.length)} 个订单）</strong></div>
        <div class="timeline-summary-card"><span>工序 / 资源行数</span><strong>${formatInt(visibleEntries.length)} / ${formatInt(data.groups.length)}</strong></div>
        <div class="timeline-summary-card"><span>已完成 / 进行中 / 未来</span><strong>${formatInt(statusCounts.completed)} / ${formatInt(statusCounts.processing)} / ${formatInt(statusCounts.future)}</strong></div>
        <div class="timeline-summary-card"><span>时间基准</span><strong>${data.hasRealBase ? "计划起始时间" : "相对小时（无 plan_start_at）"}</strong></div>
      </div>
      ${data.truncation ? `<p class="subtle-note">当前视图仍超出渲染上限：仅展示排程最多的前 ${formatInt(data.truncation.shownGroups)} / ${formatInt(data.truncation.totalGroups)} 台机器（${formatInt(data.truncation.shownOps)} / ${formatInt(data.truncation.totalOps)} 道工序），完整结果请使用导出 Excel。</p>` : ""}
      <div class="legend">
        <span class="legend-item"><span class="legend-swatch status-completed"></span>已完成</span>
        <span class="legend-item"><span class="legend-swatch status-processing"></span>进行中</span>
        <span class="legend-item"><span class="legend-swatch status-future"></span>未来排产</span>
        <span class="legend-item"><span class="legend-swatch offshift"></span>班次外</span>
        <span class="legend-item"><span class="legend-swatch planned"></span>计划停机</span>
        <span class="legend-item"><span class="legend-swatch unplanned"></span>非计划停机</span>
      </div>
      <div class="gantt-canvas" id="${escapeHtml(id)}"></div>
    </div>
  `;
}
```

替换后（After，全量）：

```js
function renderTimeline(entries, options = {}) {
  const id = options.canvasId || `gantt-${(options.title || "t").replace(/[^a-zA-Z0-9]/g, "").slice(0, 24)}`;
  const allEntries = asArray(entries);

  // 订单筛选（同图谱页的按订单聚焦逻辑）：大实例整图渲染会锁死主线程，
  // 一次只展示一个订单及其工序、涉及的机器；小实例才提供"全部订单"。
  const orderNames = new Map();
  allEntries.forEach((item) => {
    const key = item.order_id || "-";
    if (!orderNames.has(key)) orderNames.set(key, item.order_name || "");
  });
  const orderOptions = Array.from(orderNames.keys())
    .sort((a, b) => String(a).localeCompare(String(b), "zh-CN", { numeric: true }));
  const allowAll = allEntries.length <= CONFIG.GANTT_ALL_ORDERS_MAX_OPS;
  let selectedOrder = app.ganttOrderFilter[id];
  if (selectedOrder === "__all__" && !allowAll) selectedOrder = null;
  if (selectedOrder !== "__all__" && !orderOptions.includes(selectedOrder)) selectedOrder = null;
  if (!selectedOrder) selectedOrder = allowAll ? "__all__" : orderOptions[0];
  const orderEntries = selectedOrder === "__all__"
    ? allEntries
    : allEntries.filter((item) => (item.order_id || "-") === selectedOrder);

  // 订单内二级筛选：状态 / 关键字 / 时间窗（选中订单后仍可继续过滤，见问题 B）
  const entryFilter = getGanttEntryFilter(id);
  const visibleEntries = filterGanttEntries(orderEntries, entryFilter);

  const data = buildGanttData(visibleEntries, { ...options, canvasId: id });
  if (!data) {
    return renderEmptyState("暂无甘特数据", "当前方案还没有可显示的资源排程。");
  }
  // 连同已计算的 data 一起暂存：mountGantts 直接复用，避免大实例下重复构建（A3）
  app.pendingGantts.set(id, { entries: visibleEntries, options: { ...options, canvasId: id }, data });

  const orderSelector = orderOptions.length > 1 || !allowAll ? `
    <div class="field-inline">
      <span>订单</span>
      <select data-gantt-order-select data-canvas="${escapeHtml(id)}">
        ${allowAll ? `<option value="__all__" ${selectedOrder === "__all__" ? "selected" : ""}>全部订单（${formatInt(allEntries.length)} 道工序）</option>` : ""}
        ${orderOptions.map((orderId) => {
          const name = orderNames.get(orderId);
          return `<option value="${escapeHtml(orderId)}" ${orderId === selectedOrder ? "selected" : ""}>${escapeHtml(name && name !== orderId ? `${orderId} · ${name}` : orderId)}</option>`;
        }).join("")}
      </select>
    </div>
  ` : "";

  const facet = data.machineFacet;
  const machineFilterBar = facet ? `
    <div class="field-inline gantt-filter-bar">
      <span>机器类型</span>
      <select data-gantt-machine-type data-canvas="${escapeHtml(id)}">
        <option value="__all__" ${facet.filter.type === "__all__" ? "selected" : ""}>全部类型（${formatInt(facet.totalGroups)} 台）</option>
        ${facet.typeOptions.map((t) => `<option value="${escapeHtml(t)}" ${t === facet.filter.type ? "selected" : ""}>${escapeHtml(t)}</option>`).join("")}
      </select>
      <label class="gantt-filter-check"><input type="checkbox" data-gantt-downtime-only data-canvas="${escapeHtml(id)}" ${facet.filter.downtimeOnly ? "checked" : ""}> 仅含停机（${formatInt(facet.downtimeGroups)} 台）</label>
      <input type="search" class="gantt-filter-query" placeholder="搜索机器号…" value="${escapeHtml(facet.filter.query)}" data-gantt-machine-query data-canvas="${escapeHtml(id)}">
    </div>
  ` : "";
  const entryFilterBar = `
    <div class="field-inline gantt-filter-bar">
      <span>工序状态</span>
      <select data-gantt-status-filter data-canvas="${escapeHtml(id)}">
        <option value="__all__" ${entryFilter.status === "__all__" ? "selected" : ""}>全部状态</option>
        <option value="completed" ${entryFilter.status === "completed" ? "selected" : ""}>已完成</option>
        <option value="processing" ${entryFilter.status === "processing" ? "selected" : ""}>进行中</option>
        <option value="future" ${entryFilter.status === "future" ? "selected" : ""}>未来排产</option>
      </select>
      <input type="search" class="gantt-filter-query" placeholder="搜索工序/任务/机器…" value="${escapeHtml(entryFilter.query)}" data-gantt-entry-query data-canvas="${escapeHtml(id)}">
      <span>时间窗(h)</span>
      <input type="number" class="gantt-filter-num" placeholder="起" value="${escapeHtml(String(entryFilter.from))}" data-gantt-time-from data-canvas="${escapeHtml(id)}">
      <span>~</span>
      <input type="number" class="gantt-filter-num" placeholder="止" value="${escapeHtml(String(entryFilter.to))}" data-gantt-time-to data-canvas="${escapeHtml(id)}">
    </div>
  `;
  const pager = facet && facet.pageCount > 1 ? `
    <div class="gantt-pager">
      <button class="btn-ghost" data-gantt-page="${facet.page - 1}" data-canvas="${escapeHtml(id)}" ${facet.page <= 1 ? "disabled" : ""}>上一页</button>
      <span>第 ${formatInt(facet.page)} / ${formatInt(facet.pageCount)} 页 · 命中 ${formatInt(facet.filteredGroups)} / ${formatInt(facet.totalGroups)} 台机器</span>
      <button class="btn-ghost" data-gantt-page="${facet.page + 1}" data-canvas="${escapeHtml(id)}" ${facet.page >= facet.pageCount ? "disabled" : ""}>下一页</button>
    </div>
  ` : "";

  const statusCounts = data.items.reduce((acc, it) => {
    if (it.type === "background") return acc;
    const key = (it.className || "").replace("status-", "");
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, { completed: 0, processing: 0, future: 0 });

  return `
    <div class="surface-card">
      <div class="card-head">
        <h3>${escapeHtml(options.title || "资源甘特图")}</h3>
        <p>可滚轮缩放、左右拖拽平移查看全程；条块颜色区分已完成 / 进行中 / 未来排产，斜纹遮罩显示班次外与停机占用（悬停可见起止与类型）。</p>
      </div>
      ${orderSelector}
      ${machineFilterBar}
      ${entryFilterBar}
      <div class="timeline-summary-strip">
        <div class="timeline-summary-card"><span>当前展示</span><strong>${selectedOrder === "__all__" ? "全部订单" : escapeHtml(selectedOrder)}（共 ${formatInt(orderOptions.length)} 个订单）</strong></div>
        <div class="timeline-summary-card"><span>工序 / 资源行数</span><strong>${formatInt(visibleEntries.length)} / ${formatInt(data.groups.length)}${facet ? `（共 ${formatInt(facet.totalGroups)} 台）` : ""}</strong></div>
        <div class="timeline-summary-card"><span>已完成 / 进行中 / 未来</span><strong>${formatInt(statusCounts.completed)} / ${formatInt(statusCounts.processing)} / ${formatInt(statusCounts.future)}</strong></div>
        <div class="timeline-summary-card"><span>时间基准</span><strong>${data.hasRealBase ? "计划起始时间" : "相对小时（无 plan_start_at）"}</strong></div>
      </div>
      ${pager}
      <div class="legend">
        <span class="legend-item"><span class="legend-swatch status-completed"></span>已完成</span>
        <span class="legend-item"><span class="legend-swatch status-processing"></span>进行中</span>
        <span class="legend-item"><span class="legend-swatch status-future"></span>未来排产</span>
        <span class="legend-item"><span class="legend-swatch offshift"></span>班次外</span>
        <span class="legend-item"><span class="legend-swatch planned"></span>计划停机</span>
        <span class="legend-item"><span class="legend-swatch unplanned"></span>非计划停机</span>
      </div>
      <div class="gantt-canvas" id="${escapeHtml(id)}"></div>
    </div>
  `;
}
```

为何这样改：订单下拉（既有）→ 机器类型/停机/搜索（A1）→ 状态/关键字/时间窗（B）三层过滤
串成一条漏斗，全部在客户端条目数组上完成（16,536 条过滤为毫秒级）；分页器替代旧截断提示，
资源总量/命中量透明可见；`pendingGantts` 暂存 `data` 消除 A3 的重复构建。

#### 改动 6（替换）：`mountGantts` 复用暂存 data + 兼容空窗口

文件 `frontend/app_v2.js`，锚定 4385-4388 与 4399-4400。

替换前（Before）：

```js
    const payload = app.pendingGantts.get(el.id);
    if (!payload) return;
    const data = buildGanttData(payload.entries, payload.options);
    if (!data) return;
```

替换后（After）：

```js
    const payload = app.pendingGantts.get(el.id);
    if (!payload) return;
    // renderTimeline 已构建过 data（暂存在 payload），大实例下避免重复归一化/遮罩计算
    const data = payload.data || buildGanttData(payload.entries, payload.options);
    if (!data) return;
```

替换前（Before）：

```js
        start: data.window.start,
        end: data.window.end,
```

替换后（After）：

```js
        // 筛选无命中时 window 为 null，交给 vis 自适应空视图
        start: data.window ? data.window.start : undefined,
        end: data.window ? data.window.end : undefined,
```

为何这样改：消除同一次渲染里的第二遍 `buildGanttData`（16,536 条归一化 + 每机遮罩）；
`window=null` 兜底避免筛选无命中时抛 TypeError。

#### 改动 7（新增）：change / click 事件分支

文件 `frontend/app_v2.js`。
(a) change 委托：锚定 5591-5594（`data-gantt-order-select` 分支）之后追加：

```js
    if (target.matches("[data-gantt-machine-type]")) {
      const canvasId = target.dataset.canvas;
      const filter = getGanttMachineFilter(canvasId);
      filter.type = target.value;
      filter.page = 1;
      app.ganttMachineFilter[canvasId] = filter;
      return renderCurrentPage();
    }
    if (target.matches("[data-gantt-downtime-only]")) {
      const canvasId = target.dataset.canvas;
      const filter = getGanttMachineFilter(canvasId);
      filter.downtimeOnly = target.checked;
      filter.page = 1;
      app.ganttMachineFilter[canvasId] = filter;
      return renderCurrentPage();
    }
    if (target.matches("[data-gantt-machine-query]")) {
      const canvasId = target.dataset.canvas;
      const filter = getGanttMachineFilter(canvasId);
      filter.query = target.value;
      filter.page = 1;
      app.ganttMachineFilter[canvasId] = filter;
      return renderCurrentPage();
    }
    if (target.matches("[data-gantt-status-filter]")) {
      const canvasId = target.dataset.canvas;
      const filter = getGanttEntryFilter(canvasId);
      filter.status = target.value;
      app.ganttEntryFilter[canvasId] = filter;
      return renderCurrentPage();
    }
    if (target.matches("[data-gantt-entry-query]")) {
      const canvasId = target.dataset.canvas;
      const filter = getGanttEntryFilter(canvasId);
      filter.query = target.value;
      app.ganttEntryFilter[canvasId] = filter;
      return renderCurrentPage();
    }
    if (target.matches("[data-gantt-time-from], [data-gantt-time-to]")) {
      const canvasId = target.dataset.canvas;
      const filter = getGanttEntryFilter(canvasId);
      if (target.hasAttribute("data-gantt-time-from")) filter.from = target.value;
      else filter.to = target.value;
      app.ganttEntryFilter[canvasId] = filter;
      return renderCurrentPage();
    }
```

(b) click 委托：锚定 5522-5523（`document.addEventListener("click", ...)` 内、
`reviewTabTarget` 分支之前）插入：

```js
    const ganttPageTarget = event.target.closest("[data-gantt-page]");
    if (ganttPageTarget && !ganttPageTarget.disabled) {
      event.preventDefault();
      const canvasId = ganttPageTarget.dataset.canvas;
      const filter = getGanttMachineFilter(canvasId);
      filter.page = Number(ganttPageTarget.dataset.ganttPage) || 1;
      app.ganttMachineFilter[canvasId] = filter;
      renderCurrentPage();
      return;
    }
```

为何这样改：沿用现有"改状态 → `renderCurrentPage()`"的订单下拉同款联动模式（`app_v2.js:5591-5594`），
筛选/翻页后视图必然重新过滤并重渲染；搜索/时间窗用 change 事件（回车或失焦触发），避免键入抖动。

#### 改动 8（新增）：CSS

文件 `frontend/app_v2.css`，插入位置：第 2290 行（`.field-inline > select { ... }` 块）之后、
文件末尾 `.form-actions--gap` 之前（或任意甘特段落附近）：

```css
/* -- 甘特资源筛选 / 二级筛选 / 分页 -- */
.gantt-filter-bar { flex-wrap: wrap; row-gap: 6px; margin: 6px 0; }
.gantt-filter-bar > select { flex: 0 1 auto; min-width: 140px; padding: 6px 10px; }
.gantt-filter-query, .gantt-filter-num {
  border: 1px solid var(--line-strong);
  background: #fff;
  border-radius: 12px;
  padding: 6px 10px;
  color: var(--text);
  font-size: 13px;
}
.gantt-filter-query { min-width: 160px; flex: 0 1 auto; }
.gantt-filter-num { width: 76px; }
.gantt-filter-check { display: inline-flex; align-items: center; gap: 4px; color: var(--text-soft); font-size: 13px; white-space: nowrap; }
.gantt-pager { display: flex; align-items: center; gap: 10px; margin: 4px 0 10px; color: var(--text-soft); font-size: 12px; }
.gantt-pager .btn-ghost { padding: 4px 12px; border-radius: 999px; }
.gantt-group-downtime { color: var(--danger); font-weight: 700; }
```

为何这样改：复用 `.field-inline`/`.btn-ghost` 既有规范（`app_v2.css:2271, 575`），
仅补充新控件的布局细节；分组标签的 ⚠ 用危险色提示该机台含停机。

### A-验证方式

1. 数据完整性（无浏览器）：`POST /api/simulate {"rule_name":"ATC"}` 返回的
   `gantt[*]` 每条含 `op_id/task_id/order_id/machine_id/machine_name/start/end/start_at/end_at/status`
   （本诊断已逐字确认字段链路 `api/server.py:2791-2810 → 392-409`）；
   `GET /api/downtime` 返回 `downtime_type/start_time/end_time`；`GET /api/instance/details`
   的 `machines[*]` 含 `shift_windows/downtimes`（`api/server.py:419-451`）。
2. DOM/事件联动（无浏览器）：实施后用 `node -e` 或静态检查确认：
   `renderTimeline` 输出含 `data-gantt-machine-type`、`data-gantt-page`；
   change/click 委托分支命中对应 `data-*` 选择器；`buildGanttData` 对同一输入，
   筛选 `downtimeOnly=true` 时 groups 全部为含停机机器。
3. 浏览器手测：导入本实例 → 仿真页甘特：默认第 1 页含停机机器优先；
   翻页可遍历全部机器（页数 = ⌈命中/40⌉）；类型下拉只剩 53 个类型；
   悬停停机/班次外色带出现"类型 · 起 ~ 止"tooltip；条块 tooltip 的起止时刻与
   `gantt` 的 `start/end`（经 `plan_start_at` 换算）一致，缩放/平移后不变。
4. 回归：小实例（≤2000 工序）"全部订单"入口仍在（`allowAll` 逻辑未动）。

### A-实施顺序与依赖

改动 1→2→3→4→5→6→7→8 同批落地（互相引用，单独落地会 ReferenceError）。
依赖：无（纯前端）。与问题 B 共用 `getGanttEntryFilter`/`filterGanttEntries`（本计划已一并给出）。

---

## 4. 问题 B：订单下拉选择后必须可继续筛选

现象：订单下拉（`data-gantt-order-select`）选中某个订单后，只能"整单切换"，
无法在订单内按工序状态/机器/时间窗/关键字再过滤；用户在大实例（>2000 工序被强制单订单）下
面对单订单数十~数百道工序仍找不到目标。

根因定位：

```js
// frontend/app_v2.js:1492-1494（逐字）——订单过滤后即结束，无任何二级过滤
  const visibleEntries = selectedOrder === "__all__"
    ? allEntries
    : allEntries.filter((item) => (item.order_id || "-") === selectedOrder);
```

```js
// frontend/app_v2.js:5591-5594（逐字）——change 联动本身是好的（改状态→整页重渲）
    if (target.matches("[data-gantt-order-select]")) {
      app.ganttOrderFilter[target.dataset.canvas] = target.value;
      return renderCurrentPage();
    }
```

即：订单下拉的联动重渲染机制健全（`app.ganttOrderFilter[canvasId]` → `renderCurrentPage()`），
缺的是订单之下的二级过滤维度。大实例强制单订单的入口限制
（`GANTT_ALL_ORDERS_MAX_OPS=2000`，`app_v2.js:13, 1486-1490`）本身是必要的渲染保护，予以保留。

改动方案：已由问题 A 的改动一并交付——

• `filterGanttEntries`（A-改动 3）：状态 / 关键字（工序号、任务、机器）/ 时间窗（偏移小时）三维过滤；

• `renderTimeline` 中的 `entryFilterBar`（A-改动 5）：下拉 + 搜索框 + 时间窗输入，无论是否选中订单都可用，

  作用于 `orderEntries → visibleEntries`（A-改动 5 第 27-29 行）；
• change 分支（A-改动 7a）：`data-gantt-status-filter / data-gantt-entry-query / data-gantt-time-from / data-gantt-time-to`，

  与订单下拉同一联动模式，变更后必然 `renderCurrentPage()` 重过滤重渲染。

为何这样改：在既有"订单下拉 → 重渲染"的健全链路上，把过滤点从 1 个扩到 4 个，
不引入新的渲染管线；二级过滤作用于已选订单的条目子集，大实例下入口限制保持不变，
不会被规模锁死成只读单订单。

验证方式：
1. 选中某订单 → 状态下拉选"未来排产"：甘特中只剩 `status-future` 条块，
   汇总条"已完成/进行中/未来"计数同步变化（计数来自 `data.items`，已过滤）。
2. 关键字输入工序号片段 → 回车：仅匹配条目保留；清空后恢复。
3. 时间窗填 `0~200`：条块均与 [0,200] 有重叠（过滤谓词 `end>=from && start<=to`）。
4. 与 A 的联动：二级过滤 + 机器分页可同时生效（entry 过滤在前、机器分页在后，互不重置）。
5. 小实例"全部订单"入口保留（`allowAll` 未动）。

实施顺序与依赖：依赖 A-改动 3/5/7（同一批落地）。

## 5. 性能优化实现计划（16,536 工序 / 1,149 机器）

### 5.1 量化结论（代码结构 × 实测）

| 瓶颈 | 位置 | 量级估算（本实例） | Before 实测 |
|---|---|---|---|
| 派工全扫描：类型就绪桶 `list()` 复制 + 逐候选 eligible 过滤 + 每候选 17 键 features dict | `core/simulator.py:193-211` | 候选评估 10⁶–10⁷ 次/轮 | 单规则 ATC 全程 3396.6s（事件 1,717,612，约 1.9ms/事件） |
| `_trigger_idle_dispatches` 每次全扫 1,149 台机器 | `core/simulator.py:522-527` | 上限 ≈16,536×1,149 ≈ 1.9×10⁷ 次迭代 | 同上 |
| `_features` 每候选重建 dict（10/17 键静态）+ `prereq_ratio` 派工路径恒 1.0 的循环 | `core/simulator.py:652-700` | 与候选评估同量级 | 同上 |
| hybrid 精确阶段被预算截断（非 bug，机制使然） | `optimization/hybrid_nsga3_alns.py:1500-1514, 1877, 1968-1972` | 默认 90s 下 exact_evaluations ≈ 1–2 | 【待补#2】 |

结构事实（侦察确认，避免重复造轮子）：`SimulationRuntime` 复用、`_cached_sim_runtime`、
混合优化的 `SimulationRuntimePool`（`hybrid_nsga3_alns.py:324`）、进程池 worker 的
一次性 runtime（`optimization/parallel_eval.py:64-99`）均已实现——分钟级耗时的主因
是上面前两行的派工路径，不是运行时重建。

### 5.2（P3-1）派工反向索引 `machine_ready_ops`

根因：`core/simulator.py:193-199`（逐字）——每次派工把该机型的整桶就绪工序复制成列表，
再逐候选检查"本机是否在其 eligible 集合"：

```python
# core/simulator.py:193-199（Before，逐字）
                candidate_ids = list(self._ready_by_type.get(machine.type_id, set()))
                for op_id in candidate_ids:
                    op = shop.operations.get(op_id)
                    if not op or op.status != OpStatus.READY:
                        self._discard_ready(op_id, machine.type_id, ready_ops)
                        continue
                    if machine.id not in self._eligible_machine_ids.get(op.id, set()):
                        continue
```

`_ready_by_type` 全仓库仅 5 处出现（`core/simulator.py:73, 103, 193, 467, 472`），
除派工外无其他消费方，可安全整体替换为按机器的反向桶。
维护成本：`_mark_ready`/`_discard_ready` 按该工序的 eligible 机器集合增删
（`eligible_machine_ids` 已是 runtime 静态表，`core/sim_runtime.py:162-165` 构建），
每工序 O(可用机台数)（本实例均值约 22、上限 ~144），全程约 10⁶ 次 O(1) 集合操作，可忽略。

改动 1（替换）：`core/simulator.py`，`Simulator.__init__` 第 73 行：

Before：

```python
        self._ready_by_type: dict[str, set[str]] = {}
```

After：

```python
        # 机器→就绪工序反向桶：派工只扫"本机可加工且当前就绪"的工序，
        # 不再按工艺类型整桶扫描后逐候选过滤 eligible（大实例主热点）。
        self._ready_by_machine: dict[str, set[str]] = {}
```

改动 2（替换）：`core/simulator.py`，`_bind_runtime` 第 103 行：

Before：

```python
        self._ready_by_type = {}
```

After：

```python
        self._ready_by_machine = {}
```

改动 3（替换）：`core/simulator.py`，`_mark_ready`/`_discard_ready`，锚定 463-474：

Before（逐字）：

```python
    def _mark_ready(self, op: Operation, ready_ops: set[str]) -> None:
        op.status = OpStatus.READY
        ready_ops.add(op.id)
        for type_id in self._op_dispatch_type_ids.get(op.id, {op.process_type}):
            self._ready_by_type.setdefault(type_id, set()).add(op.id)

    def _discard_ready(self, op_id: str, process_type: str, ready_ops: set[str]) -> None:
        ready_ops.discard(op_id)
        for type_id in self._op_dispatch_type_ids.get(op_id, {process_type}):
            bucket = self._ready_by_type.get(type_id)
            if bucket is not None:
                bucket.discard(op_id)
```

After：

```python
    def _mark_ready(self, op: Operation, ready_ops: set[str]) -> None:
        op.status = OpStatus.READY
        ready_ops.add(op.id)
        # 按"可用机台"入桶：显式指定 eligible_machine_ids 的工序会进入多台机器、
        # 甚至多个类型的桶（sim_runtime.py:166-169 的跨类型语义原样保留）。
        for machine_id in self._eligible_machine_ids.get(op.id, ()):
            self._ready_by_machine.setdefault(machine_id, set()).add(op.id)

    def _discard_ready(self, op_id: str, process_type: str, ready_ops: set[str]) -> None:
        # process_type 形参保留（调用点签名不变），桶维护改走 eligible 机器集合。
        ready_ops.discard(op_id)
        for machine_id in self._eligible_machine_ids.get(op_id, ()):
            bucket = self._ready_by_machine.get(machine_id)
            if bucket is not None:
                bucket.discard(op_id)
```

改动 4（替换）：`core/simulator.py` 派工循环，锚定 193-199：

Before：见上方根因摘录。

After：

```python
                # 反向索引：候选 = 本机私有就绪桶（桶内工序必然本机可加工，
                # 原 199 行 eligible 成员判断随之删除）；list() 复制保留，
                # 因为循环体内 197 行仍会 _discard_ready 清理失效条目。
                candidate_ids = list(self._ready_by_machine.get(machine.id, ()))
                for op_id in candidate_ids:
                    op = shop.operations.get(op_id)
                    if not op or op.status != OpStatus.READY:
                        self._discard_ready(op_id, machine.type_id, ready_ops)
                        continue
```

（200 行起 `_earliest_feasible_start` 及其后全部不变。）

为何这样改：派工扫描量从 O(类型就绪桶) 降到 O(本机实际候选)；
最优候选由 `(score, -work_remaining, op.id)` 全序决胜（`simulator.py:221-224`），
与候选遍历顺序无关 ⇒ 排产结果逐字节不变（语义零改变的硬保证）。

### 5.3（P3-2）`_trigger_idle_dispatches` 按类型直查机器

根因：`core/simulator.py:522-527`（逐字）——每次有新工序就绪，遍历全部 1,149 台机器：

```python
# core/simulator.py:522-527（Before，逐字）
        for machine in shop.machines.values():
            if machine.state != ResourceState.IDLE:
                continue
            if process_types and machine.type_id not in process_types:
                continue
            self._schedule_dispatch(event_queue, machine.id, now)
```

`ShopFloor._machine_by_type` 索引已存在（`core/models.py:476` 声明、488-494 构建），
访问器 `get_machines_for_type`（`core/models.py:838-839`）现成。
注意事件顺序等价性：原实现按 `shop.machines` 插入序调度；改为按类型分桶后，
同 `now` 的 dispatch 事件入堆顺序可能变化（`process_types` 是 set，迭代序不稳定），
极端情况下会影响同刻派工先后 ⇒ 必须按原全局机器序重排，保证语义逐字节不变。

改动（替换）：`core/simulator.py`，锚定 515-527 整方法：

Before（逐字）：

```python
    def _trigger_idle_dispatches(
        self,
        shop: ShopFloor,
        event_queue: list[Event],
        now: float,
        process_types: set[str] | None = None,
    ) -> None:
        for machine in shop.machines.values():
            if machine.state != ResourceState.IDLE:
                continue
            if process_types and machine.type_id not in process_types:
                continue
            self._schedule_dispatch(event_queue, machine.id, now)
```

After：

```python
    def _trigger_idle_dispatches(
        self,
        shop: ShopFloor,
        event_queue: list[Event],
        now: float,
        process_types: set[str] | None = None,
    ) -> None:
        if process_types:
            # 按类型索引直查相关机器，不再全量扫描 shop.machines（1,149 台 × 每 op_done）。
            machines = [
                machine
                for type_id in process_types
                for machine in shop.get_machines_for_type(type_id)
            ]
            # 按全局机器序重排：保证同刻 dispatch 事件入堆顺序与旧的全表扫描一致，
            # 排产结果不因索引化而改变。
            machines.sort(key=self._machine_sort_key)
        else:
            machines = shop.machines.values()
        for machine in machines:
            if machine.state != ResourceState.IDLE:
                continue
            self._schedule_dispatch(event_queue, machine.id, now)
```

配套新增 1：`core/simulator.py`，`_bind_runtime` 内（锚定第 104 行
`self._dispatch_scheduled_at = dict.fromkeys(shop.machines, None)` 之后）插入一行：

```python
        self._machine_order = {machine_id: index for index, machine_id in enumerate(shop.machines)}
```

配套新增 2：`core/simulator.py`，`_trigger_idle_dispatches` 方法之后新增小方法：

```python
    def _machine_sort_key(self, machine) -> int:
        return self._machine_order.get(machine.id, 1 << 30)
```

同时在 `__init__`（锚定第 74 行 `self._dispatch_scheduled_at` 声明附近）加声明
`self._machine_order: dict[str, int] = {}`，保持与 `_bind_runtime` 对称。

为何这样改：1.9×10⁷ 次全机扫描 → 仅相关类型机器（每次数十~数百台）；
排序键恢复原事件顺序，语义零改变。

### 5.4（P3-3）`_features` 静态键缓存 + `prereq_ratio` 短路

根因：`core/simulator.py:652-700`（节选逐字）：

```python
# core/simulator.py:672-680 —— 派工路径上工序必已就绪（前驱全部完成），两个循环结果恒为 prereq_done == prereq_total
        prereq_done = 0
        prereq_total = len(op.predecessor_tasks) + len(op.predecessor_ops)
        if prereq_total > 0:
            for predecessor_task_id in op.predecessor_tasks:
                if predecessor_task_id in self._completed_tasks:
                    prereq_done += 1
            for predecessor_op_id in op.predecessor_ops:
                if predecessor_op_id in self._completed_ops:
                    prereq_done += 1
```

17 个键中 10 个静态（`processing_time/due_date/external_due_date/task_due_date/op_due_date/
priority/is_main/tooling_demand/personnel_demand/critical_slack`），
4 个随 `now` 变（`slack/urgency/wait_time`），3 个随进度/机器变（`remaining/progress/machine_busy_time`）。
11 条内置规则只读 features（`core/rules.py:25-78`），缓存静态键对规则语义透明；
唯一约束是 `compile_rule_from_code` 的进化规则可能读任意键——17 键一个不能少。

改动 1（新增）：`core/simulator.py`，`__init__` 锚定第 89 行
`self._flow_gate_cache: dict[str, float] = {}` 之后插入：

```python
        # 每工序静态特征缓存（due/priority/工时等不随仿真时刻变化的 10 个键）；
        # 动态键（slack/urgency/wait_time/remaining/progress/machine_busy_time）仍每次现算。
        self._static_features: dict[str, dict] = {}
```

并在 `_bind_runtime`（锚定第 112 行 `self._flow_gate_cache = {}` 之后）插入：

```python
        self._static_features = {}
```

改动 2（替换）：`core/simulator.py`，`_features` 锚定 652-700 整方法：

Before（逐字全量）：

```python
    def _features(self, op, task, order, machine, shop, now: float) -> dict:
        external_due = task.due_date if task else (order.due_date if order else 9999.0)
        due = op.derived_due_date if op and op.derived_due_date < float("inf") else external_due
        if task:
            remaining = self._task_remaining_work.get(task.id)
            if remaining is None:
                remaining = task.remaining_time
            total_ops = self._task_total_ops.get(task.id, 0) or len(task.operations)
            progress = (
                (total_ops - self._task_remaining_ops.get(task.id, total_ops)) / total_ops
                if total_ops > 0
                else 0.0
            )
        else:
            remaining = op.work_remaining
            progress = 0.0
        slack = due - now - remaining
        priority = order.priority if order else 1
        release_time = self._flow_ready_time(shop, op)

        prereq_done = 0
        prereq_total = len(op.predecessor_tasks) + len(op.predecessor_ops)
        if prereq_total > 0:
            for predecessor_task_id in op.predecessor_tasks:
                if predecessor_task_id in self._completed_tasks:
                    prereq_done += 1
            for predecessor_op_id in op.predecessor_ops:
                if predecessor_op_id in self._completed_ops:
                    prereq_done += 1

        return {
            "slack": slack,
            "remaining": remaining,
            "processing_time": op.work_remaining,
            "due_date": due,
            "external_due_date": external_due,
            "task_due_date": task.due_date if task else external_due,
            "op_due_date": due,
            "urgency": max(0.0, -slack),
            "progress": progress,
            "priority": priority,
            "is_main": 1.0 if (task and task.is_main) else 0.0,
            "wait_time": max(0.0, now - release_time),
            "prereq_ratio": prereq_done / prereq_total if prereq_total > 0 else 1.0,
            "machine_busy_time": machine.total_busy_time,
            "tooling_demand": float(len(op.required_tooling_types)),
            "personnel_demand": float(len(op.required_personnel_skills)),
            "critical_slack": op.critical_slack if op else float("inf"),
        }
```

After（全量）：

```python
    def _features(self, op, task, order, machine, shop, now: float) -> dict:
        static = self._static_features.get(op.id)
        if static is None:
            external_due = task.due_date if task else (order.due_date if order else 9999.0)
            due = op.derived_due_date if op and op.derived_due_date < float("inf") else external_due
            static = {
                # READY 态下 work_remaining 不变；派工后工序离开就绪桶，不会再被评估
                "processing_time": op.work_remaining,
                "due_date": due,
                "external_due_date": external_due,
                "task_due_date": task.due_date if task else external_due,
                "op_due_date": due,
                "priority": order.priority if order else 1,
                "is_main": 1.0 if (task and task.is_main) else 0.0,
                "tooling_demand": float(len(op.required_tooling_types)),
                "personnel_demand": float(len(op.required_personnel_skills)),
                "critical_slack": op.critical_slack if op else float("inf"),
            }
            self._static_features[op.id] = static
        if task:
            remaining = self._task_remaining_work.get(task.id)
            if remaining is None:
                remaining = task.remaining_time
            total_ops = self._task_total_ops.get(task.id, 0) or len(task.operations)
            progress = (
                (total_ops - self._task_remaining_ops.get(task.id, total_ops)) / total_ops
                if total_ops > 0
                else 0.0
            )
        else:
            remaining = op.work_remaining
            progress = 0.0
        slack = static["due_date"] - now - remaining
        release_time = self._flow_ready_time(shop, op)
        features = dict(static)
        features.update({
            "slack": slack,
            "remaining": remaining,
            "urgency": max(0.0, -slack),
            "progress": progress,
            "wait_time": max(0.0, now - release_time),
            # 能进入就绪桶的工序前驱必已全部完成，原统计循环恒为 1.0，直接短路
            "prereq_ratio": 1.0,
            "machine_busy_time": machine.total_busy_time,
        })
        return features
```

为何这样改：17 键全量保留、键值语义逐项一致 ⇒ 内置与进化规则行为不变；
每候选省掉 10 个键的重复推导与前驱统计循环，只剩一次浅拷贝 + 7 个动态键。

### 5.5（P3-4）hybrid 精确阶段：预算参数指引 + 后端可控 + 进度可见

实测验证配置（本诊断使用，也作为大实例推荐下限）：

```json
POST /api/optimize/hybrid
{"objective_keys": ["total_tardiness", "makespan"], "time_limit_s": 120, "coarse_time_ratio": 0.45,
 "population_size": 8, "generations": 2, "target_solution_count": 6, "parallel_workers": 0, "seed": 42}
```

结论（实测 + 代码双重确认）：四张卡片字段是优化器快照的如实转发
（`api/server.py:3078-3080` 逐字拷贝 `_snapshot_status`，非进度造假）；实测轮询（每 120s）：
`approx 1→9`（coarse，~6min）→ `exact 0→1`（exact_promotion，~48min 不动后 +1）→ done。
三个机制性事实全部实锤：
① 计数只在单次评估完成后递增（`hybrid_nsga3_alns.py:1069-1070, 1100-1101, 983-984`），
单候选仿真 48–56 分钟 ⇒ 卡片长时间不动是机制使然；
② 默认 `time_limit_s=90` 下，粗搜截止于 `time_limit×0.68`（`:1877`），
基线仿真（一次完整 DES，实测 ~48–56min）又吃掉精确阶段预算，`_budget_limited_exact_count`（`:1500-1514`）
把晋升截到 0–1 个、精修整段跳过 ⇒ 本次运行 `exact_evaluations=1`、晋升 0、精修 0、
result 仅 baseline 一解——"优化正常结束但实际没有优化"；
③ `exact_promotion` 阶段快照的 `feasible_ratio=1.0` 来自近似解群体（`:1958,1977` 传参为
`promotions or population`，均为近似评估解），不代表精确可行性——前端卡片应注明。
另实测：粗搜阶段单次近似评估 ~40–60s（`approx_eval.py:362-402` 的 beam×机台联合日历探测），
`coarse_deadline` 只能截断"不再开新代"，无法中断已提交批次。

改动 1（新增+替换）：API 暴露 `parallel_backend`（进程池不可用时用户可强制线程后端排查）。

`api/server.py`，`HybridOptimizeReq`（锚定 252-271）在最后一行 `baseline_rule_name: str = "ATC"` 之后新增：

```python
    # "process": 进程池绕开 GIL（默认）；"thread": 线程池（排查进程池故障时用）。
    parallel_backend: str = "process"
```

`api/server.py`，优化器构造（锚定 3003-3025）在 `baseline_rule_name=req.baseline_rule_name,` 之后新增一行：

```python
                    parallel_backend=req.parallel_backend,
```

改动 2（参数指引，非代码）：大实例必须显式放大预算，估算式——
`time_limit_s ≥ 单仿真秒数 × (1 + 期望精确解数 ÷ 并行度) ÷ (1 − coarse_time_ratio)`。
以单仿真 S=600s、期望 8 个精确解、并行 6、coarse 0.45 为例：≥ 600×(1+1.33)/0.55 ≈ 2,545s。
同时建议 `coarse_time_ratio` 降到 0.45（粗搜近似评估在大实例也不便宜，见
`approx_eval.py:362-402` 的 beam×机台联合日历探测）。

改动 3（可选，进度可见性）：让卡片区分"没在跑"与"在跑单个长评估"。
`hybrid_nsga3_alns.py` `_parallel_evaluate_candidates_exact`（1006-1125）提交批次时维护
`self._in_flight_exact`（submit 时 +1、future 完成时 -1），`_snapshot_status`（1689-1714）
快照加 `"in_flight_exact": self._in_flight_exact`；`api/server.py` `_progress`（3048-3097）
加一行 `task["in_flight_exact"] = snapshot.get("in_flight_exact", 0)`，
status 端点（3159-3199）返回体加同名字段；前端卡片（`app_v2.js:577-580`）
在精确评估计数后追加 `(在跑 N)`。该条不改变任何算法行为，可最后落地。

验证方式（P3-1/2/3/4 通用）：
1. 语义等价（硬性）：实施后用本实例或 `tests/shop_fixtures.py` 夹具，同一 ATC 规则跑
   优化前/后仿真，逐项比对 `makespan/total_tardiness/event_count` 与完整 gantt
   （`op_id, machine_id, start, end` 四元组序列）——必须完全一致
   （基线已存于 `C:\Users\z00426527\AppData\Local\Temp\llm4drd_e2e_out\sim_result.json`
   （Git Bash 视图 `/tmp/llm4drd_e2e_out/`）；若已被系统清理，用当前代码重跑一次 ATC 仿真重新生成）。
2. 回归测试：`LLM_API_KEY= LLM4DRD_DB=<临时> python -m pytest tests/ -x -k
   "sim_runtime or simulator or flow_gate or turnover or shift or parallel_eval"` 全绿。
3. 计时对照：同实例 ATC 仿真墙钟 before/after（填回 §1.4 与【待补#1】）。
   预期：P3-1+P3-2 把候选评估从 10⁶–10⁷ 降到 10⁵–10⁶、全机扫描从 1.9×10⁷ 降到 ~10⁵，
   仿真墙钟应降一个数量级（分钟级 → 秒级~十秒级）；P3-3 再削 features 常量成本。
4. 优化链路：短时长 hybrid 复跑，`approximate_evaluations` 持续增长、
   `exact_evaluations ≥ 2`、`feasible_ratio > 0`、result 含 ≥1 个 `feasible: true` 解。

实施顺序与依赖：P3-1 与 P3-2 相互独立，但都改 `core/simulator.py`，同批落地并一起跑等价性验证；
P3-3 依赖 P3-1（候选数降下来后收益才准确，且两者同函数域，避免交织评审）；
P3-4 独立于核心引擎，随时可落；hybrid 真实时长复测必须在 P3-1~3 之后（单仿真降速后才可行）。

---

## 6. 总体实施路线（推荐落地顺序）

| 批次 | 内容 | 风险 | 说明 |
|---|---|---|---|
| 批次 1（跑通性） | P0-1 导入响应去 `details`（1 行）；P0-2 exact-reference 阈值 + `async def→def`（约 15 行）；P0-3 details/db lite 化 + 前端调用点（约 25 行）；P0-4 export-solution `async def→def`（1 行） | 低 | 立即消除"浏览器卡死"与"后端整体卡死"四类致命点；P0-1/P0-3 同文件建议同提交 |
| 批次 2（前端 A+B） | 改动 1–8 同批：`frontend/app_v2.js`（CONFIG/状态/辅助函数/buildGanttData/renderTimeline/mountGantts/两个事件委托）+ `frontend/app_v2.css` | 中 | 纯前端、纯客户端过滤，不动数据契约；引用关系要求整批一次落地；按 A-验证 1–4 与 B-验证 1–5 逐项过 |
| 批次 3（仿真内核） | P3-1 反向索引 + P3-2 类型直查（同文件同批），跑语义等价验证；随后 P3-3 features 缓存 | 中高 | 触及 `core/simulator.py` 派工路径；等价性验证（gantt 四元组逐字节一致）是通过红线，不过则回退 |
| 批次 4（优化链路） | P3-4 改动 1（`parallel_backend` 透传）+ 参数指引进文档；可选改动 3（在跑计数） | 低 | 不改算法；随后用大预算配置做 hybrid 真实时长复测 |
| 批次 5（复测） | 全链路真实数据复测：导入→仿真→优化→评审；更新本文档 §1.4 为 after 列 | — | 四步产物齐全：实例入库、gantt+KPI、≥1 可行优化解、评审可展示 |

重点评审项：批次 3 的 `core/simulator.py` 三处替换（463-474、193-199、515-527）与
`_features` 重写——必须逐字对照本文档 Before 块，落地后先跑等价性验证再合并；
批次 2 的 `renderTimeline`/`buildGanttData` 为整函数替换，评审时对照逐字 Before 块防误伤。

---

## 7. 遗留风险与待决策事项（不纳入本次改动，但需用户/编码模型知晓）

• R-1 `/api/simulate/reference-solutions` 大实例不可用：串行跑 6 条内置规则 × 完整 DES

  （`api/server.py:2848-2858` 调 `_simulate_locked` 同款路径），本规模下 ≈6×单仿真墙钟。
  评审页"启发式参考方案"在本实例上应视为不可用。待决策：是否改为可传规则子集 / 后台任务化。
• R-2 `GET /api/instance/details` 体积：已实测 133s/394MB 并升级为 P0-3（见第 2 章），

  本行仅为索引保留。
• R-3 `POST /api/exact/solve` 同款规模保护缺失（`api/server.py:3588-3697`）：后台线程不阻塞

  事件循环，但建模仍会吃内存/CPU。建议复用 P0-2 的 `EXACT_REFERENCE_MAX_OPERATIONS` 阈值。
• R-4 进程池在 uvicorn 下的实际可用性：`parallel_eval.py` 依赖 spawn 子进程重跑主模块

  完成包注册；`_submit_to_pool`（`hybrid_nsga3_alns.py:957-968`）注释表明作者遇到过
  daemonic AssertionError 并做了线程池降级（GIL 下精确评估≈串行）。实施后务必查日志确认
  无 `hybrid: process pool unavailable` 字样；本次诊断用 launcher.py 也是为规避此问题。
• R-5 `_sim_runtime_cache_key` 每次仿真前全量排序 16,536 工序（`api/server.py:79-96`）：

  相对单仿真墙钟是小头，但 P3-1~3 落地后仿真变快，此项占比会上升，届时再评估。
• R-6 导入后前端自动 `handleBuildGraph()`（`app_v2.js:4807`）：16,536 工序建图走

  `/api/graph/build` 异步任务，图规模可能触发 `LLM4DRD_GRAPH_WARN_EDGES`（默认 30 万）；
  非本次四步主流程阻断点，未测。待决策：大实例是否默认跳过自动建图。
• R-7 筛选/翻页触发 `renderCurrentPage()` 整页重渲：与订单下拉同模式，大实例下

  整页重渲约百毫秒级，可接受；若未来卡顿，再考虑只重渲甘特卡片。
• R-8 进化规则沙箱：`compile_rule_from_code` 使用 `exec`（`core/rules.py:85-91`），

  本次未涉及自定义规则上传；生产化前必须加来源校验（AGENTS.md §9 同旨）。
• R-9 已升级为 P0-4（export-solution 241s 阻塞事件循环，见第 2 章），本行仅为索引保留。


---

## 8. 附：本诊断已确认、无需改动的正确行为（避免误修）

1. 甘特条块时间轴：`ganttOffsetToISO`（`app_v2.js:1383-1385`）与后端 `start/end` 偏移小时一致，
   vis 原生缩放/平移不影响时刻准确性。
2. 三态着色：后端 `status` → `normalizeScheduleStatus` → `status-*` className → CSS 渐变
   （`app_v2.css:2226-2228`），链路完整。
3. 停机/班次遮罩 CSS 斜纹带（`app_v2.css:2235-2237`）与图例（`app_v2.css:1583-1595`）视觉一致，
   缺的只是悬停 title（已在 A-改动 4 补上）。
4. 优化四卡片字段来源真实（优化器内部计数器快照直转），"不动"是分钟级单评估的机制结果，
   不是进度造假；`stalled` 提示文案（`frontend/optimize_progress.js:35-66`）已为此设计。
5. `_trigger_idle_dispatches` 之外的运行时复用机制（`SimulationRuntime`/`_cached_sim_runtime`/
   `SimulationRuntimePool`/进程池 worker 一次性 runtime）均已存在且正确，不要在性能批次里重写它们。
6. 停机数据双数据源（`machine.downtimes` 优先、`/api/downtime` 回退，
   `app_v2.js:989-999, 1319-1322`）真实来自后端存储，不是"只画图例不画带"——
   旧代码的问题是遮罩只画在幸存的前 40 台上且无悬停，而非数据源造假。




