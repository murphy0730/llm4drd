# 工序流转等待时间（turnover_time）设计

- 日期：2026-07-16
- 状态：已确认设计，待实施规划
- 适用范围：实例模板、实例持久化、`ShopFloor` 派生时间计算、离散事件仿真、滚动排产、近似评价、精确求解
- 实施顺序：**排在 `2026-07-16-unified-graph-context` 改造（Task 1–8）之后**，见 §7
- 代码基线：本文行号引用基于 commit `1188508`

## 1. 背景

当前模型中，后继工序的开工只受两个条件约束：前驱工序全部 `COMPLETED`（`core/simulator.py:582` `_is_op_ready`，纯布尔判断），以及任务/订单级的 `release_time`（`core/models.py:733` `get_operation_release_time`）。

真实车间里，工件在一道工序加工完成后往往不能立即流转到下一道：铸件要冷却、涂层要固化、胶接要养护、热处理后要自然时效。这段等待既不占用机床，也不消耗人力，但确实推迟了后继工序的最早开工时刻。系统目前无法表达它，只能被迫把这段时间虚增进 `processing_time`——代价是机床被错误地占用了整段等待期，产能被严重低估。

本设计为工序引入 `turnover_time` 字段，并把它作为一个**时间闸门**接入既有的就绪判定路径。

## 2. 目标与非目标

### 2.1 目标

- `operations` sheet 在 `processing_time_hrs` 之后新增 `turnover_time_hrs` 列，并更新 `docs/instance_template.xlsx`。
- `Operation` 模型承载 `turnover_time`，贯通 Excel 导入、SQLite 持久化与实例编辑接口。
- 四条排产路径（离散事件仿真、滚动排产、近似评价、CP-SAT 精确求解）一致地满足：

  ```text
  后继工序.start_time ≥ 前驱工序.end_time + 前驱工序.turnover_time
  ```

- `ShopFloor` 的关键路径、`derived_due_date`、`earliest_start_time` 等派生量计入 turnover。
- `turnover_time = 0` 时，全系统行为与改造前逐工序完全一致。

### 2.2 非目标

- 不引入车间级日历，不让 turnover 参与班次推进（见 §3 口径 1）。
- 不让 turnover 占用机器、工装或人员。
- 不把 turnover 作为派工规则特征暴露给 `core/rules.py` / `ai/evolution.py`（可另行设计）。
- 不改变任何现有优化目标、派工规则或 NSGA-III/ALNS 语义。
- 不实现按工序对（pred, succ）差异化的 turnover；turnover 是**前驱工序的属性**，对其所有后继一致生效。

## 3. 语义口径

四条口径已确认，是本设计的判定基准：

1. **自然时间流逝（wall-clock）**。turnover 不看班次。工序 22:00 完工、turnover=4h，则次日 02:00 即满足流转条件，哪怕车间不排班。实现上直接 `end_time + turnover_time`，**不经过 `_joint_compute_effective_end`**。理由：冷却、固化、干燥、时效是物理过程，不因车间下班而暂停。

2. **任务级前驱同样生效，取末工序 turnover**。后继工序等到 `max(前驱任务各工序 end_time + 各自 turnover_time)`。理由：语义对称——同一道工序被 `predecessor_ops` 引用时要等 turnover，被 `predecessor_tasks` 引用时也必须等，否则同一物理约束会因数据表达方式不同而失效。

3. **机器在 `end_time` 释放**。turnover 期间工件在等，机床不等，可立即接下一活。这正是引入本字段的核心动机（对比「把等待虚增进 `processing_time`」的现状）。`core/simulator.py:437-441` 现有逻辑已在 `now`（即 `end_time`）释放全部资源，**无需改动**。

4. **缺省值 0**。旧 xlsx 不含该列、旧 SQLite 行为 NULL 时一律取 0，行为与今天完全一致。这是零回归的锚点，也是 §8 测试的第一条断言。

## 4. 数据模型变更

### 4.1 Excel 模板

`data/template_builder.py:71` 的 `operations` headers 在 `processing_time_hrs` 与 `predecessor_ops` 之间插入 `turnover_time_hrs`，三行示例数据同步补值（建议给 `OP-0001-01-01` 一个非零值如 `2`，让模板本身即可演示该语义；其余为 `0`）。

改动后的列结构：

| op_id | task_id | op_name | process_type | processing_time_hrs | turnover_time_hrs | predecessor_ops | predecessor_tasks | eligible_machine_ids | required_tooling_types | required_personnel_skills |
|---|---|---|---|---|---|---|---|---|---|---|
| OP-0001-01-01 | T-0001-01 | Turning | turning | 5.5 | 2 | | | turning_1;turning_2 | tool_turning | skill_turning |
| OP-0001-01-02 | T-0001-01 | Milling | milling | 3.2 | 0 | OP-0001-01-01 | | milling_1 | tool_milling | skill_milling |
| OP-0001-ASM | T-0001-MAIN | Assembly | assembly | 6 | 0 | | T-0001-01 | assembly_1 | tool_assembly | skill_assembly |

`turnover_time_hrs` 单位为小时，是相对 `plan_start_at` 小时偏移量体系下的**时长增量**（与既有 `processing_time_hrs` 同口径，不涉及日期解析）。缺省 `0`，允许 `0`，拒绝负值。

`TEMPLATE_VERSION`（`data/template_builder.py:11`）递增。`docs/instance_template.xlsx` 由构建器重新生成，不手工编辑。

### 4.2 Operation 模型

`core/models.py:340` 的 `Operation` 在 `processing_time` 之后新增：

```python
turnover_time: float = 0.0
```

默认值 0 使所有既有构造点（`data/generator.py`、`tests/`、`optimization/`）无需改动即保持原行为。

### 4.3 持久化

`data/db.py` 五处：

1. `inst_operations` 建表（`:149` 附近）增加 `turnover_time REAL DEFAULT 0`。
2. 幂等迁移：比照 `:232` 的既有写法，增加 `_safe_add_column(conn, "inst_operations", "turnover_time", "REAL DEFAULT 0")`，使旧库平滑升级。
3. Excel 导入（`:480` 附近 INSERT，`:492` 附近取值）：比照 `processing_time_hrs` 的 `_float_or_default(row.get("processing_time_hrs", row.get("processing_time", 0)), 0.0)` 写法，读 `turnover_time_hrs`，缺列时落 0。
4. 单工序更新路径（`:569` 的 UPDATE 与 `:576` 的参数列表）增加该字段。
5. 行→`Operation` 加载（`:623`）增加 `turnover_time=_float_or_default(row.get("turnover_time"), 0.0)`，容忍旧库 NULL。

### 4.4 接口与校验

`api/server.py`：

- 工序写入/编辑的字段白名单与 payload 解析（比照 `:1348` 一带对 `processing_time` 的处理）。
- 导入校验（`:1630` 处已有 `processing_time <= 0` 的校验）增补：`turnover_time < 0` 判非法并报错到 `operations` sheet。注意 **turnover 允许为 0，但不允许为负**——与 `processing_time` 必须 `> 0` 的规则不同。
- 工序详情返回体（`:1496` 一带）暴露该字段，供前端展示。

## 5. 算法变更

### 5.1 仿真器：时间闸门（核心）

采用「复用既有 `release_check` 事件机制」的方案。仿真器已有完整的「算出时刻 → 压 `release_check` 事件 → 到点重查就绪」链路（`_queue_release_or_ready` at `core/simulator.py:591`，handler at `:270`），今天仅用于 task/order 的 `release_time`。本设计把闸门从「静态 release_time」泛化为「release_time 与前驱 turnover 的较大者」，事件机制原样复用。

考虑过但否决的替代方案：

- **把 turnover 建模成一道不占资源的虚拟工序**——无需改就绪逻辑，但会污染工序集合，甘特图、KPI 统计、`tools/analyze_unscheduled.py`、图谱视图全需过滤虚拟节点，且 `predecessor_ops` 要重写指向，改动面反而最大。
- **让 `_is_op_ready` 直接返回最早可开工时刻**——调用点少，但布尔语义被破坏，且 `core/models.py:740` `check_op_ready` 有独立的公开布尔语义（`api/server.py` 诊断接口在用），跟改会波及诊断路径。

`_is_op_ready` 保持纯布尔语义不变。新增闸门函数：

```python
def _flow_ready_time(self, shop: ShopFloor, op: Operation) -> float:
    """工序可开工的最早时刻：任务/订单放行 与 前驱流转完成 的较大者。

    仅在 _is_op_ready(op) 为真时有意义——此时全部前驱已 COMPLETED
    且 end_time 落定，返回值即终值，无需重算。
    """
    gate = self._release_time_cache.get(op.id, shop.get_operation_release_time(op))
    for predecessor_id in op.predecessor_ops:
        predecessor = shop.operations.get(predecessor_id)
        if predecessor is not None and predecessor.end_time is not None:
            gate = max(gate, predecessor.end_time + predecessor.turnover_time)
    for predecessor_task_id in op.predecessor_tasks:
        predecessor_task = shop.tasks.get(predecessor_task_id)
        if predecessor_task is None:
            continue
        for task_op in predecessor_task.operations:
            if task_op.end_time is not None:
                gate = max(gate, task_op.end_time + task_op.turnover_time)
    return gate
```

**闸门终值性论证**：`_queue_release_or_ready` 的唯一动态调用点是 `core/simulator.py:429`，位于 `_is_op_ready(next_op)` 为真的分支内，此刻所有前驱的 `end_time` 均已写定（`:414`）且不再变化。故闸门算一次即终值，`_release_checks_scheduled` 的去重语义继续成立。

**必须修正的时钟倒流隐患**：`_queue_release_or_ready:599` 现在判断的是 `if release_time <= 0`——与 `0` 比而非与 `now` 比，因为今天的 `release_time` 是静态的、`<= 0` 即等价于「已过期」。换成动态闸门后，若闸门值小于 `now` 就会向事件队列压入**过去时刻**的事件，`now = event.time`（`:267`）将导致仿真时钟倒流。

推演表明该场景在 `:429` 不会实际触发（闸门取 max 时含当前刚完工的前驱，故恒 `≥ now`），但这个不变量是隐式且脆弱的。本设计要求显式加固：

- `_queue_release_or_ready` 增加 `now: float = 0.0` 形参；
- 判定改为 `if gate <= now: self._mark_ready(...)`；
- 压事件改为 `self._push(event_queue, max(gate, now), "release_check", op_id=op.id)`；
- 调用点 `:257`（初始化，now=0）与 `:429`（now=事件时刻）分别传入。

**其余闸门替换点**（把 `self._release_time_cache.get(op.id, ...)` 换成 `self._flow_ready_time(shop, op)`）：

| 位置 | 用途 |
|---|---|
| `core/simulator.py:275` | `release_check` handler 的到点判定 |
| `core/simulator.py:479` | 热启动时对 `READY` 工序的复核（`<= 0` 同样需改为与闸门比较） |
| `core/simulator.py:598` | `_queue_release_or_ready` 闸门取值 |
| `core/simulator.py:719` | 派工 probe 的最早可开工时刻 |
| `core/simulator.py:748` | `wait_time` 特征计算 |

### 5.2 ShopFloor 派生时间

`core/models.py`：

- **关键路径前推**（`:631-637`）：`offset` 取 `earliest_offsets[pred] + pred.processing_time + pred.turnover_time`；`critical_path` 仍取 `offset + op.processing_time`（末工序的 turnover 不延长任务关键路径本身——它只约束**跨任务**的后继，而跨任务约束由 `task_predecessors` 边在 `:672` 的 `earliest_finish_time` 传递链上体现）。

  > **展开点**：`:672` 的任务级前推取 `predecessor.earliest_finish_time`，而任务的 `earliest_finish_time = earliest_start_time + critical_path_time` 不含末工序 turnover。为满足口径 2，`task_meta` 需额外记录 `max(末工序 offset + processing_time + turnover_time)` 口径的 `critical_path_with_turnover`，供任务级前推使用。

- **反推 `derived_start_time`**（`:699`）：`op.derived_due_date` 由后继的 `derived_start_time` 反推时，需扣掉本工序的 turnover——后继要在 `derived_start_time` 开工，则本工序必须在 `derived_start_time - turnover_time` 前完工。即 `:692-698` 的 `successor_starts` 取值改为 `successor.derived_start_time - op.turnover_time`。

- **`check_op_ready`**（`:740`）：保持纯布尔语义不变。

- **`get_ready_ops`**（`:751`）：**不改**。已勘查确认它在全仓**零调用点**（`grep -rn "get_ready_ops"` 仅命中定义自身），属既有死代码。它内部的 `release_time <= now` 判定不感知 turnover，但因无调用方而不构成正确性风险。本设计不删除它（属既有代码，超出本次请求范围），仅在此注明：**若将来复活该方法，必须先接入 §5.3 的共享闸门函数**，否则会成为第四条与其余路径语义不一致的排产入口。

### 5.3 滚动排产

`scheduling/online.py:116`（probe）、`:141`（就绪筛选）、`:319`（release_time 取值）与仿真器采用同一闸门语义。

因该模块持有 `self.sim_shop` 而非复用 `Simulator` 实例，闸门函数需提取为可共享的模块级函数（建议置于 `core/models.py`，或在职责已显拥挤时新增 `core/precedence.py`），避免两处实现漂移——这正是本设计要防的语义分裂。`Simulator._flow_ready_time` 退化为对该共享函数的薄封装（保留 `_release_time_cache` 的缓存优势）。

> **前车之鉴**：`_joint_compute_effective_end` 目前在 `core/simulator.py:892`、`optimization/approx_eval.py:34`、`data/db.py:812` 存在**三份独立副本**。这是「按引擎各写一份时间语义」的既有后果，也正是闸门必须单点实现的直接理由。本设计不重构这三份副本（属既有代码，超出本次请求范围），但明确要求 turnover 闸门不得重蹈此覆辙。turnover 按自然时间口径不经过该函数，故本次改动不受其影响。

> **展开点**：`:472-473` 在裁剪 remaining shop 时会过滤掉不在窗口内的前驱引用，需验证被过滤前驱的 turnover 约束已在窗口边界条件中体现，否则跨窗口的 turnover 会丢失。

### 5.4 近似评价

`optimization/approx_eval.py`：`:239-242` 与 `:325-328` 两处从 `predecessor_completion` / `task_completion` 取 `base_ready`，改为取值时加上对应工序的 `turnover_time`。`:78` 的 release_time 预计算保持不变（它只覆盖 task/order 放行）。

### 5.5 精确求解

`optimization/exact.py:281-294` 是最干净的一处，直接改约束：

```python
model.Add(start_var >= predecessor_end + turnover_scaled)
```

`turnover_scaled = int(round(pred.turnover_time * scale))`，与 `:165` 既有的 `scale` 取整口径一致。工序级（`:283-286`）与任务级（`:287-294`）两个循环都要改。**取整方向需与仿真器一致**，否则两引擎会在边界算例上给出不同结论（§8 用例 5 即为此设）。

### 5.6 图谱

`knowledge/graph.py:162` 的 OP 节点在 `processing_time` 旁并列 `turnover_time=op.turnover_time`。

因实施顺序排在 graph 改造之后，届时该逻辑已迁入 `CanonicalGraphBuilder`（见 §7）。

## 6. 不在本期改动

- `core/rules.py` / `ai/evolution.py` 的派工特征集不引入 turnover。
- `optimization/solution_model.py:15` 的特征列表不变。
- `frontend/app_v2.js:1707` 的特征标签映射不变（该映射服务于派工特征展示，非工序属性展示）。工序详情面板若需展示 turnover，属前端小改，随 §4.4 的接口返回体一并处理。

## 7. 与 unified graph context 改造的关系

已确认：**graph 改造（`2026-07-16-unified-graph-context`，Task 1–8）先行，turnover 在其之后实施**。

该顺序的代价是明确的，必须在实施时偿付而非事后补救：

1. **Task 1 冻结的基线会被作废**。Task 1「Freeze Legacy Behavior and Add Deterministic Fixtures」固化了 golden 实例、characterization 测试与性能基线。turnover 修改 `operations` schema 后：
   - golden fixtures 需重新生成；
   - 每个 golden 实例的 `instance_hash` 改变；
   - `docs/benchmarks/graph-context-baseline.json` 需重跑；
   - characterization 测试中涉及 operations 列的断言需更新。

2. **`builder_version` 必须 bump**（graph design §4.1）。turnover 进入 `CanonicalGraph` 的节点属性即构成构建器语义变化，按 graph design §7.4 的保守失效规则，所有既有缓存失效并重建——这是预期行为，不是缺陷。

3. **fingerprint 输入需修订**（graph design §7.2/§7.3）：
   - `feature_hash` 输入清单（§7.3）已列「加工时间」，须并列补入「流转等待时间」；
   - `instance_hash` 覆盖「完整规范化实例」，自动含该字段，无需修订；
   - `topology_hash`（§7.2）**不含** turnover——它不改变任何边的存在性，只改变边的时间权重。

4. **`CanonicalGraphBuilder` 与 `DisplayGraphProjection`** 需承载 §5.6 的节点属性，替代直接改 `knowledge/graph.py`。

> **实施前必须重新勘查**：本设计的行号引用基于 commit `1188508`。graph 改造落地后，`knowledge/graph.py` 将被降级为兼容适配器（graph plan Task 2 Step 5），`data/db.py` 会新增 `graph_context_*` 系列表（Task 4）。§5.6 与 §4.3 的落点届时需重新核对。其余各节（仿真器、exact、approx_eval、models）位于 graph 改造声明的非目标范围内（graph design §2.2「不修改 Simulator 的离散事件推进核心」），预期不受影响，但仍需以实际代码为准。

## 8. 测试设计

以 `tests/test_simulator_robustness.py:34` 既有的 shop 构造 helper 为基础扩展（该 helper 的 `ops_spec` 签名需增加 turnover 维度）。

| # | 用例 | 断言 |
|---|---|---|
| 1 | **零回归**：turnover 全为 0 | 全量 `schedule` 与改造前逐工序 `start_time` / `end_time` 完全一致 |
| 2 | **工序级前驱** | 两道串行工序，turnover=3 → `succ.start_time >= pred.end_time + 3` |
| 3 | **任务级前驱** | 前驱任务末工序 turnover=5 → 装配工序被正确推迟；多末工序时取 max |
| 4 | **资源不被占用** | turnover 期间第三道工序可占用同一机器 → 证明口径 3 |
| 5 | **双引擎一致** | 同一小算例下 `exact.py` 与仿真器结论一致 → 防取整/语义漂移 |
| 6 | **跨自然时间边界** | 工序在班次末完工、turnover 跨越非排班时段 → 后继在下个班次开头即可开工，turnover 未被班次拉长 → 证明口径 1 |
| 7 | **initial_state 已完工工序** | `end_time` 为负或 0 的历史完工工序，其 turnover 仍生效；闸门 ≤ 0 时后继立即就绪 |
| 8 | **旧数据兼容** | 不含 `turnover_time_hrs` 列的 xlsx 导入成功且全部落 0；旧 SQLite 库经迁移后 NULL 读作 0 |
| 9 | **校验** | `turnover_time < 0` 被导入校验拒绝；`= 0` 被接受 |
| 10 | **近似评价一致** | `approx_eval` 的 `base_ready` 与仿真器实际 `start_time` 在含 turnover 算例上不矛盾 |

## 9. 完成定义

- `operations` sheet 含 `turnover_time_hrs`，位于 `processing_time_hrs` 之后；`docs/instance_template.xlsx` 由构建器重新生成。
- 四条排产路径均满足 `后继.start_time ≥ 前驱.end_time + 前驱.turnover_time`，任务级前驱同样满足。
- turnover=0 时全系统零回归，有测试证明。
- 旧 xlsx 与旧 SQLite 库平滑兼容，缺失值读作 0。
- turnover 期间资源可被其他工序占用，有测试证明。
- turnover 按自然时间流逝，不被班次拉长，有测试证明。
- 闸门逻辑单点实现，仿真器与滚动排产不存在两份副本。
- `_queue_release_or_ready` 的时钟倒流隐患已加固（判定与 `now` 比较，压事件取 `max(gate, now)`）。
- graph 改造的 `builder_version` 已 bump，fingerprint 输入清单已修订，golden fixtures 与 benchmark 基线已重跑。
