---
name: whatif-planning
description: 用于排产的"如果……会怎样"推演——在内存里改车间原始数据（加/删机器、改班次、改工时、改交期、加停机），跑一条或多条派工规则，横向对比 KPI。当用户问"加一台车床能不能赶上交期""这批机器只上白班影响多大""M3 停机 4 小时会怎样""订单量涨 20% 扛不扛得住"这类假设性问题时使用。
---

# 排产 What-if 场景推演

在**不改数据库**的前提下，改一份内存里的车间数据副本，跑规则，比 KPI。
所有改动只活在后端进程内存里，服务重启即失；正式实例数据、四步流程结论都不受影响。

## 原子能力

| 工具 | 用途 |
|---|---|
| `search_planning_entities` | 读现状。`entity_type` 可取 order/operation/**machine/machine_type/tooling/personnel** |
| `create_whatif_scenario` | 建场景 → `scenario_id` |
| `apply_whatif_patch` | 唯一的改数据入口，支持 `where` 批量 |
| `describe_whatif_scenario` | 回显改了什么 + 规模变化 + 校验结果 + `apply_token` |
| `revert_whatif_patch` | 撤销最近 N 条改动 |
| `run_whatif_planning` | 在场景上跑 N 条规则 → `run_id` |
| `get_whatif_run` | 轮询（大实例一次仿真几十秒） |
| `compare_whatif_runs` | 跨场景 / 跨规则对比 |
| `apply_whatif_to_instance` | **高危落库**，见下方红线 |

## 标准流程

### 1. 先读现状，再动手

**不要凭空编造机器编号、工艺类型或班次格式。** 用户说"加一台车床"，你要先知道车床的
`type_id` 叫什么、现有车床的班次串长什么样：

```
search_planning_entities(entity_type="machine_type")   → 看有哪些工艺类型
search_planning_entities(entity_type="machine", query="车") → 拿一台现有车床当模板
```

### 2. 建场景，用业务语言命名

`create_whatif_scenario(name="车床加一台")` —— 名字会出现在对比表里，用人话。

### 3. 打 patch

一条 patch = `{op, entity, values}`，字段名与实例导入模板的列名一致。
`update` / `remove` 必须带 `id`、`ids` 或 `where` **三选一**作为选择器。

时间类字段一律是**相对计划基准时刻的偏移小时数**（不是日期）。
`shifts` 是 `"day/start_hour/hours;..."`，例：`"0/8/10;0/20/8"` = 白班 10h + 夜班 8h。

### 4. 回显并等用户确认

`describe_whatif_scenario(scenario_id)` 拿到改动清单与校验结果，用人话复述给用户：

> "已在副本上加了 1 台车床 M20（沿用 M07 的两班制），机器数 19 → 20。校验无问题。
> 要用哪几条规则跑？默认 ATC。"

**`validation.errors` 非空就停下来报告，不要硬跑**——常见于删掉了某工序唯一能上的机器，
这种情况下仿真会输出一堆排不出的工序，KPI 毫无意义。`warnings` 则提一句即可。

### 5. 跑

`run_whatif_planning(scenario_id, rule_names=["ATC","EDD"])`

- `status == "done"` → `results` 里就是结果
- `status == "running"` → 大实例还在跑，用 `run_id` 调 `get_whatif_run` 轮询
- `status == "failed"` → 看 `error`，校验不通过会在这里被拦住

**要做"改动前 vs 改动后"的对比，必须跑两次**：一次在空场景（现状基线）上，一次在改过的
场景上。空场景 = `create_whatif_scenario` 之后不打任何 patch。

### 6. 叙述对比

`compare_whatif_runs(run_ids=[基线, 场景A, ...])`

返回的每个指标带 `better` 字段——**已经按该指标是越小越好还是越大越好判定过了，直接用它，
不要自己猜方向**（利用率越高越好，延误越低越好，很容易讲反）。

叙述要给业务结论，不要只念数字：

> "加这台车床后，总延误从 12.3h 降到 8.1h（-34%），Makespan 从 96h 降到 91h。
> 代价是平均利用率从 82% 降到 76%——多出来的产能没吃满。
> 如果只是为了这批订单赶期，够用；如果是长期扩产，利用率偏低。"

## patch 速查

| 想干什么 | 怎么写 |
|---|---|
| 加一台机器 | `{op:"add", entity:"machine", values:{machine_id, machine_name, type_id, shifts}}` |
| 删一台机器 | `{op:"remove", entity:"machine", id:"M07"}` |
| 某台机器改班次 | `{op:"update", entity:"machine", id:"M07", values:{shifts:"0/8/10"}}` |
| 某类机器全改只上白班 | `{op:"update", entity:"machine", where:{type_id:"turning"}, values:{shifts:"0/8/10"}}` |
| 改工序工时 | `{op:"update", entity:"operation", id:"OP123", values:{processing_time:4.5}}` |
| 改订单交期 | `{op:"update", entity:"order", id:"ORD01", values:{due_date:120}}` |
| 整批订单延期 | `{op:"update", entity:"order", ids:["O1","O2"], values:{due_date:150}}` |
| 加停机窗口 | `{op:"add", entity:"downtime", values:{machine_id:"M07", downtime_type:"maintenance", start_time:48, end_time:56}}` |
| 加人 / 加工装 | `{op:"add", entity:"personnel"\|"tooling", values:{...}}` |
| 改计划基准时刻 | `{op:"update", entity:"planning_context", values:{plan_start_at:"2026-08-01T08:00:00+08:00"}}` |

支持的 `entity`：`order` `task` `operation` `machine` `machine_type` `tooling` `tooling_type`
`personnel` `downtime` `planning_context`。

## 红线

1. **绝不主动调 `apply_whatif_to_instance`。** 它会真写数据库、bump 实例版本号，
   导致校验/仿真/优化/评审四步流程的历史快照全部作废。只有在用户明确、直接要求"把这个改动
   落到正式数据里"时才调，且调之前必须：
   - 用 `describe_whatif_scenario` 拿到完整改动清单和 `apply_token`
   - 把清单**逐条**展示给用户
   - 明确告知"这会作废四步流程的历史结论，需要重跑"
   - 拿到用户的确认答复
2. **打 patch 前先读现状**，不要猜 id、猜工艺类型、猜班次格式。
3. **校验有 error 就停**，报告给用户，不要硬跑出一份没意义的 KPI。
4. **一次只回答一个假设**。用户如果同时问了"加机器"和"加班"两个方案，建两个场景分别跑，
   再一起对比——不要把两种改动混进一个场景，那样分不清是哪个改动起的作用。

## 常见坑

- **场景会过期**：如果推演期间有人改了正式实例数据，工具会返回 `WHATIF_BASE_STALE`。
  这时要重建场景并重新打 patch，不要试图绕过。
- **场景最多留 8 个**，超出后最旧的被淘汰，`WHATIF_SCENARIO_NOT_FOUND` 就是这个原因。
- **删机器要小心**：某工序的 `eligible_machine_ids` 可能只有那一台。校验会拦住，但更好的
  做法是删之前先看看这台机器被哪些工序依赖。
- **加机器不等于加产能**：新机器要能被工序选中，工序的 `eligible_machine_ids` 得包含它。
  如果工序按 `process_type` 匹配机器类型，加同类型机器即可；否则还要同时改工序的可用机器列表。
