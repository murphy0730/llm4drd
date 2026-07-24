# 方案评审 KPI 计算口径说明

本文档列出方案评审界面涉及的全部 KPI 计算公式，并说明各模块（KPI 对比表、机器分类利用率对比、每日利用率表、排程甘特图）之间的口径对应关系。

所有实现位置以 `文件:行号` 标注，修改指标算法时请同步更新本文档。

---

## 一、数据基础：排程条目字段语义

所有 KPI 都建立在仿真输出的排程条目（schedule entry）之上。条目由 `core/simulator.py:276-291`、`core/simulator.py:436-448` 生成，关键字段：

| 字段 | 含义 | 是否含班次外间隔 |
|---|---|---|
| `start` | 工序开始的墙钟偏移小时（相对计划起点 0） | — |
| `end` | 工序结束的墙钟偏移小时 | — |
| `duration` | **净加工时长**，不含班次外间隔 | 否 |
| `elapsed_duration` | `= end - start`，**墙钟占用时长** | 是 |

**核心区别**：一道工序若跨越班次（例如 8h 班次机器上一道 16h 的活），机床从 D0 08:00 开始，到 D1 16:00 结束。此时：

```
start = 8, end = 40
duration          = 16   （实际切削 16 小时）
elapsed_duration  = 32   （机床被占用 32 小时，中间 16 小时是下班时间，工件仍在机床上）
```

由此派生出三个不同的"时长口径"，全文档统一使用以下术语：

| 术语 | 取值 | 含义 |
|---|---|---|
| **净加工时长** | `duration` | 资源实际执行加工的时间 |
| **墙钟占用时长** | `elapsed_duration`（缺省回退 `end - start`） | 资源被这道工序占住、无法接其它活的时间 |
| **可用工时** | `Machine.available_time_between(lo, hi)`（`core/models.py:281`） | 时间窗内该机器的班次窗口时长，已扣除停机 |

---

## 二、优化目标 KPI（22 项）

定义表：`optimization/objectives.py:42-169`（`OBJECTIVE_SPECS`）
计算实现：`optimization/objectives.py:247` `build_schedule_analytics()`

### 2.1 符号约定

| 符号 | 定义 | 代码位置 |
|---|---|---|
| $C_j$ | 任务 $j$ 的完工时间 | `task_completion` |
| $d_j$ | 任务 $j$ 的交期 | `task.due_date` |
| $r_j$ | 任务 $j$ 的投放时间 | `task.release_time` |
| $C^{ord}_o$ | 订单 $o$ 完工时间 $= \max_{j \in o} C_j$ | `order_completion` |
| $T_{max}$ | 全局完工时刻 $= \max_o C^{ord}_o$ | `max_end`（`objectives.py:372`） |
| $M$ | 全部机器集合 | `shop.machines` |
| $M_c$ | 关键机器集合（`MachineType.is_critical`） | `shop.get_critical_machines()` |
| $b_m$ | 机器 $m$ 的**净加工**累计 $= \sum duration$ | `machine_busy`（`objectives.py:281`） |
| $o_m$ | 机器 $m$ 的**墙钟占用**累计 $= \sum elapsed\_duration$ | `machine_occupied`（`objectives.py:282`） |
| $[s_m, e_m]$ | 机器 $m$ 的活跃窗口 $=[\min start, \max end]$ | `machine_first_start/last_end` |
| $A_m(lo,hi)$ | 机器 $m$ 在 $[lo,hi]$ 内的可用工时 | `available_time_between` |

### 2.2 交期类指标

| KPI key | 中文标签 | 公式 | 方向 |
|---|---|---|---|
| `total_tardiness` | 总延迟 | $\sum_j \max(0,\ C_j - d_j)$ | min |
| `avg_tardiness` | 平均延迟 | $\dfrac{total\_tardiness}{\lvert J \rvert}$ | min |
| `max_tardiness` | 最大延迟 | $\max_j \max(0,\ C_j - d_j)$ | min |
| `tardy_job_count` | 延迟任务数 | $\lvert \{ j : C_j - d_j > 10^{-9} \} \rvert$ | min |
| `main_order_tardy_count` | 主订单延误数 | 主任务超期的订单数 | min |
| `main_order_tardy_total_time` | 主订单总延误 | $\sum_{o} \max(0,\ C_{main(o)} - d_o)$ | min |
| `main_order_tardy_ratio` | 主订单延误率 | $\dfrac{main\_order\_tardy\_count}{\lvert O_{main} \rvert}$ | min |

实现：`objectives.py:300-322`（任务级）、`objectives.py:340-370`（订单级）。
订单延误 `order_tardiness` 按 $\max(0, C^{ord}_o - d_o)$ 计算，用于评审界面的订单明细，不作为优化目标。

### 2.3 周期类指标

| KPI key | 中文标签 | 公式 | 方向 |
|---|---|---|---|
| `makespan` | 总周期 | $T_{max} = \max_o C^{ord}_o$ | min |
| `total_completion_time` | 总完成时间 | $\sum_j C_j$ | min |
| `avg_flowtime` | 平均流程时间 | $\dfrac{1}{\lvert J \rvert}\sum_j (C_j - r_j)$ | min |
| `max_flowtime` | 最大流程时间 | $\max_j (C_j - r_j)$ | min |
| `total_wait_time` | 总等待时间 | $\sum_j \max\left(0,\ (C_j - r_j) - \sum_{op \in j} p_{op}\right)$ | min |
| `avg_wait_time`* | 平均等待时间 | $\dfrac{total\_wait\_time}{\lvert J \rvert}$ | — |

\* `avg_wait_time` 不是优化目标，只在 `summary` 中输出供界面展示（`objectives.py:330`）。

等待时间的"理论加工量"用 `op.processing_time` 之和（工艺定额），不是排程后的 `duration`。

### 2.4 利用率指标（三套口径）

这是最容易混淆的一组。**三个指标分子分母各不相同，各自自洽，用途不同**：

#### (a) 全周期利用率 `avg_utilization` / `critical_utilization`

$$
u^{full}_m = \frac{b_m}{T_{max}}
\qquad
avg\_utilization = \frac{1}{\lvert M \rvert}\sum_{m \in M} u^{full}_m
\qquad
critical\_utilization = \frac{1}{\lvert M_c \rvert}\sum_{m \in M_c} u^{full}_m
$$

- 分子：**净加工时长**；分母：全局完工时刻（含夜班、周末、停机等全部非工作时间）
- 分母集合含**全部机器**（未排产机器按 0 计入，`objectives.py:395-400`）
- 语义：整个计划周期里，全厂机器平均有多大比例的时间在实际加工
- 实现：`objectives.py:376-379`、`objectives.py:435-440`
- 关键机器为空时退化为全部机器均值（`objectives.py:418-419`）

> 仿真器 `core/simulator.py:804-811` 也算了一份同名指标（`total_busy_time / max_end`，`total_busy_time` 累加 `productive_duration` 即净加工），与上式**口径一致**。`objectives.py:435` 优先取仿真器结果，仅在其为 0 时回退自算。

#### (b) 活跃窗口利用率 `avg_active_window_utilization` / `critical_active_window_utilization`

$$
u^{active}_m = \min\left(1,\ \frac{o_m}{e_m - s_m}\right)
\qquad
avg = \frac{1}{\lvert M \rvert}\sum_{m \in M} u^{active}_m
$$

- 分子：**墙钟占用时长**；分母：该机器**活跃窗口**跨度（首次开工到最后完工）
- 分子分母**同为墙钟量纲**，因此排满的机器算得 100%
- 语义：机器从开工到收工这段时间里，有多大比例被占着 —— **与甘特图条块视觉密度一致**
- 实现：`objectives.py:380-391`
- 未排产机器按 0 计入均值

#### (c) 净可用利用率 `avg_net_available_utilization` / `critical_net_available_utilization`

$$
u^{net}_m = \min\left(1,\ \frac{b_m}{A_m(s_m,\ e_m)}\right)
$$

- 分子：**净加工时长**；分母：活跃窗口内的**可用工时**（班次窗口，已扣停机）
- 分子分母同为"净"量纲：都排除了班次外时间
- 语义：在机器真正开着的工时里，有多大比例在加工（剔除了班次制度的影响）
- 实现：`objectives.py:388-394`

#### 三者关系

对同一台机器，一般有 $u^{full}_m \le u^{net}_m$，且 $u^{active}_m \ge u^{net}_m$（跨班次占用越多，差距越大）。挑选指标时：

- 评估**产能饱和度**（含班次制度损失）→ 全周期
- 评估**排程紧凑度**（与甘特观感一致）→ 活跃窗口
- 评估**开机效率**（剔除班次制度）→ 净可用

### 2.5 工装 / 人员利用率

| KPI key | 中文标签 | 方向 |
|---|---|---|
| `tooling_utilization` | 工装利用率 | max |
| `personnel_utilization` | 人员利用率 | max |

$$
tooling\_utilization = \frac{1}{\lvert TL \rvert}\sum_{t \in TL} \frac{o_t}{T_{max}}
\qquad
personnel\_utilization = \frac{1}{\lvert P \rvert}\sum_{p \in P} \frac{o_p}{T_{max}}
$$

其中 $o_t$、$o_p$ 为该工装 / 人员的**墙钟占用**累计（`objectives.py:286-288` 用 `occupied`）。分母集合含全部工装 / 人员，未使用的按 0 计入（`objectives.py:402-412`）。

> ⚠️ **已知口径差异**：分子用墙钟占用，而同为"全周期利用率"的机器指标 `avg_utilization` 分子用净加工时长。详见第五节。

### 2.6 均衡与同步指标

| KPI key | 中文标签 | 公式 | 方向 |
|---|---|---|---|
| `bottleneck_load_balance` | 瓶颈负载均衡 | $\operatorname{pstdev}\left(\{u^{full}_m : m \in M_c\}\right)$ | min |
| `assembly_sync_penalty` | 装配同步惩罚 | $\sum_o \sum_{j \in o,\ j \ne main} \lvert C_j - C_{main(o)} \rvert$ | min |

- `bottleneck_load_balance` 用**全周期**利用率的总体标准差；关键机器不足 2 台时记 0（`objectives.py:429`）
- 订单无主任务时，`assembly_sync_penalty` 退化为该订单内 $\max C_j - \min C_j$（`objectives.py:364-368`）
- `bottleneck_machine_ids`：按全周期利用率降序取前 5 台（`objectives.py:428-433`），供界面标注瓶颈

---

## 三、评审界面各模块口径

### 3.1 排程甘特图

- 渲染：`frontend/app_v2.js:2219-2251`（`buildGanttData`）
- 条块按 `start` → `end` **墙钟定位**；班次外时间和停机画成背景遮罩
- 机器行**只包含有排产条目的机器**，未排产机器不出现
- 因此用户视觉上的"某台机器排满" $\Longleftrightarrow \dfrac{\sum(end - start)}{\max end - \min start} \approx 1$，即 **活跃窗口利用率 = 1**

### 3.2 机器分类利用率对比（多方案横向对比）

- 实现：`api/server.py:_machine_type_utilization`、`api/review_read.py:110-141`

$$
u_m = \min\left(1,\ \frac{\sum (end - start)}{\max end - \min start}\right)
\qquad
u_{type} = \frac{1}{\lvert M^{used}_{type} \rvert}\sum_{m \in M^{used}_{type}} u_m
$$

- 分子墙钟、分母活跃窗口 → **与 2.4(b) 活跃窗口利用率同口径**
- 类型均值只统计**有排产的机器**，同时输出 `used_machines` 供界面标注规模
- 缺 `machine_id` 或未知机台的条目跳过；零长度窗口记 0

### 3.3 每日机器分类利用率表

- 实现：`api/server.py:_machine_type_daily_utilization`

$$
u_{type,\,d} = \min\left(1,\ \frac{\text{当日占用}_{type,d}}{\sum_{m \in M^{used}_{type}} A_m(24d,\ 24(d+1))}\right)
$$

- 分子：该类型当日的墙钟占用（跨天条目按天切分，`server.py:3598-3605`）
- 分母：**只统计本方案实际排产过的机器**，逐台取当日**可用工时**（班次窗口扣停机）
- 分桶：以偏移 0 为第 0 天起点，按 24h 滚动
- 当日无排产 → 返回 `null`，界面显示 `-`
- 当日可用工时为 0 但有排产（数据异常）→ 记 1.0
- 输出 `machines_used` / `machines_total`，界面在类型名下标注"已用/总台数"

**为什么分母不是"全部机器 × 24h"**：若某虚拟资源池有 1000 台而方案只用 300 台，全台数分母会把利用率恒定压到 30%，与甘特图上"这 300 台排得满满当当"直接矛盾。改为已排产机器 + 班次日历后，两者对齐。

### 3.4 方案 KPI 对比表

- 实现：`frontend/app_v2.js:renderReviewCandidateComparison`（`app_v2.js:3443`）
- 展示 `PRIMARY_KPI_LABELS`（`app_v2.js:52-72`）中的指标，值直接取自 `objective_values`，前端不做二次计算
- 每列与基线方案做 Δ 对比，最优值加粗标注

---

## 四、口径一致性核对结论

| 模块 | 分子 | 分母 | 与甘特图一致 |
|---|---|---|---|
| 甘特图条块 | `end - start`（墙钟） | 活跃窗口跨度 | 基准 |
| `machine_active_window_utilization`（`objectives.py:389`） | `elapsed_duration`（墙钟） | 活跃窗口跨度 | ✅ |
| `_machine_type_utilization`（`server.py`） | `end - start`（墙钟） | 活跃窗口跨度 | ✅ |
| `review_read.py:128-136` | `end - start`（墙钟） | 活跃窗口跨度 | ✅ |
| 每日利用率表（`server.py`） | `end - start` 按天切分（墙钟） | 已排产机器当日可用工时 | ✅（分母按班次日历折算，语义上更严格） |
| `machine_utilization`（全周期） | `duration`（净加工） | $T_{max}$ | 不适用（不同语义指标） |
| `machine_net_available_utilization` | `duration`（净加工） | 活跃窗口内可用工时 | 不适用（不同语义指标） |

**结论**：所有"活跃窗口"族指标与甘特图墙钟口径完全统一。全周期与净可用是另外两个独立语义的指标，各自分子分母量纲自洽（净/全周期、净/净），不与甘特图对齐是设计意图，不是缺陷。

由于 `elapsed_duration` 在仿真器中恒等于 `end - start`（`simulator.py:287`、`simulator.py:442`），上表前四行在数值上完全相同。

---

## 五、待决口径差异

**工装 / 人员利用率的分子与机器不一致**（`objectives.py:281-288`）：

```python
machine_busy[machine_id]   += duration    # 净加工
tooling_busy[tooling_id]   += occupied    # 墙钟占用
personnel_busy[personnel_id] += occupied  # 墙钟占用
```

三者分母同为 $T_{max}$，都叫"利用率"，但机器用净加工、工装人员用墙钟占用。跨班次工序越多，工装 / 人员利用率相对机器越偏高。

两种统一方向：

1. **都用净加工**（`occupied` → `duration`）：与 `avg_utilization` 对齐，全周期族内部一致；工装 / 人员利用率数值会下降。
2. **都用墙钟占用**：与甘特图对齐；但 `avg_utilization` 数值会上升，且与仿真器 `core/simulator.py:804` 自算的同名指标产生分歧，需同步修改仿真器。

推荐方向 1（改动面最小，且"全周期利用率"这一族的语义本就是"实际干活占比"）。此项尚未修改，因为两个指标都可作为优化目标，改动会影响历史方案的可比性。
