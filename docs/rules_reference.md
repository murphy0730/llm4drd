# LLM4DRD 派工规则参考（Dispatch Rules Reference）

> 来源：`core/rules.py`（11 条固定规则，函数定义于第 6–63 行）。
> 用途：供论文 / 汇报 / 自学引用。所有公式均取自真实代码，非杜撰。

---

## 0. 统一约定

- 规则函数签名统一为 `rule(op, machine, features, shop) -> float`，返回**分数**。
- **分数越高越先排**（`core/simulator.py:222` 算分，`candidate=(score, -work_remaining, op.id, …)`，`candidate[:3] > best[:3]` 取最大，:232）。
- 因此规则里凡要"最小值优先"的，都用**负号**把"最小"翻成"分数最大"。
- 同分兜底：剩余工时小的优先，再按 `op.id`。
- 边界兜底：CR、ATC 在可能除零 / 越界时直接返回 `1000`（最高优先）。

---

## 1. 11 条固定规则打分函数

| # | 规则 | 分数公式（代码原文） | 谁优先（人话） | 用到的特征 |
|---|------|----------------------|----------------|-----------|
| 1 | **EDD** | `-f["due_date"]` | 交期最早的工序 | `due_date` |
| 2 | **SPT** | `-f["processing_time"]` | 本工序加工最短的 | `processing_time` |
| 3 | **LPT** | `f["processing_time"]` | 本工序加工最长的 | `processing_time` |
| 4 | **CR** | `remaining<=0 → 1000`；否则 `-(f["slack"] / f["remaining"])` | 临界比最小（最紧急）的 | `slack`, `remaining` |
| 5 | **ATC** | `processing<=0 → 1000`；否则 `(1/processing) * exp(-slack / (2*processing + 0.01))` | 短工序且松弛小（紧迫）的，明星规则 | `processing_time`, `slack` |
| 6 | **FIFO** | `f.get("wait_time", 0.0)` | 在就绪队列等待最久的（先到） | `wait_time` |
| 7 | **MST** | `-f["slack"]` | 松弛最小的（最赶） | `slack` |
| 8 | **PRIORITY** | `f["priority"] * 10.0 + f["urgency"]` | 业务优先级高 + 紧急的 | `priority`, `urgency` |
| 9 | **KIT_AWARE** | `f["prereq_ratio"] * 5.0 + f["urgency"] * 2.0 + f["priority"]` | 前驱齐套度高 + 紧急 + 高优先 | `prereq_ratio`, `urgency`, `priority` |
| 10 | **BOTTLENECK** | `f["is_main"] * 8.0 + f["urgency"] * 3.0 + f["priority"] * 2.0 - f["processing_time"] * 0.1` | 主订单 + 紧急 + 高优先（长工时节轻微罚） | `is_main`, `urgency`, `priority`, `processing_time` |
| 11 | **COMPOSITE** | `urgency_bonus(=abs(slack)*3 当 slack<0) + priority*2 + prereq_ratio*4 + is_main*5 - processing_time*0.05 - tooling_demand*0.1 - personnel_demand*0.1` | 综合业务重要性（超期重赏，长 / 高耗资源罚） | `slack`, `priority`, `prereq_ratio`, `is_main`, `processing_time`, `tooling_demand`, `personnel_demand` |

---

## 2. 特征（features）速查表

特征由仿真器在派工决策时实时计算（`core/simulator.py:_features`，约 :677）。

| 特征 | 含义 | 备注 |
|------|------|------|
| `due_date` | 交期时间戳（越大 = 越晚交） | |
| `processing_time` | 本工序加工时长 | |
| `remaining` | 该任务令剩余加工时长 | |
| `slack` | 松弛时间 = `due_date - (now + remaining)`；`<0` 表示已超期 | |
| `urgency` | `max(0, -slack)`，紧急度（越超期越大） | |
| `wait_time` | 工序进入就绪队列后已等待的时长 | |
| `priority` | 订单业务优先级字段 | |
| `is_main` | 是否主订单（1 / 0） | |
| `prereq_ratio` | 前驱工序完成比例（0~1），即"齐套度" | |
| `tooling_demand` | 工装需求 | 默认 0（本项目工装 = 人员 = 0） |
| `personnel_demand` | 人员需求 | 默认 0 |
| `machine_load` | 机器负载 | 规则内未直接用于打分 |
| `predecessor_depth` / `assembly_criticality` / `shared_resource_degree` / `bottleneck_adjacency` | 图结构特征（来自 `knowledge/`） | 喂给 16 维可学习权重，固定规则未用 |

---

## 3. 固定规则 vs 16 维可学习派工函数（重要区别）

代码里存在**两套并列**的派工机制，不要混淆：

1. **固定规则（本文档）** — `core/rules.py` 的 11 条硬编码公式，UI 通过 `frontend/app_v2.js:HEURISTIC_RULES`（精选 6 条：`ATC/EDD/SPT/CR/FIFO/LPT`）暴露给"规则仿真"。
2. **可学习派工函数** — `optimization/approx_eval.py:_priority_score`（:183）做 `Σ feature_weights[i] · feature[i]`（16 维权重点积）。其权重是解 `CandidateParameters.feature_weights` 的一部分，由 NSGA-III + ALNS 在优化中搜索得到，不是写死的。

> 即：固定规则是"人写死的启发式"；16 维加权和是"算法搜出来的参数化派工函数"。两者都能喂给同一个仿真解码器跑出排程与 KPI。

---

## 4. 备注

- 后端 `api/server.py:scenario_compare` 默认对全部 11 条 `BUILTIN_RULES` 做对照仿真；前端只展示了精选的 6 条。
- 另有 `compile_rule_from_code`（rules.py:85）支持用一段 Python 代码动态生成派工函数，是 `ai/evolution.py` 中 LLM 进化规则的基础——这部分不计入固定规则。
