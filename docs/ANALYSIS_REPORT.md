# LLM4DRD 智能调度平台 — 详细分析报告

> 基于论文 *"LLM-Assisted Automatic Dispatching Rule Design for Dynamic Flexible Assembly Flow Shop Scheduling"* 的算法思路与代码实现对照分析

---

## 目录

1. [论文核心思想概述](#1-论文核心思想概述)
2. [系统架构全景分析](#2-系统架构全景分析)
3. [核心算法与代码实现映射](#3-核心算法与代码实现映射)
4. [各模块详细分析](#4-各模块详细分析)
5. [算法流程深度解读](#5-算法流程深度解读)
6. [技术亮点与创新点](#6-技术亮点与创新点)
7. [潜在改进方向与建议](#7-潜在改进方向与建议)
8. [总结](#8-总结)

---

## 1. 论文核心思想概述

### 1.1 问题定义：动态柔性装配流水车间调度 (DFAFSP)

论文研究的是 **Dynamic Flexible Assembly Flow Shop Scheduling (DFAFSP)** 问题，这是传统流水车间调度问题的多维度扩展：

- **柔性 (Flexible)**: 每道工序可在多台兼容机器上加工，机器选择本身即为决策变量
- **装配 (Assembly)**: 产品由多个零件组装而成，引入 **配套约束 (Kitting Constraint)** — 组装工序必须在所有组成零件加工完成后才能开始
- **多阶段 (Flow Shop)**: 生产流程包含加工阶段和组装阶段，具有阶段间的工序顺序约束
- **动态 (Dynamic)**: 系统运行中存在新订单到达、设备故障、加工时间波动等动态扰动事件

### 1.2 LLM4DRD 框架的核心贡献

论文提出了一个 **LLM 辅助的调度规则自动设计框架**，核心思想是：

1. **将 LLM 作为进化算子**：用大语言模型替代传统遗传编程中的交叉和变异操作，生成可解释的 PDR (Priority Dispatching Rule) 代码
2. **双专家机制 (Dual-Expert)**：
   - **LLM-A (Algorithm Expert)**: 负责生成和改进 PDR 代码
   - **LLM-S (Scheduling Expert)**: 负责评估规则质量并提供改进反馈
3. **精英知识引导初始化**: 将经典高质量调度规则作为种子注入初始种群
4. **混合评估策略**: 结合客观仿真指标 (70%) 和 LLM 主观评估 (30%)
5. **动态特征拟合进化**: 规则通过感知系统状态特征实现自适应

### 1.3 与传统方法的本质区别

| 维度 | 传统遗传编程 (GP) | LLM4DRD |
|------|------------------|---------|
| 搜索空间 | 语法树随机组合 | 语义级规则设计 |
| 交叉/变异 | 语法节点随机交换 | LLM 理解语义后智能组合 |
| 可解释性 | 树结构难以理解 | 生成可读的 Python 代码 |
| 领域知识 | 需要人工编码特征 | LLM 内含调度领域知识 |
| 评估反馈 | 仅客观适应度 | 客观+主观双重评估 |

---

## 2. 系统架构全景分析

### 2.1 工程结构概览

本平台包含 **15 个核心模块**，约 **5800+ 行 Python 代码**，完整实现了论文框架并扩展了工业化功能：

```
llm4drd_platform/
├── 数据层
│   ├── models.py              # 数据模型 (224行) — FAFSP 问题实体
│   ├── instance_generator.py  # 实例生成器 (294行) — 论文 Section 5
│   ├── config.py              # 配置管理 (116行) — 多层配置加载
│   └── db_manager.py          # 数据持久化 (261行) — SQLite WAL
│
├── 核心算法层
│   ├── heterogeneous_graph.py # 异构图建模 (265行) — 论文 Section 3
│   ├── feature_encoder.py     # 特征编码 (267行) — 论文 Section 4.3
│   ├── dispatching_rules.py   # 调度规则库 (223行) — 论文 Section 4.1
│   ├── llm_evolution.py       # LLM进化引擎 (658行) — 论文核心
│   └── simulator.py           # 仿真引擎 (432行) — 评估与在线调度
│
├── 调度应用层
│   ├── online_scheduler.py    # 在线调度 (307行) — 实时事件驱动
│   ├── rescheduler.py         # 动态重排 (335行) — 三种触发方式
│   ├── scenario_manager.py    # 场景分析 (240行) — What-if + Monte Carlo
│   └── pareto.py              # 多目标优化 (318行) — NSGA-II
│
├── 性能与服务层
│   ├── performance.py         # 性能优化 (284行) — 索引/缓存/并行
│   └── api_server.py          # REST API (266行) — FastAPI
│
└── 入口与前端
    ├── demo.py                # 演示脚本 (145行)
    └── index.html             # React 前端
```

### 2.2 分层架构设计

```
┌──────────────────────────────────────────────────────┐
│                   前端展示层                          │
│    React + Canvas 甘特图 + Recharts 帕累托图          │
├──────────────────────────────────────────────────────┤
│                   REST API 层                        │
│    FastAPI + APScheduler (api_server.py)             │
├──────────────────────────────────────────────────────┤
│                  调度应用层                            │
│  OnlineScheduler │ DynamicRescheduler │ ScenarioMgr  │
├──────────────────────────────────────────────────────┤
│                 核心算法层                             │
│  EvolutionEngine │ Simulator │ ParetoOptimizer       │
├──────────────────────────────────────────────────────┤
│                 数据抽象层                             │
│  HeteroGraph │ FeatureEncoder │ DispatchingRules     │
├──────────────────────────────────────────────────────┤
│                  基础设施层                            │
│  Models │ Config │ DBManager │ InstanceGen │ Perf    │
└──────────────────────────────────────────────────────┘
```

---

## 3. 核心算法与代码实现映射

### 3.1 论文-代码对应关系总览

| 论文章节 | 论文概念 | 实现模块 | 核心类/函数 | 代码行数 |
|---------|---------|---------|------------|---------|
| Section 3 | 有向异构图决策过程 | `heterogeneous_graph.py` | `HeterogeneousGraph` | 265 |
| Section 3 | 可行 (operation, machine) 对提取 | `heterogeneous_graph.py` | `get_feasible_pairs()` | 50 |
| Section 4.1 | 精英知识引导初始化 | `llm_evolution.py` | `initialize_population()` | 40 |
| Section 4.1 | 11种精英 PDR | `dispatching_rules.py` | `BUILTIN_RULES` | 200 |
| Section 4.2 | LLM-A 算法专家 | `llm_evolution.py` | `_crossover()`, `_mutate()` | 80 |
| Section 4.2 | LLM-S 调度专家 | `llm_evolution.py` | `_llm_evaluate_top_k()` | 30 |
| Section 4.3 | 动态特征编码器 | `feature_encoder.py` | `FeatureEncoder.encode()` | 180 |
| Section 4.4 | 混合评估 | `llm_evolution.py` | `_compute_hybrid_scores()` | 20 |
| 核心循环 | LLM4DRD Main Loop | `llm_evolution.py` | `evolve()` | 80 |
| Section 5 | 训练/测试实例生成 | `instance_generator.py` | `InstanceGenerator` | 294 |
| 在线调度 | PDR-based scheduling | `online_scheduler.py` | `on_machine_idle()` | 70 |
| 评估器 | 仿真评估 | `simulator.py` | `Simulator.run()` | 200 |

---

## 4. 各模块详细分析

### 4.1 数据模型层 (`models.py`)

#### 设计分析

数据模型采用 Python `dataclass` 实现，定义了 FAFSP 问题的完整实体体系：

```
Order (订单)
  ├── products: {product_id: quantity}
  ├── due_date, priority, release_time
  └── job_ids, assembly_group_ids

Product (产品)
  ├── bom: [{component_type, quantity}]
  └── assembly_stages

Job (作业) — 对应零件/组件的加工任务
  ├── operations: [Operation]  — 工序列表
  ├── assembly_group_id        — 配套组关联
  └── prerequisite_job_ids     — 前置依赖

AssemblyGroup (配套组) — 论文核心：Kitting Constraint
  ├── component_job_ids        — 组成零件
  ├── assembly_job_id          — 组装作业
  └── is_kitting_complete()    — 配套完成检查

Machine (机器)
  ├── capable_job_types        — 可加工类型 (柔性)
  ├── setup_time_matrix        — 换型时间矩阵
  └── speed_factor             — 加工速度因子

ShopFloor (车间) — 完整问题定义
  └── stages, machines, products, orders, jobs, assembly_groups
```

#### 关键设计决策

1. **配套约束建模**: `AssemblyGroup` 通过 `component_job_ids` 和 `assembly_job_id` 建立组装依赖关系，`is_kitting_complete()` 方法检查所有零件是否完工
2. **作业状态机**: `JobStatus` 枚举包含 PENDING → WAITING → PROCESSING → COMPLETED 和 BLOCKED 五种状态，BLOCKED 专门用于等待配套的场景
3. **柔性机器选择**: `Operation.eligible_machines` 列表允许工序在多台机器上执行

### 4.2 问题实例生成器 (`instance_generator.py`)

#### 论文对应：Section 5 — Experimental Setup

论文的实验部分需要覆盖不同规模、负荷水平和动态扰动级别的问题实例。`InstanceGenerator` 负责自动化生成满足这些要求的 FAFSP 实例，是训练和测试的数据基础。

#### 核心方法：`generate_fafsp_instance()`

该方法按以下层级自顶向下构建完整的车间调度问题实例：

```
1. 创建生产阶段 (2个)
   ├── stage_proc: 加工阶段 (sequence=1)
   └── stage_asm:  组装阶段 (sequence=2)

2. 创建机器 (可配置数量)
   ├── 加工机器 × N
   │   ├── 从 6 种作业类型中随机选取 3~6 种 (柔性建模)
   │   ├── 为每对可加工类型生成换型时间矩阵
   │   └── 速度因子 ~ Uniform(0.8, 1.2)
   └── 组装站 × M
       ├── 固定类型 "assembly"
       └── 速度因子 ~ Uniform(0.9, 1.1)

3. 创建产品模板 (5种)
   └── 每种产品的 BOM: 随机 2~5 个组件，类型从 6 种中选取

4. 创建订单 (可配置数量)
   ├── 每订单随机选 1~3 种产品
   ├── 释放时间: 立即 或 Uniform(0, spread*N*10)
   ├── 优先级: Uniform(1, 5)
   └── 对每种产品:
       ├── 创建配套组 (AssemblyGroup)
       ├── 为每个 BOM 组件创建加工作业
       │   ├── 1~3 道工序
       │   ├── 加工时间 ~ Uniform(5, 30)
       │   └── 兼容机器: 从加工机器中筛选类型匹配的
       ├── 创建组装作业
       │   ├── 加工时间 ~ Uniform(10, 40)
       │   ├── 初始状态: BLOCKED (等待配套)
       │   └── prerequisite_job_ids = 所有组件作业
       └── 设置交期
           due_date = release_time + (总加工时间/机器数) * due_date_factor

5. 所有作业继承订单交期
```

#### 可配置参数

| 参数 | 默认值 | 含义 | 对实验的影响 |
|------|-------|------|-------------|
| `num_orders` | 5 | 订单数量 | 控制问题规模 |
| `products_per_order` | (1, 3) | 每订单产品数范围 | 影响配套组复杂度 |
| `components_per_product` | (2, 5) | 每产品组件数范围 | 决定 BOM 深度和配套约束数量 |
| `num_processing_machines` | 4 | 加工机器数 | 资源充裕度 |
| `num_assembly_machines` | 2 | 组装站数 | 组装瓶颈程度 |
| `processing_time_range` | (5, 30) | 加工时间范围 | 作业异构性 |
| `setup_time_range` | (0, 5) | 换型时间范围 | 换型影响程度 |
| `due_date_factor` | 1.5 | 交期紧度因子 | >1 宽松, <1 紧张 |
| `order_arrival_spread` | 0.0 | 到达分散度 | 0=静态, >0=动态到达 |

#### 训练集/测试集生成策略

实现了论文要求的多样化实例覆盖，训练集与测试集的参数范围有意差异化以验证泛化性：

| 参数 | 训练集范围 | 测试集范围 | 设计意图 |
|------|-----------|-----------|---------|
| `num_orders` | 3~8 | 5~12 | 测试集更大规模 |
| `num_processing_machines` | 3~6 | 4~8 | 测试集更多机器 |
| `num_assembly_machines` | 1~3 | 2~4 | 测试集更多组装站 |
| `due_date_factor` | 1.0~2.0 | 0.8~1.8 | 测试集含更紧交期 |
| `order_arrival_spread` | 0.0~0.5 | 0.0~0.8 | 测试集更强动态性 |

这种设计确保在测试集上评估的规则不是过拟合到训练分布的。

#### 关键设计细节

1. **柔性机器兼容性**: 每台加工机器从 6 种作业类型中随机选取 3~6 种，实现了机器的柔性建模。当某类型找不到兼容机器时，回退到所有加工机器都可用
2. **换型时间矩阵**: 为每台机器的每对可加工类型生成独立的换型时间 `setup_time_matrix[(from_type, to_type)]`，供调度决策中 `SETUP_EDD` 等规则使用
3. **配套约束自动构建**: 组装作业自动设置 `status=BLOCKED` 和 `prerequisite_job_ids`，确保仿真引擎正确处理配套依赖
4. **交期估算公式**: `due_date = release_time + (总加工时间 / 机器数) × due_date_factor`，通过 `due_date_factor` 控制交期紧张程度
5. **可复现性**: 通过 `seed` 参数支持确定性随机，确保实验可复现

#### 生成实例的规模特征

以默认参数 (8单, 5台加工+2台组装) 为例，典型实例规模：
- 订单: 8 个
- 作业: ~40-60 个 (含加工+组装)
- 工序: ~60-120 道
- 配套组: ~12-24 个
- 异构图节点: ~150-250 个
- 异构图边: ~300-500 条

### 4.3 异构图建模 (`heterogeneous_graph.py`)

#### 论文对应：Section 3 — Directed Heterogeneous Graph-based Decision Procedure

论文将 DFAFSP 的所有调度约束统一建模为有向异构图，本模块使用 NetworkX 实现。

**节点类型 (6类):**

| 节点类型 | 前缀 | 含义 | 属性 |
|---------|------|------|------|
| `order` | `O:` | 订单 | due_date, priority, release_time |
| `product` | — | 产品 | (通过 Order→Job 边隐含) |
| `job` | `J:` | 作业 | order_id, status, due_date |
| `operation` | `OP:` | 工序 | processing_time, job_type, status |
| `machine` | `M:` | 机器 | stage_id, status |
| `assembly_group` | `AG:` | 配套组 | product_id, order_id |

**边类型 (7类):**

| 边类型 | 语义 | 约束含义 |
|-------|------|---------|
| `order_contains_product` | O→J | 订单包含关系 |
| `product_requires_job` | — | (隐含在 O→J 中) |
| `job_has_operation` | J→OP | 工序归属 |
| `operation_sequence` | OP→OP | **工序顺序约束** |
| `kitting_constraint` | J→AG | **配套约束** (零件→配套组) |
| `machine_eligible` | OP→M | **机器兼容性** |
| `assembly_depends` | AG→J | **组装依赖** (配套组→组装作业) |

#### 核心方法：`get_feasible_pairs()`

这是论文决策过程的核心实现 — 在每个决策步从图中提取所有可行的 (operation, machine) 对：

```python
def get_feasible_pairs(self, shop, current_time) -> list[(op_id, machine_id, Operation)]:
    # 对每个作业检查:
    # 1. 未完成且未在加工中
    # 2. 已到达释放时间
    # 3. 前置工序已完成 (工序顺序约束)
    # 4. 配套约束满足 (组装作业需要配套完成)
    # 5. 获取当前工序的所有空闲兼容机器
```

**约束检查链**:
```
Job → 状态检查 → 释放时间 → 前置依赖 → 配套约束 → 当前工序 → 兼容机器 → 可行对
```

#### 增量更新机制

代码支持两种图更新方式：
- `build_from_shopfloor()`: 全量重建（初始化时）
- `add_job_to_graph()` + `update_node_status()`: 增量更新（运行时），避免重建整张图的开销

### 4.4 特征编码器 (`feature_encoder.py`)

#### 论文对应：Section 4.3 — Dynamic Feature-Fitting Rule Evolution

特征编码器从系统状态中提取 **22维** 特征向量，是 PDR 规则感知系统状态的基础：

**特征体系 (4类22维):**

| 类别 | 特征名 | 维度 | 描述 |
|------|--------|------|------|
| **作业级** (JobFeatures) | slack_time | 1 | 松弛时间 = due_date - now - remaining |
| | remaining_time | 1 | 剩余加工时间 |
| | waiting_time | 1 | 已等待时间 |
| | progress_ratio | 1 | 工序完成率 (0-1) |
| | urgency_score | 1 | 紧急度 (归一化, 截断至5.0) |
| | order_priority | 1 | 订单优先级 (1-5) |
| | is_critical | 1 | 是否在关键路径 |
| **机器级** (MachineFeatures) | queue_length | 1 | 队列长度 |
| | utilization | 1 | 利用率 |
| | remaining_work | 1 | 当前作业剩余时间 |
| | setup_time | 1 | 预计换型时间 |
| | avg_processing_rate | 1 | 平均加工速率 |
| **配套级** (KittingFeatures) | completion_ratio | 1 | 组件配套完成度 |
| | bottleneck_remaining | 1 | 瓶颈组件剩余时间 |
| | component_dispersion | 1 | 同组作业离散度 (标准差) |
| | group_size | 1 | 配套组大小 |
| **系统级** (SystemFeatures) | global_wip | 1 | 全局在制品数量 |
| | bottleneck_load | 1 | 瓶颈机器负载率 |
| | avg_urgency | 1 | 平均紧急度 |
| | utilization_variance | 1 | 设备利用率方差 |
| | pending_orders | 1 | 待处理订单数 |
| | tardy_ratio | 1 | 当前延迟率 |

#### 紧急度计算公式

```python
if slack <= 0:
    urgency = 1.0 + abs(slack) / (remaining + 1)  # 已紧急：基准1 + 超期比
else:
    urgency = remaining / (slack + remaining + 1)   # 未紧急：加工占比
```

#### 配套级特征（论文关键创新）

`encode_kitting()` 方法计算：
- **completion_ratio**: 通过 `AssemblyGroup.get_completion_ratio()` 获取当前完成的组件比例
- **bottleneck_remaining**: 遍历所有未完成组件，取最大剩余加工时间
- **component_dispersion**: 组内作业剩余时间的标准差 — 衡量同组作业的同步程度

### 4.5 调度规则库 (`dispatching_rules.py`)

#### 论文对应：Section 4.1 — Elite Knowledge Guided Initialization

**11条内置 PDR**，分为两类：

**经典单目标规则 (7条):**

| 规则 | 公式核心 | 优化目标 |
|------|---------|---------|
| EDD | `priority = -due_date` | 最小化总延迟 |
| SPT | `priority = -remaining_time` | 最小化平均流程时间 |
| LPT | `priority = remaining_time` | 均衡机器负载 |
| CR | `priority = -slack/remaining` | 关键比率 |
| ATC | `priority = (1/p) * exp(-max(d-p-t,0)/(k*p_avg))` | 表观延迟成本 |
| FIFO | `priority = -release_time` | 先到先服务 |
| MST | `priority = -slack_time` | 最小松弛 |

**论文特色组合规则 (4条):**

| 规则 | 创新点 | 公式要素 |
|------|-------|---------|
| KIT_AWARE | 配套感知 | `kit_ratio² * 3 + urgency * 2 + priority * 1.5` |
| BOTTLENECK | 瓶颈感知 | 瓶颈组件优先 + 负载均衡 |
| SETUP_EDD | 换型感知 | EDD - 换型时间占比惩罚 |
| ASM_COORD | 组装协调 | 配套完整度 + 离散度 + 紧急度 |

#### 规则编译器

```python
def compile_rule_from_code(code_str, rule_name="evolved_rule"):
    namespace = {"math": math}
    exec(code_str, namespace)  # 动态编译 LLM 生成的代码
    return namespace[rule_name]
```

所有 PDR 统一签名：`def rule(job, machine, features, shop) -> float`，返回值越大优先级越高。

### 4.6 LLM 双专家进化引擎 (`llm_evolution.py`)

#### 论文对应：论文核心算法 — LLM4DRD Framework

这是整个系统最核心的模块 (658行)，完整实现了论文的 Algorithm 1。

#### 4.6.1 进化个体 (`RuleIndividual`)

```python
@dataclass
class RuleIndividual:
    id: str
    code: str                    # Python 源代码
    fitness: float = inf         # 客观适应度 (总延迟)
    llm_score: float = 0.0      # LLM-S 主观评分 (0-10)
    hybrid_score: float = inf    # 混合评估分数
    generation: int = 0
    parent_ids: list[str]        # 谱系追踪
    features_used: list[str]     # 使用的特征列表
    evaluation_details: dict     # LLM 评估反馈
```

#### 4.6.2 LLM 接口 (`LLMInterface`)

设计亮点：**无可用 API 时自动降级为模板规则生成器**

```
优先级: 构造参数 > 环境变量 > config.json
                      │
                      ▼
         ┌──── API Key 存在? ────┐
         │ Yes                   │ No
         ▼                       ▼
    _call_real_llm()       _template_fallback()
    (openai SDK)           (随机组合特征+权重)
```

模板回退机制的设计很巧妙：
- 从 14 个可用特征中随机选 4 个
- 从 6 种组合公式模板中随机选 1 种
- 随机生成权重 [-3.0, 3.0]
- 组装成完整的 PDR 函数代码

这确保了即使没有 LLM API，系统也能运行并产生有意义的规则变体。

#### 4.6.3 Prompt 工程

**LLM-A 系统提示**: 定义了角色、可用特征列表、函数签名规范

**交叉提示 (Crossover-Prompt)**:
```
给定两个高性能规则 A 和 B，设计新规则需要：
1. 保留双亲的有效特征组合
2. 引入新颖的权重平衡
3. 处理边界情况
```

**变异提示 (Improved-Prompt)**:
```
给定一个需要改进的规则及其 LLM-S 反馈：
- 调整权重
- 添加新特征组合
- 改进边界处理
- 重构优先级公式
```

**LLM-S 评估提示**: 要求从 4 个维度评分：
1. 紧急作业处理
2. 配套/组装约束考虑
3. 多目标平衡性
4. 具体改进建议

#### 4.6.4 主进化循环 (`evolve()`)

```
初始化种群 (精英PDR种子 + LLM变体)
    │
    ▼
┌─── 每一代循环 ───────────────────┐
│ Step 1: 仿真评估种群适应度         │
│         (多实例平均总延迟)          │
│ Step 2: LLM-S 对 Top-K 主观评估    │
│ Step 3: 计算混合评估分数            │
│         hybrid = 0.7*norm_fitness   │
│                + 0.3*(10-llm_score) │
│ Step 4: 精英保留                    │
│ Step 5: 生成下一代                  │
│         ├── 交叉 (概率0.6)          │
│         │   LLM-A 智能组合两父代    │
│         └── 变异 (概率0.4)          │
│             LLM-A 基于反馈改进      │
│                                    │
│ 早停检查 (patience=5代无改善)       │
└─────────────────────────────────┘
    │
    ▼
最终泛化验证 (测试实例集)
```

#### 4.6.5 混合评估公式

```python
hybrid_score = 0.7 * (fitness - min_f) / (max_f - min_f)  # 归一化客观分
             + 0.3 * (10 - llm_score) / 10                 # 归一化主观分
```

这个 7:3 权重设计平衡了客观性能和主观质量评估。

### 4.7 离散事件仿真引擎 (`simulator.py`)

#### 核心设计

采用经典的 **离散事件仿真 (DES)** 架构：

**事件类型 (8种):**

| 事件 | 触发条件 | 处理逻辑 |
|------|---------|---------|
| `ORDER_ARRIVAL` | 新订单释放时间到达 | 释放订单所有作业 |
| `MACHINE_IDLE` | 机器完成加工/修复 | **核心调度决策** |
| `OPERATION_COMPLETE` | 工序加工完成 | 推进作业状态 |
| `JOB_COMPLETE` | 作业所有工序完成 | 检查配套+订单 |
| `KITTING_CHECK` | 组件完成 | 触发组装释放 |
| `MACHINE_BREAKDOWN` | 设备故障 | 中断当前加工 |
| `MACHINE_REPAIR` | 设备修复 | 重新触发调度 |

**核心调度决策流程 (`_dispatch_on_machine`):**

```
机器空闲事件
    │
    ▼
获取所有可行 (operation, machine) 对
    │ (调用 HeterogeneousGraph.get_feasible_pairs)
    ▼
筛选该机器的可行工序
    │
    ▼
对每个候选: 编码特征 → 调用 PDR 计算分数
    │
    ▼
选择最高分 → 开始加工
    │
    ├── 计算实际加工时间 = (base_time / speed_factor) + setup_time
    ├── 更新状态: 工序=PROCESSING, 机器=BUSY
    ├── 记录调度日志
    └── 调度 OPERATION_COMPLETE 事件
```

**配套触发机制 (`_check_kitting_and_trigger_assembly`):**

```
组件作业完成
    │
    ▼
遍历所有配套组
    │
    ├── 该组件属于该配套组?
    │     └── 配套是否全部完成?
    │           └── 释放组装作业 (BLOCKED → PENDING)
    │               └── 触发组装阶段机器调度
    └── 继续检查
```

**仿真结果指标:**
- `total_tardiness`: 总延迟
- `makespan`: 最大完工时间
- `avg_tardiness / max_tardiness`: 平均/最大延迟
- `tardy_job_count`: 延迟作业数
- `avg_utilization`: 平均设备利用率
- `avg_flowtime`: 平均流程时间
- `simulation_time`: 仿真壁钟时间

### 4.8 在线调度引擎 (`online_scheduler.py`)

#### 论文对应：Online PDR-based Scheduling

在线调度器是训练好的 PDR 在生产环境中的实时执行层，特点：
- **毫秒级决策**: 无需调用 LLM，直接执行编译好的 PDR 函数
- **事件驱动**: 响应机器空闲、新订单、设备故障等事件
- **分阶段规则**: 支持为加工阶段和组装阶段配置不同 PDR

**核心方法 `on_machine_idle()`:**
```
                    事件监听
                       │
     ┌────────────────┼────────────────┐
     ▼                ▼                ▼
on_machine_idle   on_new_order   on_machine_breakdown
     │
     ▼
特征编码 → PDR计算 → 最优分配
     │
     └── DispatchDecision 记录
         (timestamp, job/op/machine_id, score, decision_time_ms)
```

**性能追踪**: 维护最近 100 次决策的平均耗时。

### 4.9 动态重排机制 (`rescheduler.py`)

#### 三种触发方式

| 触发类型 | 条件 | 场景 |
|---------|------|------|
| 周期触发 | 每 480 分钟 | 定期优化 |
| 事件触发 | 紧急插单/设备故障 | 立即响应 |
| 偏差触发 | 预计延迟率 > 15% | 自适应调整 |

#### 重排六阶段流程

```
Phase 1: 状态快照 → 记录当前机器/作业状态
Phase 2: 约束更新 → 识别可重排作业 (排除冻结窗口内、锁定的)
Phase 3: PDR快速排程 → 用训练好的规则生成新基准方案
Phase 4: 局部搜索优化 → 关键路径邻域搜索 (100次迭代, 30秒超时)
Phase 5: 稳定性检查 → 变更比例超 50% 则限制
Phase 6: 方案确认 → 返回改善指标
```

#### 局部搜索实现

当前实现为简单的随机作业交换：
```python
# 随机选两个待排作业
# 检查机器兼容性
# 仿真评估新方案
# 保留更优解
```

### 4.10 帕累托多目标优化 (`pareto.py`)

#### 实现 NSGA-II 核心算法

**支配关系判定:**
```python
def dominates(a, b, objectives):
    # a 在所有目标上不差于 b，且至少一个严格优于 b
```

**非支配排序:**
- 标准 NSGA-II O(N²) 非支配排序
- 输出分层 Pareto Front: F0 (前沿), F1, F2, ...

**拥挤度距离:**
- 保持帕累托前沿的多样性
- 边界解赋 ∞ 距离

**6个预定义目标:**
`total_tardiness`, `makespan`, `avg_utilization`, `avg_flowtime`, `tardy_count`, `max_tardiness`

**加权聚合**: 用户可指定各目标权重，获取综合排名。

### 4.11 性能优化模块 (`performance.py`)

针对工业级规模（上千订单/上百机器）的三大优化：

#### 4.11.1 可行对索引 (`FeasibilityIndex`)

```
传统方式: O(J * M) 遍历所有作业和机器
优化方式: 预建索引 machine_id → {eligible_op_ids}
查询复杂度: O(k) 其中 k 为该机器的可行工序数

增量更新接口:
  on_job_complete()     → 解除下游阻塞
  on_operation_advance() → 注册新可行工序
```

#### 4.11.2 特征缓存 (`FeatureCache`)

```
缓存键: job_id : machine_id : time_bucket(5单位粒度)
容量: 10000 条
淘汰策略: LRU (满时淘汰最旧 20%)
预期收益: 节省 30-50% 特征计算时间
```

#### 4.11.3 并行仿真评估 (`parallel_evaluate`)

```
实例数 ≤ 2 或 workers ≤ 1 → 串行执行
实例数 > 2 → ProcessPoolExecutor 多进程并行
失败回退 → 自动降级为串行

参数: max_workers=4, max_time=50000
```

### 4.12 场景分析管理器 (`scenario_manager.py`)

提供三种分析模式：

| 模式 | 方法 | 用途 |
|------|------|------|
| 规则对比 | `run_rule_comparison()` | 多 PDR 在同实例上对比 |
| What-if | `run_what_if()` | 不同扰动假设下的评估 |
| Monte Carlo | `run_monte_carlo()` | 随机扰动下的鲁棒性分析 |

**Monte Carlo 分析**:
- 默认 30 次重复
- 加工时间高斯噪声 (方差 10%)
- 随机设备故障注入 (概率 5%)
- 输出统计量: 均值、标准差、最小、最大、中位数

### 4.13 API 服务层 (`api_server.py`)

FastAPI REST 服务，提供 13 个端点：

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/instance/generate` | POST | 生成 FAFSP 实例 |
| `/api/simulate` | POST | 仿真排产 + 甘特图 |
| `/api/simulate/compare` | POST | 多规则对比 |
| `/api/pareto/optimize` | POST | 帕累托多目标优化 |
| `/api/pareto/objectives` | GET | 可用目标列表 |
| `/api/train` | POST | 后台 LLM 进化训练 |
| `/api/reschedule` | POST | 手动触发重排 |
| `/api/scenario/monte_carlo` | POST | Monte Carlo 分析 |
| `/api/config/llm` | GET/PUT | LLM 配置管理 |
| `/api/config/llm/test` | POST | 测试 LLM 连接 |
| `/api/gantt` | GET | 甘特图数据 |
| `/api/health` | GET | 健康检查 |

---

## 5. 算法流程深度解读

### 5.1 离线训练阶段 (LLM 进化)

```
                    ┌───────────────────────────┐
                    │   1. 生成训练/测试实例集    │
                    │   InstanceGenerator        │
                    └─────────┬─────────────────┘
                              │
                              ▼
                    ┌───────────────────────────┐
                    │   2. 精英初始化             │
                    │   ATC, EDD, CR, KIT_AWARE, │
                    │   BOTTLENECK → 种子个体     │
                    │   + LLM-A 生成变体填充种群   │
                    └─────────┬─────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ 个体 1   │   │ 个体 2   │   │ 个体 N   │
        │ PDR代码  │   │ PDR代码  │   │ PDR代码  │
        └──────┬───┘   └──────┬───┘   └──────┬───┘
               │              │              │
               └──────┬───────┘──────────────┘
                      │
                      ▼
            ┌──────────────────────┐
            │  3. 仿真评估          │
            │  每个个体在所有训练    │
            │  实例上运行仿真        │
            │  fitness = 平均延迟   │
            └─────────┬────────────┘
                      │
                      ▼
            ┌──────────────────────┐
            │  4. LLM-S 评估 Top-K │
            │  主观评分 0-10        │
            │  反馈弱点和建议       │
            └─────────┬────────────┘
                      │
                      ▼
            ┌──────────────────────┐
            │  5. 混合评估          │
            │  0.7 * fitness_norm  │
            │  + 0.3 * llm_norm   │
            └─────────┬────────────┘
                      │
                      ▼
            ┌──────────────────────┐
            │  6. 选择 + 生成下一代 │
            │  精英保留             │
            │  交叉 (60%): LLM-A   │
            │  变异 (40%): LLM-A   │
            └─────────┬────────────┘
                      │
                      ▼
                 收敛? ──No──→ 回到步骤3
                   │
                  Yes
                   │
                   ▼
            ┌──────────────────────┐
            │  7. 泛化验证          │
            │  在测试实例集上评估    │
            │  最优个体             │
            └──────────────────────┘
```

### 5.2 在线调度阶段

```
             生产系统运行
                 │
    ┌────────────┼────────────────┐
    │            │                │
    ▼            ▼                ▼
新订单到达    机器空闲          设备故障
    │            │                │
    ▼            ▼                ▼
增量更新图    提取可行对        中断当前作业
注册新作业    编码特征          标记故障机器
    │         PDR评分           等待修复
    │         执行分配              │
    │            │                │
    └────────────┼────────────────┘
                 │
                 ▼
           偏差检测
           ├── 延迟率 > 15%? → 动态重排
           ├── 周期到达?      → 周期重排
           └── 正常继续
```

### 5.3 温度控制策略

论文 Section 5.3 建议 0.3-1.5 的中温范围最佳，代码实现：

```python
temperature_min: float = 0.3   # 低温：保守变异
temperature_max: float = 1.5   # 高温：探索性变异

# 每次交叉/变异随机选取温度
temp = random.uniform(self.config.temperature_min, self.config.temperature_max)
```

---

## 6. 技术亮点与创新点

### 6.1 配套约束的全链路建模

从数据模型 (`AssemblyGroup`) → 异构图 (`EDGE_KITTING`, `EDGE_ASSEMBLY_DEPENDENCY`) → 特征编码 (`KittingFeatures`) → 调度规则 (`KIT_AWARE`, `ASM_COORD`) → 仿真引擎 (`_check_kitting_and_trigger_assembly`)，配套约束在每一层都得到了完整的表达和处理。

### 6.2 优雅的降级设计

LLM 接口的三层降级机制：
1. 真实 LLM API → 完整的 LLM-A/LLM-S 双专家
2. API 不可用 → 模板规则生成器 (随机组合特征和权重)
3. 模板失败 → EDD 规则兜底

### 6.3 增量图更新

避免每次状态变化都重建整张异构图：
- 新订单到达：`add_job_to_graph()` 增量添加
- 状态变更：`update_node_status()` 局部更新
- 缓存失效：`invalidate_cache()` 按需刷新

### 6.4 多模型适配

通过 OpenAI 兼容 API 接口，支持 7+ 种 LLM：
OpenAI / DeepSeek / 通义千问 / 智谱GLM / Moonshot / Ollama (本地) / vLLM (本地)

### 6.5 完整的 9 步演示流水线

`demo.py` 提供了完整的集成测试：
实例生成 → 异构图 → 特征编码 → 11规则对比 → 帕累托前沿 → LLM进化 → 在线调度 → 动态重排 → Monte Carlo

---

## 7. 潜在改进方向与建议

### 7.1 算法层面

| 改进方向 | 当前状态 | 建议 |
|---------|---------|------|
| **局部搜索** | 简单随机交换 | 实现关键路径分析导向的邻域搜索 (N5, N7 等) |
| **特征选择** | 固定 22 维 | 实现自动特征重要性分析，淘汰冗余特征 |
| **多目标进化** | 单目标适应度 + Pareto 后处理 | 将 NSGA-II 集成到进化循环中 |
| **规则组合** | 单一 PDR | 支持规则集合调度 (Ensemble PDR) |
| **深度强化学习** | 未实现 | 可在 LLM 生成的规则基础上微调 RL agent |

### 7.2 工程层面

| 改进方向 | 当前状态 | 建议 |
|---------|---------|------|
| **安全性** | `exec()` 编译规则代码 | 添加沙箱执行环境 (如 RestrictedPython) |
| **持久化** | SQLite 单文件 | 高并发场景考虑 PostgreSQL |
| **异步处理** | 部分同步调用 | 关键 API 全异步化 (async LLM 调用) |
| **测试覆盖** | 仅有演示脚本 | 添加 pytest 单元测试和集成测试 |
| **日志监控** | 基础 logging | 添加结构化日志 + 进化过程可视化 |
| **实例生成** | `InstanceGenerator` 随机生成 | 支持导入真实工业数据 (CSV/JSON)，增加更多扰动模式 |

### 7.3 LLM 集成优化

| 改进方向 | 建议 |
|---------|------|
| **Prompt 优化** | 添加 Few-shot 示例，展示好的规则长什么样 |
| **批量调用** | 交叉和变异的 LLM 调用可以批量化 |
| **成本控制** | 追踪 token 消耗，自动切换模型 |
| **多模型协作** | LLM-A 和 LLM-S 使用不同模型 |
| **结构化输出** | 使用 JSON Schema 约束 LLM 输出格式 |

### 7.4 扩展功能建议

1. **实时对接 MES**: 与制造执行系统集成，获取真实车间数据
2. **多车间协同**: 支持跨车间的订单分配和协调
3. **学习曲线**: 在进化过程中引入操作工的学习效应
4. **能耗优化**: 添加能耗作为优化目标
5. **预测性维护**: 基于设备状态预测故障并纳入调度决策

---

## 8. 总结

### 8.1 论文实现完整性评估

| 论文概念 | 实现状态 | 完整度 |
|---------|---------|-------|
| 有向异构图建模 | 完整实现，含增量更新 | 95% |
| LLM-A 算法专家 | 完整实现，含 Prompt 模板 | 90% |
| LLM-S 调度专家 | 完整实现，含评估+反馈 | 85% |
| 精英知识初始化 | 完整实现，11 种种子规则 | 100% |
| 动态特征编码 | 完整实现，22 维特征 | 95% |
| 混合评估策略 | 完整实现，7:3 权重 | 100% |
| 特征拟合进化 | 基本实现，特征通过 dict 传递 | 80% |
| 配套约束 Kitting | 完整实现，全链路支持 | 100% |
| 在线调度 | 完整实现，毫秒级事件驱动 | 95% |
| 动态重排 | 完整实现，三种触发方式 | 85% |

**总体完整度: ~92%**

### 8.2 代码质量评价

**优点:**
- 模块化设计清晰，职责边界明确
- 配套约束的全链路支持是核心亮点
- 降级机制设计优雅，确保无 LLM 也能运行
- 统一的 PDR 函数签名便于扩展
- 完整的 REST API 支持前端集成

**可改进:**
- 缺少自动化测试
- `exec()` 编译存在安全风险
- 部分模块的错误处理可以更精细
- 进化过程的可观测性可以加强

### 8.3 关键数据流总结

```
                        训练阶段
配置 → InstanceGenerator → 异构图建模 → 精英初始化
         │                         │
         ▼                         ▼
    特征编码器 ←──── LLM进化循环 ────→ 仿真评估
                        │
                        ▼
                    最优PDR规则
                        │
                        ▼
                      在线阶段
    事件流 → 特征编码 → PDR决策 → 执行分配
         └──→ 偏差检测 → 动态重排 → 更新方案
```

---

> 报告生成时间: 2026-03-05
> 分析版本: LLM4DRD Platform v2.0.0
> 代码总量: ~5800 行 Python (15 模块) + ~42000 行前端 (HTML/JS/CSS)
