# LLM4DRD 智能调度平台

> 用大语言模型（LLM）自动设计调度规则，把「动态柔性装配流水车间」的排产、优化、在线重排和可视化做成一站式工具。

论文 *"LLM-Assisted Automatic Dispatching Rule Design for Dynamic Flexible Assembly Flow Shop Scheduling"* 的完整 Python 实现：**LLM 双专家框架**自动进化调度规则，配套离散事件仿真、NSGA-II 多目标优化、OR-Tools 精确求解、事件驱动在线调度，以及一个开箱即用的 Web 前端。

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white)
![API](https://img.shields.io/badge/REST%20端点-69-informational)
![Tests](https://img.shields.io/badge/测试文件-34-success)
![Updated](https://img.shields.io/badge/最近更新-2026--07-blue)
![License](https://img.shields.io/badge/License-待补充-lightgrey)

**适合谁用**：需要在有交期、有配套（kit）、有机器停机/故障的车间里做智能排产的研究者与工程师；想复现或扩展这篇论文的算法研究团队；需要「仿真 + 优化 + 在线调度」端到端 Demo 的中文开发者。

**最短上手（60 秒）**：

```bash
git clone git@github.com:murphy0730/llm4drd.git
cd llm4drd
pip install -r requirements.txt
python run_server.py          # 自动打开浏览器 → http://127.0.0.1:8888/
```

> 无需 LLM API Key 即可运行——LLM 进化部分会自动切换到模板回退模式。

**统一入口**：仓库 <https://github.com/murphy0730/llm4drd> ｜ 问题反馈 <https://github.com/murphy0730/llm4drd/issues> ｜ 交互式 API 文档 `http://127.0.0.1:8888/docs`（服务启动后）。

---

## ✨ 项目亮点

- **LLM 自动设计调度规则**：LLM-A（算法专家）生成/交叉/变异规则代码，LLM-S（评分专家）评估质量，混合适应度 `0.7 × 目标值 + 0.3 × LLM 评分`。没有 API Key 也能跑（模板回退）。
- **四种求解范式一套数据**：11 条内置优先级规则（PDR）、帕累托枚举 / NSGA-II 真实前沿、OR-Tools CP-SAT 精确求解、事件驱动在线调度，共享同一份车间实例与仿真引擎。
- **面向真实车间**：支持子件加工 + 总装检测的层次结构、工序前置约束、按工艺分组的机器、绝对时间段停机窗口、运行中注入机器故障并动态重排。
- **能扛大实例**：统一规范图 + 不可变计算投影 + SQLite 两级缓存（进程内 L1 / 磁盘 L2），针对约 1000 订单、2 万+ 工序的规模做过性能优化与保护阈值。
- **开箱即用的 Web 端**：React 单页前端（`frontend/`）提供甘特图、帕累托前沿、在线调度与评审视图；后端暴露 **69 个 REST 端点**，自带 Swagger 文档。

---

## 📦 安装

```bash
pip install -r requirements.txt        # 核心依赖
pip install "ortools>=9.7"             # 可选：启用 OR-Tools CP-SAT 精确求解
```

- 开发与测试环境：**Python 3.14**（建议 Python ≥ 3.10）。
- 核心依赖：`networkx` `fastapi` `uvicorn` `pydantic v2` `openai`（兼容任意 OpenAI 协议端点）`openpyxl` `python-multipart`。

---

## 🚀 快速开始

### 方式一：启动 Web 平台（推荐）

```bash
python run_server.py
# macOS / Linux；自动检测端口占用并打开浏览器 → http://127.0.0.1:8888/
```

Windows 可用 `start_server.ps1`。前端页面为 `frontend/index_v2.html`（服务根路径 `/` 即返回该页）。

### 方式二：命令行完整演示

```bash
# 无需 LLM API Key（模板回退）
python -m llm4drd_platform

# 使用真实 LLM（任意 OpenAI 兼容端点）
export LLM_API_KEY=sk-xxx
export LLM_BASE_URL=https://api.deepseek.com/v1   # 或 OpenAI / 其它兼容端点
export LLM_MODEL=deepseek-chat
python -m llm4drd_platform
```

`python -m llm4drd_platform` 会运行 `demo.py`，串联「实例生成 → 仿真 → 帕累托 → 精确求解 → LLM 进化 → 在线调度」的完整流程。

---

## 🧭 使用示例

生成实例并用内置规则跑一次仿真：

```python
from llm4drd_platform.data.generator import InstanceGenerator
from llm4drd_platform.core.simulator import Simulator
from llm4drd_platform.core.rules import BUILTIN_RULES

shop = InstanceGenerator(seed=42).generate(num_orders=10)
result = Simulator(shop, BUILTIN_RULES["ATC"]).run()
print(result.to_dict())   # Makespan / 总延迟 / 主订单延误率 / 资源利用率 …
```

NSGA-II 搜索真实帕累托前沿：

```python
from llm4drd_platform.optimization.pareto import NSGA2Optimizer

nsga2 = NSGA2Optimizer(shop, ["total_tardiness", "makespan"], pop_size=30, generations=20)
solutions = nsga2.run()
pareto = [s for s in solutions if s.rank == 0]
```

LLM 双专家进化调度规则（自动读取 `LLM_API_KEY`，无 Key 则模板回退）：

```python
from llm4drd_platform.ai.evolution import EvolutionEngine, EvolutionConfig, LLMInterface

engine = EvolutionEngine(
    shop=shop, llm=LLMInterface(),
    config=EvolutionConfig(population_size=8, max_generations=15),
    db_path="llm4drd.db", objective="total_tardiness",
)
result = engine.run()
print(f"最优 fitness: {result.best_fitness}")
```

在线调度：推进时间、注入故障、动态重排：

```python
from llm4drd_platform.scheduling.online import OnlineSchedulerV3

scheduler = OnlineSchedulerV3(shop, rule_name="ATC")
scheduler.advance(20.0)                        # 推进 20 小时
scheduler.on_breakdown("turning_1", repair_at=30.0)   # 注入故障
scheduler.on_repair("turning_1")
status = scheduler.reschedule("COMPOSITE")     # 动态切换规则重排
```

### 内置调度规则（11 条）

11 条固定派工规则定义在 `core/rules.py` 的 `BUILTIN_RULES`（第 6–63 行），函数签名统一为
`rule(op, machine, features, shop)`，返回**分数越高越优先排**（见下方"派工统一约定"）。

#### 11 条规则及其打分公式

| # | 规则 | 分数公式（代码原文） | 谁优先（人话） | 用到的特征 |
|---|------|----------------------|----------------|-----------|
| 1 | **EDD** | `-f["due_date"]` | 交期最早的工序 | `due_date` |
| 2 | **SPT** | `-f["processing_time"]` | 本工序加工最短的 | `processing_time` |
| 3 | **LPT** | `f["processing_time"]` | 本工序加工最长的 | `processing_time` |
| 4 | **CR** | `remaining<=0 → 1000`；否则 `-(f["slack"]/f["remaining"])` | 临界比最小（最紧急）的 | `slack`, `remaining` |
| 5 | **ATC** | `processing<=0 → 1000`；否则 `(1/processing)·exp(-slack/(2·processing+0.01))` | 短工序且松弛小（紧迫）的，明星规则 | `processing_time`, `slack` |
| 6 | **FIFO** | `f.get("wait_time", 0.0)` | 在就绪队列等待最久的（先到） | `wait_time` |
| 7 | **MST** | `-f["slack"]` | 松弛最小的（最赶） | `slack` |
| 8 | **PRIORITY** | `f["priority"]*10 + f["urgency"]` | 业务优先级高 + 紧急的 | `priority`, `urgency` |
| 9 | **KIT_AWARE** | `f["prereq_ratio"]*5 + f["urgency"]*2 + f["priority"]` | 前驱齐套度高 + 紧急 + 高优先 | `prereq_ratio`, `urgency`, `priority` |
| 10 | **BOTTLENECK** | `f["is_main"]*8 + f["urgency"]*3 + f["priority"]*2 - f["processing_time"]*0.1` | 主订单 + 紧急 + 高优先 | `is_main`, `urgency`, `priority`, `processing_time` |
| 11 | **COMPOSITE** | `urgency_bonus + priority*2 + prereq_ratio*4 + is_main*5 - processing_time*0.05 - tooling_demand*0.1 - personnel_demand*0.1`（其中 `urgency_bonus = abs(slack)*3` 当 `slack<0`，否则 0） | 综合业务重要性（超期重赏，长/高耗资源罚） | `slack`, `priority`, `prereq_ratio`, `is_main`, `processing_time`, `tooling_demand`, `personnel_demand` |

> **派工统一约定**：`core/simulator.py:222` 算分后用 `candidate[:3] > best[:3]` 取**最大分数**优先，
> 同分按剩余工时小、再 `op.id` 兜底。因此规则里凡"最小者优先"都用**负号**翻成"分数最大优先"。
> CR、ATC 在可能除零/越界时直接给 `1000`（最高优先）做兜底。

#### 16 维特征及其含义

这 16 维特征（`optimization/solution_model.py` 的 `FEATURE_NAMES`）是派工决策的输入表征，
由仿真状态（`core/simulator.py:_features`）与知识图谱（`knowledge/context.py:operation_features`）提取：

| # | 特征名 | 含义 | 来源 | 越大表示 |
|---|--------|------|------|----------|
| 1 | `urgency` | 紧迫度 = `max(0, -slack)`，已超期时间 | 仿真状态 | 越紧急 |
| 2 | `slack` | 松弛时间 = `due_date - now - remaining` | 仿真状态 | 越宽裕（<0 已超期） |
| 3 | `remaining` | 所属任务令剩余总工时 | 仿真状态 | 活越多 |
| 4 | `processing_time` | 本工序自身加工工时 = `op.work_remaining` | 仿真状态 | 本工序越长 |
| 5 | `priority` | 业务优先级 = `order.priority` | 仿真状态 | 业务越重要 |
| 6 | `is_main` | 是否主订单（0/1） | 仿真状态 | 1=主订单 |
| 7 | `wait_time` | 进入就绪队列后的已等待时间 | 仿真状态 | 等得越久 |
| 8 | `prereq_ratio` | 前驱齐套比（前驱完成比例，当前恒为 1.0） | 仿真状态 | 越齐套 |
| 9 | `machine_load` | 候选机器累计忙时 = `machine.total_busy_time` | 仿真状态 | 机器越忙 |
| 10 | `tooling_demand` | 所需工装类型数 | 仿真状态 | 工装占用越多 |
| 11 | `personnel_demand` | 所需人员技能数 | 仿真状态 | 人力占用越多 |
| 12 | `predecessor_depth` | 依赖图上该工序最长前驱链深度（÷操作数归一化） | 知识图谱 | 前驱越多越深 |
| 13 | `assembly_criticality` | 装配关键度：主订单 1.0 / 主订单祖先 0.78 / 含主任务 0.32 / 其他 0.18 | 知识图谱 | 装配越关键 |
| 14 | `shared_resource_degree` | 共享资源争用度 = `min(3, 1/可上机数 + 工装稀缺 + 人员稀缺 + 0.25·工装类型数 + 0.25·人员数)` | 知识图谱 | 资源争用越强 |
| 15 | `bottleneck_adjacency` | 瓶颈邻接度 = `min(1, 0.55·关键设备比 + 0.45·(1/可上机数))` | 知识图谱 | 越靠近瓶颈设备 |
| 16 | `due_date` | 交期时间戳（工序派生交期，缺失回退任务令/订单交期） | 仿真状态 | 交期越晚 |

> 注：第 1–11、16 共 12 个来自车间/仿真状态；第 12–15 这 4 个来自知识图谱的异构图结构特征。
> `machine_load` 在 `FEATURE_NAMES` 里的名字，实际由 `machine_busy_time` 别名喂入。

#### 规则与特征的关系

**方向：特征在上游，规则在下游。** 16 维特征是对工序当前状态的"体检报告"，由车间/图谱状态
独立提取；11 条规则（以及 16 维权重点积可学习派工函数 `approx_eval._priority_score`）都是这些
特征的**消费者**：

- 每条规则只读取它关心的几个特征算分（如 EDD 只读 `due_date`，ATC 读 `processing_time`+`slack`），
  规则**不生成**任何特征；
- 16 维权重点积函数读取**全部 16 维**特征做加权求和，权重是优化解 `CandidateParameters.feature_weights`
  的一部分，是可学习的；
- 唯一"相遇"处是 seed 混合（`approx_eval.py:185`）：可学习函数可把某条固定规则的分数以 0.12
  权重叠加进自身分数做初始化——仍是"规则→分数"，不是"规则→特征"。

> 简言之：**11 条规则 ≠ 16 维特征的来源**；二者共享同一份特征输入。规则是写死的"指标组合公式"
> （只读其中几个特征），可学习函数则是算法搜出来的"指标组合公式"（读全部 16 维特征）。

自定义规则（编译一段 Python 打分函数即可，复用同样的 `features` 输入）：

```python
from llm4drd_platform.core.rules import compile_rule_from_code

fn = compile_rule_from_code("""
def my_rule(op, machine, features, shop):
    return features['priority'] * 3 - features['processing_time'] * 0.1
""")
```

`features` 可用键：即上方 **16 维特征** 全部可用（`urgency` `slack` `remaining` `processing_time`
`priority` `is_main` `wait_time` `prereq_ratio` `machine_load` `tooling_demand` `personnel_demand`
`predecessor_depth` `assembly_criticality` `shared_resource_degree` `bottleneck_adjacency` `due_date`）；
此外仿真器还会附带 `progress`、`critical_slack`、`external_due_date` 等辅助键。

---

## 📁 项目结构

```
llm4drd/
├── run_server.py         Web 服务入口（注册包 + 自动打开浏览器）
├── __main__.py           CLI 入口：python -m llm4drd_platform → demo.py
├── demo.py               端到端演示脚本
├── config.py / config.json   LLM 与数据库配置
│
├── core/                 数据模型 + 离散事件仿真引擎 + 11 条调度规则
├── knowledge/            规范异构图（订单→任务→工序→机器）+ 计算投影 GraphContext
├── scheduling/           事件驱动在线调度引擎（故障注入 / 动态重排）
├── optimization/         帕累托 / NSGA-II / OR-Tools CP-SAT 精确求解
├── ai/                   LLM 双专家进化引擎（LLM-A 算法 + LLM-S 评分）
├── data/                 SQLite 数据层（规则库/实例/停机/图投影）+ 实例生成器
├── api/                  FastAPI REST 服务（server.py，69 个端点）
├── frontend/             React 单页前端（index_v2.html / app_v2.js / app_v2.css）
├── tools/                基准测试与校验脚本（benchmark_* / verify_*）
├── tests/                34 个测试文件（pytest）
└── docs/                 当前优化方案、规则参考与性能基准
```

---

## 🌐 REST API

服务启动后打开 `http://127.0.0.1:8888/docs` 查看交互式文档（Swagger）。共 **69 个端点**，按模块划分：

| 模块前缀 | 能力 |
|----------|------|
| `/api/instance/*` | 实例生成、详情、CSV/Excel 导入导出、订单/任务/工序/机器编辑、校验 |
| `/api/graph/*` | 后台图谱构建、状态查询、节点/边/邻居检索、按订单过滤 |
| `/api/simulate/*` | 仿真、对比、参考解、Excel 导出 |
| `/api/optimize/hybrid/*` | 混合优化任务：状态/结果/排程/机器利用率/评审数据 |
| `/api/pareto/*` · `/api/exact/*` | 帕累托 / NSGA-II / OR-Tools CP-SAT（后台任务 + 进度查询）|
| `/api/ai/pareto/*` · `/api/train` | LLM 推荐/问答 / 规则进化训练与日志 |
| `/api/online/*` | 在线调度：推进、故障、恢复、重排、状态 |
| `/api/workflow/*` | 四步流程进度与评审 |
| `/api/downtime/*` · `/api/config/llm` · `/api/health` | 停机记录管理、LLM 配置、健康检查 |

---

## ⚙️ 配置

`config.json` 或环境变量（环境变量优先级更高）：

| 环境变量 | 说明 |
|----------|------|
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | LLM 端点配置（OpenAI 兼容协议）|
| `LLM4DRD_DB` / `LLM4DRD_CONFIG` | 数据库与配置文件路径 |
| `LLM4DRD_GRAPH_CONTEXT_MODE` | 计算图上下文模式：`legacy`（默认）/ `shadow`（影子对比）/ `active`（直接使用缓存不可变上下文）|
| `LLM4DRD_GRAPH_TIMEOUT_S` | 图谱构建 + 保存总超时，默认 180 秒 |
| `LLM4DRD_GRAPH_WARN_EDGES` / `LLM4DRD_GRAPH_MAX_EDGES` | 关系边预警（默认 30 万）/ 安全上限（默认 200 万）|
| `LLM4DRD_GRAPH_MAX_NODES` | 节点安全上限，默认 10 万 |

> ⚠️ 安全提示：`config.json` 请勿写入真实 API Key 后提交——本仓库会自动提交推送到 GitHub，密钥会外泄。建议改用环境变量。

---

## ❓ 适用场景

- **算法研究**：复现/扩展 LLM 自动设计调度规则的论文实验，对比 PDR、元启发式与精确解。
- **车间排产原型**：为带交期、配套与停机约束的柔性装配流水车间快速搭建可视化排产 Demo。
- **在线调度实验**：模拟机器故障、动态到单等扰动，验证不同规则的动态重排效果。
- **多目标权衡分析**：在 makespan、总延迟、主订单延误率、资源利用率等目标间求帕累托前沿。

---

## 🛠️ 开发

```bash
pytest tests/                 # 运行测试（共 34 个测试文件）

# 图上下文缓存基准（中/大型实例温启动至少 2× 加速为验收阈值）
python -m llm4drd.tools.benchmark_graph_context \
    --sizes 80,500,2500 --runs 7 --warmup 2 --seed 42 \
    --mode compare --output-dir docs/benchmarks
```

`tools/` 下另有 `benchmark_simulation_perf.py`、`verify_v2_full.py` 等基准与端到端校验脚本。

---

## 🤝 贡献

欢迎通过 [Issue](https://github.com/murphy0730/llm4drd/issues) 反馈问题或提交 PR。提交前请先运行 `pytest tests/` 确保测试通过。

---

## 📝 License

待补充（仓库暂未包含 LICENSE 文件）。
