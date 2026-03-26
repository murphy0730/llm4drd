# LLM4DRD 智能调度平台

基于论文 *"LLM-Assisted Automatic Dispatching Rule Design for Dynamic Flexible Assembly Flow Shop Scheduling"* 的完整 Python 实现。

使用大语言模型（LLM）双专家框架自动设计调度规则，结合仿真引擎、多目标优化、在线调度与精确求解，提供从实例生成到 Web 可视化的一站式解决方案。

---

## 目录结构

```
llm4drd_platform/
├── config.py              配置管理 (config.json + 环境变量)
├── config.json            LLM 及数据库配置
├── demo.py                完整演示脚本 (9 个步骤)
├── __main__.py            入口: python -m llm4drd_platform
│
├── core/                  核心层
│   ├── models.py          数据模型 (Order/Task/Operation/Machine/Downtime)
│   ├── simulator.py       离散事件仿真引擎 (堆优化, 增量就绪队列)
│   └── rules.py           11 条内置调度规则 + 规则编译器
│
├── knowledge/             知识表示层
│   └── graph.py           有向异构图 (订单→任务→工序→机器)
│
├── scheduling/            调度引擎层
│   └── online.py          在线调度引擎 (事件驱动, 支持故障注入与动态重排)
│
├── optimization/          多目标优化层
│   ├── pareto.py          帕累托优化器 + NSGA-II 真实前沿搜索
│   └── exact.py           OR-Tools CP-SAT 精确求解器
│
├── ai/                    AI 进化层
│   └── evolution.py       LLM 双专家进化引擎 (LLM-A 算法 + LLM-S 评分)
│
├── data/                  数据层
│   ├── db.py              SQLite 数据库 (规则库/实例/停机记录)
│   └── generator.py       可配置问题实例生成器
│
├── api/                   API 服务层
│   └── server.py          FastAPI REST 服务 (~20 个端点)
│
└── frontend/
    └── index.html         React 单页前端 (甘特图/帕累托/在线调度)
```

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
# 可选: 精确求解功能
pip install ortools>=9.7
```

### 运行演示

```bash
# 无需 LLM API Key (使用模板回退)
python -m llm4drd_platform

# 使用真实 LLM
export LLM_API_KEY=sk-xxx
export LLM_BASE_URL=https://api.openai.com/v1   # 或任意 OpenAI 兼容端点
export LLM_MODEL=gpt-4o
python -m llm4drd_platform
```

### 启动 Web 服务

```bash
uvicorn llm4drd_platform.api.server:app --reload --port 8000
# 前端: http://localhost:8000
```

---

## 核心功能

### 1. 数据模型 (`core/models.py`)

业务驱动的层次结构：

```
订单 (Order)
  └── 任务 (Task)  [子件加工 + 主任务(总装检测)]
        └── 工序 (Operation)  [前置关系, 指定工艺类型/机器]
机器 (Machine)  [按工艺类别分组, 含停机窗口 Downtime]
```

- `ShopFloor`：完整车间模型，含加速索引
- `Downtime`：绝对时间段停机（计划性/非计划性）
- `Machine.next_start_time(t)` / `compute_effective_end(start, dur)`：自动绕过停机窗口

### 2. 仿真引擎 (`core/simulator.py`)

高性能离散事件仿真：

- 基于 `heapq` 的事件队列，增量就绪队列（避免全量扫描）
- 自动跳过机器停机时段
- KPI：Makespan、总延迟、主订单延误率、关键资源利用率、总等待时间

```python
from llm4drd_platform.data.generator import InstanceGenerator
from llm4drd_platform.core.simulator import Simulator
from llm4drd_platform.core.rules import BUILTIN_RULES

shop = InstanceGenerator(seed=42).generate(num_orders=10)
result = Simulator(shop, BUILTIN_RULES["ATC"]).run()
print(result.to_dict())
```

### 3. 内置调度规则 (`core/rules.py`)

11 条优先级调度规则（PDR）：

| 规则 | 说明 |
|------|------|
| EDD | 最早交期优先 |
| SPT | 最短加工时间 |
| LPT | 最长加工时间 |
| CR | 关键比率 |
| ATC | 表观延迟成本 |
| FIFO | 先到先服务 |
| MST | 最小松弛时间 |
| PRIORITY | 订单优先级 |
| KIT_AWARE | 配套感知 |
| BOTTLENECK | 瓶颈感知 |
| COMPOSITE | 加权综合 |

自定义规则：

```python
from llm4drd_platform.core.rules import compile_rule_from_code

fn = compile_rule_from_code("""
def my_rule(op, machine, features, shop):
    return features['priority'] * 3 - features['processing_time'] * 0.1
""")
```

特征字典 `features` 包含：`slack`, `remaining`, `processing_time`, `due_date`, `urgency`, `progress`, `priority`, `is_main`, `wait_time`, `prereq_ratio`, `machine_busy_time`

### 4. 多目标优化 (`optimization/`)

**帕累托枚举**（`ParetoOptimizer`）：对 11 条内置规则做非支配排序

**NSGA-II**（`NSGA2Optimizer`）：加权集成规则空间搜索，真实帕累托前沿

```python
from llm4drd_platform.optimization.pareto import NSGA2Optimizer

nsga2 = NSGA2Optimizer(shop, ["total_tardiness", "makespan"], pop_size=30, generations=20)
solutions = nsga2.run()
pareto = [s for s in solutions if s.rank == 0]
```

支持目标：`total_tardiness`, `makespan`, `main_order_tardy_count`, `main_order_tardy_ratio`, `avg_utilization`, `critical_utilization`, `total_wait_time`, `avg_flowtime`, `max_tardiness`

**精确求解**（`ExactSolver`）：OR-Tools CP-SAT，IntervalVar + NoOverlap + 前置约束

```python
from llm4drd_platform.optimization.exact import ExactSolver

result = ExactSolver(shop, objectives=["makespan"], time_limit_s=60).solve()
# result.status: OPTIMAL | FEASIBLE | INFEASIBLE | UNKNOWN | ERROR
```

### 5. AI 进化引擎 (`ai/evolution.py`)

LLM 双专家框架：

- **LLM-A**（算法专家）：生成、交叉、变异调度规则 Python 代码
- **LLM-S**（评分专家）：评估规则的调度质量
- 混合适应度：`0.7 × 目标值 + 0.3 × LLM评分`
- 无 API Key 时自动切换为模板回退模式

```python
from llm4drd_platform.ai.evolution import EvolutionEngine, EvolutionConfig, LLMInterface

llm = LLMInterface()  # 自动读取 LLM_API_KEY 环境变量
engine = EvolutionEngine(shop=shop, llm=llm,
                         config=EvolutionConfig(population_size=8, max_generations=15),
                         db_path="llm4drd.db", objective="total_tardiness")
result = engine.run()
print(f"最优 fitness: {result.best_fitness}")
```

### 6. 在线调度 (`scheduling/online.py`)

事件驱动在线调度器：

```python
from llm4drd_platform.scheduling.online import OnlineSchedulerV3

scheduler = OnlineSchedulerV3(shop, rule_name="ATC")
status = scheduler.advance(20.0)        # 推进 20 小时
scheduler.on_breakdown("turning_1", repair_at=30.0)  # 注入故障
status = scheduler.advance(15.0)
scheduler.on_repair("turning_1")
status = scheduler.reschedule("COMPOSITE")  # 动态切换规则
```

### 7. 异构图 (`knowledge/graph.py`)

NetworkX 有向异构图，5 类节点（order/task/operation/machine）、6 类边，用于图神经网络扩展。

### 8. 数据库 (`data/db.py`)

SQLite（WAL 模式）：

- `RuleStore`：规则库增删改查
- `InstanceStore`：车间实例序列化存储
- `GraphStore`：图数据存储
- `DowntimeStore`：机器停机记录管理

---

## REST API

启动后访问 `http://localhost:8000/docs` 查看交互式 API 文档。

| 路径 | 方法 | 说明 |
|------|------|------|
| `/api/generate` | POST | 生成问题实例 |
| `/api/simulate` | POST | 运行仿真 |
| `/api/pareto` | POST | 计算帕累托前沿 |
| `/api/pareto/nsga2` | POST | 启动 NSGA-II (后台任务) |
| `/api/pareto/nsga2/status/{id}` | GET | 查询进度 |
| `/api/exact/solve` | POST | 启动精确求解 |
| `/api/exact/status/{id}` | GET | 查询进度 |
| `/api/train` | POST | 启动 LLM 进化 |
| `/api/online/start` | POST | 初始化在线调度器 |
| `/api/online/advance` | POST | 推进时间 |
| `/api/online/breakdown` | POST | 注入机器故障 |
| `/api/online/repair` | POST | 恢复机器 |
| `/api/online/reschedule` | POST | 动态重排 |
| `/api/online/status` | GET | 获取当前状态 |
| `/api/downtime` | GET/POST | 停机记录管理 |
| `/api/downtime/{id}` | PUT/DELETE | 停机记录编辑 |

---

## 配置

`config.json` 或环境变量：

```json
{
  "llm": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "",
    "model": "gpt-4o",
    "max_tokens": 2048,
    "timeout": 60
  },
  "database": {
    "path": "llm4drd.db"
  }
}
```

环境变量（优先级高于配置文件）：`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM4DRD_DB`, `LLM4DRD_CONFIG`

---

## 依赖

```
networkx>=3.0       # 异构图
fastapi>=0.100.0    # Web 框架
uvicorn>=0.23.0     # ASGI 服务器
pydantic>=2.0.0     # 数据验证
openai>=1.0.0       # LLM 客户端 (兼容任意 OpenAI 协议端点)
python-multipart    # 文件上传
openpyxl>=3.1.0     # Excel 导出
ortools>=9.7        # CP-SAT 精确求解 (可选)
```
