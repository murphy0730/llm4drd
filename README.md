# LLM4DRD 智能调度平台 — 合并工程 v2

基于论文 *"LLM-Assisted Automatic Dispatching Rule Design for Dynamic Flexible Assembly Flow Shop Scheduling"* 的完整 Python 实现，含性能优化、帕累托多目标、React 前端和多模型配置。

---

## 工程结构 (14 个核心模块)

```
llm4drd_platform/
│
├── config.json              # 大模型 & 平台配置 (支持 7+ 种 LLM)
├── config.py                # 配置加载 (文件 → 环境变量 → 默认值)
├── requirements.txt         # 依赖清单
│
├── models.py                # 数据模型: 订单/作业/机器/BOM/配套组
├── heterogeneous_graph.py   # 异构图建模 (NetworkX) — 论文 Section 3
├── feature_encoder.py       # 动态特征编码器 (22维) — 论文 Section 4.3
├── dispatching_rules.py     # 11 种内置 PDR + 规则编译器
├── simulator.py             # 离散事件仿真引擎
│
├── llm_evolution.py         # LLM 双专家进化 (LLM-A + LLM-S) — 论文核心
├── online_scheduler.py      # 事件驱动在线调度器 (毫秒级)
├── rescheduler.py           # 动态重排 (周期/事件/偏差触发)
├── scenario_manager.py      # What-if 分析 & Monte Carlo
│
├── pareto.py                # 帕累托多目标 (NSGA-II 非支配排序)
├── performance.py           # 性能优化: 可行对索引/特征缓存/并行仿真
├── db_manager.py            # SQLite 规则库 & 结果存储
├── api_server.py            # FastAPI REST 服务 + APScheduler
│
├── demo.py                  # 9 步完整演示脚本
├── __main__.py              # python -m llm4drd_platform 入口
└── frontend/                # React 前端 (甘特图/帕累托/配置)
```

---

## 快速开始

### 1. 安装

```bash
pip install networkx fastapi uvicorn pydantic apscheduler openai
```

### 2. 运行演示 (无需 LLM)

```bash
python -m llm4drd_platform
```

输出 9 步集成测试: 实例生成 → 异构图 → 特征编码 → 11规则对比 → 帕累托前沿 → LLM进化 → 在线调度 → 动态重排 → Monte Carlo。

### 3. 启动 API 服务

```bash
uvicorn llm4drd_platform.api_server:app --host 0.0.0.0 --port 8000
```

Swagger 文档: http://localhost:8000/docs

### 4. 配置大模型 (支持任意 OpenAI 兼容 API)

编辑 `config.json` 或设置环境变量:

```bash
# DeepSeek
export LLM_API_KEY=sk-xxx
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_MODEL=deepseek-chat

# 或通义千问
export LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export LLM_MODEL=qwen-max

# 或本地 Ollama
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=qwen2.5:72b
export LLM_API_KEY=ollama
```

config.json 内已预置 OpenAI / DeepSeek / 通义千问 / 智谱GLM / Moonshot / Ollama / vLLM 七种示例。

---

## 论文对应关系

| 论文概念 | 实现模块 | 说明 |
|---------|---------|------|
| 有向异构图 | `heterogeneous_graph.py` | NetworkX 图, 5类节点, 4类边 |
| LLM-A 算法专家 | `llm_evolution.py` | 生成 PDR 代码的 Prompt |
| LLM-S 调度专家 | `llm_evolution.py` | 评估规则质量的 Prompt |
| 精英初始化 | `dispatching_rules.py` | 11种内置规则作为种子 |
| 动态特征编码器 | `feature_encoder.py` | 22维特征 (作业/机器/配套/系统) |
| 混合评估 | `_compute_hybrid_scores()` | 目标函数70% + LLM评分30% |
| 特征拟合进化 | 规则通过 features dict 自适应 | 不同状态不同行为 |
| 配套约束 Kitting | `AssemblyGroup` + `KittingFeatures` | 多层组装依赖 |

## 内置 11 条调度规则

| 规则 | 缩写 | 特色 |
|------|------|------|
| Earliest Due Date | EDD | 经典延迟最小化 |
| Shortest Processing Time | SPT | 最小化平均流程时间 |
| Longest Processing Time | LPT | 均衡机器负载 |
| Critical Ratio | CR | 紧急度比率 |
| Apparent Tardiness Cost | ATC | 经典组合规则 |
| First In First Out | FIFO | 先到先服务 |
| Minimum Slack Time | MST | 最小松弛 |
| **Kitting-Aware** | KIT_AWARE | 配套感知 (论文特色) |
| **Bottleneck-Aware** | BOTTLENECK | 瓶颈感知 |
| **Setup-Aware EDD** | SETUP_EDD | 换型感知 |
| **Assembly Coordination** | ASM_COORD | 组装协调 |

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/instance/generate` | POST | 生成 FAFSP 实例 (支持千级订单) |
| `/api/simulate` | POST | 仿真排产 + 甘特图数据 |
| `/api/simulate/compare` | POST | 11 规则对比 |
| `/api/pareto/optimize` | POST | 帕累托多目标优化 |
| `/api/pareto/objectives` | GET | 可用优化目标列表 |
| `/api/train` | POST | 启动 LLM 进化训练 (后台) |
| `/api/reschedule` | POST | 手动触发动态重排 |
| `/api/scenario/monte_carlo` | POST | Monte Carlo 鲁棒性分析 |
| `/api/config/llm` | GET/PUT | 查看/修改大模型配置 |
| `/api/config/llm/test` | POST | 测试大模型连接 |
| `/api/gantt` | GET | 获取甘特图数据 |
| `/api/rules` | GET | 规则库列表 |
| `/api/health` | GET | 健康检查 |

## 技术栈

| 组件 | 选型 |
|------|------|
| 图引擎 | NetworkX |
| 后端框架 | FastAPI |
| 定时任务 | APScheduler |
| 数据库 | SQLite (WAL 模式) |
| LLM 接口 | openai SDK (兼容任意 API) |
| 前端 | React + Recharts + Canvas 甘特图 |
| 部署 | 本地直接运行 |
