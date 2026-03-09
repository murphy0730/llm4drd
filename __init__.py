"""
LLM4DRD 智能调度平台 - 基于论文核心框架的完整实现
=================================================
论文: LLM-Assisted Automatic Dispatching Rule Design for 
      Dynamic Flexible Assembly Flow Shop Scheduling

核心架构:
  1. 异构图建模层 (heterogeneous_graph.py)
  2. 离散事件仿真引擎 (simulator.py)
  3. 特征工程模块 (feature_encoder.py)
  4. LLM 双专家进化引擎 (llm_evolution.py)
  5. 在线调度引擎 (online_scheduler.py)
  6. 动态重排机制 (rescheduler.py)
  7. 方案模拟与对比 (scenario_manager.py)
  8. Web API 服务 (api_server.py)
  9. 数据库管理 (db_manager.py)

技术栈: Python + NetworkX + SQLite + APScheduler + FastAPI
"""
