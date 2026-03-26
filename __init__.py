"""
LLM4DRD 智能调度平台 v3
=======================
基于大语言模型驱动的动态柔性装配流水车间智能调度系统。

包结构:
  core/          数据模型 (Order/Task/Operation/Machine) + 仿真引擎 + 调度规则库
  knowledge/     异构图建模 (订单→任务→工序→机器 四层有向图)
  scheduling/    在线调度引擎 (事件驱动 + 动态重排)
  optimization/  多目标优化 (NSGA-II 帕累托 + OR-Tools CP-SAT 精确求解)
  ai/            LLM 双专家进化引擎 (LLM-A 算法专家 + LLM-S 评分专家)
  data/          数据层 (SQLite 规则库 + 问题实例生成器)
  api/           FastAPI REST 服务

快速开始:
  from llm4drd_platform.data.generator import InstanceGenerator
  from llm4drd_platform.core.simulator import Simulator
  from llm4drd_platform.core.rules import BUILTIN_RULES
  shop = InstanceGenerator().generate(num_orders=5)
  result = Simulator(shop, BUILTIN_RULES["ATC"]).run()
  print(result.to_dict())
"""
