"""
方案模拟与对比模块
=================
支持 What-if 分析、多方案并行评估、Monte Carlo 鲁棒性分析

论文支撑: 仿真器是 LLM4DRD 框架不可或缺的组件，
既用于训练评估也用于在线方案验证。
"""
import copy
import random
import logging
from typing import Callable, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from .models import ShopFloor, MachineStatus
from .simulator import Simulator, SimulationResult
from .dispatching_rules import BUILTIN_RULES, compile_rule_from_code

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfig:
    """场景配置"""
    name: str
    rule_name: Optional[str] = None
    rule_code: Optional[str] = None
    # 扰动设置
    machine_breakdown_prob: float = 0.0    # 设备故障概率
    processing_time_variance: float = 0.0  # 加工时间波动系数
    demand_multiplier: float = 1.0         # 需求量倍数
    extra_orders: int = 0                  # 额外新增订单数


@dataclass
class ScenarioResult:
    """场景评估结果"""
    scenario_name: str
    sim_result: SimulationResult
    config: ScenarioConfig

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario_name,
            "config": {
                "rule": self.config.rule_name or "custom",
                "breakdown_prob": self.config.machine_breakdown_prob,
                "pt_variance": self.config.processing_time_variance,
            },
            "results": self.sim_result.to_dict(),
        }


class ScenarioManager:
    """
    方案模拟管理器
    
    功能:
    1. 多规则对比: 不同 PDR 在相同实例上的效果对比
    2. 参数敏感性: 调整规则权重观察影响
    3. 场景分析: 不同需求/扰动假设下的表现
    4. Monte Carlo: 随机扰动下的鲁棒性评估
    """

    def __init__(self, base_shop: ShopFloor):
        self.base_shop = base_shop
        self.scenario_results: dict[str, list[ScenarioResult]] = {}

    def run_rule_comparison(
        self,
        rule_names: list[str] = None,
        custom_rules: dict[str, str] = None,
    ) -> list[ScenarioResult]:
        """
        多规则对比
        在相同实例上运行多种 PDR，对比 KPI
        """
        results = []
        rules_to_test = {}

        # 内置规则
        if rule_names:
            for name in rule_names:
                if name in BUILTIN_RULES:
                    rules_to_test[name] = BUILTIN_RULES[name]

        # 自定义规则
        if custom_rules:
            for name, code in custom_rules.items():
                try:
                    rules_to_test[name] = compile_rule_from_code(code)
                except Exception as e:
                    logger.error(f"Failed to compile rule {name}: {e}")

        if not rules_to_test:
            rules_to_test = {k: v for k, v in list(BUILTIN_RULES.items())[:5]}

        for name, func in rules_to_test.items():
            logger.info(f"Evaluating rule: {name}")
            sim = Simulator(self.base_shop, func)
            sim_result = sim.run(max_time=50000)

            config = ScenarioConfig(name=name, rule_name=name)
            result = ScenarioResult(
                scenario_name=name,
                sim_result=sim_result,
                config=config,
            )
            results.append(result)

        self.scenario_results["rule_comparison"] = results
        return results

    def run_what_if(self, scenarios: list[ScenarioConfig]) -> list[ScenarioResult]:
        """
        What-if 场景分析
        每个场景可以配置不同的扰动和规则
        """
        results = []

        for scenario in scenarios:
            logger.info(f"Running scenario: {scenario.name}")

            # 构建场景实例
            shop = self._build_scenario_shop(scenario)

            # 获取规则
            if scenario.rule_name and scenario.rule_name in BUILTIN_RULES:
                func = BUILTIN_RULES[scenario.rule_name]
            elif scenario.rule_code:
                func = compile_rule_from_code(scenario.rule_code)
            else:
                func = BUILTIN_RULES["ATC"]

            # 运行仿真
            sim = Simulator(shop, func)

            # 注入扰动事件
            if scenario.machine_breakdown_prob > 0:
                self._inject_breakdowns(sim, shop, scenario.machine_breakdown_prob)

            sim_result = sim.run(max_time=50000)

            result = ScenarioResult(
                scenario_name=scenario.name,
                sim_result=sim_result,
                config=scenario,
            )
            results.append(result)

        self.scenario_results["what_if"] = results
        return results

    def run_monte_carlo(
        self,
        rule_func: Callable,
        num_replications: int = 30,
        breakdown_prob: float = 0.05,
        pt_variance: float = 0.1,
    ) -> dict:
        """
        Monte Carlo 鲁棒性分析
        多次随机仿真评估规则的稳定性
        """
        tardiness_samples = []
        makespan_samples = []

        for rep in range(num_replications):
            shop = copy.deepcopy(self.base_shop)

            # 随机扰动加工时间
            if pt_variance > 0:
                for job in shop.jobs.values():
                    for op in job.operations:
                        noise = random.gauss(1.0, pt_variance)
                        op.processing_time *= max(0.5, noise)

            sim = Simulator(shop, rule_func)

            # 随机设备故障
            if breakdown_prob > 0:
                self._inject_breakdowns(sim, shop, breakdown_prob)

            result = sim.run(max_time=50000)
            tardiness_samples.append(result.total_tardiness)
            makespan_samples.append(result.makespan)

        # 统计分析
        import statistics
        return {
            "num_replications": num_replications,
            "tardiness": {
                "mean": round(statistics.mean(tardiness_samples), 2),
                "std": round(statistics.stdev(tardiness_samples), 2) if len(tardiness_samples) > 1 else 0,
                "min": round(min(tardiness_samples), 2),
                "max": round(max(tardiness_samples), 2),
                "median": round(statistics.median(tardiness_samples), 2),
            },
            "makespan": {
                "mean": round(statistics.mean(makespan_samples), 2),
                "std": round(statistics.stdev(makespan_samples), 2) if len(makespan_samples) > 1 else 0,
                "min": round(min(makespan_samples), 2),
                "max": round(max(makespan_samples), 2),
                "median": round(statistics.median(makespan_samples), 2),
            },
        }

    def _build_scenario_shop(self, config: ScenarioConfig) -> ShopFloor:
        """构建场景化的车间实例"""
        shop = copy.deepcopy(self.base_shop)

        # 加工时间波动
        if config.processing_time_variance > 0:
            for job in shop.jobs.values():
                for op in job.operations:
                    noise = random.gauss(1.0, config.processing_time_variance)
                    op.processing_time *= max(0.5, noise)

        return shop

    def _inject_breakdowns(self, sim: Simulator, shop: ShopFloor, prob: float):
        """注入随机设备故障事件"""
        from .simulator import EventType
        for mid in shop.machines:
            if random.random() < prob:
                breakdown_time = random.uniform(50, 500)
                repair_time = random.uniform(10, 50)
                sim.inject_event(
                    breakdown_time,
                    EventType.MACHINE_BREAKDOWN,
                    mid,
                    repair_time=repair_time,
                )

    def get_comparison_summary(self, key: str = "rule_comparison") -> list[dict]:
        """获取对比摘要"""
        results = self.scenario_results.get(key, [])
        return [r.to_dict() for r in results]
