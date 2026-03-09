"""
在线调度引擎
===========
训练好的 PDR 在生产环境中的实时执行层。
毫秒级事件驱动决策，无需调用 LLM。

论文 Section 3: Online PDR-based scheduling
每个决策步: 提取特征 → 选择规则 → 计算优先级 → 执行分配
"""
import time
import logging
from typing import Optional, Callable
from dataclasses import dataclass, field

from .models import (
    ShopFloor, Job, Machine, Operation, Order, AssemblyGroup,
    JobStatus, MachineStatus, StageType, generate_uid
)
from .heterogeneous_graph import HeterogeneousGraph
from .feature_encoder import FeatureEncoder
from .dispatching_rules import BUILTIN_RULES, compile_rule_from_code

logger = logging.getLogger(__name__)


@dataclass
class DispatchDecision:
    """一次调度决策的记录"""
    timestamp: float
    job_id: str
    operation_id: str
    machine_id: str
    priority_score: float
    decision_time_ms: float  # 决策耗时（毫秒）
    features_snapshot: dict = field(default_factory=dict)


@dataclass
class SchedulerState:
    """调度器运行状态"""
    is_running: bool = False
    current_time: float = 0.0
    total_decisions: int = 0
    avg_decision_time_ms: float = 0.0
    active_rule_name: str = ""
    last_decision: Optional[DispatchDecision] = None


class OnlineScheduler:
    """
    在线调度器 - 事件驱动的实时调度引擎
    
    工作流程:
    1. 监听事件（机器空闲、作业到达、设备故障等）
    2. 提取当前系统特征
    3. 用已训练的 PDR 计算所有可行 (作业,机器) 对的优先级
    4. 执行最优分配
    5. 更新图状态
    """

    def __init__(self, shop: ShopFloor):
        self.shop = shop
        self.graph = HeterogeneousGraph()
        self.encoder = FeatureEncoder(shop)
        self.state = SchedulerState()

        # 当前活跃的调度规则（可以为不同阶段配置不同规则）
        self.stage_rules: dict[str, Callable] = {}  # {stage_id: pdr_function}
        self.default_rule: Optional[Callable] = None

        # 调度历史
        self.decision_history: list[DispatchDecision] = []

        # 事件回调
        self.event_callbacks: dict[str, list[Callable]] = {}

        # 构建初始图
        self.graph.build_from_shopfloor(shop)

    def set_rule(self, rule_name_or_code: str, stage_id: Optional[str] = None):
        """
        设置调度规则
        支持内置规则名称或自定义代码
        可为特定阶段设置专属规则（论文中加工和组装阶段使用不同 PDR）
        """
        # 尝试从内置规则获取
        if rule_name_or_code in BUILTIN_RULES:
            func = BUILTIN_RULES[rule_name_or_code]
        else:
            func = compile_rule_from_code(rule_name_or_code)

        if stage_id:
            self.stage_rules[stage_id] = func
            logger.info(f"Set rule for stage {stage_id}")
        else:
            self.default_rule = func
            self.state.active_rule_name = rule_name_or_code[:50]
            logger.info(f"Set default rule: {rule_name_or_code[:50]}")

    def set_rule_function(self, func: Callable, stage_id: Optional[str] = None):
        """直接设置规则函数"""
        if stage_id:
            self.stage_rules[stage_id] = func
        else:
            self.default_rule = func

    def _get_rule_for_job(self, job: Job) -> Callable:
        """获取作业对应阶段的规则"""
        if job.stage_id in self.stage_rules:
            return self.stage_rules[job.stage_id]
        return self.default_rule or BUILTIN_RULES["ATC"]

    # ============================================================
    # 事件处理
    # ============================================================

    def on_machine_idle(self, machine_id: str, current_time: float) -> Optional[DispatchDecision]:
        """
        事件: 机器空闲 → 触发调度决策
        论文核心决策过程
        """
        t_start = time.time()
        self.state.current_time = current_time

        machine = self.shop.machines.get(machine_id)
        if not machine or machine.status != MachineStatus.IDLE:
            return None

        # 获取所有可行 (operation, machine) 对
        all_feasible = self.graph.get_feasible_pairs(self.shop, current_time)
        machine_feasible = [
            (op_id, mid, op) for op_id, mid, op in all_feasible
            if mid == machine_id
        ]

        if not machine_feasible:
            return None

        # 用 PDR 计算每对的优先级
        best_score = float('-inf')
        best_op = None
        best_job = None
        best_features = {}

        for op_id, mid, op in machine_feasible:
            job = self.shop.jobs.get(op.job_id)
            if not job:
                continue

            rule = self._get_rule_for_job(job)
            features = self.encoder.encode(job, machine, current_time)
            feat_dict = features.to_dict()

            try:
                score = rule(job, machine, feat_dict, self.shop)
            except Exception:
                score = 0.0

            if score > best_score:
                best_score = score
                best_op = op
                best_job = job
                best_features = feat_dict

        if not best_op or not best_job:
            return None

        # 执行分配
        decision_time = (time.time() - t_start) * 1000  # ms

        decision = DispatchDecision(
            timestamp=current_time,
            job_id=best_job.id,
            operation_id=best_op.id,
            machine_id=machine_id,
            priority_score=best_score,
            decision_time_ms=decision_time,
            features_snapshot=best_features,
        )

        self.decision_history.append(decision)
        self.state.total_decisions += 1
        self.state.last_decision = decision

        # 更新平均决策时间
        total_time = sum(d.decision_time_ms for d in self.decision_history[-100:])
        count = min(len(self.decision_history), 100)
        self.state.avg_decision_time_ms = total_time / count

        return decision

    def on_new_order(self, order: Order, jobs: list[Job],
                     assembly_groups: list[AssemblyGroup] = None):
        """
        事件: 新订单到达
        增量更新图结构，不需要重建
        """
        # 加入车间数据
        self.shop.orders[order.id] = order
        for job in jobs:
            self.shop.jobs[job.id] = job
            self.graph.add_job_to_graph(job, self.shop)
            order.job_ids.append(job.id)

        if assembly_groups:
            for ag in assembly_groups:
                self.shop.assembly_groups[ag.id] = ag
                order.assembly_group_ids.append(ag.id)

        logger.info(f"New order {order.id} added: {len(jobs)} jobs")

    def on_job_complete(self, job_id: str, current_time: float):
        """事件: 作业完成"""
        job = self.shop.jobs.get(job_id)
        if job:
            job.status = JobStatus.COMPLETED
            job.completion_time = current_time
            self.graph.update_node_status(f"J:{job_id}", status="completed")

    def on_machine_breakdown(self, machine_id: str, current_time: float,
                              estimated_repair_time: float):
        """事件: 设备故障"""
        machine = self.shop.machines.get(machine_id)
        if not machine:
            return

        machine.status = MachineStatus.BREAKDOWN
        self.graph.update_node_status(f"M:{machine_id}", status="breakdown")

        # 当前作业中断
        if machine.current_job_id:
            job = self.shop.jobs.get(machine.current_job_id)
            if job:
                for op in job.operations:
                    if op.status == JobStatus.PROCESSING and op.assigned_machine_id == machine_id:
                        op.status = JobStatus.PENDING
                        op.assigned_machine_id = None
                job.status = JobStatus.PENDING
            machine.current_job_id = None

        logger.warning(
            f"Machine {machine_id} breakdown at {current_time}, "
            f"estimated repair: {estimated_repair_time}"
        )

    def on_machine_repair(self, machine_id: str, current_time: float):
        """事件: 设备修复"""
        machine = self.shop.machines.get(machine_id)
        if machine:
            machine.status = MachineStatus.IDLE
            self.graph.update_node_status(f"M:{machine_id}", status="idle")

    # ============================================================
    # 状态查询
    # ============================================================

    def get_current_status(self) -> dict:
        """获取当前调度状态摘要"""
        idle_machines = sum(
            1 for m in self.shop.machines.values()
            if m.status == MachineStatus.IDLE
        )
        busy_machines = sum(
            1 for m in self.shop.machines.values()
            if m.status == MachineStatus.BUSY
        )
        pending_jobs = sum(
            1 for j in self.shop.jobs.values()
            if j.status in (JobStatus.PENDING, JobStatus.WAITING)
        )
        completed_jobs = sum(
            1 for j in self.shop.jobs.values()
            if j.status == JobStatus.COMPLETED
        )

        return {
            "scheduler_state": {
                "is_running": self.state.is_running,
                "current_time": self.state.current_time,
                "total_decisions": self.state.total_decisions,
                "avg_decision_time_ms": round(self.state.avg_decision_time_ms, 3),
                "active_rule": self.state.active_rule_name,
            },
            "shop_status": {
                "idle_machines": idle_machines,
                "busy_machines": busy_machines,
                "pending_jobs": pending_jobs,
                "completed_jobs": completed_jobs,
                "total_jobs": len(self.shop.jobs),
                "total_orders": len(self.shop.orders),
            },
            "graph_stats": self.graph.get_graph_stats(),
        }

    def get_schedule_gantt_data(self) -> list[dict]:
        """获取甘特图数据"""
        return [
            {
                "job_id": d.job_id,
                "operation_id": d.operation_id,
                "machine_id": d.machine_id,
                "start_time": d.timestamp,
                "priority_score": d.priority_score,
            }
            for d in self.decision_history
        ]
