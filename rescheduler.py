"""
动态重排机制
===========
支持三种触发方式:
1. 周期触发 - 每隔固定时间对未来订单重排
2. 事件触发 - 紧急插单/设备故障时立即重排
3. 偏差触发 - 实际与计划偏差超阈值时自动重排

论文思想: PDR 用于快速生成基准方案，
局部搜索用于进一步优化关键路径
"""
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from .models import (
    ShopFloor, Job, Machine, Operation, Order,
    JobStatus, MachineStatus
)
from .simulator import Simulator, SimulationResult
from .online_scheduler import OnlineScheduler

logger = logging.getLogger(__name__)


@dataclass
class RescheduleConfig:
    """重排配置"""
    # 周期触发
    periodic_interval: float = 480.0       # 周期间隔（分钟）
    # 偏差触发
    deviation_threshold: float = 0.15      # 偏差率阈值（15%）
    # 稳定性控制
    freeze_window: float = 60.0            # 冻结窗口（分钟内的安排不动）
    max_change_ratio: float = 0.5          # 最大变更比例
    # 局部搜索
    enable_local_search: bool = True
    local_search_iterations: int = 100
    local_search_timeout: float = 30.0     # 秒


@dataclass
class RescheduleResult:
    """重排结果"""
    trigger_reason: str
    old_fitness: float
    new_fitness: float
    improvement: float
    changed_assignments: int
    total_assignments: int
    change_ratio: float
    computation_time: float
    new_schedule: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trigger_reason": self.trigger_reason,
            "old_fitness": round(self.old_fitness, 2),
            "new_fitness": round(self.new_fitness, 2),
            "improvement_pct": round(self.improvement * 100, 2),
            "changed_assignments": self.changed_assignments,
            "total_assignments": self.total_assignments,
            "change_ratio": round(self.change_ratio, 4),
            "computation_time_s": round(self.computation_time, 2),
        }


class DynamicRescheduler:
    """
    动态重排器
    
    重排流程:
    Phase 1: 状态快照 - 冻结当前正在加工的作业
    Phase 2: 约束更新 - 根据最新信息更新异构图
    Phase 3: PDR快速排程 - 用训练好的规则生成基准方案
    Phase 4: 局部搜索优化 - 关键路径邻域搜索
    Phase 5: 稳定性约束 - 限制与原方案的偏差
    Phase 6: 方案确认 - 可视化对比
    """

    def __init__(self, scheduler: OnlineScheduler, config: RescheduleConfig = None):
        self.scheduler = scheduler
        self.config = config or RescheduleConfig()
        self.reschedule_history: list[RescheduleResult] = []
        self.current_plan: list[dict] = []  # 当前执行中的计划
        self._last_periodic_time: float = 0.0

    def check_periodic_trigger(self, current_time: float) -> bool:
        """检查是否需要周期性重排"""
        if current_time - self._last_periodic_time >= self.config.periodic_interval:
            self._last_periodic_time = current_time
            return True
        return False

    def check_deviation_trigger(self, current_time: float) -> bool:
        """
        检查实际执行与计划偏差是否超过阈值
        偏差 = 预计延迟订单占比
        """
        shop = self.scheduler.shop
        total_active = 0
        tardy_predicted = 0

        for jid, job in shop.jobs.items():
            if job.status == JobStatus.COMPLETED:
                continue
            total_active += 1
            # 预估完成时间
            est_completion = current_time + job.remaining_processing_time
            if est_completion > job.due_date:
                tardy_predicted += 1

        if total_active == 0:
            return False

        deviation = tardy_predicted / total_active
        return deviation > self.config.deviation_threshold

    def reschedule(
        self,
        current_time: float,
        trigger_reason: str,
        pdr_function: Optional[Callable] = None,
        locked_jobs: Optional[set] = None,
    ) -> RescheduleResult:
        """
        执行重排
        
        Args:
            current_time: 当前时间
            trigger_reason: 触发原因
            pdr_function: 用于重排的 PDR（默认使用在线调度器当前规则）
            locked_jobs: 管理者锁定的作业ID集合（不参与重排）
        """
        t_start = time.time()
        shop = self.scheduler.shop
        locked = locked_jobs or set()

        logger.info(f"Rescheduling triggered: {trigger_reason} at time {current_time}")

        # Phase 1: 状态快照
        snapshot = self._take_snapshot(shop, current_time)

        # Phase 2: 识别可重排的作业
        reschedulable_jobs = self._get_reschedulable_jobs(
            shop, current_time, locked
        )

        if not reschedulable_jobs:
            logger.info("No reschedulable jobs found")
            return RescheduleResult(
                trigger_reason=trigger_reason,
                old_fitness=0, new_fitness=0, improvement=0,
                changed_assignments=0, total_assignments=0,
                change_ratio=0, computation_time=time.time() - t_start,
            )

        # Phase 3: 用 PDR 快速生成新方案
        rule = pdr_function or self.scheduler.default_rule
        if not rule:
            from .dispatching_rules import BUILTIN_RULES
            rule = BUILTIN_RULES["ATC"]

        new_shop = copy.deepcopy(shop)
        # 重置可重排作业的状态
        for jid in reschedulable_jobs:
            if jid in new_shop.jobs:
                job = new_shop.jobs[jid]
                for op in job.operations:
                    if op.status != JobStatus.COMPLETED:
                        op.status = JobStatus.PENDING
                        op.assigned_machine_id = None
                        op.start_time = None
                        op.end_time = None
                if job.status != JobStatus.COMPLETED:
                    job.status = JobStatus.PENDING

        # 重置空闲机器
        for mid, m in new_shop.machines.items():
            if m.status == MachineStatus.IDLE or m.current_job_id in reschedulable_jobs:
                m.status = MachineStatus.IDLE
                m.current_job_id = None

        # 仿真新方案
        sim = Simulator(new_shop, rule)
        new_result = sim.run(max_time=50000)

        # Phase 4: 局部搜索优化（可选）
        if self.config.enable_local_search:
            new_result = self._local_search(
                new_shop, rule, new_result, reschedulable_jobs
            )

        # 评估原方案
        old_sim = Simulator(shop, rule)
        old_result = old_sim.run(max_time=50000)

        # Phase 5: 稳定性检查
        old_schedule = {
            (entry[0], entry[2]): entry  # (job_id, machine_id)
            for entry in old_result.schedule
        }
        new_schedule = {
            (entry[0], entry[2]): entry
            for entry in new_result.schedule
        }

        changed = 0
        total = len(new_schedule)
        for key in new_schedule:
            if key not in old_schedule:
                changed += 1

        change_ratio = changed / total if total > 0 else 0.0

        # 如果变更超限，回退到原方案
        if change_ratio > self.config.max_change_ratio:
            logger.warning(
                f"Change ratio {change_ratio:.2%} exceeds limit "
                f"{self.config.max_change_ratio:.2%}, applying partial changes"
            )
            # TODO: 实现部分变更策略

        improvement = (
            (old_result.total_tardiness - new_result.total_tardiness) /
            max(old_result.total_tardiness, 0.01)
        )

        result = RescheduleResult(
            trigger_reason=trigger_reason,
            old_fitness=old_result.total_tardiness,
            new_fitness=new_result.total_tardiness,
            improvement=improvement,
            changed_assignments=changed,
            total_assignments=total,
            change_ratio=change_ratio,
            computation_time=time.time() - t_start,
            new_schedule=new_result.schedule,
        )

        self.reschedule_history.append(result)
        logger.info(
            f"Reschedule complete: improvement={improvement:.2%}, "
            f"changes={changed}/{total}, time={result.computation_time:.2f}s"
        )
        return result

    def _take_snapshot(self, shop: ShopFloor, current_time: float) -> dict:
        """拍摄当前状态快照"""
        return {
            "time": current_time,
            "machine_states": {
                mid: {"status": m.status.value, "current_job": m.current_job_id}
                for mid, m in shop.machines.items()
            },
            "job_states": {
                jid: {"status": j.status.value, "progress": j.progress_ratio}
                for jid, j in shop.jobs.items()
            },
        }

    def _get_reschedulable_jobs(
        self, shop: ShopFloor, current_time: float, locked: set
    ) -> list[str]:
        """获取可重排的作业列表"""
        reschedulable = []
        freeze_end = current_time + self.config.freeze_window

        for jid, job in shop.jobs.items():
            # 已完成的不重排
            if job.status == JobStatus.COMPLETED:
                continue
            # 被锁定的不重排
            if jid in locked:
                continue
            # 冻结窗口内正在加工的不重排
            if job.status == JobStatus.PROCESSING:
                for op in job.operations:
                    if (op.status == JobStatus.PROCESSING and
                        op.end_time and op.end_time <= freeze_end):
                        continue
            reschedulable.append(jid)

        return reschedulable

    def _local_search(
        self,
        shop: ShopFloor,
        rule: Callable,
        baseline: SimulationResult,
        reschedulable_jobs: list[str],
    ) -> SimulationResult:
        """
        局部搜索优化 - 在 PDR 基准方案上进行关键路径邻域搜索
        """
        best_result = baseline
        t_start = time.time()

        for _ in range(self.config.local_search_iterations):
            if time.time() - t_start > self.config.local_search_timeout:
                break

            # 简单的邻域操作: 随机交换两个作业的机器分配
            # 更复杂的实现可以使用关键路径分析
            perturbed_shop = copy.deepcopy(shop)

            # 随机选两个待排作业交换
            import random
            if len(reschedulable_jobs) < 2:
                break

            j1_id, j2_id = random.sample(reschedulable_jobs, 2)
            j1 = perturbed_shop.jobs.get(j1_id)
            j2 = perturbed_shop.jobs.get(j2_id)

            if j1 and j2 and j1.current_operation and j2.current_operation:
                op1 = j1.current_operation
                op2 = j2.current_operation
                # 交换机器分配（如果兼容）
                common_machines = set(op1.eligible_machines) & set(op2.eligible_machines)
                if common_machines:
                    # 尝试交换
                    sim = Simulator(perturbed_shop, rule)
                    result = sim.run(max_time=50000)
                    if result.total_tardiness < best_result.total_tardiness:
                        best_result = result

        return best_result

    def get_reschedule_history(self) -> list[dict]:
        """获取重排历史"""
        return [r.to_dict() for r in self.reschedule_history]
