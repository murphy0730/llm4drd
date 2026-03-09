"""
特征工程模块
===========
论文核心: 动态特征编码器 (Dynamic Feature Encoder)
提取当前系统状态的关键特征，供 PDR 规则使用。
特征拟合进化的基础 - 规则通过这些特征感知系统状态。
"""
import math
from dataclasses import dataclass
from typing import Optional
from .models import (
    ShopFloor, Job, Machine, Operation, AssemblyGroup,
    JobStatus, MachineStatus
)


@dataclass
class JobFeatures:
    """作业级特征"""
    slack_time: float          # 松弛时间 = due_date - now - remaining_processing_time
    remaining_time: float      # 剩余加工时间
    waiting_time: float        # 已等待时间 = now - release_time
    progress_ratio: float      # 工序完成率
    urgency_score: float       # 紧急度 (归一化)
    order_priority: int        # 订单优先级
    is_critical: bool          # 是否在关键路径上


@dataclass
class MachineFeatures:
    """机器级特征"""
    queue_length: int          # 队列长度
    utilization: float         # 利用率
    remaining_work: float      # 当前作业剩余时间
    setup_time: float          # 预计换型时间
    avg_processing_rate: float # 平均加工速率


@dataclass
class KittingFeatures:
    """配套级特征 - 论文中的关键创新"""
    completion_ratio: float    # 组件配套完成度
    bottleneck_remaining: float  # 瓶颈组件剩余时间
    component_dispersion: float  # 同组作业离散度
    group_size: int            # 配套组大小


@dataclass
class SystemFeatures:
    """系统级特征"""
    global_wip: int            # 全局在制品数量
    bottleneck_load: float     # 瓶颈机器负载率
    avg_urgency: float         # 平均紧急度
    utilization_variance: float  # 设备利用率方差
    pending_orders: int        # 待处理订单数
    tardy_ratio: float         # 当前延迟率


@dataclass
class FeatureVector:
    """完整特征向量 - 供 PDR 使用"""
    job: JobFeatures
    machine: MachineFeatures
    kitting: Optional[KittingFeatures]
    system: SystemFeatures

    def to_dict(self) -> dict:
        d = {}
        for prefix, feat in [("job", self.job), ("machine", self.machine), ("system", self.system)]:
            for k, v in feat.__dict__.items():
                d[f"{prefix}_{k}"] = v
        if self.kitting:
            for k, v in self.kitting.__dict__.items():
                d[f"kitting_{k}"] = v
        return d


class FeatureEncoder:
    """
    特征编码器
    论文 Section 4.3: Dynamic Feature-Fitting Rule Evolution
    从当前系统状态中提取特征向量
    """

    def __init__(self, shop: ShopFloor):
        self.shop = shop

    def encode_system(self, current_time: float) -> SystemFeatures:
        """编码系统级特征"""
        jobs = self.shop.jobs
        machines = self.shop.machines

        # 全局 WIP
        wip = sum(
            1 for j in jobs.values()
            if j.status in (JobStatus.PENDING, JobStatus.WAITING, JobStatus.PROCESSING)
        )

        # 机器利用率
        utils = []
        for m in machines.values():
            if current_time > 0:
                utils.append(m.total_busy_time / current_time)
            else:
                utils.append(0.0)

        avg_util = sum(utils) / len(utils) if utils else 0.0
        max_util = max(utils) if utils else 0.0
        util_var = (
            sum((u - avg_util) ** 2 for u in utils) / len(utils)
            if utils else 0.0
        )

        # 紧急度
        urgencies = []
        tardy_count = 0
        pending_count = 0
        for j in jobs.values():
            if j.status == JobStatus.COMPLETED:
                continue
            pending_count += 1
            slack = j.due_date - current_time - j.remaining_processing_time
            if slack < 0:
                tardy_count += 1
            urgencies.append(max(0, -slack))

        avg_urgency = sum(urgencies) / len(urgencies) if urgencies else 0.0

        # 待处理订单
        pending_orders = sum(
            1 for o in self.shop.orders.values() if o.status != "completed"
        )

        return SystemFeatures(
            global_wip=wip,
            bottleneck_load=max_util,
            avg_urgency=avg_urgency,
            utilization_variance=util_var,
            pending_orders=pending_orders,
            tardy_ratio=tardy_count / pending_count if pending_count > 0 else 0.0,
        )

    def encode_job(self, job: Job, current_time: float) -> JobFeatures:
        """编码作业级特征"""
        remaining = job.remaining_processing_time
        slack = job.due_date - current_time - remaining
        waiting = current_time - job.release_time

        # 紧急度归一化
        if slack <= 0:
            urgency = 1.0 + abs(slack) / (remaining + 1)
        else:
            urgency = remaining / (slack + remaining + 1)

        order = self.shop.orders.get(job.order_id)
        priority = order.priority if order else 1

        return JobFeatures(
            slack_time=slack,
            remaining_time=remaining,
            waiting_time=max(0, waiting),
            progress_ratio=job.progress_ratio,
            urgency_score=min(urgency, 5.0),  # 截断
            order_priority=priority,
            is_critical=slack < 0,
        )

    def encode_machine(self, machine: Machine, job: Job, current_time: float) -> MachineFeatures:
        """编码机器级特征"""
        queue_len = len(machine.queue)

        # 利用率
        util = machine.total_busy_time / current_time if current_time > 0 else 0.0

        # 当前作业剩余时间
        if machine.status == MachineStatus.BUSY:
            remaining = max(0, machine.current_job_finish_time - current_time)
        else:
            remaining = 0.0

        # 换型时间
        setup = machine.get_setup_time(job.job_type)

        return MachineFeatures(
            queue_length=queue_len,
            utilization=min(util, 1.0),
            remaining_work=remaining,
            setup_time=setup,
            avg_processing_rate=machine.speed_factor,
        )

    def encode_kitting(self, job: Job, current_time: float) -> Optional[KittingFeatures]:
        """编码配套级特征 - 论文的关键贡献"""
        if not job.assembly_group_id:
            return None

        ag = self.shop.assembly_groups.get(job.assembly_group_id)
        if not ag:
            return None

        jobs_dict = self.shop.jobs
        completion_ratio = ag.get_completion_ratio(jobs_dict)

        # 瓶颈组件剩余时间
        max_remaining = 0.0
        remaining_times = []
        for cjid in ag.component_job_ids:
            if cjid in jobs_dict:
                cj = jobs_dict[cjid]
                if cj.status != JobStatus.COMPLETED:
                    rt = cj.remaining_processing_time
                    remaining_times.append(rt)
                    max_remaining = max(max_remaining, rt)

        # 同组作业离散度（标准差）
        if len(remaining_times) > 1:
            mean_rt = sum(remaining_times) / len(remaining_times)
            dispersion = math.sqrt(
                sum((r - mean_rt) ** 2 for r in remaining_times) / len(remaining_times)
            )
        else:
            dispersion = 0.0

        return KittingFeatures(
            completion_ratio=completion_ratio,
            bottleneck_remaining=max_remaining,
            component_dispersion=dispersion,
            group_size=len(ag.component_job_ids),
        )

    def encode(self, job: Job, machine: Machine, current_time: float) -> FeatureVector:
        """编码完整特征向量"""
        return FeatureVector(
            job=self.encode_job(job, current_time),
            machine=self.encode_machine(machine, job, current_time),
            kitting=self.encode_kitting(job, current_time),
            system=self.encode_system(current_time),
        )

    def encode_global_state(self, current_time: float) -> dict:
        """编码全局状态摘要（用于 LLM-S 评估）"""
        sys_feat = self.encode_system(current_time)
        
        # 各阶段负载
        stage_loads = {}
        for sid, stage in self.shop.stages.items():
            machines_in_stage = [
                self.shop.machines[mid]
                for mid in stage.machine_ids
                if mid in self.shop.machines
            ]
            busy = sum(1 for m in machines_in_stage if m.status == MachineStatus.BUSY)
            total = len(machines_in_stage)
            stage_loads[sid] = busy / total if total > 0 else 0.0

        # 配套状态
        kitting_status = {}
        for agid, ag in self.shop.assembly_groups.items():
            kitting_status[agid] = ag.get_completion_ratio(self.shop.jobs)

        return {
            "system": sys_feat.__dict__,
            "stage_loads": stage_loads,
            "kitting_status": kitting_status,
            "current_time": current_time,
        }
