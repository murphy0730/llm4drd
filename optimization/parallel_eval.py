"""进程池评估 worker
====================
子进程 initializer 收一次 pickle 的 shop / 图特征 / 尺度，进程内自建
SimulationRuntime 与近似评估器；每个任务只传 CandidateParameters，
收回 SimResult / OptimizationSolution。spawn 启动方式要求本模块的
函数可按限定名导入，故全部为模块级定义。
"""
from __future__ import annotations

import pickle

from ..core.rules import BUILTIN_RULES
from ..core.sim_runtime import SimulationRuntime
from ..core.simulator import SimResult, Simulator
from .approx_eval import ApproximateScheduleEvaluator
from .solution_model import FEATURE_NAMES, CandidateParameters, OptimizationSolution

_STATE: dict = {}


def build_candidate_rule(candidate, graph_features, time_scale, busy_scale,
                         priority_scale, due_scale):
    """由候选参数构造派工规则。

    主进程与子进程共用此单点——HybridNSGA3ALNSOptimizer._dispatch_rule 委托到这里。
    """
    built_in = BUILTIN_RULES.get(candidate.seed_rule_name or "")

    def _rule(op, machine, features, shop):
        graph_values = graph_features.get(op.id, {})
        score_components = {
            "urgency": features["urgency"] / time_scale,
            "slack": -features["slack"] / time_scale,
            "remaining": -features["remaining"] / time_scale,
            "processing_time": -features["processing_time"] / time_scale,
            "priority": features["priority"] / priority_scale,
            "is_main": features["is_main"],
            "wait_time": features["wait_time"] / time_scale,
            "prereq_ratio": features["prereq_ratio"],
            "machine_load": -features["machine_busy_time"] / busy_scale,
            "tooling_demand": -features["tooling_demand"],
            "personnel_demand": -features["personnel_demand"],
            "predecessor_depth": graph_values.get("predecessor_depth", 0.0),
            "assembly_criticality": graph_values.get("assembly_criticality", 0.0),
            "shared_resource_degree": -graph_values.get("shared_resource_degree", 0.0),
            "bottleneck_adjacency": graph_values.get("bottleneck_adjacency", 0.0),
            "due_date": -features["due_date"] / due_scale,
        }
        score = sum(
            candidate.feature_weights.get(name, 0.0) * score_components.get(name, 0.0)
            for name in FEATURE_NAMES
        )
        score += candidate.op_bias.get(op.id, 0.0)
        if built_in is not None:
            try:
                score += 0.18 * built_in(op, machine, features, shop)
            except Exception:
                pass
        return score

    return _rule


def init_worker(payload_path: str) -> None:
    """从磁盘载入一次 payload，进程内自建 runtime 与近似评估器。

    走文件而非 bytes 形参：ProcessPoolExecutor 的 initargs 会被 worker 主循环
    的栈帧一直引用，大实例的 payload（百 MB 级）会在每个 worker 里常驻到进程
    退出，白白占满内存。
    """
    with open(payload_path, "rb") as payload_file:
        payload = pickle.load(payload_file)
    scales = payload["scales"]
    # runtime 直接接管这份刚反序列化出来的私有 shop（不再深拷贝），approx 也复用
    # 同一个对象：三份 shop 拷贝正是大实例 worker 初始化 MemoryError 的来源。
    runtime = SimulationRuntime(payload["shop"], copy_shop=False)
    _STATE["shop"] = runtime.shop
    _STATE["graph_features"] = payload["graph_features"]
    _STATE["scales"] = scales
    _STATE["runtime"] = runtime
    _STATE["approx"] = ApproximateScheduleEvaluator(
        runtime.shop,
        payload["graph_features"],
        scales["time_scale"],
        scales["due_scale"],
        scales["priority_scale"],
        keep_schedule_limit=0,
    )


def run_exact_simulation(candidate: CandidateParameters) -> SimResult:
    scales = _STATE["scales"]
    rule = build_candidate_rule(
        candidate, _STATE["graph_features"],
        scales["time_scale"], scales["busy_scale"],
        scales["priority_scale"], scales["due_scale"],
    )
    simulator = Simulator(_STATE["shop"], rule, runtime=_STATE["runtime"])
    return simulator.run()


def run_approx_evaluation(candidate: CandidateParameters, source: str,
                          generation: int) -> OptimizationSolution:
    # approx 与 runtime 共用同一个 shop（省一份大实例深拷贝），而上一次精确仿真
    # 会把它留在"全部完成"的终态；approx 的 analytics 要读 op.status/end_time，
    # 故必须先复位回初始状态。reset() 是 O(N) 的，相对一次评估可忽略。
    _STATE["runtime"].reset()
    return _STATE["approx"].evaluate(candidate, source, generation)
