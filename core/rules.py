"""
调度规则库 v3 — 适配新数据模型
==============================
每个 PDR: (operation, machine, features_dict, shop) -> float (越大越优先)
"""
import math

def edd_rule(op, machine, f, shop):
    """EDD 最早交期优先"""
    return -f["due_date"]

def spt_rule(op, machine, f, shop):
    """SPT 最短加工时间"""
    return -f["processing_time"]

def lpt_rule(op, machine, f, shop):
    """LPT 最长加工时间"""
    return f["processing_time"]

def cr_rule(op, machine, f, shop):
    """CR 关键比率"""
    r = f["remaining"]
    if r <= 0: return 1000.0
    return -(f["slack"] / r)

def atc_rule(op, machine, f, shop):
    """ATC 表观延迟成本"""
    p = f["processing_time"]
    if p <= 0: return 1000.0
    k = 2.0
    slack = max(f["slack"], 0)
    return (1.0 / p) * math.exp(-slack / (k * p + 0.01))

def fifo_rule(op, machine, f, shop):
    """FIFO 先到先服务"""
    return -f.get("wait_time", 0)

def mst_rule(op, machine, f, shop):
    """MST 最小松弛时间"""
    return -f["slack"]

def priority_rule(op, machine, f, shop):
    """按订单优先级"""
    return f["priority"] * 10 + f["urgency"]

def kit_aware_rule(op, machine, f, shop):
    """配套感知 — 前置完成比高的优先"""
    return f["prereq_ratio"] * 5 + f["urgency"] * 2 + f["priority"]

def bottleneck_rule(op, machine, f, shop):
    """瓶颈感知 — 主任务+高紧急度优先"""
    return f["is_main"] * 8 + f["urgency"] * 3 + f["priority"] * 2 - f["processing_time"] * 0.1

def composite_rule(op, machine, f, shop):
    """综合规则 — 加权组合多因素"""
    slack = f["slack"]
    if slack < 0:
        urg = abs(slack) * 3.0
    else:
        urg = 0.0
    return (urg + f["priority"] * 2 + f["prereq_ratio"] * 4
            + f["is_main"] * 5 - f["processing_time"] * 0.05)


BUILTIN_RULES = {
    "EDD": edd_rule, "SPT": spt_rule, "LPT": lpt_rule,
    "CR": cr_rule, "ATC": atc_rule, "FIFO": fifo_rule,
    "MST": mst_rule, "PRIORITY": priority_rule,
    "KIT_AWARE": kit_aware_rule, "BOTTLENECK": bottleneck_rule,
    "COMPOSITE": composite_rule,
}

def get_all_rule_names(): return list(BUILTIN_RULES.keys())

def compile_rule_from_code(code_str: str, name: str = "evolved_rule"):
    ns = {"math": math}
    exec(code_str, ns)
    for k, v in ns.items():
        if callable(v) and not k.startswith("_") and k not in ("math",):
            return v
    raise ValueError("No callable found in code")
