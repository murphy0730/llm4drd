from __future__ import annotations

import math


def edd_rule(op, machine, f, shop):
    return -f["due_date"]


def spt_rule(op, machine, f, shop):
    return -f["processing_time"]


def lpt_rule(op, machine, f, shop):
    return f["processing_time"]


def cr_rule(op, machine, f, shop):
    remaining = f["remaining"]
    if remaining <= 0:
        return 1000.0
    return -(f["slack"] / remaining)


def atc_rule(op, machine, f, shop):
    processing = f["processing_time"]
    if processing <= 0:
        return 1000.0
    slack = max(f["slack"], 0.0)
    return (1.0 / processing) * math.exp(-slack / (2.0 * processing + 0.01))


def fifo_rule(op, machine, f, shop):
    return f.get("wait_time", 0.0)


def mst_rule(op, machine, f, shop):
    return -f["slack"]


def priority_rule(op, machine, f, shop):
    return f["priority"] * 10.0 + f["urgency"]


def kit_aware_rule(op, machine, f, shop):
    return f["prereq_ratio"] * 5.0 + f["urgency"] * 2.0 + f["priority"]


def bottleneck_rule(op, machine, f, shop):
    return f["is_main"] * 8.0 + f["urgency"] * 3.0 + f["priority"] * 2.0 - f["processing_time"] * 0.1


def composite_rule(op, machine, f, shop):
    urgency_bonus = abs(f["slack"]) * 3.0 if f["slack"] < 0 else 0.0
    return (
        urgency_bonus
        + f["priority"] * 2.0
        + f["prereq_ratio"] * 4.0
        + f["is_main"] * 5.0
        - f["processing_time"] * 0.05
        - f.get("tooling_demand", 0.0) * 0.1
        - f.get("personnel_demand", 0.0) * 0.1
    )


BUILTIN_RULES = {
    "EDD": edd_rule,
    "SPT": spt_rule,
    "LPT": lpt_rule,
    "CR": cr_rule,
    "ATC": atc_rule,
    "FIFO": fifo_rule,
    "MST": mst_rule,
    "PRIORITY": priority_rule,
    "KIT_AWARE": kit_aware_rule,
    "BOTTLENECK": bottleneck_rule,
    "COMPOSITE": composite_rule,
}


def get_all_rule_names():
    return list(BUILTIN_RULES.keys())


def compile_rule_from_code(code_str: str, name: str = "evolved_rule"):
    namespace = {"math": math}
    exec(code_str, namespace)
    for key, value in namespace.items():
        if callable(value) and not key.startswith("_") and key not in {"math"}:
            return value
    raise ValueError("No callable found in code")
