"""延误归因：把「订单为什么晚」拆成五笔各自对应不同处置手段的账。

存在的理由：此前回答「哪台机器是瓶颈」用的是 rank_machines_by_utilization——
那只是全体机器按全周期利用率排的榜，既不看订单是否延误，也不看机器是否真的
卡住了谁。一台只在白班开工、利用率天然偏低的机器会被误判成不是瓶颈；反过来
一台闲着的机器也会因为排在榜上而被当成瓶颈。

这里换一套口径：只看**延误订单**的工序，逐道算它从可开工到真正开工之间等了多久，
再把这段等待按成因拆开——

    capacity_bound  被分配的机器在干别的活，且**同期其他可用机器也都忙**
                    → 真·产能不足，加机台有用
    dispatch_bound  被分配的机器忙，但同期有其他可用机器空着
                    → 选机/派工问题，加机台没用，得改规则或放宽选机范围
    off_shift       那段时间根本没排班        → 工时不足，加班次有用
    downtime        设备停机                  → 检修计划问题
    idle            被分配的机器空着却没开工   → 卡在工装 / 人员 / 前置资源

五者之和恒等于总等待，可直接当成"这次延误该怪谁"的分账表。

实现上一律走**区间集合**而不是时长累加：工序跨班次时墙钟跨度远大于真实占机
（班中休息夹在里面），只有把占机区间与机器可用窗口求交才是真实占用——否则会
虚增 capacity 并把真实空闲吞掉。
"""

from __future__ import annotations

import math

from ..core.models import (
    Operation,
    ShopFloor,
    _merge_windows,
    _subtract_windows,
)


MAX_TOP_WAITS = 20

WAIT_CAUSES = ("capacity_bound", "dispatch_bound", "off_shift", "downtime", "idle")

METHOD_NOTE = (
    "只统计延误订单的工序等待（可开工时刻→实际开工时刻），按成因五分："
    "capacity_bound=被分配的机器在干别的活且同期其他可用机器也都忙（真·产能不足，加机台有用）、"
    "dispatch_bound=被分配的机器忙但同期有其他可用机器空着（选机/派工问题，加机台没用）、"
    "off_shift=该时段没排班（工时不足）、downtime=设备停机、"
    "idle=被分配的机器空着却没开工（卡在工装/人员/前置资源）。"
    "capacity_wait_hours 是判定瓶颈的依据；"
    "利用率排行榜（machine_utilization_ranking）不能用来回答瓶颈问题。"
    "注意 total_order_tardiness_hours 是**订单级**延误（按订单最后一道工序完工 vs 订单交期），"
    "与方案 KPI 里的 total_tardiness（**任务级**，逐 task 累加）口径不同，两个数字对不上是正常的，"
    "不要混着报给用户。"
    "未完整排入本方案的订单不参与等待归因，单独列在 unscheduled_order_ids / "
    "partially_scheduled_order_ids 里——它们的延误未知，比已知延误更需要处理。"
)


def analyze_delay_attribution(
    shop: ShopFloor,
    schedule: list[dict],
    *,
    machine_limit: int = 20,
) -> dict:
    """按机器汇总延误归因，按 capacity_wait_hours 降序。

    machine_limit 只截断展示条数；wait_breakdown 始终基于全部机器统计。
    """
    index = _ScheduleIndex(shop, schedule)
    machines: dict[str, dict] = {}
    totals = dict.fromkeys(WAIT_CAUSES, 0.0)

    for split in index.iter_tardy_waits():
        bucket = machines.get(split.machine_id)
        if bucket is None:
            bucket = machines[split.machine_id] = _machine_bucket(shop, split.machine_id)
        bucket["capacity_wait_hours"] += split.capacity_bound
        bucket["dispatch_wait_hours"] += split.dispatch_bound
        bucket["off_shift_wait_hours"] += split.off_shift
        bucket["downtime_wait_hours"] += split.downtime
        bucket["idle_wait_hours"] += split.idle
        bucket["total_wait_hours"] += split.wait
        bucket["operation_count"] += 1
        bucket["_orders"].add(split.order_id)
        for key in totals:
            totals[key] += getattr(split, key)

    ranked = sorted(
        machines.values(),
        key=lambda item: (-item["capacity_wait_hours"], item["machine_id"]),
    )
    for bucket in ranked:
        orders = bucket.pop("_orders")
        bucket["affected_order_ids"] = sorted(orders)
        bucket["affected_order_count"] = len(orders)
        bucket["saturation"] = index.saturation(bucket["machine_id"])
        for key, value in bucket.items():
            if key.endswith("_hours"):
                bucket[key] = round(value, 3)

    return {
        "tardy_order_count": len(index.tardy_orders),
        "total_order_tardiness_hours": round(sum(index.tardy_orders.values()), 3),
        "unscheduled_order_count": len(index.unscheduled_order_ids),
        "unscheduled_order_ids": index.unscheduled_order_ids,
        "partially_scheduled_order_count": len(index.partially_scheduled_order_ids),
        "partially_scheduled_order_ids": index.partially_scheduled_order_ids,
        "wait_breakdown": {key: round(value, 3) for key, value in totals.items()},
        "machines": ranked[: max(1, int(machine_limit))] if ranked else [],
        "method_note": METHOD_NOTE,
    }


def explain_order_delay(shop: ShopFloor, schedule: list[dict], order_id: str) -> dict:
    """单订单延误归因：先分出工艺本身的必然延误，再把剩下的等待按成因拆开。"""
    order = shop.orders.get(order_id)
    if order is None:
        raise KeyError(order_id)

    index = _ScheduleIndex(shop, schedule)
    completion = index.order_completion.get(order_id)
    scheduled, total = index.coverage(order_id)
    planned = total > 0 and scheduled == total
    # 没排全的订单，completion 只是"已排部分的最晚完工"，据此算延误会系统性低估——
    # 最坏的情况（一道没排）反而会报成准时。宁可给 None，也不给一个偏乐观的数。
    tardiness = index.tardy_orders.get(order_id, 0.0) if planned else None

    attribution = dict.fromkeys(WAIT_CAUSES, 0.0)
    top_waits = []
    if tardiness:
        for split in index.iter_waits(order_id):
            for key in attribution:
                attribution[key] += getattr(split, key)
            machine = shop.machines.get(split.machine_id)
            operation = shop.operations.get(split.operation_id)
            top_waits.append({
                "operation_id": split.operation_id,
                "operation_name": operation.name if operation is not None else "",
                "machine_id": split.machine_id,
                "machine_name": machine.name if machine is not None else "",
                "ready_hours": round(split.ready, 3),
                "ready_at": shop.time_label(split.ready),
                "start_hours": round(split.start, 3),
                "start_at": shop.time_label(split.start),
                "wait_hours": round(split.wait, 3),
                "capacity_bound_hours": round(split.capacity_bound, 3),
                "dispatch_bound_hours": round(split.dispatch_bound, 3),
                "off_shift_hours": round(split.off_shift, 3),
                "downtime_hours": round(split.downtime, 3),
                "idle_hours": round(split.idle, 3),
            })
        top_waits.sort(key=lambda item: (-item["wait_hours"], item["operation_id"]))
        del top_waits[MAX_TOP_WAITS:]

    return {
        "order_id": order_id,
        "order_name": order.name,
        "planned": planned,
        "scheduled_operation_count": scheduled,
        "total_operation_count": total,
        "due_hours": _rounded(order.due_date),
        "due_at": shop.time_label(_finite(order.due_date)),
        "completion_hours": _rounded(completion),
        "completion_at": shop.time_label(completion),
        "tardiness_hours": None if tardiness is None else round(tardiness, 3),
        "inevitable_tardiness_hours": round(_inevitable_tardiness(shop, order_id), 3),
        "attribution": {key: round(value, 3) for key, value in attribution.items()},
        "top_waits": top_waits,
        "method_note": METHOD_NOTE,
    }


class _WaitSplit:
    """一道工序的等待时长及其五段成因。"""

    __slots__ = (
        "operation_id", "order_id", "machine_id", "ready", "start", "wait",
        *WAIT_CAUSES,
    )

    def __init__(self, **fields) -> None:
        for key, value in fields.items():
            setattr(self, key, value)


class _ScheduleIndex:
    """把一个方案的 schedule 预处理成归因需要的几张索引。"""

    def __init__(self, shop: ShopFloor, schedule: list[dict]) -> None:
        self._shop = shop
        self._entries: list[dict] = []
        self._ends: dict[str, float] = {}
        self._by_machine: dict[str, list[tuple[float, float, str]]] = {}
        self._by_order: dict[str, list[dict]] = {}
        self.order_completion: dict[str, float] = {}

        horizon = 0.0
        for raw in schedule or []:
            operation_id = str(raw.get("op_id") or "")
            start = _finite(raw.get("start"))
            end = _finite(raw.get("end"))
            if not operation_id or start is None or end is None:
                continue
            order_id = self._entry_order_id(raw, operation_id)
            entry = {
                "op_id": operation_id,
                "machine_id": str(raw.get("machine_id") or ""),
                "start": start,
                "end": end,
                "order_id": order_id,
            }
            self._entries.append(entry)
            self._ends[operation_id] = end
            self._by_machine.setdefault(entry["machine_id"], []).append((start, end, operation_id))
            if order_id:
                self._by_order.setdefault(order_id, []).append(entry)
                previous = self.order_completion.get(order_id)
                self.order_completion[order_id] = end if previous is None else max(previous, end)
            horizon = max(horizon, end)

        self._horizon = horizon
        self._scheduled_ops = set(self._ends)
        self.tardy_orders = self._tardy_orders()
        self.unscheduled_order_ids, self.partially_scheduled_order_ids = self._coverage_gaps()

    # --- 对外 ---------------------------------------------------------------

    def iter_tardy_waits(self):
        for order_id in self.tardy_orders:
            yield from self.iter_waits(order_id)

    def iter_waits(self, order_id: str):
        for entry in self._by_order.get(order_id, []):
            split = self._split(entry)
            if split is not None:
                yield split

    def coverage(self, order_id: str) -> tuple[int, int]:
        """(已排工序数, 订单工序总数)。"""
        order = self._shop.orders.get(order_id)
        if order is None:
            return 0, 0
        total = scheduled = 0
        for task_id in order.task_ids:
            task = self._shop.tasks.get(task_id)
            if task is None:
                continue
            for operation in task.operations:
                total += 1
                scheduled += operation.id in self._scheduled_ops
        return scheduled, total

    def saturation(self, machine_id: str) -> float | None:
        """该机器在计划跨度内的占用率：真实占机 / 净可用工时（均已扣班次外与停机）。

        分子必须把占机区间与可用窗口求交：跨班次的工序墙钟跨度含班中休息，
        直接累加 end-start 会把休息算成忙碌，饱和度虚高到 1.0。
        """
        machine = self._shop.machines.get(machine_id)
        if machine is None or self._horizon <= 0.0:
            return None
        available = machine.available_time_between(0.0, self._horizon)
        if available <= 1e-9:
            return None
        busy = sum(
            machine.available_time_between(start, end)
            for start, end, _ in self._by_machine.get(machine_id, [])
        )
        return round(min(1.0, busy / available), 4)

    # --- 内部 ---------------------------------------------------------------

    def _coverage_gaps(self) -> tuple[list[str], list[str]]:
        unscheduled, partial = [], []
        for order_id in self._shop.orders:
            scheduled, total = self.coverage(order_id)
            if total == 0 or scheduled == total:
                continue
            (unscheduled if scheduled == 0 else partial).append(order_id)
        return sorted(unscheduled), sorted(partial)

    def _tardy_orders(self) -> dict[str, float]:
        tardy: dict[str, float] = {}
        for order_id, completion in self.order_completion.items():
            order = self._shop.orders.get(order_id)
            if order is None:
                continue
            due = _finite(order.due_date)
            if due is None:
                continue
            tardiness = completion - due
            if tardiness > 1e-9:
                tardy[order_id] = tardiness
        return tardy

    def _entry_order_id(self, raw: dict, operation_id: str) -> str:
        explicit = raw.get("order_id")
        if explicit:
            return str(explicit)
        operation = self._shop.operations.get(operation_id)
        task = self._shop.tasks.get(operation.task_id) if operation else None
        return task.order_id if task else ""

    def _ready_time(self, operation: Operation) -> float:
        """可开工时刻：订单/任务放行 与 全部前驱流转完成 的较大者。

        与 ShopFloor.get_operation_flow_ready_time 同口径，但那份读的是 op.end_time
        运行时状态；这里必须从某个方案的 schedule 反推，故单独实现。
        """
        gate = self._shop.get_operation_release_time(operation)
        for predecessor_id in operation.predecessor_ops:
            predecessor = self._shop.operations.get(predecessor_id)
            end = self._ends.get(predecessor_id)
            if predecessor is not None and end is not None:
                gate = max(gate, end + predecessor.turnover_time)
        for task_id in operation.predecessor_tasks:
            task = self._shop.tasks.get(task_id)
            if task is None:
                continue
            for task_op in task.operations:
                end = self._ends.get(task_op.id)
                if end is not None:
                    gate = max(gate, end + task_op.turnover_time)
        return gate

    def _split(self, entry: dict) -> _WaitSplit | None:
        operation = self._shop.operations.get(entry["op_id"])
        if operation is None:
            return None
        ready = self._ready_time(operation)
        start = entry["start"]
        wait = start - ready
        if wait <= 1e-9:
            return None

        machine = self._shop.machines.get(entry["machine_id"])
        if machine is None:
            # 机器不在实例里（历史方案引用了已删除设备），无法归因到成因，整段记为 idle。
            return self._make_split(entry, ready, start, wait, idle=wait)

        # 逐段做区间运算而不是累加时长：跨班次的工序墙钟跨度含班中休息，
        # 只有与可用窗口求交才是真实占机。
        available_windows = machine.available_windows_between(ready, start)
        available = _windows_length(available_windows)
        shift_time = machine.shift_time_between(ready, start)
        off_shift = max(0.0, wait - shift_time)
        downtime = max(0.0, shift_time - available)

        busy_windows = _intersect(
            available_windows, self._occupancy(entry["machine_id"], ready, start, entry["op_id"])
        )
        idle = max(0.0, available - _windows_length(busy_windows))
        if not busy_windows:
            return self._make_split(
                entry, ready, start, wait,
                off_shift=off_shift, downtime=downtime, idle=idle,
            )

        # 分配的机器忙着——但同期若有别的可用机器空着，这段等待就是选机问题而非产能不足。
        alternatives = self._alternatives_free(operation, entry["machine_id"], ready, start)
        dispatch_windows = _intersect(busy_windows, alternatives)
        dispatch = _windows_length(dispatch_windows)
        return self._make_split(
            entry, ready, start, wait,
            capacity_bound=max(0.0, _windows_length(busy_windows) - dispatch),
            dispatch_bound=dispatch,
            off_shift=off_shift, downtime=downtime, idle=idle,
        )

    @staticmethod
    def _make_split(entry: dict, ready: float, start: float, wait: float, **causes) -> _WaitSplit:
        return _WaitSplit(
            operation_id=entry["op_id"], order_id=entry["order_id"],
            machine_id=entry["machine_id"], ready=ready, start=start, wait=wait,
            **{key: causes.get(key, 0.0) for key in WAIT_CAUSES},
        )

    def _occupancy(
        self, machine_id: str, window_start: float, window_end: float, exclude_op: str = "",
    ) -> list[tuple[float, float]]:
        """该机器在窗口内被占用的区间（墙钟口径，调用方负责与可用窗口求交）。"""
        clipped = []
        for start, end, operation_id in self._by_machine.get(machine_id, []):
            if operation_id == exclude_op:
                continue
            overlap_start, overlap_end = max(start, window_start), min(end, window_end)
            if overlap_end > overlap_start:
                clipped.append((overlap_start, overlap_end))
        return _merge_windows(clipped)

    def _alternatives_free(
        self, operation: Operation, assigned_machine_id: str, ready: float, start: float,
    ) -> list[tuple[float, float]]:
        """除被分配机器外，其他合格机器"可用且空闲"的时段并集。"""
        windows: list[tuple[float, float]] = []
        for machine in self._shop.get_eligible_machines(operation):
            if machine.id == assigned_machine_id:
                continue
            free = _subtract_windows(
                machine.available_windows_between(ready, start),
                self._occupancy(machine.id, ready, start),
            )
            windows.extend(free)
        return _merge_windows(windows)


def _machine_bucket(shop: ShopFloor, machine_id: str) -> dict:
    machine = shop.machines.get(machine_id)
    machine_type = shop.machine_types.get(machine.type_id) if machine else None
    return {
        "machine_id": machine_id,
        "machine_name": machine.name if machine else "",
        "type_id": machine.type_id if machine else "",
        "type_name": machine_type.name if machine_type else "",
        "is_critical": bool(machine_type.is_critical) if machine_type else False,
        "capacity_wait_hours": 0.0,
        "dispatch_wait_hours": 0.0,
        "off_shift_wait_hours": 0.0,
        "downtime_wait_hours": 0.0,
        "idle_wait_hours": 0.0,
        "total_wait_hours": 0.0,
        "operation_count": 0,
        "_orders": set(),
    }


def _inevitable_tardiness(shop: ShopFloor, order_id: str) -> float:
    """即使资源无限也无法避免的延误——只受释放时刻与工艺依赖约束。

    与 /api/instance/validate 的「关键路径理想最早完工晚于交期」同源。
    """
    order = shop.orders.get(order_id)
    due = _finite(order.due_date) if order is not None else None
    if order is None or due is None:
        return 0.0
    shop.derive_internal_targets()
    task_ids = [order.main_task_id] if order.main_task_id else list(order.task_ids)
    finishes = [
        shop.tasks[task_id].earliest_finish_time
        for task_id in task_ids
        if task_id in shop.tasks and math.isfinite(shop.tasks[task_id].earliest_finish_time)
    ]
    if not finishes:
        return 0.0
    return max(0.0, max(finishes) - due)


def _windows_length(windows: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in windows)


def _intersect(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """A ∩ B = A − (A − B)，用现成的 _subtract_windows 组合，不新造区间原语。"""
    if not left or not right:
        return []
    return _subtract_windows(left, _subtract_windows(left, right))


def _finite(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 3)
