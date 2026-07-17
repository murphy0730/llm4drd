"""可复用仿真运行时
=================
把 Simulator 每次 run() 都要重建的静态数据——深拷贝、日历编译、派生时刻、
候选资源缓存、依赖环检测——提为一次性构建；动态字段快照后 reset() 以 O(N)
恢复初始状态。

一个 runtime 同一时刻只能被一个 Simulator 使用（非线程安全）；并行评估时
用 SimulationRuntimePool 为每个 worker 提供互不共享的实例。
"""
from __future__ import annotations

import copy
from collections import defaultdict
from queue import Empty, SimpleQueue
from threading import Lock

from .models import ShopFloor


def _iterative_tarjan_scc(nodes, adj):
    """迭代式 Tarjan 强连通分量，避免大实例递归爆栈。

    adj: dict[node -> 可迭代的前驱节点]。返回 list[list[node]]，其中长度>1
    或含自环的单节点分量即为依赖环。
    """
    index_counter = [0]
    index: dict = {}
    lowlink: dict = {}
    on_stack: dict = {}
    stack: list = []
    result: list = []
    for start in nodes:
        if start in index:
            continue
        work = [(start, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                index[v] = index_counter[0]
                lowlink[v] = index_counter[0]
                index_counter[0] += 1
                stack.append(v)
                on_stack[v] = True
            recurse = False
            neighbors = list(adj[v])
            i = pi
            while i < len(neighbors):
                w = neighbors[i]
                if w not in index:
                    work[-1] = (v, i + 1)
                    work.append((w, 0))
                    recurse = True
                    break
                elif on_stack.get(w):
                    lowlink[v] = min(lowlink[v], index[w])
                i += 1
            if recurse:
                continue
            if lowlink[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == v:
                        break
                result.append(comp)
            work.pop()
            if work:
                u = work[-1][0]
                lowlink[u] = min(lowlink[u], lowlink[v])
    return result


def detect_dependency_cycles(shop: ShopFloor) -> list[dict]:
    """仿真前用 Tarjan SCC 检测依赖环，并区分环类型。

    工序就绪需 predecessor_ops 全完成 + predecessor_tasks 全完成(任务完成=
    该任务所有工序完成)。故依赖图边 = 工序前驱 + 任务前驱展开为该任务全部工序。
    SCC(长度>1 或自环) 即依赖环。按环内边来源区分:
      - op   : 仅工序级前驱(predecessor_ops)闭合——改 predecessor_ops 打断
      - task : 仅任务前驱(predecessor_tasks)展开闭合——改任务令前驱打断
      - mixed: 两者都有
    返回 [{"kind","size","ops","tasks"}, ...]，按 size 降序。
    """
    task_op_ids: dict[str, list[str]] = defaultdict(list)
    for op in shop.operations.values():
        task_op_ids[op.task_id].append(op.id)

    adj: dict[str, set[str]] = {op.id: set() for op in shop.operations.values()}
    for op in shop.operations.values():
        for p in op.predecessor_ops:
            if p in adj and p != op.id:
                adj[op.id].add(p)
        for t in op.predecessor_tasks:
            for po in task_op_ids.get(t, []):
                if po in adj and po != op.id:
                    adj[op.id].add(po)

    sccs = _iterative_tarjan_scc(list(adj.keys()), adj)
    cycles: list[dict] = []
    for comp in sccs:
        if len(comp) <= 1:
            nid = comp[0] if comp else None
            if nid and nid in adj.get(nid, set()):
                comp = [nid]
            else:
                continue
        comp_set = set(comp)
        op_edges = 0
        task_edges = 0
        for oid in comp:
            op_obj = shop.operations[oid]
            op_edges += sum(1 for p in op_obj.predecessor_ops if p in comp_set and p != oid)
            for t in op_obj.predecessor_tasks:
                task_edges += sum(1 for po in task_op_ids.get(t, []) if po in comp_set and po != oid)
        if op_edges and task_edges:
            kind = "mixed"
        elif task_edges:
            kind = "task"
        else:
            kind = "op"
        cycles.append({
            "kind": kind,
            "size": len(comp),
            "ops": sorted(comp),
            "tasks": sorted({shop.operations[oid].task_id for oid in comp}),
        })
    cycles.sort(key=lambda c: c["size"], reverse=True)
    return cycles


class SimulationRuntime:
    def __init__(self, shop: ShopFloor):
        self.shop = copy.deepcopy(shop)
        self.shop.build_indexes()
        self.dependency_cycles = detect_dependency_cycles(self.shop)

        self.eligible_machine_ids: dict[str, set[str]] = {}
        self.op_dispatch_type_ids: dict[str, set[str]] = {}
        self.tooling_candidates: dict[str, dict[str, list]] = {}
        self.personnel_candidates: dict[str, dict[str, list]] = {}
        self.release_time_cache: dict[str, float] = {}
        self.dependent_ops_by_op: dict[str, list[str]] = {
            op_id: [] for op_id in self.shop.operations
        }
        self.dependent_ops_by_task: dict[str, list[str]] = {
            task_id: [] for task_id in self.shop.tasks
        }
        self.task_op_counts: dict[str, int] = {
            task_id: len(task.operations) for task_id, task in self.shop.tasks.items()
        }
        for op_id, op in self.shop.operations.items():
            eligible_machines = self.shop.get_eligible_machines(op)
            self.eligible_machine_ids[op_id] = {machine.id for machine in eligible_machines}
            # 派工桶必须按"可用机台的实际类型"建立：工序显式指定 eligible_machine_ids 时，
            # 机台类型可能不等于工序的 process_type——若仍按 process_type 建桶，
            # 指定机台所在类型的机器永远扫描不到该工序，导致其被永久饿死
            # （表现为"前驱已完成但抢不到资源"，且与派工规则无关）。
            dispatch_types = {machine.type_id for machine in eligible_machines}
            self.op_dispatch_type_ids[op_id] = dispatch_types or {op.process_type}
            self.tooling_candidates[op_id] = {
                tooling_type: list(self.shop.get_toolings_for_type(tooling_type))
                for tooling_type in op.required_tooling_types
            }
            self.personnel_candidates[op_id] = {
                skill_id: list(self.shop.get_personnel_for_skill(skill_id))
                for skill_id in op.required_personnel_skills
            }
            self.release_time_cache[op_id] = self.shop.get_operation_release_time(op)
            for predecessor_id in op.predecessor_ops:
                self.dependent_ops_by_op.setdefault(predecessor_id, []).append(op_id)
            for predecessor_task_id in op.predecessor_tasks:
                self.dependent_ops_by_task.setdefault(predecessor_task_id, []).append(op_id)

        self._op_snapshot = {
            op_id: (
                op.status,
                op.assigned_machine_id,
                tuple(op.assigned_tooling_ids),
                tuple(op.assigned_personnel_ids),
                op.start_time,
                op.end_time,
                op.remaining_processing_time,
            )
            for op_id, op in self.shop.operations.items()
        }
        self._task_snapshot = {
            task_id: task.completion_time for task_id, task in self.shop.tasks.items()
        }
        # 键必须带资源类型：机器/工装/人员各存一个 dict，模型不保证 ID 跨类型唯一，
        # 只按 id 索引会让同名资源互相覆盖（且静默篡改初始状态）。
        self._resource_snapshot = {
            (kind, resource.id): (
                resource.state,
                resource.current_op_id,
                resource.current_finish_time,
                resource.total_busy_time,
            )
            for kind, resource in self._iter_resources()
        }

    def _iter_resources(self):
        """产出 (kind, resource)；kind 用于区分跨类型同名 ID。"""
        for machine in self.shop.machines.values():
            yield "machine", machine
        for tooling in self.shop.toolings.values():
            yield "tooling", tooling
        for person in self.shop.personnel.values():
            yield "personnel", person

    def reset(self) -> None:
        for op_id, op in self.shop.operations.items():
            (status, machine_id, tooling_ids, personnel_ids,
             start, end, remaining) = self._op_snapshot[op_id]
            op.status = status
            op.assigned_machine_id = machine_id
            op.assigned_tooling_ids = list(tooling_ids)
            op.assigned_personnel_ids = list(personnel_ids)
            op.start_time = start
            op.end_time = end
            op.remaining_processing_time = remaining
        for task_id, task in self.shop.tasks.items():
            task.completion_time = self._task_snapshot[task_id]
        for kind, resource in self._iter_resources():
            (resource.state, resource.current_op_id,
             resource.current_finish_time, resource.total_busy_time) = (
                self._resource_snapshot[(kind, resource.id)]
            )


class SimulationRuntimePool:
    """为并行评估提供互不共享的 runtime；懒创建，至多 max_size 个。"""

    def __init__(self, shop: ShopFloor, max_size: int):
        self._shop = shop
        self._max_size = max(1, max_size)
        self._created = 0
        self._lock = Lock()
        self._idle: SimpleQueue = SimpleQueue()

    def acquire(self) -> SimulationRuntime:
        try:
            return self._idle.get_nowait()
        except Empty:
            pass
        with self._lock:
            if self._created < self._max_size:
                self._created += 1
                return SimulationRuntime(self._shop)
        return self._idle.get()

    def release(self, runtime: SimulationRuntime) -> None:
        self._idle.put(runtime)
