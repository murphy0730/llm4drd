# 仿真与优化算法性能优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变任何仿真/优化结果（同 seed 逐位一致）的前提下，消除每次评估的重复构建开销、转库闸门的内环重算，并把评估并行从 GIL 线程换成进程，使优化总耗时下降 5–10 倍。

**Architecture:** 五个独立可验证的改动，按依赖排序：① 仿真器内 memo 化转库闸门；② 近似评估器增量维护任务流转完成时刻；③ 新建 `SimulationRuntime`（一次深拷贝 + 一次静态构建 + O(N) 重置），优化器经运行时池复用；④ 精确/近似批量评估切到 `ProcessPoolExecutor`（worker 子进程各持一份 runtime）；⑤ 解克隆共享不可变负载。全程行为保持不变，由既有冻结基线测试 + 新增等价性测试守护。

**Tech Stack:** Python 标准库（`concurrent.futures`、`pickle`、`queue`），无新第三方依赖。测试用 unittest（与仓库现有测试一致），fixture 复用 `tests/shop_fixtures.py`。

## Global Constraints

- 行为保持：同 seed 下 `hybrid_result_signature`（`tests/shop_fixtures.py:65`）与仿真 `SimResult`（排除 `wall_time_ms`）逐位一致；`tests/test_graph_legacy_baseline.py`、`tests/test_turnover_time.py`、`tests/test_simulator_robustness.py` 全程必须保持通过。
- 测试运行方式：仓库内测试用绝对包名导入（`from llm4drd.tests.shop_fixtures import ...`），所有 pytest 命令从仓库父目录运行：`cd /Users/zhouwentao/Desktop && python -m pytest llm4drd/tests/<file> -q`。
- 开工前从 main 切分支：`git checkout -b perf/simulation-optimizer`。
- 无新第三方依赖；不改 `LLM4DRD_GRAPH_CONTEXT_MODE` 的默认值（图上下文模式切换由用户以环境变量控制，不在本计划范围内）。
- 遵守用户 CLAUDE.md：只改与本计划直接相关的行，不顺手重构无关代码。
- 基准命令（Task 1 记录基线、Task 6 记录结果，两次必须用同一台机器同一命令）：

```bash
cd /Users/zhouwentao/Desktop && python - <<'PY'
import time
from llm4drd.tests.shop_fixtures import make_graph_context_shop
from llm4drd.optimization.hybrid_nsga3_alns import HybridConfig, HybridNSGA3ALNSOptimizer

config = HybridConfig(
    objective_keys=["total_tardiness", "makespan"],
    population_size=12, generations=4, alns_iterations_per_candidate=2,
    time_limit_s=120, parallel_workers=4, seed=17,
)
start = time.perf_counter()
result = HybridNSGA3ALNSOptimizer(make_graph_context_shop(), config).run()
print("elapsed_s:", round(time.perf_counter() - start, 2))
print("exact_evaluations:", result.exact_evaluations,
      "approx_evaluations:", result.approximate_evaluations,
      "exact_eval_time_total:", round(result.exact_eval_time_total, 2))
PY
```

---

### Task 1: 仿真器转库闸门 memo 缓存

转库改造后 `Simulator._flow_ready_time`（`core/simulator.py:591`）在派工内环（`_earliest_feasible_start`、`_features`）里每次调用都委托 `ShopFloor.get_operation_flow_ready_time`（`core/models.py:816`）全量遍历前驱工序与前驱任务的全部工序。关键不变量：**所有调用点都在 `_is_op_ready(op)` 为真之后**（`run()` 的 release_check 分支、`_seed_initial_state`、`_queue_release_or_ready` 均有 `and` 短路保护；`_earliest_feasible_start`/`_features` 只作用于 READY 工序）——此时全部前驱已 COMPLETED、`end_time` 已固定，闸门值是终值。因此首查即可缓存，永不失效。

**Files:**
- Modify: `core/simulator.py`（`__init__` ~136 行、`_init_runtime_caches` ~141 行、`_flow_ready_time` :591-597）
- Test: `tests/test_flow_gate_cache.py`（新建）

**Interfaces:**
- Consumes: `ShopFloor.get_operation_flow_ready_time(op, release_time=None)`（已存在，不改）
- Produces: `Simulator._flow_gate_cache: dict[str, float]`（实例属性，Task 3 的 `_bind_runtime` 会接管其重置）

- [ ] **Step 0: 记录优化前基线耗时**

运行 Global Constraints 中的基准命令，把输出的 `elapsed_s` / `exact_evaluations` / `exact_eval_time_total` 记录到本文件末尾的「基准记录」小节（追加一行 `baseline: ...`）。

- [ ] **Step 1: 写失败测试——闸门底层函数每工序至多计算一次**

创建 `tests/test_flow_gate_cache.py`：

```python
import unittest
from collections import Counter

from llm4drd.core.models import ShopFloor
from llm4drd.core.rules import BUILTIN_RULES
from llm4drd.core.simulator import Simulator
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class FlowGateCacheTests(unittest.TestCase):
    def test_flow_gate_computed_at_most_once_per_operation(self):
        shop = make_graph_context_shop()
        calls: Counter = Counter()
        original = ShopFloor.get_operation_flow_ready_time

        def counting(self, op, release_time=None):
            calls[op.id] += 1
            return original(self, op, release_time=release_time)

        ShopFloor.get_operation_flow_ready_time = counting
        try:
            result = Simulator(shop, BUILTIN_RULES["ATC"]).run()
        finally:
            ShopFloor.get_operation_flow_ready_time = original

        self.assertTrue(result.feasible)
        self.assertTrue(calls, "simulation should query the flow gate")
        worst = max(calls.values())
        self.assertLessEqual(
            worst, 1,
            f"flow gate recomputed {worst} times for one operation: {calls}",
        )

    def test_gate_cache_does_not_change_schedule(self):
        shop = make_graph_context_shop()
        shop.operations["OP-11"].turnover_time = 4.0
        shop.build_indexes()
        result = Simulator(shop, BUILTIN_RULES["ATC"]).run()
        entries = {entry["op_id"]: entry for entry in result.schedule}
        # OP-12 前驱 OP-11：开工不得早于 end + turnover
        self.assertGreaterEqual(
            entries["OP-12"]["start"] + 1e-9,
            entries["OP-11"]["end"] + 4.0,
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认第一条失败**

Run: `cd /Users/zhouwentao/Desktop && python -m pytest llm4drd/tests/test_flow_gate_cache.py -v`
Expected: `test_flow_gate_computed_at_most_once_per_operation` FAIL（当前每个被派工的工序至少在入队、派工评估、特征计算三处各查一次，计数 ≥ 2）；`test_gate_cache_does_not_change_schedule` PASS。

- [ ] **Step 3: 实现 memo 缓存**

`core/simulator.py` 三处修改。

`__init__`（`self._op_dispatch_type_ids: dict[str, set[str]] = {}` 之后）加一行：

```python
        self._flow_gate_cache: dict[str, float] = {}
```

`_init_runtime_caches`（`self._pdr_error_logged = False` 之后）加一行：

```python
        self._flow_gate_cache = {}
```

`_flow_ready_time` 整体替换为：

```python
    def _flow_ready_time(self, shop: ShopFloor, op: Operation) -> float:
        """闸门取值，首查后 memo——所有调用点都在 _is_op_ready 为真之后，
        此时前驱 end_time 已终值化，闸门不再变化。

        实现单点在 ShopFloor.get_operation_flow_ready_time——此处只做缓存加速。
        """
        gate = self._flow_gate_cache.get(op.id)
        if gate is None:
            release_time = self._release_time_cache.get(op.id)
            gate = shop.get_operation_flow_ready_time(op, release_time=release_time)
            self._flow_gate_cache[op.id] = gate
        return gate
```

- [ ] **Step 4: 运行新测试与转库/健壮性回归**

Run: `cd /Users/zhouwentao/Desktop && python -m pytest llm4drd/tests/test_flow_gate_cache.py llm4drd/tests/test_turnover_time.py llm4drd/tests/test_simulator_robustness.py -q`
Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add tests/test_flow_gate_cache.py core/simulator.py
git commit -m "perf: memoize the simulator turnover flow gate"
```

---

### Task 2: 近似评估器增量维护任务流转完成时刻

`optimization/approx_eval.py` 中 `push_ready`（:345-355）和 `_plan_operation`（:245-259）对每个 `predecessor_tasks` 前驱任务遍历其全部工序求 `max(completion + turnover)`。工序只在前驱任务全部工序调度完毕后才会入堆（`_combined_predecessors` 计数归零），因此该 max 可在每道工序完工时增量维护为 `task_flow_ready[task_id]`，两处退化为 O(前驱任务数) 查表。语义完全等价。

**Files:**
- Modify: `optimization/approx_eval.py`（`_plan_operation` :229-259、`evaluate` :314-331 与 :426-430、`push_ready` :334-358、`_plan_operation` 调用点 :380-387）
- Test: `tests/test_approx_flow_ready.py`（新建）

**Interfaces:**
- Consumes: 无外部新依赖
- Produces: `_plan_operation` 新签名 `(self, op, predecessor_completion, task_completion, task_flow_ready, machine_ready_time, tooling_ready_time, personnel_ready_time)`（仅文件内部使用）

- [ ] **Step 1: 写行为守护测试（重构前后都应通过）**

创建 `tests/test_approx_flow_ready.py`：

```python
import unittest

from llm4drd.optimization.approx_eval import ApproximateScheduleEvaluator
from llm4drd.optimization.solution_model import CandidateParameters
from llm4drd.tests.shop_fixtures import make_graph_context_shop


class ApproxFlowReadyTests(unittest.TestCase):
    def test_successor_waits_for_slowest_flow_of_predecessor_task(self):
        shop = make_graph_context_shop()
        # OP-13 依赖整个 T-11（含 OP-11、OP-12）；给两道前驱不同的转库时间
        shop.operations["OP-11"].turnover_time = 4.0
        shop.operations["OP-12"].turnover_time = 1.0
        shop.build_indexes()
        evaluator = ApproximateScheduleEvaluator(
            shop, {}, 1.0, 1.0, 1.0, keep_schedule_limit=16,
        )
        candidate = CandidateParameters(
            feature_weights={}, destroy_weights={}, repair_weights={},
        )
        solution = evaluator.evaluate(candidate, "test", 0)
        entries = {entry["op_id"]: entry for entry in solution.schedule}
        self.assertIn("OP-13", entries)
        gate = max(
            entries["OP-11"]["end"] + 4.0,
            entries["OP-12"]["end"] + 1.0,
        )
        self.assertGreaterEqual(entries["OP-13"]["start"] + 1e-9, gate)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认当前实现通过（这是重构守护，不是失败测试）**

Run: `cd /Users/zhouwentao/Desktop && python -m pytest llm4drd/tests/test_approx_flow_ready.py -v`
Expected: PASS（若 FAIL 说明夹具假设有误，先修测试再动实现）。

- [ ] **Step 3: 重构为增量维护**

`optimization/approx_eval.py` 四处修改。

(a) `evaluate()` 中 `task_completion: dict[str, float] = {}`（:321）后加一行：

```python
        task_flow_ready: dict[str, float] = {}
```

(b) `evaluate()` 完工登记处（:426-430），在 `task_completion[task_id] = max(...)` 之后加一行：

```python
            task_flow_ready[task_id] = max(
                task_flow_ready.get(task_id, 0.0), end + chosen_op.turnover_time
            )
```

(c) `push_ready` 闭包中 `if op.predecessor_tasks:` 块（:345-355）整体替换为：

```python
            if op.predecessor_tasks:
                for task_id in op.predecessor_tasks:
                    if task_id not in self.shop.tasks:
                        base_ready = max(base_ready, task_completion.get(task_id, 0.0))
                        continue
                    base_ready = max(base_ready, task_flow_ready.get(task_id, 0.0))
```

(d) `_plan_operation` 增加 `task_flow_ready` 参数（放在 `task_completion` 之后），其 `if op.predecessor_tasks:` 块（:245-259）整体替换为：

```python
        if op.predecessor_tasks:
            task_ready = 0.0
            for task_id in op.predecessor_tasks:
                if task_id not in self.shop.tasks:
                    task_ready = max(task_ready, task_completion.get(task_id, float("inf")))
                    continue
                task_ready = max(task_ready, task_flow_ready.get(task_id, 0.0))
            if task_ready == float("inf"):
                return None
            base_ready = max(base_ready, task_ready)
```

同步更新 `evaluate()` 里唯一的 `_plan_operation` 调用点（:380-387），在 `task_completion,` 之后加 `task_flow_ready,`。

- [ ] **Step 4: 运行守护测试与转库回归**

Run: `cd /Users/zhouwentao/Desktop && python -m pytest llm4drd/tests/test_approx_flow_ready.py llm4drd/tests/test_turnover_time.py llm4drd/tests/test_graph_legacy_baseline.py -q`
Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add tests/test_approx_flow_ready.py optimization/approx_eval.py
git commit -m "perf: track task flow-finish incrementally in approx eval"
```

---

### Task 3: SimulationRuntime——消除每次评估的深拷贝与静态重建

`Simulator.run()`（`core/simulator.py:239-244`）每次执行 `deepcopy(shop)` → `build_indexes()`（重排班次、重编全部资源日历、重跑 `derive_internal_targets`）→ `_init_runtime_caches` → Tarjan SCC。这些只依赖车间静态结构，对同一优化任务的几千次评估完全相同。新建 `SimulationRuntime`：构建一次，快照动态字段，`reset()` O(N) 恢复；优化器经 `SimulationRuntimePool`（每 worker 一个实例）复用。

仿真器只在这些动态字段上写：`Operation.{status, assigned_machine_id, assigned_tooling_ids, assigned_personnel_ids, start_time, end_time, remaining_processing_time}`、`Task.completion_time`、资源（机器/工装/人员）`.{state, current_op_id, current_finish_time, total_busy_time}`。`flow_release_floor`、派生时刻、日历编译产物均为静态输入，不进快照。

**Files:**
- Create: `core/sim_runtime.py`
- Modify: `core/simulator.py`（`__init__` :118-139、`run` :239-244、删除 `_init_runtime_caches` :141-180 与 `_detect_dependency_cycles` :182-237 与 `_iterative_tarjan_scc` :17-69）
- Modify: `optimization/hybrid_nsga3_alns.py`（`__init__` 加运行时池、`_simulate_candidate` :749、`_evaluate_builtin_rule` :1010）
- Modify: `api/server.py`（`/api/simulate/compare` :2565-2575）
- Test: `tests/test_sim_runtime.py`（新建）

**Interfaces:**
- Consumes: `ShopFloor`（deepcopy + `build_indexes()`）
- Produces:
  - `SimulationRuntime(shop: ShopFloor)`：属性 `shop`、`dependency_cycles: list[dict]`、`eligible_machine_ids: dict[str, set[str]]`、`op_dispatch_type_ids: dict[str, set[str]]`、`tooling_candidates: dict[str, dict[str, list]]`、`personnel_candidates: dict[str, dict[str, list]]`、`release_time_cache: dict[str, float]`、`dependent_ops_by_op: dict[str, list[str]]`、`dependent_ops_by_task: dict[str, list[str]]`、`task_op_counts: dict[str, int]`；方法 `reset() -> None`
  - `SimulationRuntimePool(shop, max_size)`：方法 `acquire() -> SimulationRuntime`、`release(runtime) -> None`
  - `Simulator(shop, pdr, runtime: SimulationRuntime | None = None)`（Task 4 依赖此签名）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_sim_runtime.py`：

```python
import unittest

from llm4drd.core.rules import BUILTIN_RULES
from llm4drd.core.sim_runtime import SimulationRuntime, SimulationRuntimePool
from llm4drd.core.simulator import Simulator
from llm4drd.tests.shop_fixtures import make_graph_context_shop


def _comparable(result):
    payload = result.to_dict()
    payload["wall_time_ms"] = 0.0
    return payload


class SimulationRuntimeTests(unittest.TestCase):
    def test_reuse_produces_identical_results(self):
        shop = make_graph_context_shop()
        runtime = SimulationRuntime(shop)
        rule = BUILTIN_RULES["ATC"]
        first = Simulator(shop, rule, runtime=runtime).run()
        second = Simulator(shop, rule, runtime=runtime).run()
        self.assertEqual(_comparable(first), _comparable(second))
        self.assertEqual(first.schedule, second.schedule)

    def test_interleaved_rules_do_not_leak_state(self):
        shop = make_graph_context_shop()
        runtime = SimulationRuntime(shop)
        names = sorted(BUILTIN_RULES)
        self.assertGreaterEqual(len(names), 2)
        first = Simulator(shop, BUILTIN_RULES[names[0]], runtime=runtime).run()
        Simulator(shop, BUILTIN_RULES[names[1]], runtime=runtime).run()
        third = Simulator(shop, BUILTIN_RULES[names[0]], runtime=runtime).run()
        self.assertEqual(first.schedule, third.schedule)
        self.assertEqual(_comparable(first), _comparable(third))

    def test_matches_standalone_simulator(self):
        shop = make_graph_context_shop()
        runtime = SimulationRuntime(shop)
        pooled = Simulator(shop, BUILTIN_RULES["ATC"], runtime=runtime).run()
        standalone = Simulator(shop, BUILTIN_RULES["ATC"]).run()
        self.assertEqual(pooled.schedule, standalone.schedule)
        self.assertEqual(_comparable(pooled), _comparable(standalone))

    def test_original_shop_not_mutated(self):
        shop = make_graph_context_shop()
        before = {op_id: (op.status, op.start_time, op.end_time)
                  for op_id, op in shop.operations.items()}
        runtime = SimulationRuntime(shop)
        Simulator(shop, BUILTIN_RULES["ATC"], runtime=runtime).run()
        after = {op_id: (op.status, op.start_time, op.end_time)
                 for op_id, op in shop.operations.items()}
        self.assertEqual(before, after)

    def test_pool_lazily_creates_up_to_max(self):
        shop = make_graph_context_shop()
        pool = SimulationRuntimePool(shop, max_size=2)
        first = pool.acquire()
        second = pool.acquire()
        self.assertIsNot(first, second)
        pool.release(first)
        third = pool.acquire()
        self.assertIs(first, third)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Users/zhouwentao/Desktop && python -m pytest llm4drd/tests/test_sim_runtime.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'llm4drd.core.sim_runtime'`。

- [ ] **Step 3: 创建 core/sim_runtime.py**

```python
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
    # [从 core/simulator.py:17-69 原样移入，函数体零改动]
    ...


def detect_dependency_cycles(shop: ShopFloor) -> list[dict]:
    # [从 core/simulator.py Simulator._detect_dependency_cycles(:182-237) 移入：
    #  去掉 self 参数，其余函数体（含 docstring）零改动]
    ...


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
            # 派工桶必须按"可用机台的实际类型"建立（语义说明见原 Simulator._init_runtime_caches）
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
        self._resource_snapshot = {
            resource.id: (
                resource.state,
                resource.current_op_id,
                resource.current_finish_time,
                resource.total_busy_time,
            )
            for resource in self._iter_resources()
        }

    def _iter_resources(self):
        yield from self.shop.machines.values()
        yield from self.shop.toolings.values()
        yield from self.shop.personnel.values()

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
        for resource in self._iter_resources():
            (resource.state, resource.current_op_id,
             resource.current_finish_time, resource.total_busy_time) = (
                self._resource_snapshot[resource.id]
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
```

两处 `# [...原样移入...]` 必须执行为真实的代码搬移（复制原函数体后从 `core/simulator.py` 删除原定义），不得留占位注释。`detect_dependency_cycles` 原函数体不引用 `self`，去参即可。移入后 `sim_runtime.py` 需要 `from collections import defaultdict`（`detect_dependency_cycles` 用到）。

- [ ] **Step 4: 改造 core/simulator.py**

(a) 搬移后，删除文件顶部的 `_iterative_tarjan_scc`（:17-69）、`Simulator._init_runtime_caches`（:141-180）、`Simulator._detect_dependency_cycles`（:182-237）。先确认无外部引用：

Run: `cd /Users/zhouwentao/Desktop/llm4drd && grep -rn "_iterative_tarjan_scc\|_detect_dependency_cycles\|_init_runtime_caches" --include="*.py" . | grep -v "core/simulator.py\|core/sim_runtime.py"`
Expected: 无输出。若有输出，在 `core/simulator.py` 保留一行兼容再导出 `from .sim_runtime import _iterative_tarjan_scc, detect_dependency_cycles` 并更新引用方。

(b) 顶部 import 增加：

```python
from .sim_runtime import SimulationRuntime
```

（若 `copy` 与 `defaultdict` 在删除后不再被本文件其他代码使用，则一并从 import 移除——用 grep 确认后再删。）

(c) `__init__` 增加 `runtime` 参数并保存（其余字段声明保持不变）：

```python
    def __init__(self, shop: ShopFloor, pdr: PDRFunc, runtime: SimulationRuntime | None = None):
        self.orig_shop = shop
        self.pdr = pdr
        self._runtime = runtime
```

(d) `run()` 的前 5 行（:239-244）替换为：

```python
    def run(self, max_time: float = 999999) -> SimResult:
        started_at = wall_time.time()
        runtime = self._runtime
        if runtime is None:
            runtime = self._runtime = SimulationRuntime(self.orig_shop)
        runtime.reset()
        shop = runtime.shop
        self._bind_runtime(runtime)
        self._dependency_cycles = list(runtime.dependency_cycles)
```

（`reset()` 对新建实例是恒等操作，统一执行以保证复用路径与首用路径一致。）

(e) 新增 `_bind_runtime`（放在原 `_init_runtime_caches` 的位置）：

```python
    def _bind_runtime(self, runtime: SimulationRuntime) -> None:
        """静态缓存共享 runtime 的（本轮 run 只读），动态结构每轮新建。"""
        shop = runtime.shop
        self._eligible_machine_ids = runtime.eligible_machine_ids
        self._op_dispatch_type_ids = runtime.op_dispatch_type_ids
        self._tooling_candidates = runtime.tooling_candidates
        self._personnel_candidates = runtime.personnel_candidates
        self._release_time_cache = runtime.release_time_cache
        self._dependent_ops_by_op = runtime.dependent_ops_by_op
        self._dependent_ops_by_task = runtime.dependent_ops_by_task
        self._ready_by_type = {}
        self._dispatch_scheduled_at = dict.fromkeys(shop.machines, None)
        self._release_checks_scheduled = set()
        self._completed_ops = set()
        self._completed_tasks = set()
        self._task_remaining_ops = dict(runtime.task_op_counts)
        self._unschedulable_ops = set()
        self._flow_gate_cache = {}
        self._pdr_error_logged = False
        self._dependency_cycles = []
```

- [ ] **Step 5: 运行仿真器测试**

Run: `cd /Users/zhouwentao/Desktop && python -m pytest llm4drd/tests/test_sim_runtime.py llm4drd/tests/test_flow_gate_cache.py llm4drd/tests/test_turnover_time.py llm4drd/tests/test_simulator_robustness.py -q`
Expected: 全部 PASS。

- [ ] **Step 6: 优化器接入运行时池**

`optimization/hybrid_nsga3_alns.py` 四处修改。

(a) 顶部 import 增加：

```python
from ..core.sim_runtime import SimulationRuntimePool
```

(b) `__init__` 中 `self.refine_parallel_workers = self._phase_parallel_workers("refine")` 之后加：

```python
        self._runtime_pool = SimulationRuntimePool(
            self.shop,
            max(
                self.parallel_workers,
                self.exact_parallel_workers,
                self.refine_parallel_workers,
            ),
        )
```

(c) `_simulate_candidate`（:749-761）开头两行替换为借还池：

```python
    def _simulate_candidate(self, candidate: CandidateParameters, source: str, generation: int) -> OptimizationSolution:
        runtime = self._runtime_pool.acquire()
        try:
            simulator = Simulator(self.shop, self._dispatch_rule(candidate), runtime=runtime)
            sim_result = simulator.run()
        finally:
            self._runtime_pool.release(runtime)
```

（其后 `schedule = self._enrich_schedule(...)` 起保持不变。）

(d) `_evaluate_builtin_rule`（:1010）中 `simulator = Simulator(self.shop, BUILTIN_RULES[rule_name])` 与 `sim_result = simulator.run()` 两行同样替换为借还池写法：

```python
        runtime = self._runtime_pool.acquire()
        try:
            simulator = Simulator(self.shop, BUILTIN_RULES[rule_name], runtime=runtime)
            sim_result = simulator.run()
        finally:
            self._runtime_pool.release(runtime)
```

- [ ] **Step 7: /api/simulate/compare 复用 runtime**

`api/server.py`：顶部 import 增加 `from ..core.sim_runtime import SimulationRuntime`；`/api/simulate/compare`（:2565-2575）中循环体里的 `r = Simulator(current_shop, BUILTIN_RULES[n]).run()` 改为循环外建一次 runtime、循环内复用：

```python
    runtime = SimulationRuntime(current_shop)
    ...（原有循环保持）
        r = Simulator(current_shop, BUILTIN_RULES[n], runtime=runtime).run()
```

（`runtime = ...` 放在 rule 循环开始之前、请求校验之后。）

- [ ] **Step 8: 运行优化器冻结基线与全量测试**

Run: `cd /Users/zhouwentao/Desktop && python -m pytest llm4drd/tests -q`
Expected: 全部 PASS（重点看 `test_graph_legacy_baseline.py` 的 `test_legacy_hybrid_is_deterministic` 与 `test_hybrid_graph_context.py`——它们守护优化器行为不变）。

- [ ] **Step 9: Commit**

```bash
git add core/sim_runtime.py core/simulator.py optimization/hybrid_nsga3_alns.py api/server.py tests/test_sim_runtime.py
git commit -m "perf: reuse simulation runtime across evaluations"
```

---

### Task 4: 精确/近似批量评估切换到进程池

`_parallel_evaluate_candidates_exact`（:911）与 `_parallel_evaluate_candidates_approx`（:979）目前用 `ThreadPoolExecutor` 跑纯 Python CPU 计算，GIL 下无真实并行。改为 `ProcessPoolExecutor`：worker 子进程 initializer 收一次 pickle 的 shop + 图特征 + 尺度，进程内自建 `SimulationRuntime` 与 `ApproximateScheduleEvaluator`；每个任务只传 `CandidateParameters`、收回 `SimResult` / `OptimizationSolution`（均为纯数据，可 pickle）。refine 阶段（:1378）的线程池保持不变（其任务体是含闭包的 ALNS 局部搜索，本任务不动，作为已知限制记录）。

**Files:**
- Create: `optimization/parallel_eval.py`
- Modify: `optimization/hybrid_nsga3_alns.py`（`HybridConfig` :54-73、`_dispatch_rule` :764、`_simulate_candidate` 拆分、两个 parallel 方法、`run()` 收尾、`__init__`）
- Test: `tests/test_parallel_eval.py`（新建）

**Interfaces:**
- Consumes: `SimulationRuntime`（Task 3）、`Simulator(shop, pdr, runtime=...)`、`ApproximateScheduleEvaluator(shop, graph_features, time_scale, due_scale, priority_scale, keep_schedule_limit=0)`、`CandidateParameters`（picklable dataclass）
- Produces:
  - `parallel_eval.build_candidate_rule(candidate, graph_features, time_scale, busy_scale, priority_scale, due_scale) -> Callable`
  - `parallel_eval.init_worker(payload_bytes: bytes) -> None`
  - `parallel_eval.run_exact_simulation(candidate: CandidateParameters) -> SimResult`
  - `parallel_eval.run_approx_evaluation(candidate: CandidateParameters, source: str, generation: int) -> OptimizationSolution`
  - `HybridConfig.parallel_backend: str = "process"`（可选值 `"process"` / `"thread"`）
  - `HybridNSGA3ALNSOptimizer._solution_from_sim_result(candidate, sim_result, source, generation) -> OptimizationSolution`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_parallel_eval.py`：

```python
import pickle
import unittest

from llm4drd.optimization.hybrid_nsga3_alns import HybridConfig, HybridNSGA3ALNSOptimizer
from llm4drd.optimization import parallel_eval
from llm4drd.tests.shop_fixtures import make_graph_context_shop


def _small_config(**overrides) -> HybridConfig:
    params = dict(
        objective_keys=["total_tardiness", "makespan"],
        target_solution_count=2, population_size=4, generations=1,
        alns_iterations_per_candidate=0, time_limit_s=60,
        parallel_workers=1, seed=17,
    )
    params.update(overrides)
    return HybridConfig(**params)


class ParallelEvalTests(unittest.TestCase):
    def test_worker_matches_inline_simulation(self):
        # 在当前进程直接调用 worker 的初始化与任务函数（不真正 spawn），
        # 验证 worker 路径与主进程 _simulate_candidate 产出一致。
        shop = make_graph_context_shop()
        optimizer = HybridNSGA3ALNSOptimizer(shop, _small_config())
        candidate = optimizer._default_candidate(optimizer.config.baseline_rule_name)

        parallel_eval.init_worker(optimizer._worker_payload_bytes())
        worker_sim = parallel_eval.run_exact_simulation(candidate)
        worker_solution = optimizer._solution_from_sim_result(
            candidate, worker_sim, "test", 0,
        )
        inline_solution = optimizer._simulate_candidate(candidate, "test", 0)

        def comparable(solution):
            metrics = dict(solution.metrics)
            metrics["wall_time_ms"] = 0.0
            return metrics, solution.schedule_signature

        self.assertEqual(comparable(worker_solution), comparable(inline_solution))

    def test_worker_payload_is_picklable(self):
        shop = make_graph_context_shop()
        optimizer = HybridNSGA3ALNSOptimizer(shop, _small_config())
        payload = pickle.loads(optimizer._worker_payload_bytes())
        self.assertIn("shop", payload)
        self.assertIn("graph_features", payload)
        self.assertEqual(
            sorted(payload["scales"]),
            ["busy_scale", "due_scale", "priority_scale", "time_scale"],
        )

    def test_process_backend_runs_end_to_end(self):
        shop = make_graph_context_shop()
        config = _small_config(parallel_workers=2, parallel_backend="process")
        result = HybridNSGA3ALNSOptimizer(shop, config).run()
        self.assertGreaterEqual(result.found_solution_count, 1)
        for solution in result.solutions:
            self.assertTrue(solution["metrics"]["feasible"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Users/zhouwentao/Desktop && python -m pytest llm4drd/tests/test_parallel_eval.py -v`
Expected: FAIL，`ImportError`（`parallel_eval` 不存在）。

- [ ] **Step 3: 抽出模块级派工规则工厂并创建 worker 模块**

创建 `optimization/parallel_eval.py`：

```python
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

    实现体从 HybridNSGA3ALNSOptimizer._dispatch_rule 原样搬移（self.* 换为
    入参），主进程与子进程共用此单点。
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


def init_worker(payload_bytes: bytes) -> None:
    payload = pickle.loads(payload_bytes)
    shop = payload["shop"]
    scales = payload["scales"]
    _STATE["shop"] = shop
    _STATE["graph_features"] = payload["graph_features"]
    _STATE["scales"] = scales
    _STATE["runtime"] = SimulationRuntime(shop)
    _STATE["approx"] = ApproximateScheduleEvaluator(
        shop,
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
    return _STATE["approx"].evaluate(candidate, source, generation)
```

**搬移校验**：`_rule` 闭包体必须与仓库现有 `HybridNSGA3ALNSOptimizer._dispatch_rule`（`hybrid_nsga3_alns.py:764` 起）逐行一致（仅 `self.graph_features/self.time_scale/self.busy_scale/self.priority_scale/self.due_scale` 换成参数）。搬移时以现有实现为准——若上方代码块与现有实现有任何出入（例如 `built_in` 分支细节），以现有实现为准。完成后把 `_dispatch_rule` 改为委托：

```python
    def _dispatch_rule(self, candidate: CandidateParameters):
        return build_candidate_rule(
            candidate, self.graph_features, self.time_scale,
            self.busy_scale, self.priority_scale, self.due_scale,
        )
```

并在 `hybrid_nsga3_alns.py` 顶部加 `from .parallel_eval import build_candidate_rule, init_worker, run_approx_evaluation, run_exact_simulation` 与 `import pickle`、`from concurrent.futures import ProcessPoolExecutor` 和 `from concurrent.futures.process import BrokenProcessPool`。

- [ ] **Step 4: 优化器接入进程池**

`optimization/hybrid_nsga3_alns.py` 七处修改。

(a) `HybridConfig` 加字段（`baseline_rule_name` 之后）：

```python
    parallel_backend: str = "process"
```

(b) `__init__` 中 `self._runtime_pool = ...` 之后加：

```python
        self._process_pool = None
        self._process_backend_failed = False
```

(c) 拆分 `_simulate_candidate`：把借还池 + 仿真之后的解构建部分抽成 `_solution_from_sim_result`：

```python
    def _simulate_candidate(self, candidate: CandidateParameters, source: str, generation: int) -> OptimizationSolution:
        runtime = self._runtime_pool.acquire()
        try:
            simulator = Simulator(self.shop, self._dispatch_rule(candidate), runtime=runtime)
            sim_result = simulator.run()
        finally:
            self._runtime_pool.release(runtime)
        return self._solution_from_sim_result(candidate, sim_result, source, generation)

    def _solution_from_sim_result(self, candidate: CandidateParameters, sim_result, source: str, generation: int) -> OptimizationSolution:
        schedule = self._enrich_schedule(sim_result.schedule)
        analytics = build_schedule_analytics(self.shop, sim_result)
        metrics = sim_result.to_dict()
        metrics.update({key: round(value, 6) for key, value in analytics.objective_values.items()})
        metrics["completed_operations"] = analytics.completed_operations
        metrics["total_operations"] = len(self.shop.operations)
        metrics["feasible"] = analytics.feasible
        metrics["evaluation_mode"] = "exact"
        return self._make_solution(candidate, source, generation, schedule, analytics, metrics)
```

(d) 进程池管理方法（放在 `_solution_from_sim_result` 之后）：

```python
    def _worker_payload_bytes(self) -> bytes:
        return pickle.dumps(
            {
                "shop": self.shop,
                "graph_features": self.graph_features,
                "scales": {
                    "time_scale": self.time_scale,
                    "busy_scale": self.busy_scale,
                    "priority_scale": self.priority_scale,
                    "due_scale": self.due_scale,
                },
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    def _ensure_process_pool(self, worker_count: int):
        if self._process_backend_failed or self.config.parallel_backend != "process":
            return None
        if self._process_pool is None:
            try:
                self._process_pool = ProcessPoolExecutor(
                    max_workers=worker_count,
                    initializer=init_worker,
                    initargs=(self._worker_payload_bytes(),),
                )
            except Exception as exc:
                logging.warning("hybrid: process pool unavailable (%s), using threads", exc)
                self._process_backend_failed = True
                return None
        return self._process_pool

    def _shutdown_process_pool(self) -> None:
        if self._process_pool is not None:
            self._process_pool.shutdown(cancel_futures=True)
            self._process_pool = None

    def _abandon_process_backend(self, exc: BaseException) -> None:
        logging.warning("hybrid: process backend failed (%s), falling back to threads", exc)
        self._process_backend_failed = True
        self._shutdown_process_pool()
```

(e) `_parallel_evaluate_candidates_exact` 的 `if worker_count > 1:` 线程分支（:939-961）前插入进程分支，结构变为：

```python
        pending = list(unique_candidates.items())
        if pending:
            worker_count = self._worker_count_for_batch("exact", len(pending))
            executor = self._ensure_process_pool(worker_count) if worker_count > 1 else None
            if executor is not None:
                batch_started = time.time()
                new_exact = 0
                try:
                    futures = {
                        executor.submit(run_exact_simulation, candidate): signature
                        for signature, candidate in pending
                    }
                    for future in as_completed(futures):
                        signature = futures[future]
                        sim_result = future.result()
                        solution = self._solution_from_sim_result(
                            unique_candidates[signature], sim_result, source, generation,
                        )
                        with self.cache_lock:
                            existing = self.exact_cache.get(signature)
                            if existing is None:
                                self.exact_cache[signature] = solution.clone()
                                self.total_evaluations += 1
                                self.exact_evaluations += 1
                                new_exact += 1
                                results_by_signature[signature] = solution
                            else:
                                results_by_signature[signature] = self._clone_solution(existing, source, generation)
                except (BrokenProcessPool, pickle.PicklingError) as exc:
                    self._abandon_process_backend(exc)
                if new_exact > 0:
                    self.exact_eval_time_total += max(0.0, time.time() - batch_started)
                # 进程池失败时把未完成的候选走缓存感知的串行路径补齐
                for signature, candidate in pending:
                    if signature not in results_by_signature:
                        results_by_signature[signature] = self._evaluate_candidate(candidate, source, generation)
            elif worker_count > 1:
                ...（原 ThreadPoolExecutor 分支整体保持不变，仅缩进随 if/elif 调整）
            else:
                ...（原串行分支整体保持不变）
```

（`...（原分支保持不变）` 指现有代码原地保留、不得改动其内容，仅缩进层级随新的 if/elif/else 结构调整。）

(f) `_parallel_evaluate_candidates_approx`（:979-999）改为先按签名查缓存去重、进程分支、失败回退，新增缓存登记辅助方法：

```python
    def _register_approx_solution(self, signature: str, solution: OptimizationSolution, source: str, generation: int) -> OptimizationSolution:
        with self.cache_lock:
            existing = self.approx_cache.get(signature)
            if existing is None:
                self.approx_cache[signature] = solution.clone()
                self.total_evaluations += 1
                self.approximate_evaluations += 1
                return solution
            return self._clone_solution(existing, source, generation)

    def _parallel_evaluate_candidates_approx(self, candidates, source, generation):
        if not candidates:
            return []
        worker_count = self._worker_count_for_batch("approx", len(candidates))
        executor = self._ensure_process_pool(worker_count) if worker_count > 1 else None
        if executor is not None:
            results: list[OptimizationSolution] = []
            remaining: list[CandidateParameters] = []
            for candidate in candidates:
                signature = candidate.signature()
                with self.cache_lock:
                    cached = self.approx_cache.get(signature)
                if cached is not None:
                    results.append(self._clone_solution(cached, source, generation))
                else:
                    remaining.append(candidate)
            submitted = list(remaining)
            try:
                futures = {
                    executor.submit(run_approx_evaluation, candidate, source, generation): candidate
                    for candidate in remaining
                }
                for future in as_completed(futures):
                    candidate = futures[future]
                    solution = future.result()
                    results.append(self._register_approx_solution(candidate.signature(), solution, source, generation))
                    submitted.remove(candidate)
            except (BrokenProcessPool, pickle.PicklingError) as exc:
                self._abandon_process_backend(exc)
                for candidate in submitted:
                    results.append(self._evaluate_candidate_approx(candidate, source, generation))
        elif worker_count > 1:
            ...(原 ThreadPoolExecutor 分支整体保持不变，仅缩进随 if/elif 调整)
        else:
            results = [self._evaluate_candidate_approx(candidate, source, generation) for candidate in candidates]
        return [self._clone_solution(solution, source, generation) for solution in results]
```

(g) `run(self, progress_callback=None)`（:1386 起）方法体最外层套 try/finally（原有代码整体成为 try 块，内容不变），finally 中调用：

```python
        finally:
            self._shutdown_process_pool()
```

- [ ] **Step 5: 运行新测试**

Run: `cd /Users/zhouwentao/Desktop && python -m pytest llm4drd/tests/test_parallel_eval.py -v`
Expected: 3 个测试 PASS（`test_process_backend_runs_end_to_end` 首次运行会 spawn 子进程，耗时数秒属正常）。

- [ ] **Step 6: 全量回归**

Run: `cd /Users/zhouwentao/Desktop && python -m pytest llm4drd/tests -q`
Expected: 全部 PASS。特别注意 `test_graph_legacy_baseline.py::test_legacy_hybrid_is_deterministic` 用 `parallel_workers=1`（不触发进程池），必须逐位一致。

- [ ] **Step 7: Commit**

```bash
git add optimization/parallel_eval.py optimization/hybrid_nsga3_alns.py tests/test_parallel_eval.py
git commit -m "perf: evaluate candidates in a process pool"
```

---

### Task 5: 解克隆共享不可变负载

`OptimizationSolution.clone()`（`optimization/solution_model.py:134-149`）深拷贝 `schedule`（每工序一个 dict）与 `analytics_summary`。克隆发生在缓存写入、缓存命中、并行收集、归档四条链路上，大排程下是纯 CPU/内存 churn。`schedule` 与 `analytics_summary` 在解构建后从不被原地修改（本任务先审计确认），改为按不可变共享。

**Files:**
- Modify: `optimization/solution_model.py`（`clone` :134-149）
- Test: `tests/test_solution_clone.py`（新建）

**Interfaces:**
- Consumes: 无
- Produces: `clone()` 语义变化——`schedule` / `analytics_summary` 与源对象共享引用（约定为构建后不可变）

- [ ] **Step 1: 审计确认无原地修改**

Run: `cd /Users/zhouwentao/Desktop/llm4drd && grep -rn "\.schedule\.append\|\.schedule\.extend\|\.schedule\[\|\.schedule +=\|\.analytics_summary\[\|\.analytics_summary\.update\|\.analytics_summary\.pop" --include="*.py" optimization api core scheduling ai`

Expected: 输出中不存在对 `OptimizationSolution` 实例的 `.schedule` / `.analytics_summary` 的写操作（`SimResult.schedule.append` 等仿真器内部对自身列表的构建性写入不算——它们发生在解对象创建之前）。**若发现对解对象的写点：在该写点先 `list(solution.schedule)` / `dict(solution.analytics_summary)` 复制再改，然后才能继续 Step 2。**

- [ ] **Step 2: 写失败测试**

创建 `tests/test_solution_clone.py`：

```python
import unittest

from llm4drd.optimization.solution_model import CandidateParameters, OptimizationSolution


def _make_solution() -> OptimizationSolution:
    return OptimizationSolution(
        solution_id="S-1",
        source="test",
        generation=0,
        candidate=CandidateParameters(
            feature_weights={}, destroy_weights={}, repair_weights={},
        ),
        objectives={"makespan": 10.0},
        metrics={"makespan": 10.0, "feasible": True},
        schedule=[{"op_id": "OP-1", "start": 0.0, "end": 1.0}],
        feasible=True,
        schedule_signature="sig-1",
        analytics_summary={"completed_operations": 1},
    )


class SolutionCloneTests(unittest.TestCase):
    def test_clone_shares_immutable_payload(self):
        solution = _make_solution()
        cloned = solution.clone()
        self.assertIs(cloned.schedule, solution.schedule)
        self.assertIs(cloned.analytics_summary, solution.analytics_summary)

    def test_clone_still_isolates_mutable_fields(self):
        solution = _make_solution()
        cloned = solution.clone()
        cloned.metrics["makespan"] = 99.0
        cloned.objectives["makespan"] = 99.0
        cloned.candidate.feature_weights["urgency"] = 1.0
        self.assertEqual(solution.metrics["makespan"], 10.0)
        self.assertEqual(solution.objectives["makespan"], 10.0)
        self.assertEqual(solution.candidate.feature_weights, {})


if __name__ == "__main__":
    unittest.main()
```

字段名以仓库现有 `OptimizationSolution` dataclass 定义为准；若构造还需其他必填字段，按其默认值补齐。

- [ ] **Step 3: 运行确认第一条失败**

Run: `cd /Users/zhouwentao/Desktop && python -m pytest llm4drd/tests/test_solution_clone.py -v`
Expected: `test_clone_shares_immutable_payload` FAIL（当前 deepcopy 产生新对象）；第二条 PASS。

- [ ] **Step 4: 修改 clone**

`optimization/solution_model.py` 中 `clone()` 的两行替换：

```python
            schedule=copy.deepcopy(self.schedule),
            analytics_summary=copy.deepcopy(self.analytics_summary),
```

改为：

```python
            # schedule / analytics_summary 构建后按不可变约定共享，克隆不复制；
            # 任何需要修改的调用方必须先自行 list()/dict() 复制（审计见
            # docs/superpowers/plans/2026-07-16-sim-optimizer-performance.md Task 5）
            schedule=self.schedule,
            analytics_summary=self.analytics_summary,
```

若 `copy` import 因此在本文件不再被使用（grep 确认 `copy.` 无其他引用），一并移除。

- [ ] **Step 5: 运行测试与全量回归**

Run: `cd /Users/zhouwentao/Desktop && python -m pytest llm4drd/tests/test_solution_clone.py llm4drd/tests -q`
Expected: 全部 PASS（冻结基线守护优化器输出未变）。

- [ ] **Step 6: Commit**

```bash
git add optimization/solution_model.py tests/test_solution_clone.py
git commit -m "perf: share immutable payloads across solution clones"
```

---

### Task 6: 全量验证与基准对比

**Files:**
- Modify: `docs/superpowers/plans/2026-07-16-sim-optimizer-performance.md`（本文件「基准记录」小节）

**Interfaces:**
- Consumes: Task 1 Step 0 记录的基线数字
- Produces: 优化前后耗时对比记录

- [ ] **Step 1: 全量测试**

Run: `cd /Users/zhouwentao/Desktop && python -m pytest llm4drd/tests -q`
Expected: 全部 PASS。

- [ ] **Step 2: 复跑基准命令**

运行 Global Constraints 中的同一条基准命令，把结果以 `optimized: ...` 追加到「基准记录」小节。Expected: `elapsed_s` 相对 baseline 显著下降（固定 fixture 较小，加速比会低于真实大数据集；量级参考：Task 1–3 生效后单次评估成本应降 40% 以上，Task 4 生效后 4 worker 下 `elapsed_s` 应再降）。若 `elapsed_s` 未下降，用 `py-spy record --output profile.svg -- python <基准脚本>` 定位后再回看对应 Task。

- [ ] **Step 3: 转库语义抽查**

Run: `cd /Users/zhouwentao/Desktop && python -m pytest llm4drd/tests/test_turnover_time.py -q`
Expected: 全部 PASS（转库闸门、派生时刻、CP-SAT、在线调度语义未被性能改动破坏）。

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-07-16-sim-optimizer-performance.md
git commit -m "docs: record perf benchmark before/after"
```

---

## 基准记录

（Task 1 Step 0 与 Task 6 Step 2 在此追加）

基准脚本：`tools/benchmark_simulation_perf.py`（在仓库内，可复现）。

```
cd /Users/zhouwentao/Desktop && python -m llm4drd.tools.benchmark_simulation_perf
```

实例：`InstanceGenerator(seed=17)`，30 订单 / 109 任务 / 366 工序 / 19 机器。
计划正文 Global Constraints 里的原始命令有误——`exact_eval_time_total` 在 optimizer 实例上而非 `HybridResult` 上
（直接 AttributeError），且 `make_graph_context_shop()` 仅 4 道工序、0.05 秒跑完，测不出差异。以本脚本为准。

- baseline (2026-07-16, main@bd8d9da):
  - single_simulation_median_s: 0.0492（events=1110, feasible=True）
  - optimizer_elapsed_s: 6.95
  - exact_evaluations: 85, approx_evaluations: 60, exact_eval_time_total: 13.02
  - found_solution_count: 1
- optimized (2026-07-17, code review 修复后):
  - single_simulation_median_s: 0.0502（events=1110, feasible=True）
  - optimizer_elapsed_s: **3.12**（baseline 6.95 → **-55%**，2.2x）
  - workers: approx=4 exact=4 refine=4 pool_capacity=4；process_backend_failed=False
  - exact_evaluations: 85, approx_evaluations: 60, exact_eval_time_total: **6.88**（baseline 13.02 → -47%）
  - found_solution_count: 1

后端对照（同实例、同 seed）：进程池 **3.12s** vs 线程池 **4.99s** —— 证明 GIL 确实是瓶颈，
且进程池的加速不是测量噪声。

分阶段实测（同脚本、同机器）：

| 阶段 | optimizer_elapsed_s | exact_eval_time_total |
|---|---|---|
| baseline | 6.95 | 13.02 |
| Task 1+2+3（runtime 复用） | 4.83 (-30%) | 7.29 (-44%) |
| Task 4（进程池） | 3.02 (-57%) | 6.57 |
| Task 5（克隆共享） | 2.88 (-59%) | 6.50 |
| **review 修复后** | **3.12 (-55%)** | **6.88** |

review 修复让端到端从 2.88 回升到 3.12：Finding 3 的修复取消了"隐式 runtime 永久缓存"，
未显式传 runtime 的 `Simulator` 恢复为每次 run() 重建——那部分加速本来就是靠破坏语义换来的，不该保留。
优化器内部仍走显式 runtime 池，收益不受影响。

注：单次仿真耗时（~50ms）基本持平——本计划的收益集中在**跨评估的重复构建开销**与**并行度**，
而非单次仿真内环。评估次数（85 精确 / 60 近似）全程不变，证明搜索行为未被改动。

## 实施记录：与计划的偏差

1. **pytest 不存在**：venv 里没装 pytest，所有 `python -m pytest X` 实际用
   `cd /Users/zhouwentao/Desktop && python -m unittest llm4drd.tests.X` 执行。
2. **基准命令有误**：计划正文里 `result.exact_eval_time_total` 不存在（该计数在 optimizer 实例上），
   且 `make_graph_context_shop()` 只有 4 道工序、0.05 秒跑完，测不出差异。改用
   `scratchpad/bench_perf.py`（生成器造的 366 工序实例）。
3. **Task 4 的测试补强**：计划里的 `test_process_backend_runs_end_to_end` 是**假通过**——
   进程池 spawn 失败会静默回退线程池，断言照样成立。已加 `assertFalse(_process_backend_failed)`
   显式确认进程后端未被放弃，并新增 `test_process_backend_falls_back_when_pool_breaks`
   验证回退路径能把候选补齐。
4. **`approx_parallel_workers` 随工序规模变化**（本条为 code review 后的更正，原先写反了）：
   `_phase_parallel_workers("approx")` 仅在 `op_count < 250` 时返回 1；基准实例有 366 工序，
   `parallel_workers=4` 时返回 **4**——近似评估同样并行。此前文档断言"恒为 1、进程池收益全部来自
   精确评估"，那是用 4 工序的 `make_graph_context_shop()` 验证后错误外推到 366 工序实例的结论，已作废。
5. **`tools/analyze_unscheduled.py` 自带一份 `_iterative_tarjan_scc` 副本**（本地定义，非从 simulator 导入）。
   属预先存在的重复，按「不动无关代码」原则未合并。

## Code review 修复（2026-07-17）

首版实现有 5 处缺陷，均已修复，每处先写复现测试：

1. **[P1] 资源快照跨类型 ID 冲突**（`core/sim_runtime.py`）：快照只按 `resource.id` 索引，而机器/工装/人员
   存在三个独立 dict、模型不保证 ID 跨类型唯一。同名时 `reset()` 会静默把机器状态恢复成工装的
   （实测 `total_busy_time` 11 → 3），在第一次仿真前就篡改初始状态。改用 `(kind, id)` 为键。
   这是本计划引入的**回归**——旧代码每次 deepcopy 重建，不存在此问题。
2. **[P1] 进程池回退漏掉启动期异常**（`hybrid_nsga3_alns.py`）：worker 通常到首次 `submit()` 才启动，
   届时抛的是 `RuntimeError` / `OSError` / `AssertionError`（如 daemon 进程内不允许再起子进程——
   本服务跑在 FastAPI worker 里正是该场景），而非 `BrokenProcessPool`。原实现只捕获后两者，异常直接
   炸穿且 `_process_backend_failed` 仍为 False。新增 `_submit_to_pool` + `_ProcessBackendUnavailable`
   把「基础设施故障」（回退）与「任务内部异常」（上抛）分开。
3. **[P2] 隐式 runtime 冻结 shop**（`core/simulator.py`）：首次 `run()` 把自建 runtime 永久存到
   `self._runtime`，此后调用方修改 shop 也不生效（实测 9 / 9 / 45）。改为仅复用显式传入的 runtime，
   隐式的每次 run() 现建——恢复重构前语义。
4. **[P2] 进程池容量被首批固定**（`hybrid_nsga3_alns.py`）：懒建的池以首个批次的 worker 数定容，
   后续阶段要求更高并发也拿不到。改为按各阶段最大并发 `_max_pool_workers` 一次建池。
5. **[P3] 基准不可复现 + 文档结论错误**：基准脚本原在会话临时目录，已移入
   `tools/benchmark_simulation_perf.py`；文档原称"366 工序下 approx worker 恒为 1、进程池收益全部来自
   精确评估"，实为用 4 工序夹具验证后错误外推——`_phase_parallel_workers` 只在 `op_count < 250` 时返回 1，
   366 工序时返回 4。该结论已作废，基准脚本现在会直接打印各阶段实际并发数。

## 已知限制（本计划不处理）

- refine 阶段（`hybrid_nsga3_alns.py:1378`）的线程池仍受 GIL 限制：其任务体是含闭包的 ALNS 局部搜索，进程化需要更大的重构，收益评估后再立项。
- `Simulator` 单发使用（如 `/api/simulate`）仍是每请求一次深拷贝——与改动前持平，非回归。
- 近似评估并行的结果收集顺序沿用现状（completion order），与线程版行为一致。
