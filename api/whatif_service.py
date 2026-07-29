"""What-if 场景的编排：物化 rows → ShopFloor、跑规则、对比 KPI、（高危）落库。

隔离红线：本模块只读数据库（load_all / list_all），绝不写 inst_* 表、绝不 bump
inst_version——唯一例外是显式调用的 apply_to_instance()。也绝不触碰 api/server.py 的
全局 shop / last_result / last_sim_payload / _sim_runtime_cache / workflow_progress，
所以推演不会污染四步流程的任何结论。
"""
from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..core.rules import BUILTIN_RULES, get_all_rule_names
from ..core.simulator import Simulator
from ..core.sim_runtime import SimulationRuntime
from ..data.db import DowntimeStore, InstanceStore, get_instance_version
from ..optimization.objectives import OBJECTIVE_SPECS, build_schedule_analytics
from ..core.time_utils import ensure_aware
from .whatif_scenario import (
    PatchError,
    Scenario,
    ScenarioStore,
    apply_patches,
    rows_scale,
)

logger = logging.getLogger(__name__)

DEFAULT_METRIC_KEYS = ["total_tardiness", "makespan", "main_order_tardy_total_time"]


class WhatIfError(ValueError):
    def __init__(self, code: str, message: str, suggestions: list | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.suggestions = list(suggestions or [])


@dataclass
class WhatIfRun:
    run_id: str
    scenario_id: str
    scenario_name: str
    rule_names: list[str]
    include_baseline: bool = False
    status: str = "running"          # running | done | failed
    created_at: str = ""
    finished_at: str | None = None
    results: list[dict] = field(default_factory=list)
    error: dict | None = None
    warnings: list[dict] = field(default_factory=list)

    def to_dict(self, *, include_results: bool = True) -> dict:
        payload = {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "rule_names": list(self.rule_names),
            "include_baseline": self.include_baseline,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "warnings": list(self.warnings),
            "error": self.error,
        }
        if include_results:
            payload["results"] = list(self.results)
        return payload


class WhatIfService:
    def __init__(
        self,
        inst_store: InstanceStore,
        downtime_store: DowntimeStore,
        *,
        max_scenarios: int = 8,
        max_runs: int = 16,
    ):
        self.inst_store = inst_store
        self.downtime_store = downtime_store
        self.scenarios = ScenarioStore(max_size=max_scenarios)
        self._runs: OrderedDict[str, WhatIfRun] = OrderedDict()
        self._max_runs = max_runs
        self._futures: dict[str, Future] = {}
        # 单槽物化缓存：大实例一份 shop + 一份 runtime 深拷贝已 130MB 量级，多槽会炸内存。
        self._materialized: tuple[str, object, SimulationRuntime] | None = None
        self._lock = threading.Lock()
        # 单 worker：推演之间串行，避免并发物化把内存翻倍。
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whatif")

    # --- 基线读取 -------------------------------------------------------------

    def current_instance_version(self) -> int:
        return get_instance_version(self.inst_store.db_path)

    def load_base_rows(self) -> dict:
        """基线 rows = load_all() + downtimes。只读，不加锁（SQLite 自己管并发）。"""
        rows = self.inst_store.load_all()
        rows["downtimes"] = self.downtime_store.list_all()
        return rows

    # --- 场景 -----------------------------------------------------------------

    def create_scenario(self, name: str) -> Scenario:
        if not self.inst_store.has_data():
            raise WhatIfError("WHATIF_INSTANCE_NOT_FOUND", "当前没有可用的排产实例，请先导入或生成实例")
        return self.scenarios.create(name, self.current_instance_version())

    def require_scenario(self, scenario_id: str) -> Scenario:
        scenario = self.scenarios.get(scenario_id)
        if scenario is None:
            raise WhatIfError(
                "WHATIF_SCENARIO_NOT_FOUND",
                f"场景 {scenario_id} 不存在或已被淘汰（场景只活在内存里，服务重启即失）",
                [{"scenario_id": item.scenario_id, "name": item.name} for item in self.scenarios.list()],
            )
        return scenario

    def assert_base_fresh(self, scenario: Scenario) -> None:
        current = self.current_instance_version()
        if current != scenario.base_instance_version:
            raise WhatIfError(
                "WHATIF_BASE_STALE",
                f"场景基线已过期：创建时实例版本为 {scenario.base_instance_version}，"
                f"当前为 {current}（推演期间原始数据被改过）。请重建场景。",
            )

    def add_patches(self, scenario: Scenario, patches: list[dict]) -> dict:
        """追加 patch。先在副本上试算一遍，通过了才落到场景上——保证场景永远处于可物化状态。"""
        self.assert_base_fresh(scenario)
        if not isinstance(patches, list) or not patches:
            raise WhatIfError("INVALID_ARGUMENT", "patches 不能为空")
        merged = list(scenario.patches) + list(patches)
        try:
            patched_rows, effects = apply_patches(self.load_base_rows(), merged)
        except PatchError as error:
            raise WhatIfError(error.code, error.message, error.suggestions) from error
        self.scenarios.set_patches(scenario.scenario_id, merged)
        self._invalidate_materialized()
        return {
            "scenario": scenario.to_dict(),
            "applied": effects[-len(patches):],
            "scale": rows_scale(patched_rows),
        }

    def revert_patches(self, scenario: Scenario, count: int) -> dict:
        if count <= 0:
            raise WhatIfError("INVALID_ARGUMENT", "count 必须大于 0")
        if not scenario.patches:
            raise WhatIfError("INVALID_ARGUMENT", "该场景还没有任何改动可撤销")
        remaining = scenario.patches[:-count] if count < len(scenario.patches) else []
        self.scenarios.set_patches(scenario.scenario_id, remaining)
        self._invalidate_materialized()
        return self.describe(self.require_scenario(scenario.scenario_id))

    def describe(self, scenario: Scenario) -> dict:
        """场景详情：改了什么 + 规模变化 + 校验结果 + 落库确认令牌。"""
        base_rows = self.load_base_rows()
        base_scale = rows_scale(base_rows)
        stale = self.current_instance_version() != scenario.base_instance_version
        payload = {
            **scenario.to_dict(),
            "base_stale": stale,
            "current_instance_version": self.current_instance_version(),
            "base_scale": base_scale,
        }
        if stale:
            payload["validation"] = None
            payload["scale"] = None
            payload["apply_token"] = None
            return payload
        try:
            patched_rows, effects = apply_patches(base_rows, scenario.patches)
        except PatchError as error:
            raise WhatIfError(error.code, error.message, error.suggestions) from error
        scale = rows_scale(patched_rows)
        shop = self._materialize_shop(scenario, patched_rows)
        payload.update({
            "changes": effects,
            "scale": scale,
            "scale_delta": {
                key: scale[key] - base_scale.get(key, 0)
                for key in scale if scale[key] != base_scale.get(key, 0)
            },
            "validation": _validate(shop),
            "apply_token": _apply_token(scenario),
        })
        return payload

    # --- 物化 -----------------------------------------------------------------

    def _cache_key(self, scenario: Scenario) -> str:
        return f"{scenario.base_instance_version}|{scenario.digest()}"

    def _invalidate_materialized(self) -> None:
        with self._lock:
            self._materialized = None

    def _materialize_shop(self, scenario: Scenario, patched_rows: dict | None = None):
        """rows → ShopFloor，单槽缓存。

        **不建 SimulationRuntime**：那是一次整实例深拷贝（大实例十几秒），而
        describe / 校验只需要读 shop。构建 runtime 的成本只有真要跑仿真时才付，
        否则 describe_whatif_scenario 会在生产规模实例上直接撑爆 MCP 的调用超时。
        """
        key = self._cache_key(scenario)
        with self._lock:
            cached = self._materialized
            if cached is not None and cached[0] == key:
                return cached[1]
        if patched_rows is None:
            try:
                patched_rows, _ = apply_patches(self.load_base_rows(), scenario.patches)
            except PatchError as error:
                raise WhatIfError(error.code, error.message, error.suggestions) from error
        downtimes_by_machine = _downtime_rows_to_objects(patched_rows.get("downtimes") or [])
        shop = self.inst_store.build_shopfloor_from_rows(patched_rows, downtimes_by_machine)
        with self._lock:
            self._materialized = (key, shop, None)
        return shop

    def _materialize(self, scenario: Scenario, patched_rows: dict | None = None):
        """rows → ShopFloor + SimulationRuntime，单槽缓存。仅跑仿真时才走这条。

        同一场景第二次 run 可省掉 build_shopfloor(~2s) 与 runtime 深拷贝(十几秒)。
        """
        key = self._cache_key(scenario)
        with self._lock:
            cached = self._materialized
            if cached is not None and cached[0] == key and cached[2] is not None:
                return cached[1], cached[2]
        shop = self._materialize_shop(scenario, patched_rows)
        runtime = SimulationRuntime(shop)
        with self._lock:
            # shop 必须与 runtime 同源：期间若缓存已被别的场景顶掉，这里连同 shop 一起写回。
            self._materialized = (key, shop, runtime)
        return shop, runtime

    # --- 跑规则 ---------------------------------------------------------------

    def submit_run(
        self,
        scenario: Scenario,
        rule_names: list[str],
        include_baseline: bool = False,
    ) -> WhatIfRun:
        self.assert_base_fresh(scenario)
        rules = _normalize_rule_names(rule_names)
        run = WhatIfRun(
            run_id=f"wr-{uuid.uuid4().hex[:12]}",
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
            rule_names=rules,
            # 场景没有任何改动时，它自己就是基线，再跑一遍纯属浪费。
            include_baseline=bool(include_baseline) and bool(scenario.patches),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._runs[run.run_id] = run
            while len(self._runs) > self._max_runs:
                evicted, _ = self._runs.popitem(last=False)
                self._futures.pop(evicted, None)
        future = self._executor.submit(self._execute_run, scenario, run)
        with self._lock:
            self._futures[run.run_id] = future
        return run

    def wait_for(self, run_id: str, seconds: float) -> None:
        """在预算内等一等：小实例毫秒级完成，一次调用就能拿到结果，不必轮询。"""
        with self._lock:
            future = self._futures.get(run_id)
        if future is None or seconds <= 0:
            return
        try:
            future.result(timeout=seconds)
        except TimeoutError:
            pass
        except Exception:  # 失败详情已记录在 run.error 上
            pass

    def _execute_run(self, scenario: Scenario, run: WhatIfRun) -> None:
        try:
            shop, runtime = self._materialize(scenario)
            validation = _validate(shop)
            run.warnings = validation["warnings"]
            if validation["errors"]:
                run.status = "failed"
                run.error = {
                    "code": "WHATIF_VALIDATION_FAILED",
                    "message": f"改动后的实例有 {len(validation['errors'])} 个错误级问题，仿真结果不可信，已拒绝执行",
                    "suggestions": validation["errors"][:20],
                }
                return
            results = self._run_rules(run, shop, runtime, "scenario", run.scenario_name)
            if run.include_baseline:
                # 同一次调用里把未改动的基线也跑掉：否则调用方要多花「建场景 + 再跑一次
                # + 对比」三步，而宿主 Agent 的单次 Run 步数预算很容易不够。
                base_shop, base_runtime = self._materialize_baseline()
                results += self._run_rules(run, base_shop, base_runtime, "baseline", "现状（基线）")
            results.sort(key=lambda item: item["metrics"].get("total_tardiness", 0.0))
            run.results = results
            run.status = "done"
        except WhatIfError as error:
            run.status = "failed"
            run.error = {"code": error.code, "message": error.message, "suggestions": error.suggestions}
        except Exception as error:  # noqa: BLE001 — 后台线程边界，必须把失败落到 run 上
            logger.exception("whatif run %s failed", run.run_id)
            run.status = "failed"
            run.error = {"code": "WHATIF_RUN_FAILED", "message": f"{type(error).__name__}: {error}", "suggestions": []}
        finally:
            run.finished_at = datetime.now(timezone.utc).isoformat()

    def _run_rules(self, run: WhatIfRun, shop, runtime, variant: str, label: str) -> list[dict]:
        """在一份实例上逐规则跑仿真。

        与 /api/simulate/compare 完全同形：一个 runtime 逐规则复用，
        保证 KPI 口径与主流程逐字一致。
        """
        results = []
        for rule_name in run.rule_names:
            result = Simulator(shop, BUILTIN_RULES[rule_name], runtime=runtime).run()
            analytics = build_schedule_analytics(shop, result)
            metrics = result.to_dict()
            metrics.update({
                key: round(float(value), 4)
                for key, value in analytics.objective_values.items()
            })
            metrics["completed_operations"] = analytics.completed_operations
            metrics["total_operations"] = len(shop.operations)
            metrics["feasible"] = analytics.feasible
            results.append({
                "rule_name": rule_name,
                "variant": variant,
                "label": label,
                "metrics": metrics,
                "scheduled_operations": len(result.schedule or []),
            })
            logger.info(
                "whatif[%s/%s] variant=%s rule=%s total_tardiness=%.2f makespan=%.2f feasible=%s",
                run.run_id, run.scenario_name, variant, rule_name,
                metrics.get("total_tardiness", 0.0), metrics.get("makespan", 0.0),
                metrics.get("feasible"),
            )
        return results

    def _materialize_baseline(self):
        """未打任何 patch 的基线实例。

        刻意不走单槽缓存：那一槽留给当前场景，否则每次带基线的推演都会把场景那份
        顶掉，下一轮又要重建一次（大实例十几秒）。
        """
        rows = self.load_base_rows()
        downtimes = _downtime_rows_to_objects(rows.get("downtimes") or [])
        shop = self.inst_store.build_shopfloor_from_rows(rows, downtimes)
        return shop, SimulationRuntime(shop)

    def get_run(self, run_id: str) -> WhatIfRun:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise WhatIfError("WHATIF_RUN_NOT_FOUND", f"推演任务 {run_id} 不存在或已被淘汰")
        return run

    def list_runs(self) -> list[WhatIfRun]:
        with self._lock:
            return list(reversed(self._runs.values()))

    # --- 对比 -----------------------------------------------------------------

    def compare_runs(self, run_ids: list[str], metric_keys: list[str] | None = None) -> dict:
        if not run_ids:
            raise WhatIfError("INVALID_ARGUMENT", "run_ids 不能为空")
        runs = [self.get_run(run_id) for run_id in dict.fromkeys(run_ids)]
        pending = [run.run_id for run in runs if run.status == "running"]
        if pending:
            raise WhatIfError("WHATIF_RUN_NOT_READY", f"以下推演还没跑完：{', '.join(pending)}")
        failed = [run for run in runs if run.status == "failed"]
        if failed:
            raise WhatIfError(
                "WHATIF_RUN_FAILED",
                f"以下推演执行失败，无法参与对比：{', '.join(run.run_id for run in failed)}",
                [run.error for run in failed if run.error],
            )
        keys = [key for key in (metric_keys or DEFAULT_METRIC_KEYS) if key in OBJECTIVE_SPECS]
        if not keys:
            raise WhatIfError(
                "INVALID_ARGUMENT",
                "metric_keys 里没有任何已知指标",
                [{"key": spec.key, "label": spec.label, "direction": spec.direction}
                 for spec in OBJECTIVE_SPECS.values()],
            )

        entries = []
        for run in runs:
            for item in run.results:
                variant = item.get("variant", "scenario")
                entries.append({
                    "run_id": run.run_id,
                    "scenario_id": run.scenario_id,
                    "scenario_name": item.get("label") or run.scenario_name,
                    "variant": variant,
                    "rule_name": item["rule_name"],
                    "values": {key: item["metrics"].get(key) for key in keys},
                })
        if not entries:
            raise WhatIfError("WHATIF_RUN_NOT_READY", "所选推演里没有任何结果")

        # 若结果里带着未改动的基线，它才是天然的比较锚点——否则「变好/变坏」
        # 会以某个已改动场景为准，方向全反。同规则优先，避免拿 EDD 当 ATC 的基准。
        baseline = next(
            (item for item in entries
             if item["variant"] == "baseline" and item["rule_name"] == entries[0]["rule_name"]),
            next((item for item in entries if item["variant"] == "baseline"), entries[0]),
        )
        for entry in entries:
            entry["deltas"] = {
                key: _delta(baseline["values"].get(key), entry["values"].get(key), OBJECTIVE_SPECS[key].direction)
                for key in keys
            }
        return {
            "baseline": {
                "run_id": baseline["run_id"],
                "scenario_name": baseline["scenario_name"],
                "variant": baseline["variant"],
                "rule_name": baseline["rule_name"],
            },
            "metrics": [
                {"key": key, "label": OBJECTIVE_SPECS[key].label, "direction": OBJECTIVE_SPECS[key].direction}
                for key in keys
            ],
            "entries": entries,
        }

    # --- 高危：落库 -----------------------------------------------------------

    def apply_to_instance(self, scenario: Scenario, confirm_token: str) -> dict:
        """把场景改动真正写入 inst_* 表。会 bump inst_version，作废四步流程快照。

        四道闸：token 匹配 → 基线新鲜 → 校验无 error → 写前自动备份。
        """
        self.assert_base_fresh(scenario)
        expected = _apply_token(scenario)
        if not confirm_token or confirm_token != expected:
            raise WhatIfError(
                "WHATIF_CONFIRM_REQUIRED",
                "confirm_token 缺失或与当前场景状态不符。请先调用场景详情拿到最新 apply_token，"
                "并向用户展示改动清单取得明确确认。",
            )
        if not scenario.patches:
            raise WhatIfError("INVALID_ARGUMENT", "该场景没有任何改动，无需落库")
        try:
            patched_rows, effects = apply_patches(self.load_base_rows(), scenario.patches)
        except PatchError as error:
            raise WhatIfError(error.code, error.message, error.suggestions) from error

        shop = self._materialize_shop(scenario, patched_rows)
        validation = _validate(shop)
        if validation["errors"]:
            raise WhatIfError(
                "WHATIF_VALIDATION_FAILED",
                f"改动后的实例有 {len(validation['errors'])} 个错误级问题，拒绝落库",
                validation["errors"][:20],
            )

        backup_path = self._backup_current_instance()
        save_args = rows_to_save_args(patched_rows)
        self.inst_store.save_from_csv(**save_args)
        self.downtime_store.replace_all(patched_rows.get("downtimes") or [])
        self._invalidate_materialized()
        new_version = self.current_instance_version()
        logger.warning(
            "whatif apply: scenario=%s patches=%d inst_version -> %d backup=%s",
            scenario.scenario_id, len(scenario.patches), new_version, backup_path,
        )
        return {
            "applied": True,
            "scenario_id": scenario.scenario_id,
            "changes": effects,
            "instance_version": new_version,
            "backup_path": backup_path,
            "warnings": validation["warnings"],
            "side_effect_notice": (
                "实例数据已被覆盖，inst_version 已递增：校验 / 仿真 / 优化 / 方案评审"
                "四步流程的历史快照已全部作废，需要重新跑一遍。"
            ),
        }

    def _backup_current_instance(self) -> str:
        """整库快照。用 SQLite 原生 backup API 而非文件拷贝——库跑在 WAL 模式下，
        直接 copy 主库文件会漏掉尚未 checkpoint 的 WAL 内容。"""
        import sqlite3
        from pathlib import Path

        backup_dir = Path(self.inst_store.db_path).resolve().parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = backup_dir / f"llm4drd-before-whatif-{stamp}.db"
        source = sqlite3.connect(self.inst_store.db_path)
        try:
            target = sqlite3.connect(str(path))
            try:
                source.backup(target)
            finally:
                target.close()
        finally:
            source.close()
        return str(path)


# --- 辅助 ---------------------------------------------------------------------

def _normalize_rule_names(rule_names: list[str] | None) -> list[str]:
    names = [str(name).strip().upper() for name in (rule_names or []) if str(name).strip()]
    names = list(dict.fromkeys(names)) or ["ATC"]
    unknown = [name for name in names if name not in BUILTIN_RULES]
    if unknown:
        raise WhatIfError(
            "PLANNING_RULE_NOT_FOUND",
            f"未找到内置排产规则：{', '.join(unknown)}",
            [{"rule_name": name} for name in get_all_rule_names()],
        )
    return names


def _validate(shop) -> dict:
    # 延迟导入：api.server 在模块级构造 WhatIfService，顶层导入会循环。
    from .server import _validate_instance

    report = _validate_instance(shop)
    return {"errors": report.get("errors") or [], "warnings": report.get("warnings") or []}


def _apply_token(scenario: Scenario) -> str:
    return f"{scenario.base_instance_version}-{scenario.digest()}"


def _delta(baseline_value, value, direction: str) -> dict:
    if baseline_value is None or value is None:
        return {"abs": None, "pct": None, "better": None}
    diff = float(value) - float(baseline_value)
    pct = (diff / abs(float(baseline_value)) * 100.0) if float(baseline_value) else None
    if abs(diff) < 1e-9:
        better = None
    else:
        better = (diff < 0) if direction == "min" else (diff > 0)
    return {
        "abs": round(diff, 4),
        "pct": round(pct, 2) if pct is not None else None,
        "better": better,
    }


def _downtime_rows_to_objects(rows: list[dict]) -> dict:
    from ..core.models import Downtime

    result: dict[str, list] = {}
    for row in rows:
        downtime = Downtime(
            id=str(row.get("id")),
            machine_id=str(row.get("machine_id") or ""),
            downtime_type=str(row.get("downtime_type") or "planned"),
            start_time=float(row.get("start_time") or 0.0),
            end_time=float(row.get("end_time") or 0.0),
        )
        result.setdefault(downtime.machine_id, []).append(downtime)
    return result


_INITIAL_FIELDS = (
    "initial_status", "initial_start_time", "initial_end_time",
    "initial_remaining_processing_time", "initial_assigned_machine_id",
    "initial_assigned_tooling_ids", "initial_assigned_personnel_ids",
)


def rows_to_save_args(rows: dict) -> dict:
    """把 rows 适配成 InstanceStore.save_from_csv 的入参。

    两处形状差异必须在这里抹平：
      1. load_all() 把 initial_* 内联在 operations 行里，save_from_csv 要求单独的
         initial_state_rows（按 op_id 关联）；
      2. save_from_csv 完全不管停机，machine_downtime 得由调用方另走 DowntimeStore。
    """
    operations = rows.get("operations") or []
    initial_state_rows = [
        {"op_id": row.get("op_id"), **{field: row.get(field) for field in _INITIAL_FIELDS}}
        for row in operations
        if any(_has_value(row.get(field)) for field in _INITIAL_FIELDS)
    ]
    plan_start_raw = (rows.get("planning_context") or {}).get("plan_start_at")
    plan_start_at = None
    if plan_start_raw:
        plan_start_at = ensure_aware(datetime.fromisoformat(str(plan_start_raw).replace("Z", "+00:00")))
    return {
        "orders_rows": rows.get("orders") or [],
        "tasks_rows": rows.get("tasks") or [],
        "ops_rows": operations,
        "machine_type_rows": rows.get("machine_types") or [],
        "machine_rows": rows.get("machines") or [],
        "tooling_type_rows": rows.get("tooling_types") or [],
        "tooling_rows": rows.get("toolings") or [],
        "personnel_rows": rows.get("personnel") or [],
        "initial_state_rows": initial_state_rows,
        "plan_start_at": plan_start_at,
    }


def _has_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True
