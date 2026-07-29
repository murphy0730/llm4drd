from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from ..core.models import Operation, Order, ShopFloor
from ..optimization.objectives import bounded_bottleneck_limit, rank_bottleneck_machines


MAX_SOLUTIONS = 4
MAX_SEARCH_RESULTS = 50
MAX_ORDER_OPERATIONS = 200
CHINESE_NUMERALS = ("零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十")
SOLUTION_CATEGORY_LABELS = {
    "baseline": "基线方案",
    "pareto": "优化候选方案",
    "exact_reference": "精确参考方案",
    "heuristic": "启发式参考方案",
}


@dataclass
class SolutionCatalog:
    """一次查询里的全部可选方案，顺序与评审页候选列表一致（基线 → 优化候选 → 参考方案）。"""

    solutions: dict[str, dict]
    names: dict[str, str]
    categories: dict[str, str]


@dataclass
class PlanningQueryError(Exception):
    code: str
    message: str
    suggestions: list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "suggestions": list(self.suggestions),
            },
        }


class PlanningQueryService:
    def __init__(
        self,
        *,
        shop: ShopFloor,
        tasks: dict[str, dict],
        latest_task_id: str | None,
        instance_version: int,
    ) -> None:
        self._shop = shop
        self._tasks = tasks
        self._latest_task_id = latest_task_id
        self._instance_version = int(instance_version)

    def get_overview(self, task_id: str | None = None) -> dict:
        resolved_id, task, payload = self._resolve_task(task_id)
        catalog = self._solution_catalog(task, payload)
        return {
            "instance_version": self._instance_version,
            "task_id": resolved_id,
            "status": task.get("status", "done"),
            # solutions 列出全部可选方案；candidate_count 仍只数优化候选，与优化页口径一致。
            "solution_count": len(catalog.solutions),
            "candidate_count": len(payload.get("solutions") or []),
            "archive_size": int(payload.get("archive_size", 0) or 0),
            "baseline_count": 1 if (payload.get("baseline") or {}).get("solution_id") else 0,
            "reference_count": len(self._reference_solutions(task, payload)),
            "solutions": [
                self._solution_summary(solution, catalog)
                for solution in catalog.solutions.values()
            ],
        }

    def compare_solutions(
        self,
        task_id: str | None = None,
        solution_ids: list[str] | None = None,
        metric_keys: list[str] | None = None,
        bottleneck_limit: int | None = None,
    ) -> dict:
        resolved_id, task, payload = self._resolve_task(task_id)
        catalog = self._solution_catalog(task, payload)
        solutions, truncated = self._select_solutions(catalog, solution_ids)
        keys = metric_keys or [
            "total_tardiness",
            "main_order_tardy_total_time",
            "makespan",
        ]
        limit = bounded_bottleneck_limit(bottleneck_limit)
        rows = []
        sources = set()
        for solution in solutions:
            metrics = {
                key: self._metric_value(solution, key)
                for key in keys
            }
            machines, source = self._bottleneck_machines(solution, limit)
            sources.add(source)
            rows.append({
                **self._solution_summary(solution, catalog),
                "metric_values": metrics,
                "bottleneck_machines": machines,
            })
        return {
            "task_id": resolved_id,
            "solution_count": len(rows),
            "truncated": truncated,
            "metric_keys": list(keys),
            "bottleneck_limit": limit,
            # 任一方案降级就标 legacy_top5，提醒调用方这批数据的排名不完整。
            "bottleneck_source": "legacy_top5" if "legacy_top5" in sources else "utilization_ranking",
            "solutions": rows,
        }

    def _bottleneck_machines(self, solution: dict, limit: int) -> tuple[list[dict], str]:
        """方案里利用率最高的若干台机器。

        返回 (列表, 数据来源)。老任务的 payload 没有逐机利用率，只能退回 summary 里那
        份 top5 的 id——它没有利用率数值，也排不满调用方要的条数，用 legacy_top5 标出来。
        """
        summary = solution.get("summary") or {}
        utilization = summary.get("machine_utilization")
        if isinstance(utilization, dict) and utilization:
            return rank_bottleneck_machines(utilization, self._shop, limit), "utilization_ranking"
        legacy_ids = [str(item) for item in (summary.get("bottleneck_machine_ids") or [])][:limit]
        items = []
        for machine_id in legacy_ids:
            machine = self._shop.machines.get(machine_id)
            machine_type = self._shop.machine_types.get(machine.type_id) if machine else None
            items.append({
                "machine_id": machine_id,
                "machine_name": machine.name if machine else "",
                "type_id": machine.type_id if machine else "",
                "type_name": machine_type.name if machine_type else "",
                "is_critical": bool(machine_type.is_critical) if machine_type else False,
                "utilization": None,
            })
        return items, "legacy_top5"

    def search_orders(self, query: str, limit: int = 20) -> list[dict]:
        needle = str(query or "").strip().lower()
        matches = []
        for order in self._shop.orders.values():
            order_id = str(order.id)
            name = str(order.name or "")
            match_type = self._match_type(needle, order_id, name)
            if match_type is None:
                continue
            matches.append({
                "order_id": order_id,
                "order_name": name,
                "match_type": match_type,
            })
        matches.sort(key=lambda item: (
            self._match_rank(item["match_type"]),
            self._natural_key(item["order_id"]),
        ))
        return matches[: self._bounded_limit(limit)]

    def get_order_planning(
        self,
        order_id: str,
        *,
        task_id: str | None = None,
        solution_ids: list[str] | None = None,
        max_operations: int = MAX_ORDER_OPERATIONS,
    ) -> dict:
        order = self._shop.orders.get(order_id)
        if order is None:
            raise PlanningQueryError(
                "ORDER_NOT_FOUND",
                f"未找到订单: {order_id}",
                self.search_orders(order_id, 10),
            )
        resolved_id, task, payload = self._resolve_task(task_id)
        catalog = self._solution_catalog(task, payload)
        solutions, solutions_truncated = self._select_solutions(catalog, solution_ids)
        operation_limit = max(1, min(int(max_operations), MAX_ORDER_OPERATIONS))
        planning_rows = []
        for solution in solutions:
            entries = [
                entry
                for entry in list(solution.get("schedule") or [])
                if self._entry_order_id(entry) == order.id
            ]
            entries.sort(key=lambda item: (
                self._finite(item.get("start"), math.inf),
                str(item.get("op_id") or ""),
            ))
            finite_ends = [
                value
                for entry in entries
                if (value := self._finite(entry.get("end"))) is not None
            ]
            completion = max(finite_ends, default=None)
            tardiness = (
                max(0.0, completion - float(order.due_date))
                if completion is not None and math.isfinite(float(order.due_date))
                else None
            )
            planning_rows.append({
                "solution_id": solution.get("solution_id"),
                "solution_name": self._solution_name(solution, catalog),
                "feasible": self._feasible(solution),
                "planned": bool(entries),
                "operation_count": len(entries),
                "truncated": len(entries) > operation_limit,
                "order_completion_hours": self._rounded(completion),
                "order_completion_at": self._time_label(completion),
                "order_tardiness_hours": self._rounded(tardiness),
                "total_tardiness_hours": self._rounded(
                    self._metric_value(solution, "total_tardiness")
                ),
                "operations": [
                    self._operation_entry(entry) for entry in entries[:operation_limit]
                ],
            })
        return {
            "task_id": resolved_id,
            "order": self._order_payload(order),
            "solution_count": len(planning_rows),
            "solutions_truncated": solutions_truncated,
            "solutions": planning_rows,
        }

    def search_operations(
        self,
        query: str,
        *,
        order_id: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        needle = str(query or "").strip().lower()
        matches = []
        for operation in self._shop.operations.values():
            task = self._shop.tasks.get(operation.task_id)
            current_order_id = task.order_id if task else ""
            if order_id and current_order_id != order_id:
                continue
            fields = (
                str(operation.id),
                str(operation.name or ""),
                str(operation.task_id or ""),
                str(current_order_id or ""),
            )
            match_type = self._match_type(needle, *fields)
            if match_type is None:
                continue
            matches.append({
                **self._operation_payload(operation),
                "match_type": match_type,
            })
        matches.sort(key=lambda item: (
            self._match_rank(item["match_type"]),
            self._natural_key(item["operation_id"]),
        ))
        return matches[: self._bounded_limit(limit)]

    def get_operation_planning(
        self,
        operation_id: str,
        *,
        task_id: str | None = None,
        solution_ids: list[str] | None = None,
    ) -> dict:
        operation = self._shop.operations.get(operation_id)
        if operation is None:
            raise PlanningQueryError(
                "OPERATION_NOT_FOUND",
                f"未找到工序: {operation_id}",
                self.search_operations(operation_id, limit=10),
            )
        resolved_id, task, payload = self._resolve_task(task_id)
        catalog = self._solution_catalog(task, payload)
        solutions, truncated = self._select_solutions(catalog, solution_ids)
        placements = []
        for solution in solutions:
            entry = next(
                (
                    item for item in list(solution.get("schedule") or [])
                    if str(item.get("op_id") or "") == operation_id
                ),
                None,
            )
            if entry is None:
                placements.append({
                    "solution_id": solution.get("solution_id"),
                    "solution_name": self._solution_name(solution, catalog),
                    "planned": False,
                    "reason": "operation_not_present_in_solution",
                })
                continue
            placements.append({
                "solution_id": solution.get("solution_id"),
                "solution_name": self._solution_name(solution, catalog),
                "planned": True,
                **self._operation_entry(entry, include_identity=False),
            })
        return {
            "task_id": resolved_id,
            "operation": self._operation_payload(operation),
            "solution_count": len(placements),
            "solutions_truncated": truncated,
            "placements": placements,
        }

    def _resolve_task(self, task_id: str | None) -> tuple[str, dict, dict]:
        resolved_id = task_id or self._latest_task_id
        if not resolved_id:
            resolved_id = next(
                (
                    key for key, value in reversed(list(self._tasks.items()))
                    if value.get("status") == "done" and value.get("result")
                ),
                None,
            )
        if not resolved_id or resolved_id not in self._tasks:
            raise PlanningQueryError(
                "PLANNING_TASK_NOT_FOUND",
                "未找到可用的排产优化结果，请先完成一次混合优化",
            )
        task = self._tasks[resolved_id]
        task_instance_version = task.get("instance_version")
        if (
            task_instance_version is not None
            and int(task_instance_version) != self._instance_version
        ):
            raise PlanningQueryError(
                "PLANNING_SNAPSHOT_STALE",
                f"排产结果属于实例版本 {task_instance_version}，当前版本为 {self._instance_version}",
            )
        if task.get("status") != "done" or not task.get("result"):
            raise PlanningQueryError(
                "PLANNING_TASK_NOT_READY",
                f"排产优化任务 {resolved_id} 尚未完成",
            )
        payload = task.get("export_result") or task.get("result") or {}
        return resolved_id, task, payload

    def _select_solutions(
        self,
        catalog: SolutionCatalog,
        solution_ids: list[str] | None,
    ) -> tuple[list[dict], bool]:
        available = catalog.solutions
        if solution_ids:
            requested = list(dict.fromkeys(str(item) for item in solution_ids if item))
            if len(requested) > MAX_SOLUTIONS:
                raise PlanningQueryError(
                    "INVALID_ARGUMENT",
                    f"单次最多比较 {MAX_SOLUTIONS} 个方案",
                )
            missing = [item for item in requested if item not in available]
            if missing:
                raise PlanningQueryError(
                    "SOLUTION_NOT_FOUND",
                    f"未找到方案: {', '.join(missing)}",
                    [
                        {
                            "solution_id": key,
                            "solution_name": catalog.names.get(key, key),
                        }
                        for key in available
                    ],
                )
            return [available[item] for item in requested], False
        # 不指定方案时按评审页顺序取前几个，基线方案排在最前，作为其余方案的对照。
        values = list(available.values())
        return values[:MAX_SOLUTIONS], len(values) > MAX_SOLUTIONS

    def _solution_catalog(self, task: dict, payload: dict) -> SolutionCatalog:
        entries: list[tuple[str, dict, str]] = []
        baseline = payload.get("baseline")
        if isinstance(baseline, dict) and baseline.get("solution_id"):
            entries.append((str(baseline["solution_id"]), baseline, "baseline"))
        for item in payload.get("solutions") or []:
            if isinstance(item, dict) and item.get("solution_id"):
                entries.append((str(item["solution_id"]), item, self._category(item, "pareto")))
        for item in self._reference_solutions(task, payload):
            entries.append((str(item["solution_id"]), item, self._category(item, "heuristic")))
        solutions: dict[str, dict] = {}
        categories: dict[str, str] = {}
        for solution_id, item, category in entries:
            if solution_id in solutions:
                continue
            solutions[solution_id] = item
            categories[solution_id] = category
        names = {
            solution_id: self._scheme_display_name(index)
            for index, solution_id in enumerate(solutions)
        }
        return SolutionCatalog(solutions=solutions, names=names, categories=categories)

    @staticmethod
    def _category(solution: dict, fallback: str) -> str:
        """与前端 candidateCategory 同口径：source 是算法内部标记，只有这几个值有业务语义。"""
        source = str(solution.get("source") or "")
        if source == "baseline":
            return "baseline"
        if source.startswith("exact_reference"):
            return "exact_reference"
        if source == "heuristic_rule":
            return "heuristic"
        return fallback

    @staticmethod
    def _reference_solutions(task: dict, payload: dict) -> list[dict]:
        merged: dict[str, dict] = {}
        for source in (
            payload.get("reference_solutions") or [],
            task.get("reference_solutions") or [],
        ):
            for item in source:
                if isinstance(item, dict) and item.get("solution_id"):
                    merged[str(item["solution_id"])] = item
        return list(merged.values())

    @staticmethod
    def _solution_name(solution: dict, catalog: SolutionCatalog) -> str:
        solution_id = str(solution.get("solution_id") or "")
        return catalog.names.get(solution_id, solution_id)

    @staticmethod
    def _scheme_display_name(index: int) -> str:
        number = index + 1
        if number <= 10:
            numeral = CHINESE_NUMERALS[number]
        elif number < 20:
            numeral = f"十{CHINESE_NUMERALS[number - 10]}"
        else:
            ones = CHINESE_NUMERALS[number % 10] if number % 10 else ""
            numeral = f"{CHINESE_NUMERALS[number // 10]}十{ones}"
        return f"方案{numeral}"

    def _solution_summary(
        self,
        solution: dict,
        catalog: SolutionCatalog,
    ) -> dict:
        solution_id = str(solution.get("solution_id") or "")
        category = catalog.categories.get(solution_id, "pareto")
        return {
            "solution_id": solution.get("solution_id"),
            "solution_name": self._solution_name(solution, catalog),
            "category": category,
            "category_label": SOLUTION_CATEGORY_LABELS[category],
            # 基线方案与启发式参考方案带规则名（如 ATC），Pareto 方案没有。
            "rule_name": solution.get("rule_name"),
            "source": solution.get("source") or category,
            "feasible": self._feasible(solution),
            "total_tardiness_hours": self._rounded(
                self._metric_value(solution, "total_tardiness")
            ),
            "main_order_tardiness_hours": self._rounded(
                self._metric_value(solution, "main_order_tardy_total_time")
            ),
            "makespan_hours": self._rounded(
                self._metric_value(solution, "makespan")
            ),
        }

    @staticmethod
    def _feasible(solution: dict) -> bool:
        # 基线方案的 payload 没有顶层 feasible 字段，可行性只记在 metrics 里。
        value = solution.get("feasible")
        if value is None:
            value = (solution.get("metrics") or {}).get("feasible")
        return bool(value)

    @staticmethod
    def _metric_value(solution: dict, key: str) -> float | int | None:
        for source in (solution.get("objectives") or {}, solution.get("metrics") or {}):
            value = source.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return value
        return None

    def _entry_order_id(self, entry: dict) -> str:
        explicit = entry.get("order_id")
        if explicit:
            return str(explicit)
        operation = self._shop.operations.get(str(entry.get("op_id") or ""))
        task = self._shop.tasks.get(operation.task_id) if operation else None
        return task.order_id if task else ""

    def _operation_entry(self, entry: dict, *, include_identity: bool = True) -> dict:
        operation_id = str(entry.get("op_id") or "")
        operation = self._shop.operations.get(operation_id)
        start = self._finite(entry.get("start"))
        end = self._finite(entry.get("end"))
        payload = {
            "machine_id": entry.get("machine_id"),
            "start_hours": self._rounded(start),
            "end_hours": self._rounded(end),
            "start_at": entry.get("start_at") or self._time_label(start),
            "end_at": entry.get("end_at") or self._time_label(end),
            "tooling_ids": list(entry.get("tooling_ids") or []),
            "personnel_ids": list(entry.get("personnel_ids") or []),
        }
        if include_identity:
            payload = {
                "operation_id": operation_id,
                "operation_name": (
                    operation.name if operation is not None else entry.get("op_name", "")
                ),
                "task_id": (
                    operation.task_id if operation is not None else entry.get("task_id", "")
                ),
                **payload,
            }
        return payload

    def _operation_payload(self, operation: Operation) -> dict:
        task = self._shop.tasks.get(operation.task_id)
        return {
            "operation_id": operation.id,
            "operation_name": operation.name,
            "task_id": operation.task_id,
            "order_id": task.order_id if task else "",
            "process_type": operation.process_type,
            "predecessor_operation_ids": list(operation.predecessor_ops),
        }

    def _order_payload(self, order: Order) -> dict:
        due = self._finite(order.due_date)
        return {
            "order_id": order.id,
            "order_name": order.name,
            "priority": order.priority,
            "due_hours": self._rounded(due),
            "due_at": self._time_label(due),
        }

    def _time_label(self, value: float | None) -> str | None:
        return self._shop.time_label(value) if value is not None else None

    @staticmethod
    def _finite(value: object, default: float | None = None) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if math.isfinite(result) else default

    @staticmethod
    def _rounded(value: float | int | None) -> float | int | None:
        if value is None or isinstance(value, bool):
            return value
        return round(float(value), 4)

    @staticmethod
    def _match_type(needle: str, *fields: str) -> str | None:
        normalized = [str(field or "").lower() for field in fields]
        if not needle:
            return "all"
        if normalized and normalized[0] == needle:
            return "exact_id"
        if normalized and normalized[0].startswith(needle):
            return "id_prefix"
        if any(needle in field for field in normalized):
            return "contains"
        return None

    @staticmethod
    def _match_rank(value: str) -> int:
        return {"exact_id": 0, "id_prefix": 1, "contains": 2, "all": 3}.get(value, 4)

    @staticmethod
    def _natural_key(value: str) -> tuple:
        return tuple(
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(value))
        )

    @staticmethod
    def _bounded_limit(limit: int) -> int:
        return max(1, min(int(limit), MAX_SEARCH_RESULTS))
