"""What-if 场景的领域逻辑：把「对原始数据的改动」表达成打在 rows 上的 patch。

设计要点：patch 停在 rows 层（load_all() 的返回形状，与 DB 列名、Excel 模板同构），
而不是直接改 ShopFloor 对象图。原因：
  - rows 是纯 dict，改完丢给 InstanceStore.build_shopfloor_from_rows 就白拿了
    build_indexes / ensure_calendar_capacity / 在制状态还原这一整套；
  - 直接改对象图则要手工补 _machine_by_type 反向索引与日历编译缓存，且无法回滚、
    无法向用户回显「我到底改了什么」。

本模块是纯函数 + 内存容器，不依赖 FastAPI，也绝不触碰数据库。
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from ..data.db import normalize_shifts_field


class PatchError(ValueError):
    """patch 语法/语义错误。带 code 与 suggestions，供 MCP 层转成业务错误让模型自行澄清。"""

    def __init__(self, code: str, message: str, suggestions: list | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.suggestions = list(suggestions or [])


# --- 字段归一化 ---------------------------------------------------------------

def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _num(value: Any) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return 0.0
    return float(value)


def _num_or_none(value: Any):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return float(value)


def _int(value: Any) -> int:
    if value is None or (isinstance(value, str) and not value.strip()):
        return 0
    return int(float(value))


def _bool01(value: Any) -> int:
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "yes", "y", "是"} else 0
    return 1 if value else 0


def _shifts(value: Any) -> str:
    """班次统一成 "day/start_hour/hours;..." 存储串；数组形式也接受。"""
    return normalize_shifts_field(value)


def _idlist(value: Any) -> str:
    """分号分隔的 id 串；数组形式也接受。中英文逗号一并归一成分号。"""
    if isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value]
    else:
        raw = _text(value).replace(",", ";").replace("，", ";")
        items = [item.strip() for item in raw.split(";")]
    return ";".join(item for item in items if item)


# --- 实体定义 -----------------------------------------------------------------

@dataclass(frozen=True)
class EntitySpec:
    rows_key: str
    pk: str
    label: str
    columns: dict[str, Callable[[Any], Any]]
    singleton: bool = False
    autoincrement_pk: bool = False


ENTITY_SPECS: dict[str, EntitySpec] = {
    "order": EntitySpec("orders", "order_id", "订单", {
        "order_id": _text, "order_name": _text,
        "release_time": _num, "due_date": _num, "priority": _int,
    }),
    "task": EntitySpec("tasks", "task_id", "任务令", {
        "task_id": _text, "order_id": _text, "task_name": _text,
        "is_main": _bool01, "predecessor_task_ids": _idlist,
        "release_time": _num, "due_date": _num,
    }),
    "operation": EntitySpec("operations", "op_id", "工序", {
        "op_id": _text, "task_id": _text, "op_name": _text, "process_type": _text,
        "processing_time": _num, "turnover_time": _num,
        "predecessor_ops": _idlist, "predecessor_tasks": _idlist,
        "eligible_machine_ids": _idlist,
        "required_tooling_types": _idlist, "required_personnel_skills": _idlist,
        "initial_status": _text,
        "initial_start_time": _num_or_none, "initial_end_time": _num_or_none,
        "initial_remaining_processing_time": _num_or_none,
        "initial_assigned_machine_id": _text,
        "initial_assigned_tooling_ids": _idlist,
        "initial_assigned_personnel_ids": _idlist,
    }),
    "machine_type": EntitySpec("machine_types", "type_id", "机器类型", {
        "type_id": _text, "type_name": _text, "is_critical": _bool01,
    }),
    "machine": EntitySpec("machines", "machine_id", "机器", {
        "machine_id": _text, "machine_name": _text, "type_id": _text, "shifts": _shifts,
    }),
    "tooling_type": EntitySpec("tooling_types", "type_id", "工装类型", {
        "type_id": _text, "type_name": _text,
    }),
    "tooling": EntitySpec("toolings", "tooling_id", "工装", {
        "tooling_id": _text, "tooling_name": _text, "type_id": _text, "shifts": _shifts,
    }),
    "personnel": EntitySpec("personnel", "personnel_id", "人员", {
        "personnel_id": _text, "personnel_name": _text, "skills": _idlist, "shifts": _shifts,
    }),
    "downtime": EntitySpec("downtimes", "id", "停机窗口", {
        "id": _int, "machine_id": _text, "downtime_type": _text,
        "start_time": _num, "end_time": _num,
    }, autoincrement_pk=True),
    "planning_context": EntitySpec("planning_context", "id", "计划基准时间", {
        "plan_start_at": _text,
    }, singleton=True),
}

VALID_OPS = ("add", "update", "remove")


# --- patch 应用 ---------------------------------------------------------------

def _loose_eq(left: Any, right: Any) -> bool:
    """rows 里的值来自 SQLite（str/int/float），选择器里的值来自 JSON。放宽类型比较。"""
    if left is None or right is None:
        return left is right
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left).strip() == str(right).strip()


def _normalize_values(spec: EntitySpec, values: dict, *, require_pk: bool) -> dict:
    if not isinstance(values, dict) or not values:
        raise PatchError("INVALID_PATCH", f"{spec.label} 的 values 不能为空")
    unknown = [key for key in values if key not in spec.columns]
    if unknown:
        raise PatchError(
            "UNKNOWN_FIELD",
            f"{spec.label} 不存在字段 {', '.join(unknown)}",
            [{"field": name} for name in spec.columns],
        )
    if require_pk and not spec.autoincrement_pk and not _text(values.get(spec.pk)):
        raise PatchError("INVALID_PATCH", f"新增{spec.label}必须提供 {spec.pk}")
    normalized = {}
    for key, raw in values.items():
        try:
            normalized[key] = spec.columns[key](raw)
        except (TypeError, ValueError) as error:
            raise PatchError("INVALID_FIELD_VALUE", f"{spec.label}.{key} 取值非法：{raw}") from error
    return normalized


def _select(rows: list[dict], spec: EntitySpec, patch: dict) -> list[int]:
    """返回命中的行下标。三种选择器互斥：id / ids / where。"""
    target_id = patch.get("id")
    ids = patch.get("ids")
    where = patch.get("where")
    provided = [name for name, value in (("id", target_id), ("ids", ids), ("where", where)) if value]
    if len(provided) > 1:
        raise PatchError("INVALID_PATCH", f"id / ids / where 只能提供一个，收到 {', '.join(provided)}")
    if target_id:
        return [i for i, row in enumerate(rows) if _loose_eq(row.get(spec.pk), target_id)]
    if ids:
        if not isinstance(ids, (list, tuple)):
            raise PatchError("INVALID_PATCH", "ids 必须是数组")
        return [i for i, row in enumerate(rows) if any(_loose_eq(row.get(spec.pk), item) for item in ids)]
    if where:
        if not isinstance(where, dict):
            raise PatchError("INVALID_PATCH", "where 必须是对象")
        unknown = [key for key in where if key not in spec.columns]
        if unknown:
            raise PatchError("UNKNOWN_FIELD", f"where 里 {spec.label} 不存在字段 {', '.join(unknown)}")
        return [
            i for i, row in enumerate(rows)
            if all(_loose_eq(row.get(key), value) for key, value in where.items())
        ]
    raise PatchError("INVALID_PATCH", f"{patch.get('op')} 操作必须提供 id、ids 或 where 之一")


def _next_downtime_id(rows: list[dict]) -> int:
    existing = [int(row.get("id") or 0) for row in rows]
    return (max(existing) if existing else 0) + 1


def apply_patches(rows: dict, patches: list[dict]) -> tuple[dict, list[dict]]:
    """把 patch 序列应用到 rows 上，返回 (新 rows, 每条 patch 的影响说明)。

    纯函数：深拷贝入参，绝不就地修改调用方的 rows。
    rows 形状 = InstanceStore.load_all() 的返回值 + "downtimes"（DowntimeStore.list_all()）。
    """
    result = copy.deepcopy(rows)
    result.setdefault("downtimes", [])
    effects: list[dict] = []

    for index, patch in enumerate(patches):
        if not isinstance(patch, dict):
            raise PatchError("INVALID_PATCH", f"第 {index + 1} 条 patch 必须是对象")
        op = _text(patch.get("op")).lower()
        entity = _text(patch.get("entity")).lower()
        if op not in VALID_OPS:
            raise PatchError("INVALID_PATCH", f"第 {index + 1} 条 patch 的 op 必须是 {'/'.join(VALID_OPS)}")
        spec = ENTITY_SPECS.get(entity)
        if spec is None:
            raise PatchError(
                "UNKNOWN_ENTITY",
                f"不支持的实体类型 {entity or '(空)'}",
                [{"entity": name, "label": item.label} for name, item in ENTITY_SPECS.items()],
            )

        if spec.singleton:
            if op != "update":
                raise PatchError("INVALID_PATCH", f"{spec.label}只支持 update")
            values = _normalize_values(spec, patch.get("values") or {}, require_pk=False)
            result[spec.rows_key] = {**(result.get(spec.rows_key) or {}), **values}
            effects.append(_effect(index, op, spec, 1, f"已更新{spec.label}"))
            continue

        rows_list = result.setdefault(spec.rows_key, [])

        if op == "add":
            values = _normalize_values(spec, patch.get("values") or {}, require_pk=True)
            if spec.autoincrement_pk:
                values.setdefault(spec.pk, _next_downtime_id(rows_list))
            else:
                new_id = values[spec.pk]
                if any(_loose_eq(row.get(spec.pk), new_id) for row in rows_list):
                    raise PatchError("DUPLICATE_ID", f"{spec.label} {new_id} 已存在")
            row = {column: func(None) for column, func in spec.columns.items()}
            row.update(values)
            rows_list.append(row)
            effects.append(_effect(index, op, spec, 1, f"已新增{spec.label} {row[spec.pk]}"))
            continue

        matched = _select(rows_list, spec, patch)
        if not matched:
            raise PatchError(
                "TARGET_NOT_FOUND",
                f"没有匹配到任何{spec.label}（第 {index + 1} 条 patch）",
            )

        if op == "remove":
            removed = [rows_list[i].get(spec.pk) for i in matched]
            for i in reversed(matched):
                rows_list.pop(i)
            effects.append(_effect(
                index, op, spec, len(matched),
                f"已删除 {len(matched)} 个{spec.label}：{_preview(removed)}",
            ))
            continue

        values = _normalize_values(spec, patch.get("values") or {}, require_pk=False)
        if spec.pk in values and len(matched) > 1:
            raise PatchError("INVALID_PATCH", f"批量更新不能同时修改主键 {spec.pk}")
        for i in matched:
            rows_list[i].update(values)
        effects.append(_effect(
            index, op, spec, len(matched),
            f"已更新 {len(matched)} 个{spec.label}的 {', '.join(values)}：{_preview([rows_list[i].get(spec.pk) for i in matched])}",
        ))

    return result, effects


def _effect(index: int, op: str, spec: EntitySpec, matched: int, message: str) -> dict:
    return {
        "index": index,
        "op": op,
        "entity": spec.rows_key,
        "entity_label": spec.label,
        "matched": matched,
        "message": message,
    }


def _preview(ids: list, limit: int = 5) -> str:
    shown = [str(item) for item in ids[:limit]]
    return "、".join(shown) + (f" 等 {len(ids)} 个" if len(ids) > limit else "")


def rows_scale(rows: dict) -> dict:
    """各实体行数，用于向用户回显「机器 19 → 20」这类规模变化。"""
    return {
        spec.rows_key: len(rows.get(spec.rows_key) or [])
        for spec in ENTITY_SPECS.values()
        if not spec.singleton
    }


# --- 场景容器 -----------------------------------------------------------------

@dataclass
class Scenario:
    scenario_id: str
    name: str
    base_instance_version: int
    created_at: str
    patches: list[dict] = field(default_factory=list)

    def digest(self) -> str:
        payload = json.dumps(self.patches, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "base_instance_version": self.base_instance_version,
            "created_at": self.created_at,
            "patch_count": len(self.patches),
            "patches": list(self.patches),
            "digest": self.digest(),
        }


class ScenarioStore:
    """进程内存 LRU。场景只存 patch（KB 级），物化出来的 ShopFloor 不在这里。"""

    def __init__(self, max_size: int = 8):
        self._max_size = max_size
        self._items: OrderedDict[str, Scenario] = OrderedDict()
        self._lock = threading.Lock()

    def create(self, name: str, base_instance_version: int) -> Scenario:
        scenario = Scenario(
            scenario_id=f"wf-{uuid.uuid4().hex[:12]}",
            name=_text(name) or "未命名场景",
            base_instance_version=int(base_instance_version),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._items[scenario.scenario_id] = scenario
            while len(self._items) > self._max_size:
                self._items.popitem(last=False)
        return scenario

    def get(self, scenario_id: str) -> Scenario | None:
        with self._lock:
            scenario = self._items.get(scenario_id)
            if scenario is not None:
                self._items.move_to_end(scenario_id)
            return scenario

    def list(self) -> list[Scenario]:
        with self._lock:
            return list(reversed(self._items.values()))

    def delete(self, scenario_id: str) -> bool:
        with self._lock:
            return self._items.pop(scenario_id, None) is not None

    def set_patches(self, scenario_id: str, patches: list[dict]) -> None:
        with self._lock:
            scenario = self._items.get(scenario_id)
            if scenario is not None:
                scenario.patches = list(patches)
                self._items.move_to_end(scenario_id)
