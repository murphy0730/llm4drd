from __future__ import annotations

import json
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import ProxyHandler, Request, build_opener

from .errors import PlanningAPIError


class PlanningAPIClient:
    def __init__(self, base_url: str, timeout_seconds: float = 10.0) -> None:
        normalized = str(base_url or "").strip().rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("PLANNING_API_BASE_URL 必须是本机 http 地址")
        self._base_url = normalized
        self._timeout = max(0.1, min(float(timeout_seconds), 60.0))
        self._opener = build_opener(ProxyHandler({}))

    def get_overview(self, task_id: str | None = None) -> dict:
        return self._get("/api/query/planning/overview", self._optional(task_id=task_id))

    def list_rules(self) -> dict:
        return self._get("/api/query/planning/rules", {})

    def run_rule_planning(self, rule_name: str) -> dict:
        return self._post(
            "/api/command/planning/run-rule",
            {"rule_name": rule_name},
        )

    def compare_solutions(
        self,
        task_id: str | None = None,
        solution_ids: list[str] | None = None,
        metric_keys: list[str] | None = None,
    ) -> dict:
        return self._get(
            "/api/query/planning/solutions",
            self._optional(
                task_id=task_id,
                solution_ids=self._csv(solution_ids),
                metric_keys=self._csv(metric_keys),
            ),
        )

    def search_orders(self, query: str, limit: int = 20) -> dict:
        return self._get(
            "/api/query/planning/orders/search",
            {"q": query, "limit": limit},
        )

    def get_order_planning(
        self,
        order_id: str,
        task_id: str | None = None,
        solution_ids: list[str] | None = None,
    ) -> dict:
        return self._get(
            f"/api/query/planning/order/{self._path_segment(order_id)}",
            self._optional(task_id=task_id, solution_ids=self._csv(solution_ids)),
        )

    def search_operations(
        self,
        query: str,
        order_id: str | None = None,
        limit: int = 20,
    ) -> dict:
        return self._get(
            "/api/query/planning/operations/search",
            self._optional(q=query, order_id=order_id, limit=limit),
        )

    def get_operation_planning(
        self,
        operation_id: str,
        task_id: str | None = None,
        solution_ids: list[str] | None = None,
    ) -> dict:
        return self._get(
            f"/api/query/planning/operation/{self._path_segment(operation_id)}",
            self._optional(task_id=task_id, solution_ids=self._csv(solution_ids)),
        )

    def _get(self, path: str, params: dict) -> dict:
        query = urlencode(params)
        url = f"{self._base_url}{path}" + (f"?{query}" if query else "")
        return self._open(url)

    def _post(self, path: str, payload: dict) -> dict:
        request = Request(
            f"{self._base_url}{path}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._open(request)

    def _open(self, request: str | Request) -> dict:
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                raw = response.read()
        except HTTPError as error:
            raw = error.read()
            payload = self._decode(raw, allow_failure=True)
            detail = payload.get("detail") if isinstance(payload, dict) else None
            if error.code in {400, 404, 409, 422} and isinstance(detail, dict):
                return detail
            raise PlanningAPIError(
                "PLANNING_API_UNAVAILABLE",
                f"排产 API 返回 HTTP {error.code}",
            ) from error
        except (TimeoutError, socket.timeout) as error:
            raise PlanningAPIError(
                "PLANNING_API_TIMEOUT",
                f"排产 API 请求超时（>{self._timeout:g}s）",
            ) from error
        except URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                raise PlanningAPIError(
                    "PLANNING_API_TIMEOUT",
                    f"排产 API 请求超时（>{self._timeout:g}s）",
                ) from error
            raise PlanningAPIError(
                "PLANNING_API_UNAVAILABLE",
                f"无法连接排产 API: {error.reason}",
            ) from error
        return self._decode(raw)

    @staticmethod
    def _decode(raw: bytes, *, allow_failure: bool = False) -> dict:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if allow_failure:
                return {}
            raise PlanningAPIError(
                "PLANNING_API_INVALID_RESPONSE",
                "排产 API 返回了无法解析的 JSON",
            ) from error
        if not isinstance(payload, dict):
            if allow_failure:
                return {}
            raise PlanningAPIError(
                "PLANNING_API_INVALID_RESPONSE",
                "排产 API 返回值不是 JSON 对象",
            )
        return payload

    @staticmethod
    def _optional(**values) -> dict:
        return {key: value for key, value in values.items() if value is not None}

    @staticmethod
    def _csv(values: list[str] | None) -> str | None:
        return ",".join(values) if values else None

    @staticmethod
    def _path_segment(value: str) -> str:
        from urllib.parse import quote

        return quote(str(value), safe="")
