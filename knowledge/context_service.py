from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from ..core.models import ShopFloor
from .canonical import CanonicalGraphBuilder, GraphFingerprint, compute_graph_fingerprint
from .context import (
    ComputeGraphProjection,
    DisplayGraphProjection,
    GraphContext,
    GraphContextBuildError,
    GraphContextCorruptError,
    GraphContextStaleError,
    validate_graph_context,
)

if TYPE_CHECKING:
    from ..data.graph_artifact_store import GraphArtifactStore

logger = logging.getLogger(__name__)


class GraphContextMode(str, Enum):
    LEGACY = "legacy"
    SHADOW = "shadow"
    ACTIVE = "active"


@dataclass(frozen=True)
class GraphContextDiagnostics:
    cache_level: str
    cache_hit: bool
    fingerprint: GraphFingerprint
    load_time_ms: float
    build_time_ms: float
    validation_time_ms: float
    operation_count: int
    relation_count: int
    invalid_reason: str = ""


def _relation_count(context: GraphContext) -> int:
    return (
        len(context.predecessor_indices)
        + len(context.successor_indices)
        + len(context.eligible_machine_indices)
    )


class GraphContextService:
    def __init__(
        self,
        store: GraphArtifactStore,
        builder=None,
        compute_projection=None,
    ):
        self.store = store
        self.builder = builder or CanonicalGraphBuilder()
        self.compute_projection = compute_projection or ComputeGraphProjection()
        self._lock = threading.Condition()
        self._memory_context: GraphContext | None = None
        self._building: set[GraphFingerprint] = set()
        self._failures: dict[GraphFingerprint, BaseException] = {}

    @staticmethod
    def _diagnostics(
        context: GraphContext,
        cache_level: str,
        *,
        load_time_ms: float = 0.0,
        build_time_ms: float = 0.0,
        validation_time_ms: float = 0.0,
        invalid_reason: str = "",
    ) -> GraphContextDiagnostics:
        return GraphContextDiagnostics(
            cache_level=cache_level,
            cache_hit=cache_level in {"l1", "sqlite"},
            fingerprint=context.fingerprint,
            load_time_ms=load_time_ms,
            build_time_ms=build_time_ms,
            validation_time_ms=validation_time_ms,
            operation_count=len(context.operation_ids),
            relation_count=_relation_count(context),
            invalid_reason=invalid_reason,
        )

    def _build_once(
        self,
        shop: ShopFloor,
        fingerprint: GraphFingerprint,
        progress_callback,
        deadline: float | None,
        current_fingerprint_provider,
    ) -> tuple[GraphContext, float, float]:
        started = time.perf_counter()
        canonical = self.builder.build(
            shop,
            progress_callback=progress_callback,
            deadline=deadline,
        )
        if canonical.fingerprint != fingerprint:
            raise GraphContextStaleError("instance changed during graph build")
        display = DisplayGraphProjection.from_canonical(canonical)
        context = self.compute_projection.build(shop, canonical)
        build_time_ms = (time.perf_counter() - started) * 1000.0

        validation_started = time.perf_counter()
        validate_graph_context(shop, context)
        validation_time_ms = (time.perf_counter() - validation_started) * 1000.0

        precommit_check = None
        if current_fingerprint_provider is not None:
            def precommit_check() -> None:
                if current_fingerprint_provider() != fingerprint:
                    raise GraphContextStaleError(
                        "instance changed during graph build"
                    )

        self.store.save_artifacts(
            display,
            context,
            precommit_check=precommit_check,
        )
        return context, build_time_ms, validation_time_ms

    def _load_or_build(
        self,
        shop: ShopFloor,
        fingerprint: GraphFingerprint,
        *,
        force_rebuild: bool,
        progress_callback,
        deadline: float | None,
        current_fingerprint_provider,
    ) -> tuple[GraphContext, GraphContextDiagnostics]:
        invalid_reason = ""
        original_corruption: GraphContextCorruptError | None = None
        if not force_rebuild:
            load_started = time.perf_counter()
            try:
                context = self.store.load_context(fingerprint)
                load_time_ms = (time.perf_counter() - load_started) * 1000.0
                if context is not None:
                    validation_started = time.perf_counter()
                    validate_graph_context(shop, context)
                    validation_time_ms = (
                        time.perf_counter() - validation_started
                    ) * 1000.0
                    return context, self._diagnostics(
                        context,
                        "sqlite",
                        load_time_ms=load_time_ms,
                        validation_time_ms=validation_time_ms,
                    )
            except GraphContextCorruptError as exc:
                original_corruption = exc
                invalid_reason = str(exc)
                self.store.clear_all()
            else:
                meta = self.store.load_context_meta()
                if meta:
                    invalid_reason = meta.get("invalid_reason", "")

        attempts = 0
        while attempts < 2:
            attempts += 1
            try:
                context, build_time_ms, validation_time_ms = self._build_once(
                    shop,
                    fingerprint,
                    progress_callback,
                    deadline,
                    current_fingerprint_provider,
                )
                return context, self._diagnostics(
                    context,
                    "built",
                    build_time_ms=build_time_ms,
                    validation_time_ms=validation_time_ms,
                    invalid_reason=invalid_reason,
                )
            except GraphContextCorruptError as exc:
                if original_corruption is None:
                    original_corruption = exc
                    invalid_reason = str(exc)
                    self.store.clear_all()
                    continue
                error = GraphContextBuildError(
                    f"graph context rebuild failed: {exc}"
                )
                raise error from original_corruption
            except GraphContextStaleError:
                raise
            except BaseException as exc:
                if original_corruption is not None:
                    error = GraphContextBuildError(
                        f"graph context rebuild failed: {exc}"
                    )
                    raise error from original_corruption
                raise

        raise GraphContextBuildError("graph context rebuild failed")

    def get_or_build(
        self,
        shop: ShopFloor,
        *,
        force_rebuild: bool = False,
        progress_callback=None,
        deadline: float | None = None,
        current_fingerprint_provider=None,
    ) -> tuple[GraphContext, GraphContextDiagnostics]:
        fingerprint = compute_graph_fingerprint(shop)
        with self._lock:
            if (
                not force_rebuild
                and self._memory_context is not None
                and self._memory_context.fingerprint == fingerprint
            ):
                return self._memory_context, self._diagnostics(
                    self._memory_context, "l1"
                )

            if fingerprint in self._building:
                while fingerprint in self._building:
                    self._lock.wait()
                failure = self._failures.get(fingerprint)
                if failure is not None:
                    raise failure
                if (
                    self._memory_context is not None
                    and self._memory_context.fingerprint == fingerprint
                ):
                    return self._memory_context, self._diagnostics(
                        self._memory_context, "l1"
                    )

            self._failures.pop(fingerprint, None)
            self._building.add(fingerprint)

        try:
            context, diagnostics = self._load_or_build(
                shop,
                fingerprint,
                force_rebuild=force_rebuild,
                progress_callback=progress_callback,
                deadline=deadline,
                current_fingerprint_provider=current_fingerprint_provider,
            )
        except BaseException as exc:
            with self._lock:
                self._failures[fingerprint] = exc
                self._building.discard(fingerprint)
                self._lock.notify_all()
            raise

        with self._lock:
            self._memory_context = context
            self._failures.pop(fingerprint, None)
            self._building.discard(fingerprint)
            self._lock.notify_all()
        return context, diagnostics

    def invalidate(self, reason: str) -> None:
        with self._lock:
            self._memory_context = None
        self.store.mark_invalid(reason)

    def clear_memory_cache(self) -> None:
        with self._lock:
            self._memory_context = None


def resolve_graph_context_mode() -> GraphContextMode:
    value = os.environ.get("LLM4DRD_GRAPH_CONTEXT_MODE", "legacy").strip().lower()
    try:
        return GraphContextMode(value)
    except ValueError:
        logger.warning(
            "Invalid LLM4DRD_GRAPH_CONTEXT_MODE=%r; falling back to legacy",
            value,
        )
        return GraphContextMode.LEGACY
