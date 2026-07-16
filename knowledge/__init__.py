"""知识表示模块: 异构图建模"""
from .canonical import (
    GRAPH_BUILDER_VERSION,
    GRAPH_SCHEMA_VERSION,
    CanonicalEdge,
    CanonicalGraph,
    CanonicalGraphBuilder,
    CanonicalNode,
    GraphFingerprint,
    compute_graph_fingerprint,
)
from .context import (
    ComputeGraphProjection,
    DisplayGraphProjection,
    GraphContext,
    GraphContextBuildError,
    GraphContextCorruptError,
    GraphContextDiff,
    GraphContextError,
    GraphContextPersistenceError,
    GraphContextStaleError,
    compare_legacy_context,
    validate_graph_context,
)
from .context_service import (
    GraphContextDiagnostics,
    GraphContextMode,
    GraphContextService,
    resolve_graph_context_mode,
)
from .graph import HeterogeneousGraph

__all__ = [
    "GRAPH_BUILDER_VERSION",
    "GRAPH_SCHEMA_VERSION",
    "CanonicalEdge",
    "CanonicalGraph",
    "CanonicalGraphBuilder",
    "CanonicalNode",
    "ComputeGraphProjection",
    "DisplayGraphProjection",
    "GraphContext",
    "GraphContextBuildError",
    "GraphContextCorruptError",
    "GraphContextDiff",
    "GraphContextDiagnostics",
    "GraphContextError",
    "GraphContextMode",
    "GraphContextPersistenceError",
    "GraphContextService",
    "GraphContextStaleError",
    "GraphFingerprint",
    "HeterogeneousGraph",
    "compare_legacy_context",
    "compute_graph_fingerprint",
    "resolve_graph_context_mode",
    "validate_graph_context",
]
