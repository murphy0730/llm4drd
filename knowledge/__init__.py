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
    "GraphContextError",
    "GraphContextPersistenceError",
    "GraphContextStaleError",
    "GraphFingerprint",
    "HeterogeneousGraph",
    "compare_legacy_context",
    "compute_graph_fingerprint",
    "validate_graph_context",
]
