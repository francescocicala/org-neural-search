"""Organizational neural search laboratory: NK graph engine, organization templates, and web lab."""

from typing import Any

from .org_graph import (
    FEEDBACK_REGIME_DEFAULTS,
    GRAPH_PRESET_METADATA,
    GRAPH_TEMPLATES,
    MAX_B_FIRMS,
    MAX_N_BITS,
    MAX_T_STEPS,
    MIN_N_BITS,
    VALID_DECISION_RIGHTS,
    VALID_FEEDBACK_REGIMES,
    VALID_NK_PATTERNS,
    VALID_NODE_CAPACITIES,
    VALID_NODE_FEEDBACK,
    VALID_NODE_MODEL_TYPES,
    VALID_NODE_ROLES,
    VALID_TURBULENCE,
    VALID_UPWARD_AGGS,
    MarketConfig,
    OrgEdge,
    OrgGraph,
    OrgNode,
    rescale_jurisdiction_bits,
)

# The engine imports JAX, so it is loaded on first use
_ENGINE_EXPORTS = {
    "run_graph",
    "compare_graphs",
    "compare_multi_graphs",
    "stream_graph",
    "stream_graph_comparison",
    "stream_multi_graph_comparison",
    "stream_graph_sweep",
    "predict_graph_representations",
    "graph_representation_ids",
    "summarize_history",
    "summarize_trajectories",
    "build_landscape_batch",
    "GRAPH_SWEEP_PARAMS",
    "MAX_CONFIGURATIONS",
}


def __getattr__(name: str) -> Any:
  if name in _ENGINE_EXPORTS:
    from . import engine

    return getattr(engine, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(
    _ENGINE_EXPORTS
    | {
        "FEEDBACK_REGIME_DEFAULTS",
        "GRAPH_PRESET_METADATA",
        "GRAPH_TEMPLATES",
        "MAX_B_FIRMS",
        "MAX_N_BITS",
        "MAX_T_STEPS",
        "MIN_N_BITS",
        "VALID_DECISION_RIGHTS",
        "VALID_FEEDBACK_REGIMES",
        "VALID_NK_PATTERNS",
        "VALID_NODE_CAPACITIES",
        "VALID_NODE_FEEDBACK",
        "VALID_NODE_MODEL_TYPES",
        "VALID_NODE_ROLES",
        "VALID_TURBULENCE",
        "VALID_UPWARD_AGGS",
        "MarketConfig",
        "OrgEdge",
        "OrgGraph",
        "OrgNode",
        "rescale_jurisdiction_bits",
    }
)
