"""Local web server and interactive laboratory for org_neural_search."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import socket
import threading
from typing import Any, Optional
import urllib.parse
import webbrowser

try:
  from . import org_graph as _org_graph_mod
  from .org_graph import (
      GRAPH_PRESET_METADATA,
      GRAPH_TEMPLATES,
      MAX_B_FIRMS,
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
      OrgGraph,
  )
except ImportError:
  import org_graph as _org_graph_mod  # type: ignore
  from org_graph import (  # type: ignore
      GRAPH_PRESET_METADATA,
      GRAPH_TEMPLATES,
      MAX_B_FIRMS,
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
      OrgGraph,
  )

MAX_T_STEPS = getattr(_org_graph_mod, "MAX_T_STEPS", 20000)

# Version of the experiment file format written by the web laboratory and read by `cli.py run`.
SCHEMA_VERSION = "0.1.0"


def _detect_default_backend() -> str:
  """Name of JAX's default backend; runs pick the hardware automatically."""
  if importlib.util.find_spec("jax") is None:
    return "CPU"
  try:
    import jax

    return jax.default_backend().upper()
  except Exception:
    return "CPU"


def build_catalog_payload() -> dict[str, Any]:
  """Build what the web UI needs: templates, field choices, defaults, limits, and devices."""
  return {
      "schema_version": SCHEMA_VERSION,
      "graph_templates": {k: v.to_dict() for k, v in GRAPH_TEMPLATES.items()},
      "graph_preset_metadata": dict(GRAPH_PRESET_METADATA),
      "market_defaults": MarketConfig().to_dict(),
      "field_choices": {
          "node_roles": list(VALID_NODE_ROLES),
          "node_model_types": list(VALID_NODE_MODEL_TYPES),
          "node_capacities": list(VALID_NODE_CAPACITIES),
          "node_feedback": list(VALID_NODE_FEEDBACK),
          "upward_aggs": list(VALID_UPWARD_AGGS),
          "feedback_regimes": list(VALID_FEEDBACK_REGIMES),
          "decision_rights": list(VALID_DECISION_RIGHTS),
          "nk_patterns": list(VALID_NK_PATTERNS),
          "turbulence": list(VALID_TURBULENCE),
      },
      "limits": {"max_firms": MAX_B_FIRMS, "max_t_steps": getattr(_org_graph_mod, "MAX_T_STEPS", 20000)},
      "devices": {"default_backend": _detect_default_backend()},
  }


def resolve_graph_request(data: dict[str, Any]) -> tuple[OrgGraph, MarketConfig]:
  """Parse an {org_graph, market_config} payload into a validated OrgGraph and MarketConfig."""
  graph_raw = data.get("org_graph")
  if not isinstance(graph_raw, dict):
    raise ValueError("The request needs an 'org_graph' object.")
  mkt_raw = data.get("market_config")
  if mkt_raw is None:
    mkt_raw = {}
  if not isinstance(mkt_raw, dict):
    raise ValueError("'market_config' must be an object.")
  market = MarketConfig.from_dict(mkt_raw)
  graph = OrgGraph.from_dict(graph_raw).validate(n_bits=market.n_bits)
  return graph, market


def compile_config_payload(data: dict[str, Any]) -> dict[str, Any]:
  """Validate an {org_graph, market_config} request and return the engine's explanation of it."""
  unknown = sorted(set(data) - {"org_graph", "market_config"})
  if unknown:
    raise ValueError(
        f"Unknown request field(s): {', '.join(unknown)}. /api/compile accepts org_graph and market_config."
    )
  graph, market = resolve_graph_request(data)
  return {
      "org_graph": graph.to_dict(),
      "market_config": market.to_dict(),
      "graph_explanation": graph.explain_graph(market),
  }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Organizational Search &amp; Mental Representations: Organizational Neural Search Laboratory</title>
<style>
  :root {
    --bg-canvas: #fafafa;
    --bg-card: #ffffff;
    --bg-subtle: #f4f4f5;
    --bg-accent-subtle: #eff6ff;
    --bg-warn-subtle: #fffbeb;
    --border-color: #e4e4e7;
    --border-strong: #d4d4d8;
    --text-primary: #18181b;
    --text-secondary: #3f3f46;
    --text-muted: #71717a;
    --navy: #1d4ed8;
    --navy-hover: #1e40af;
    --navy-band: rgba(29, 78, 216, 0.12);
    --terracotta: #b45309;
    --terracotta-band: rgba(180, 83, 9, 0.12);
    --forest: #15803d;
    --crimson: #b91c1c;
    --font-sans: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --font-serif: Georgia, Cambria, "Times New Roman", serif;
    --font-mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  }

  * {
    box-sizing: border-box;
  }

  body {
    margin: 0;
    padding: 0;
    background-color: var(--bg-canvas);
    color: var(--text-primary);
    font-family: var(--font-sans);
    font-size: 13px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }

  .app-shell {
    max-width: 1480px;
    margin: 0 auto;
    padding: 16px 24px 36px;
  }

  .top-header {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 16px;
  }

  .brand-title {
    font-family: var(--font-sans);
    font-size: 18px;
    font-weight: 600;
    letter-spacing: -0.01em;
    margin: 0;
    color: var(--text-primary);
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    font-family: var(--font-sans);
    font-size: 12.5px;
    font-weight: 500;
    padding: 6px 12px;
    border-radius: 4px;
    border: 1px solid var(--border-strong);
    background: var(--bg-card);
    color: var(--text-primary);
    cursor: pointer;
  }

  .btn:hover:not(:disabled) {
    background: var(--bg-subtle);
  }

  .btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .btn-primary {
    background: var(--navy);
    border-color: var(--navy);
    color: #ffffff;
    font-weight: 600;
  }

  .btn-primary:hover:not(:disabled) {
    background: var(--navy-hover);
    border-color: var(--navy-hover);
  }

  .btn-danger {
    background: #ffffff;
    border-color: var(--border-strong);
    color: var(--crimson);
  }

  .btn-danger:hover:not(:disabled) {
    background: #fef2f2;
    border-color: var(--crimson);
  }

  .view-switcher {
    display: inline-flex;
    border-bottom: none;
    gap: 2px;
    margin-right: 6px;
  }

  .view-tab {
    border: 1px solid transparent;
    background: transparent;
    color: var(--text-secondary);
    font-family: var(--font-sans);
    font-size: 13px;
    font-weight: 500;
    padding: 5px 12px;
    border-radius: 4px;
    cursor: pointer;
  }

  .view-tab:hover {
    color: var(--text-primary);
    background: var(--bg-subtle);
  }

  .view-tab.active {
    background: var(--bg-subtle);
    border-color: var(--border-strong);
    color: var(--text-primary);
    font-weight: 600;
  }

  /* Stacked dashboard: market, configurations, design, and results each take the full width. */
  .design-top {
    display: grid;
    grid-template-columns: minmax(260px, 2fr) minmax(0, 3fr);
    gap: 10px 16px;
    align-items: start;
  }

  .design-top .design-readout-box {
    margin-top: 0;
  }

  /* Editor about 60 percent on the left; organization settings and the inspector on the right. */
  .design-grid {
    display: grid;
    grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
    grid-template-rows: auto 1fr;
    column-gap: 18px;
    align-items: start;
    margin-top: 12px;
  }

  .design-grid #editor-home {
    grid-column: 1;
    grid-row: 1 / span 2;
    min-width: 0;
  }

  .design-grid .editor {
    margin-top: 0;
  }

  .design-grid .org-settings-box {
    grid-column: 2;
    grid-row: 1;
    margin-top: 0;
  }

  .design-grid #node-inspector-panel {
    grid-column: 2;
    grid-row: 2;
  }

  .results-head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px 20px;
  }

  .results-head .progress-container {
    flex: 0 1 320px;
  }

  .results-head .legend-row {
    flex-wrap: wrap;
    margin-left: auto;
  }

  @media (max-width: 1100px) {
    .design-top,
    .design-grid {
      grid-template-columns: minmax(0, 1fr);
    }

    .design-grid #editor-home,
    .design-grid .org-settings-box,
    .design-grid #node-inspector-panel {
      grid-column: 1;
    }

    .design-grid #editor-home {
      grid-row: 1;
    }

    .design-grid .org-settings-box {
      grid-row: 2;
      margin-top: 10px;
    }

    .design-grid #node-inspector-panel {
      grid-row: 3;
    }
  }

  .market-top-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    justify-content: space-between;
    gap: 12px 16px;
    padding: 12px 16px;
    margin-bottom: 14px;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 4px;
  }

  .market-controls-row {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    gap: 10px 14px;
  }

  .market-controls-row .form-group {
    min-width: 104px;
  }

  .chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 2px;
  }

  .chip {
    font-family: var(--font-sans);
    font-size: 11px;
    line-height: 1.5;
    padding: 0 7px;
    border: 1px solid var(--border-strong);
    border-radius: 9px;
    background: #ffffff;
    color: var(--text-secondary);
    cursor: pointer;
    white-space: nowrap;
  }

  .chip:hover {
    background: var(--bg-subtle);
  }

  .chip.active {
    border-color: var(--navy);
    background: var(--bg-accent-subtle);
    color: var(--navy);
  }

  .config-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px 8px;
    padding: 8px 12px;
    margin-bottom: 14px;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 4px;
  }

  .config-bar-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
    margin-right: 2px;
  }

  .config-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    max-width: 270px;
    padding: 3px 4px 3px 9px;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: var(--bg-subtle);
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
  }

  .config-chip.active {
    background: #ffffff;
    color: var(--text-primary);
    font-weight: 600;
    box-shadow: inset 0 -2px 0 var(--chip-color, var(--navy));
  }

  .config-swatch {
    flex: none;
    width: 9px;
    height: 9px;
    border-radius: 50%;
  }

  .config-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .config-chip input.config-name-input {
    width: 160px;
    padding: 1px 4px;
    font-size: 12px;
  }

  .chip-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 2px 4px;
    border: none;
    border-radius: 3px;
    background: transparent;
    color: var(--text-muted);
    font-size: 13px;
    line-height: 1;
    cursor: pointer;
  }

  .chip-icon:hover {
    background: var(--border-color);
    color: var(--text-primary);
  }

  .config-bar-aside {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-left: auto;
  }

  .config-bar-aside select {
    width: auto;
    min-width: 240px;
    font-size: 12px;
  }

  .editor {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin: 10px 0 4px;
  }

  .editor-toolbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 5px;
  }

  .editor-toolbar .btn {
    padding: 4px 9px;
    font-size: 12px;
  }

  .editor-toolbar .spacer {
    flex: 1;
  }

  .editor-hint {
    margin: 0;
    font-size: 11.5px;
    line-height: 1.45;
    color: var(--text-muted);
  }

  .editor-canvas-wrap {
    position: relative;
    height: 490px;
    background: #fbfbfa;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    overflow: hidden;
  }

  .editor-canvas-wrap svg {
    display: block;
    width: 100%;
    height: 100%;
    touch-action: none;
    user-select: none;
    outline: none;
  }

  .editor-canvas-wrap:focus-within {
    border-color: #93a4c3;
  }

  .editor-msg {
    position: absolute;
    left: 8px;
    bottom: 8px;
    max-width: calc(100% - 16px);
    padding: 4px 8px;
    border: 1px solid #fca5a5;
    border-radius: 3px;
    background: #fef2f2;
    color: var(--crimson);
    font-size: 11.5px;
  }

  .canvas-tip {
    position: absolute;
    z-index: 3;
    padding: 3px 7px;
    border-radius: 3px;
    background: #1f2937;
    color: #ffffff;
    font-size: 11.5px;
    white-space: nowrap;
    pointer-events: none;
  }

  .edge-editor-title {
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .edge-editor label {
    display: block;
    margin-top: 6px;
  }

  .edge-editor .range-row {
    height: 24px;
  }

  .edge-editor-note {
    margin: 2px 0 10px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .edge-editor-actions {
    display: flex;
    gap: 6px;
  }

  .edge-editor-actions .btn {
    padding: 3px 12px;
    font-size: 11.5px;
  }

  .editor-legend {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px 14px;
    font-size: 11.5px;
    color: var(--text-secondary);
  }

  .legend-entry {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }

  .overlay {
    position: fixed;
    inset: 0;
    z-index: 50;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(15, 23, 42, 0.45);
  }

  .overlay[hidden] {
    display: none;
  }

  .overlay-panel {
    display: flex;
    flex-direction: column;
    width: min(1320px, 94vw);
    height: 90vh;
    padding: 12px 16px;
    border-radius: 6px;
    background: var(--bg-card);
  }

  .overlay-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }

  .overlay-slot {
    display: flex;
    flex: 1;
    gap: 14px;
    min-height: 0;
  }

  .overlay-slot .editor {
    flex: 1;
    min-width: 0;
    min-height: 0;
    margin: 0;
  }

  .overlay-slot .editor-canvas-wrap {
    flex: 1;
    height: auto;
  }

  .overlay-slot .node-inspector-box {
    flex: none;
    width: 400px;
    margin-top: 0;
    overflow-y: auto;
  }

  #org-canvas .node {
    cursor: grab;
  }

  #org-canvas.dragging .node {
    cursor: grabbing;
  }

  #org-canvas .node .handle {
    opacity: 0;
    cursor: crosshair;
  }

  #org-canvas .node:hover .handle,
  #org-canvas .node.selected .handle {
    opacity: 1;
  }

  #org-canvas .edge-hit {
    cursor: pointer;
  }

  .node-inspector-box {
    background: #fafafa;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 10px 12px;
    margin-top: 8px;
  }

  .node-inspector-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 6px;
    margin-bottom: 8px;
    border-bottom: 1px solid var(--border-color);
  }

  .role-pill {
    font-family: var(--font-mono);
    font-size: 10.5px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 3px;
    background: var(--bg-subtle);
    color: var(--text-secondary);
    border: 1px solid var(--border-strong);
    text-transform: uppercase;
  }

  .bit-pills-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(22px, 1fr));
    gap: 3px;
    margin-top: 4px;
  }

  .bit-pill {
    padding: 4px 0;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    text-align: center;
    border: 1px solid var(--border-strong);
    border-radius: 3px;
    background: #ffffff;
    color: var(--text-muted);
    cursor: pointer;
    user-select: none;
  }

  .bit-pill.active {
    background: var(--navy);
    border-color: var(--navy);
    color: #ffffff;
  }

  .links-list {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }

  .links-subhead {
    margin-top: 2px;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
  }

  .link-row {
    display: grid;
    grid-template-columns: 1fr auto auto auto;
    align-items: center;
    gap: 6px;
    padding: 2px 3px 2px 8px;
    border: 1px solid var(--border-color);
    border-radius: 3px;
    background: #ffffff;
    font-size: 11.5px;
  }

  .link-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .link-row label {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .link-row input[type="number"] {
    width: 58px;
    padding: 2px 4px;
    font-size: 11.5px;
  }

  .links-empty {
    font-size: 11.5px;
    color: var(--text-muted);
  }

  .card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 14px 16px;
    margin-bottom: 14px;
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border-color);
  }

  .card-title {
    font-family: var(--font-sans);
    font-size: 13.5px;
    font-weight: 600;
    margin: 0;
    color: var(--text-primary);
  }

  .section-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
    margin: 0 0 8px 0;
  }

  .badge {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
    background: transparent;
    border: none;
    padding: 0;
  }

  .card-header .badge {
    display: none;
  }

  .form-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px 10px;
  }

  .form-group {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }

  .form-group.full-width {
    grid-column: 1 / -1;
  }

  label {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
  }

  select, input[type="number"], input[type="text"] {
    width: 100%;
    padding: 5px 7px;
    font-family: var(--font-sans);
    font-size: 12.5px;
    color: var(--text-primary);
    background: #ffffff;
    border: 1px solid var(--border-strong);
    border-radius: 3px;
    font-variant-numeric: tabular-nums;
  }

  select:focus, input:focus {
    outline: none;
    border-color: var(--navy);
  }

  .notice-box {
    background: var(--bg-warn-subtle);
    border: 1px solid #fcd34d;
    color: #78350f;
    padding: 6px 8px;
    border-radius: 3px;
    font-size: 11.5px;
    margin-top: 8px;
  }

  .error-box {
    background: #fef2f2;
    border: 1px solid #fca5a5;
    color: var(--crimson);
    padding: 6px 8px;
    border-radius: 3px;
    font-size: 11.5px;
    margin-top: 8px;
  }

  .table-wrap {
    width: 100%;
    overflow-x: auto;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }

  th, td {
    text-align: left;
    padding: 6px 8px;
    border-bottom: 1px solid var(--border-color);
  }

  th {
    font-weight: 600;
    color: var(--text-secondary);
    background: var(--bg-subtle);
  }

  td.mono, th.mono {
    font-family: var(--font-mono);
    font-size: 11px;
  }

  tr.group-row td {
    background: var(--bg-subtle);
    color: var(--text-primary);
  }

  .exec-controls {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .progress-container {
    flex: 1;
    min-width: 200px;
  }

  .progress-meta {
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--text-muted);
    margin-bottom: 4px;
  }

  .progress-track {
    width: 100%;
    height: 6px;
    background: var(--bg-subtle);
    border-radius: 3px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    width: 0%;
    background: var(--navy);
    transition: width 0.1s linear;
  }

  /* Moving stripe shown while the model is prepared, before the first steps arrive. */
  .progress-fill.indeterminate {
    width: 30% !important;
    transition: none;
    animation: progress-slide 1.2s ease-in-out infinite;
  }

  @keyframes progress-slide {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(340%); }
  }

  .charts-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin-bottom: 14px;
  }

  @media (max-width: 1100px) {
    .charts-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (max-width: 700px) {
    .charts-grid {
      grid-template-columns: minmax(0, 1fr);
    }
  }

  .chart-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 12px 14px;
  }

  .chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }

  .chart-title {
    font-family: var(--font-sans);
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
  }

  .chart-header .badge {
    margin-left: 8px;
    text-align: right;
  }

  .chart-header .badge span {
    white-space: nowrap;
  }

  .chart-canvas-wrap {
    position: relative;
    width: 100%;
    height: 250px;
  }

  /* Reference lines of the payoff chart: their own small legend under the chart title. */
  .chart-ref-legend {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px 14px;
    margin: -2px 0 6px;
    font-size: 11.5px;
    color: var(--text-secondary);
  }

  .chart-ref-legend .legend-entry {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  canvas.traj-canvas {
    width: 100%;
    height: 100%;
    display: block;
  }

  .legend-row {
    display: flex;
    align-items: center;
    gap: 14px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-weight: 500;
  }

  .swatch {
    width: 12px;
    height: 2.5px;
    display: inline-block;
  }

  .sig-pos {
    color: var(--forest);
    font-weight: 600;
  }

  .sig-neg {
    color: var(--crimson);
    font-weight: 600;
  }

  .guide-container {
    max-width: 880px;
  }

  .guide-container p {
    font-size: 13.5px;
    line-height: 1.6;
    color: var(--text-secondary);
    margin: 0 0 10px 0;
  }

  .guide-container p:last-child {
    margin-bottom: 0;
  }

  .design-readout-box {
    background: #fafafa;
    border: 1px solid var(--border-color);
    border-radius: 3px;
    font-size: 12px;
    line-height: 1.45;
    padding: 8px 10px;
    margin-top: 10px;
    color: var(--text-secondary);
  }

  .design-readout-focus {
    margin-bottom: 6px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border-color);
    color: var(--text-primary);
  }

  .design-readout-summary {
    color: var(--text-secondary);
  }

  .design-readout-link, .help-jump-link {
    display: inline-block;
    margin-top: 5px;
    font-size: 11.5px;
    color: var(--navy);
    text-decoration: none;
    cursor: pointer;
    background: none;
    border: none;
    padding: 0;
    font-family: var(--font-sans);
  }

  .design-readout-link:hover, .help-jump-link:hover {
    text-decoration: underline;
  }

  .guide-subhead {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 16px 0 6px 0;
  }

  .guide-subhead:first-of-type {
    margin-top: 4px;
  }

  .market-bar-title {
    flex-basis: 100%;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
    margin-bottom: -4px;
  }

  .market-bar-subtitle {
    font-weight: 400;
    text-transform: none;
    letter-spacing: 0;
  }

  .exec-area {
    display: flex;
    flex-basis: 100%;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px 16px;
  }

  .exec-area .btn,
  .exec-area .badge {
    white-space: nowrap;
  }

  .market-summary {
    flex-basis: 100%;
    font-size: 12px;
    color: var(--text-secondary);
    border-top: 1px solid var(--border-color);
    padding-top: 7px;
    margin-top: -2px;
  }

  .help {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 15px;
    height: 15px;
    margin-left: 4px;
    padding: 0;
    border: 1px solid var(--border-strong);
    border-radius: 50%;
    background: #ffffff;
    color: var(--text-muted);
    font-family: var(--font-sans);
    font-size: 10px;
    font-weight: 600;
    line-height: 1;
    vertical-align: 1px;
    cursor: pointer;
  }

  .help:hover,
  .help[aria-expanded="true"] {
    border-color: var(--navy);
    color: var(--navy);
  }

  .help:focus-visible {
    outline: 2px solid var(--navy);
    outline-offset: 1px;
  }

  .help-pop {
    position: fixed;
    z-index: 60;
    max-width: 390px;
    padding: 8px 11px;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    background: #ffffff;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.14);
    color: var(--text-secondary);
    font-size: 12px;
    font-weight: 400;
    line-height: 1.48;
  }

  .help-pop[hidden] {
    display: none;
  }

  .help-pop-title {
    margin-bottom: 3px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .field-hint {
    font-size: 11.5px;
    line-height: 1.4;
    color: var(--text-muted);
  }

  .range-row {
    display: flex;
    align-items: center;
    gap: 8px;
    height: 30px;
  }

  input[type="range"] {
    flex: 1;
    min-width: 70px;
    margin: 0;
    accent-color: var(--navy);
  }

  .range-value {
    min-width: 34px;
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--text-secondary);
    text-align: right;
  }

  .org-settings-box {
    background: #fafafa;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 10px 12px;
    margin: 4px 0 8px;
  }

  .org-settings-title {
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-primary);
    padding-bottom: 6px;
    margin-bottom: 8px;
    border-bottom: 1px solid var(--border-color);
  }

  .fb-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 15px;
    height: 13px;
    border: 1.2px solid currentColor;
    border-radius: 3px;
    background: #ffffff;
    font-family: var(--font-mono);
    font-size: 9px;
    font-weight: 700;
    line-height: 1;
  }

  .fb-badge.fb-firm {
    color: var(--navy);
  }

  .fb-badge.fb-domain {
    color: var(--forest);
  }

  .fb-badge.fb-none {
    color: var(--text-muted);
  }

  .acc-tag {
    padding: 0 4px;
    border: 1px solid #c7d2e3;
    border-radius: 3px;
    background: #ffffff;
    color: var(--text-secondary);
    font-family: var(--font-mono);
    font-size: 10px;
  }

  details.inspector-advanced,
  details.advanced-details {
    grid-column: 1 / -1;
    border-top: 1px solid var(--border-color);
    padding-top: 6px;
    margin-top: 4px;
  }

  details.inspector-advanced summary,
  details.advanced-details summary {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
    cursor: pointer;
    user-select: none;
  }

  details.advanced-details summary:hover {
    color: var(--navy);
  }

  details.inspector-advanced .form-group {
    margin-top: 6px;
  }

  .org-settings-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px 10px;
  }

  .guide-container ol.guide-steps,
  .guide-container ul.guide-notes {
    font-size: 13.5px;
    line-height: 1.6;
    color: var(--text-secondary);
    margin: 0 0 10px 0;
    padding-left: 22px;
  }

  .guide-container ol.guide-steps li,
  .guide-container ul.guide-notes li {
    margin-bottom: 6px;
  }

  .guide-anchor {
    color: var(--text-muted);
    font-size: 12px;
  }

  .guide-refs {
    font-size: 12.5px;
    line-height: 1.55;
    color: var(--text-secondary);
    margin: 0;
    padding-left: 18px;
  }
  .acc-row {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .acc-row select {
    width: auto;
    min-width: 200px;
  }

  .acc-row .range-row {
    flex: 1;
  }

  [hidden] {
    display: none !important;
  }
</style>
</head>
<body>
<div class="app-shell">
  <header class="top-header">
    <h1 class="brand-title">Organizational Search &amp; Mental Representations</h1>
    <div class="header-actions">
      <div class="view-switcher" role="navigation" aria-label="Views">
        <button type="button" class="view-tab active" id="view-tab-lab">Simulator</button>
        <button type="button" class="view-tab" id="view-tab-guide">Model &amp; Parameters Guide</button>
      </div>
      <input type="file" id="file-load-input" accept=".json" hidden>
      <button type="button" class="btn" id="btn-load-json">Import JSON</button>
      <button type="button" class="btn" id="btn-download-json">Export JSON</button>
      <button type="button" class="help" data-help="files" aria-label="Help: experiment files">?</button>
    </div>
  </header>

  <!-- ==================== SIMULATOR VIEW ==================== -->
  <div id="view-lab">

    <!-- Market environment shared by every configuration, and run controls -->
    <div class="market-top-bar">
      <div class="market-bar-title">Market environment <span class="market-bar-subtitle">(shared by all configurations, so comparisons are matched)</span></div>
      <div class="market-controls-row">
        <div class="form-group" style="min-width:96px">
          <label for="ctrl-n-bits">Choices N<button type="button" class="help" data-help="n_bits" aria-label="Help: Choices N">?</button></label>
          <input type="number" id="ctrl-n-bits" min="2" max="16" step="1" value="12">
          <div class="chip-row" id="nbits-chips">
            <button type="button" class="chip" data-value="8">8</button>
            <button type="button" class="chip" data-value="12">12</button>
            <button type="button" class="chip" data-value="16">16</button>
          </div>
        </div>
        <div class="form-group" style="min-width:140px">
          <label for="ctrl-nk-k">Interdependence K<button type="button" class="help" data-help="nk_k" aria-label="Help: Interdependence K">?</button></label>
          <div class="range-row">
            <input type="range" id="ctrl-nk-k" min="0" max="11" step="1" value="3">
            <span class="range-value" id="ctrl-nk-k-value">3</span>
          </div>
        </div>
        <div class="form-group">
          <label for="ctrl-nk-pattern">Interactions<button type="button" class="help" data-help="nk_pattern" aria-label="Help: Interactions">?</button></label>
          <select id="ctrl-nk-pattern">
            <option value="random" selected>Random</option>
            <option value="modular">Modular</option>
          </select>
        </div>
        <div class="form-group" style="min-width:150px">
          <label for="ctrl-noise">Feedback noise<button type="button" class="help" data-help="sigma_obs" aria-label="Help: Feedback noise">?</button></label>
          <input type="number" id="ctrl-noise" min="0" max="3" step="0.05" value="0.65">
          <div class="chip-row" id="noise-chips">
            <button type="button" class="chip" data-value="0">0</button>
            <button type="button" class="chip" data-value="0.3">0.30</button>
            <button type="button" class="chip" data-value="0.65">0.65</button>
            <button type="button" class="chip" data-value="1">1.00</button>
          </div>
        </div>
        <div class="form-group">
          <label for="ctrl-turbulence">Turbulence<button type="button" class="help" data-help="turbulence" aria-label="Help: Turbulence">?</button></label>
          <select id="ctrl-turbulence">
            <option value="none" selected>None</option>
            <option value="random">Random</option>
            <option value="periodic">Periodic</option>
          </select>
        </div>
        <div class="form-group" id="wrap-pshock" hidden>
          <label for="ctrl-pshock">Probability per period<button type="button" class="help" data-help="p_shock" aria-label="Help: Probability per period">?</button></label>
          <input type="number" id="ctrl-pshock" min="0" max="0.1" step="0.005" value="0.02">
        </div>
        <div class="form-group" id="wrap-shock-period" hidden>
          <label for="ctrl-shock-period">Every T periods<button type="button" class="help" data-help="shock_period" aria-label="Help: Every T periods">?</button></label>
          <input type="number" id="ctrl-shock-period" min="1" max="20000" step="1" value="100">
        </div>
        <div class="form-group" style="min-width:176px">
          <label for="ctrl-firms">Iterations (firms)<button type="button" class="help" data-help="b_firms" aria-label="Help: Iterations (firms)">?</button></label>
          <input type="number" id="ctrl-firms" min="1" max="2048" step="1" value="50">
          <div class="chip-row" id="firms-chips">
            <button type="button" class="chip" data-value="20">20 (fast)</button>
            <button type="button" class="chip" data-value="50">50</button>
            <button type="button" class="chip" data-value="100">100 (slow)</button>
          </div>
        </div>
        <div class="form-group" style="min-width:86px">
          <label for="ctrl-tsteps">Horizon<button type="button" class="help" data-help="t_steps" aria-label="Help: Horizon">?</button></label>
          <input type="number" id="ctrl-tsteps" min="10" max="20000" step="10" value="600">
        </div>
        <div class="form-group" style="min-width:96px">
          <label for="ctrl-eval-every">Log every<button type="button" class="help" data-help="eval_every" aria-label="Help: Log every">?</button></label>
          <input type="number" id="ctrl-eval-every" min="1" max="20000" step="1" value="25">
        </div>
        <div class="form-group" style="min-width:122px">
          <label for="ctrl-seed">Seed<button type="button" class="help" data-help="seed" aria-label="Help: Seed">?</button></label>
          <input type="number" id="ctrl-seed" min="0" max="4294967295" step="1" value="2026">
        </div>
      </div>

      <div class="exec-area">
        <div class="exec-controls">
          <button type="button" class="btn btn-primary" id="btn-run">Run Simulation</button>
          <button type="button" class="btn btn-danger" id="btn-stop" disabled>Stop</button>
          <span class="badge" id="device-active-badge"></span>
        </div>
      </div>
      <div class="market-summary" id="market-summary-line"></div>
    </div>

    <!-- Configurations: organization designs that run together on the shared market -->
    <div class="config-bar" id="config-bar">
      <span class="config-bar-label">Configurations<button type="button" class="help" data-help="configurations" aria-label="Help: Configurations">?</button></span>
      <button type="button" class="btn" id="btn-add-config" style="padding:3px 9px;font-size:12px">+ Add configuration</button>
      <div class="config-bar-aside">
        <select id="ctrl-experiment" aria-label="Load a ready-made experiment">
          <option value="">Load a ready-made experiment...</option>
        </select>
        <button type="button" class="help" data-help="experiments" aria-label="Help: Ready-made experiments">?</button>
      </div>
    </div>

    <!-- Design of the active configuration: template and summary on top, then the editor beside the settings and inspector -->
    <div class="card" id="design-section">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px">
        <div class="section-label" style="margin:0">Design of <span id="active-config-name">Configuration 1</span></div>
        <button type="button" class="help-jump-link" id="link-org-design-help" style="margin:0">Settings guide</button>
      </div>

      <div class="design-top">
        <div class="form-group">
          <label for="ctrl-graph-template">Template<button type="button" class="help" data-help="template" aria-label="Help: Template">?</button></label>
          <select id="ctrl-graph-template"></select>
          <div class="field-hint" id="template-hint"></div>
        </div>
        <div class="design-readout-box" id="design-readout-box"></div>
      </div>
      <div id="constraint-notes-area"></div>
      <div id="compile-error-area"></div>

      <!-- Grid placement is by id, so the editor and the inspector can move to the Expand overlay and back -->
      <div class="design-grid">
          <!-- Organization settings, stored in the configuration's org_graph -->
          <div class="org-settings-box" id="org-settings-panel">
            <div class="org-settings-title">Organization settings</div>
            <div class="form-grid">
              <div class="form-group full-width">
                <label for="ctrl-feedback-regime">Who sees market results<button type="button" class="help" data-help="feedback_regime" aria-label="Help: Who sees market results">?</button></label>
                <select id="ctrl-feedback-regime">
                  <option value="dc_firm_units_domain" selected>DC sees firm results, units see their own domain</option>
                  <option value="everyone_firm">Everyone sees firm results</option>
                  <option value="dc_only">Only the DC sees results</option>
                </select>
                <div class="field-hint" id="hint-feedback-regime"></div>
              </div>
              <div class="form-group full-width">
                <label for="ctrl-decision-rights">Decision rights<button type="button" class="help" data-help="decision_rights" aria-label="Help: Decision rights">?</button></label>
                <select id="ctrl-decision-rights">
                  <option value="centralized" selected>Coalition decides</option>
                  <option value="propose_approve">Units propose, coalition approves</option>
                  <option value="delegated">Units decide in their domains</option>
                </select>
                <div class="field-hint" id="hint-decision-rights"></div>
              </div>
              <div class="form-group full-width">
                <label for="ctrl-unit-acc-mode">Initial accuracy of units<button type="button" class="help" data-help="initial_accuracy" aria-label="Help: Initial accuracy of units">?</button></label>
                <div class="acc-row">
                  <select id="ctrl-unit-acc-mode">
                    <option value="uninformed" selected>Uninformed (random beliefs)</option>
                    <option value="set">Set accuracy</option>
                  </select>
                  <div class="range-row" id="wrap-unit-acc" hidden>
                    <input type="range" id="ctrl-unit-acc" min="0" max="1" step="0.05" value="0.5" aria-label="Initial accuracy of units">
                    <span class="range-value" id="ctrl-unit-acc-value">0.50</span>
                  </div>
                </div>
              </div>
              <details class="advanced-details" id="org-advanced-details">
                <summary>Advanced settings (experiments, shortfalls, track record)</summary>
                <div class="org-settings-grid" style="margin-top:8px">
                  <div class="form-group">
                    <label for="ctrl-experiment-rate">Unit experiments<button type="button" class="help" data-help="experiment_rate" aria-label="Help: Unit experiments">?</button></label>
                    <input type="number" id="ctrl-experiment-rate" min="0" max="1" step="0.05" value="0.50">
                  </div>
                  <div class="form-group">
                    <label for="ctrl-aspiration">Search after shortfalls<button type="button" class="help" data-help="aspiration_sensitivity" aria-label="Help: Search after shortfalls">?</button></label>
                    <input type="number" id="ctrl-aspiration" min="0" max="5" step="0.25" value="0.00">
                  </div>
                  <div class="form-group full-width">
                    <label for="ctrl-org-lambda">Track-record learning<button type="button" class="help" data-help="lambda_track" aria-label="Help: Track-record learning">?</button></label>
                    <input type="number" id="ctrl-org-lambda" min="0" max="5" step="0.25" value="0.00">
                  </div>
                </div>
              </details>
            </div>
          </div>

          <!-- Organization editor; the Expand button moves this block into a large overlay -->
          <div id="editor-home">
            <div class="editor" id="editor">
              <div class="editor-toolbar">
                <button type="button" class="btn" id="btn-add-unit">+ Unit</button>
                <button type="button" class="btn" id="btn-add-manager">+ Manager</button>
                <button type="button" class="btn" id="btn-auto-layout">Auto-layout</button>
                <button type="button" class="btn" id="btn-fit">Fit</button>
                <button type="button" class="btn" id="btn-undo" disabled>Undo</button>
                <button type="button" class="btn btn-danger" id="btn-delete">Delete</button>
                <span class="spacer"></span>
                <button type="button" class="btn" id="btn-expand">Expand</button>
              </div>
              <p class="editor-hint">
                <strong>How to build an organization.</strong> Add a unit or manager with the buttons, or double-click empty space.<br>
                Drag from the dot on a node's right edge to another node to add a link; the arrow points to the node that learns.<br>
                Click a node or link to edit it; Delete removes it and Ctrl+Z (Cmd+Z) undoes.
              </p>
              <div class="editor-canvas-wrap" id="editor-canvas-wrap">
                <svg id="org-canvas" tabindex="0" aria-label="Organization graph editor"></svg>
                <div class="canvas-tip" id="canvas-tip" hidden></div>
                <div class="editor-msg" id="editor-msg" role="status" hidden></div>
              </div>
              <div class="editor-legend">
                <span class="legend-entry"><svg width="36" height="10" aria-hidden="true"><line x1="2" y1="5" x2="27" y2="5" stroke="#475569" stroke-width="2"/><path d="M27 1 L35 5 L27 9 Z" fill="#475569"/></svg>A to B: B learns from A</span>
                <span class="legend-entry"><svg width="44" height="10" aria-hidden="true"><line x1="2" y1="5" x2="18" y2="5" stroke="#475569" stroke-width="1"/><line x1="24" y1="5" x2="42" y2="5" stroke="#475569" stroke-width="5"/></svg>Thicker: higher permeability P</span>
                <span class="legend-entry"><svg width="30" height="10" aria-hidden="true"><line x1="2" y1="5" x2="28" y2="5" stroke="#94a3b8" stroke-width="2" stroke-dasharray="5 4"/></svg>Dashed: commensurability C below 0.6</span>
                <span class="legend-entry"><span class="fb-badge fb-firm">F</span><span class="fb-badge fb-domain">D</span><span class="fb-badge fb-none">-</span>Sees firm, own-domain, or no market results (filled: set on the node)</span>
                <span class="legend-entry"><span class="acc-tag">acc 0.6</span>Informed start (initial accuracy)</span>
              </div>
            </div>
          </div>

          <!-- Inspector: the selected link replaces the node fields until a node or empty canvas is clicked -->
          <div class="node-inspector-box" id="node-inspector-panel">
            <div class="edge-editor" id="edge-editor" hidden>
              <div class="node-inspector-header">
                <span class="edge-editor-title" id="edge-editor-title">Link</span>
                <span style="display:flex;align-items:center;gap:6px">
                  <span class="role-pill">Link</span>
                  <button type="button" class="chip-icon" id="edge-editor-close" aria-label="Close the link editor">&times;</button>
                </span>
              </div>
              <label for="edge-ctrl-p">Permeability P<button type="button" class="help" data-help="permeability" aria-label="Help: Permeability">?</button></label>
              <div class="range-row">
                <input type="range" id="edge-ctrl-p" min="0" max="1" step="0.05" value="0.25">
                <span class="range-value" id="edge-ctrl-p-value">0.25</span>
              </div>
              <label for="edge-ctrl-c">Commensurability C<button type="button" class="help" data-help="commensurability" aria-label="Help: Commensurability">?</button></label>
              <div class="range-row">
                <input type="range" id="edge-ctrl-c" min="0" max="1" step="0.05" value="0.85">
                <span class="range-value" id="edge-ctrl-c-value">0.85</span>
              </div>
              <p class="edge-editor-note">C is shared by both directions of a pair of nodes.</p>
              <div class="edge-editor-actions">
                <button type="button" class="btn" id="edge-btn-reverse">Reverse</button>
                <button type="button" class="btn btn-danger" id="edge-btn-delete">Delete</button>
              </div>
              <p class="edge-editor-note" style="margin:10px 0 0">Click a node or empty canvas to return to node editing.</p>
            </div>

            <div class="node-inspector-header" id="node-inspector-head">
              <strong id="node-inspector-title" style="font-size:12.5px">Dominant Coalition</strong>
              <span class="role-pill" id="node-inspector-role-pill">DC</span>
            </div>

            <div class="form-grid" id="node-form">
              <div class="form-group full-width" id="node-inspector-body">
                <label for="node-ctrl-label">Label</label>
                <input type="text" id="node-ctrl-label" value="Dominant Coalition">
              </div>
              <div class="form-group">
                <label for="node-ctrl-model">Representation<button type="button" class="help" data-help="model_type" aria-label="Help: Representation">?</button></label>
                <select id="node-ctrl-model">
                  <option value="neural" selected>Neural network</option>
                  <option value="linear">Linear (additive)</option>
                  <option value="tabular">Lookup table</option>
                </select>
              </div>
              <div class="form-group">
                <label for="node-ctrl-h">Capacity (H)<button type="button" class="help" data-help="h_agent" aria-label="Help: Capacity">?</button></label>
                <select id="node-ctrl-h">
                  <option value="0">Linear (H = 0)</option>
                  <option value="4">Simplifier (H = 4)</option>
                  <option value="8">Compact (H = 8)</option>
                  <option value="16" selected>Moderate (H = 16)</option>
                  <option value="32">Expanded (H = 32)</option>
                  <option value="64">Complexifier (H = 64)</option>
                </select>
              </div>

              <!-- Choices (bits 1-N) the node attends to -->
              <div class="form-group full-width" id="node-jurisdiction-wrap">
                <div style="display:flex;justify-content:space-between;align-items:center">
                  <label for="node-jurisdiction-preset">Domain: choices it attends to<button type="button" class="help" data-help="domain" aria-label="Help: Domain">?</button></label>
                  <select id="node-jurisdiction-preset" style="width:auto;padding:2px 6px;font-size:11px">
                    <option value="all">All 12 (1-12)</option>
                    <option value="first_half">First half (1-6)</option>
                    <option value="second_half">Second half (7-12)</option>
                    <option value="custom">Custom</option>
                  </select>
                </div>
                <div class="bit-pills-row" id="node-bit-pills"></div>
              </div>

              <div class="form-group full-width" id="node-wrap-feedback">
                <label for="node-ctrl-feedback">Sees market results<button type="button" class="help" data-help="node_feedback" aria-label="Help: Sees market results">?</button></label>
                <select id="node-ctrl-feedback">
                  <option value="">Organization default</option>
                  <option value="firm">Firm results</option>
                  <option value="domain">Own-domain results</option>
                  <option value="none">None</option>
                </select>
                <div class="field-hint" id="node-feedback-note" hidden></div>
              </div>

              <div class="form-group full-width">
                <label for="node-ctrl-acc-mode">Initial accuracy<button type="button" class="help" data-help="initial_accuracy" aria-label="Help: Initial accuracy">?</button></label>
                <div class="acc-row">
                  <select id="node-ctrl-acc-mode">
                    <option value="default">Organization default</option>
                    <option value="uninformed">Uninformed</option>
                    <option value="set">Set accuracy</option>
                  </select>
                  <div class="range-row" id="wrap-node-acc" hidden>
                    <input type="range" id="node-ctrl-acc" min="0" max="1" step="0.05" value="0.5" aria-label="Initial accuracy of this node">
                    <span class="range-value" id="node-ctrl-acc-value">0.50</span>
                  </div>
                </div>
                <div class="field-hint" id="node-acc-note"></div>
              </div>

              <div class="form-group">
                <label for="node-ctrl-lr">Market learning rate<button type="button" class="help" data-help="lr" aria-label="Help: Market learning rate">?</button></label>
                <input type="number" id="node-ctrl-lr" min="0.01" max="1.0" step="0.02" value="0.14">
                <div class="field-hint">Updates toward market results only; learning from colleagues uses link Permeability P.</div>
              </div>
              <div class="form-group" id="node-wrap-rho">
                <label for="node-ctrl-rho">Distant search (rho)<button type="button" class="help" data-help="rho" aria-label="Help: Distant search (rho)">?</button></label>
                <input type="number" id="node-ctrl-rho" min="0.00" max="1.00" step="0.05" value="0.00">
              </div>
              <div class="form-group full-width" id="node-search-note" hidden>
                <div class="field-hint" id="node-search-text"></div>
              </div>

              <div class="form-group full-width" id="node-links-wrap">
                <label>Links<button type="button" class="help" data-help="links" aria-label="Help: Links">?</button></label>
                <div class="links-list" id="node-links-box"></div>
              </div>

              <details class="advanced-details" id="node-advanced-details">
                <summary>Advanced node settings (learning noise, headcount, multi-step search, aggregation)</summary>
                <div class="org-settings-grid" style="margin-top:8px">
                  <div class="form-group">
                    <label for="node-ctrl-sigma-learn">Learning noise<button type="button" class="help" data-help="sigma_learn" aria-label="Help: Learning noise">?</button></label>
                    <input type="number" id="node-ctrl-sigma-learn" min="0.00" max="1.50" step="0.05" value="0.05">
                  </div>
                  <div class="form-group">
                    <label for="node-ctrl-count">Headcount<button type="button" class="help" data-help="count" aria-label="Help: Headcount">?</button></label>
                    <input type="number" id="node-ctrl-count" min="1" max="32" step="1" value="1">
                  </div>
                  <div class="form-group" id="node-wrap-search-steps">
                    <label for="node-ctrl-search-steps">Search choices per period<button type="button" class="help" data-help="search_steps" aria-label="Help: Search choices per period">?</button></label>
                    <input type="number" id="node-ctrl-search-steps" min="1" max="16" step="1" value="1">
                  </div>
                  <div class="form-group" id="node-wrap-agg">
                    <label for="node-ctrl-agg">Combining several sources<button type="button" class="help" data-help="upward_agg" aria-label="Help: Combining several sources">?</button></label>
                    <select id="node-ctrl-agg">
                      <option value="mean" selected>Permeability-weighted mean (default)</option>
                      <option value="max">Most optimistic source (max)</option>
                      <option value="min">Most skeptical source (min)</option>
                    </select>
                  </div>
                </div>
              </details>
            </div>
          </div>

      </div>
    </div>

    <!-- Results of the last run: progress and legend, six live charts, paired tests, and checkpoints -->
    <section id="results-section">
      <div class="card" style="padding:10px 16px">
        <div class="results-head">
          <h2 class="card-title">Results</h2>
          <div class="progress-container">
            <div class="progress-meta">
              <span id="progress-status-text">Ready</span>
              <span id="progress-time-text">0.00s</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" id="progress-fill-bar"></div>
            </div>
          </div>
          <div class="legend-row" id="chart-legend-row"></div>
        </div>
        <div class="field-hint" id="results-stale-note" style="margin-top:6px" hidden>Results are from an earlier setup; press Run Simulation to update.</div>
      </div>

        <!-- Six synchronized live trajectory charts -->
        <div class="charts-grid">
          <div class="chart-card">
            <div class="chart-header">
              <h3 class="chart-title">Enacted Payoff V(s_t)<button type="button" class="help" data-help="m_v" aria-label="Help: Enacted payoff">?</button></h3>
              <span class="badge" id="stat-badge-v">--</span>
            </div>
            <div class="chart-ref-legend" id="payoff-ref-legend" hidden>
              <span class="legend-entry"><svg width="26" height="8" aria-hidden="true"><line x1="1" y1="4" x2="25" y2="4" stroke="#52525b" stroke-width="1.4" stroke-dasharray="6 3"/></svg>Best strategy</span>
              <span class="legend-entry"><svg width="26" height="8" aria-hidden="true"><line x1="1" y1="4" x2="25" y2="4" stroke="#a1a1aa" stroke-width="1.4" stroke-dasharray="2 3"/></svg>Local search with perfect knowledge</span>
              <button type="button" class="help" data-help="payoff_refs" aria-label="Help: Reference lines">?</button>
            </div>
            <div class="chart-canvas-wrap">
              <canvas class="traj-canvas" id="canvas-v"></canvas>
            </div>
          </div>

          <div class="chart-card">
            <div class="chart-header">
              <h3 class="chart-title">Global Epistemic Accuracy (dc_acc_global)<button type="button" class="help" data-help="m_acc" aria-label="Help: Global epistemic accuracy">?</button></h3>
              <span class="badge" id="stat-badge-acc">--</span>
            </div>
            <div class="chart-canvas-wrap">
              <canvas class="traj-canvas" id="canvas-acc"></canvas>
            </div>
          </div>

          <div class="chart-card">
            <div class="chart-header">
              <h3 class="chart-title">Representational Complexity (dc_complexity)<button type="button" class="help" data-help="m_comp" aria-label="Help: Representational complexity">?</button></h3>
              <span class="badge" id="stat-badge-comp">--</span>
            </div>
            <div class="chart-canvas-wrap">
              <canvas class="traj-canvas" id="canvas-comp"></canvas>
            </div>
          </div>

          <div class="chart-card">
            <div class="chart-header">
              <h3 class="chart-title">Representational Heterogeneity<button type="button" class="help" data-help="m_orth" aria-label="Help: Representational heterogeneity">?</button></h3>
              <span class="badge" id="stat-badge-orth">--</span>
            </div>
            <div class="chart-canvas-wrap">
              <canvas class="traj-canvas" id="canvas-orth"></canvas>
            </div>
          </div>

          <div class="chart-card">
            <div class="chart-header">
              <h3 class="chart-title">Local Gradient Correlation (r_local)<button type="button" class="help" data-help="m_r" aria-label="Help: Local gradient correlation">?</button></h3>
              <span class="badge" id="stat-badge-r">--</span>
            </div>
            <div class="chart-canvas-wrap">
              <canvas class="traj-canvas" id="canvas-r"></canvas>
            </div>
          </div>

          <div class="chart-card">
            <div class="chart-header">
              <h3 class="chart-title"><span id="chart4-title">False Belief Stall Rate</span><button type="button" class="help" data-help="m_c6" aria-label="Help: Sixth chart">?</button></h3>
              <div style="display:flex;align-items:center;gap:8px">
                <select id="chart4-metric-select" style="padding:2px 6px;font-size:11.5px;width:auto" aria-label="Metric shown in the sixth chart">
                  <option value="false_stall_mean" selected>False Belief Stall Rate</option>
                  <option value="v_percentile_mean">Share of strategies beaten</option>
                  <option value="orthogonality_linked_mean">Heterogeneity (linked pairs)</option>
                  <option value="at_local_peak_mean">At a local peak (share of firms)</option>
                  <option value="mean_permeability_mean">Mean Permeability (P_ab)</option>
                  <option value="emp_div_local_slope_mean">Subunit Slope Diversity</option>
                  <option value="v_pct_max_mean">% of Landscape Max</option>
                </select>
                <span class="badge" id="stat-badge-fs">--</span>
              </div>
            </div>
            <div class="chart-canvas-wrap">
              <canvas class="traj-canvas" id="canvas-fs"></canvas>
            </div>
          </div>
        </div>

        <!-- Paired comparison with the first configuration (shown with two or more configurations) -->
        <div class="card" id="paired-test-card" hidden>
          <div class="card-header">
            <h2 class="card-title" id="paired-test-title">Paired comparison with Configuration 1 (same landscapes and noise draws)</h2>
          </div>
          <div class="table-wrap">
            <table>
              <thead id="paired-test-thead"></thead>
              <tbody id="paired-test-tbody"></tbody>
            </table>
          </div>
        </div>

        <!-- Trajectory checkpoints for every configuration -->
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">Trajectory Checkpoints</h2>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr id="checkpoint-thead-row">
                  <th>Configuration</th>
                  <th>Step</th>
                  <th>V(s_t) (SE)</th>
                  <th>Strategies beaten</th>
                  <th>Global Acc</th>
                  <th>Complexity</th>
                  <th>Heterogeneity</th>
                  <th>DC r_local</th>
                  <th>False Stall</th>
                </tr>
              </thead>
              <tbody id="checkpoint-tbody">
                <tr><td colspan="9" style="color:var(--text-muted)">Press Run Simulation to stream trajectories.</td></tr>
              </tbody>
            </table>
          </div>
        </div>
    </section>
  </div>

  <!-- ==================== MODEL & PARAMETERS GUIDE VIEW ==================== -->
  <div id="view-guide" class="guide-container" hidden>
    <div class="card">
      <div class="card-header">
        <h2 class="card-title">How the Model Works</h2>
      </div>
      <p>
        Each simulated firm chooses a strategy made of <code>N</code> binary choices (2 to 16, default 12), so there are <code>2^N</code> possible strategies (4,096 at <code>N = 12</code>). An organization is a directed graph of <em>representations</em>: each node holds a mental model of how strategies pay off. The <strong>dominant coalition (DC)</strong> adopts the firm's strategy; managers and units attend to a subset of the choices, their domain. Links carry views from one node to another. All configurations in a run face the same landscapes, noise draws, and shocks, so the differences between them are paired.
      </p>
      <p>
        <strong>The landscape.</strong> Payoffs follow an NK landscape: each choice's payoff contribution depends on the choice itself and on <code>K</code> other choices, and the firm payoff is the average contribution, standardized over all <code>2^N</code> strategies (Levinthal 1997).
      </p>
      <p>
        <strong>Before period 1.</strong> Each node starts uninformed, from random beliefs (the default), or with an initial accuracy <code>a</code> between 0 and 1: the correlation between its starting beliefs and the payoff it observes, as far as its own choices can explain it. That explainable part is the observed payoff averaged over the choices outside the node's domain; for the DC and any other node whose domain covers every choice, it is the observed payoff itself. An informed node starts from beliefs fitted to <code>t(s) = 0.5 * (a * z_obs(s) + sqrt(1 - a^2) * z_mis(s))</code>, where <code>z_obs</code> is the standardized payoff the node will observe (the firm payoff, or the average contribution of its own domain) and <code>z_mis</code> is an independent NK landscape with the same interaction structure: the node knows which choices interact but holds partly wrong beliefs about how. The factor 0.5 keeps starting beliefs weak, so market feedback can overwrite them (Puranam and Swamy 2016).
      </p>
      <p>
        When interactions cross domains, a unit's own choices explain only part of its result. For a domain of 6 of the 12 choices, the correlation between its own-domain payoff and the part its choices explain is about 0.55 with random interactions and <code>K = 3</code>, about 0.85 with modular interactions and <code>K = 3</code>, and 1 when the domain consists of whole interaction blocks. Even a fully accurate unit therefore cannot predict its results perfectly. Initial accuracy is thus within-domain accuracy: a unit's ability to predict its actual results is about its initial accuracy times that correlation.
      </p>
      <p>
        Informed nodes are an explicit starting endowment: each informed node brings its own independent prior, so an organization with more informed units starts with more pooled knowledge. This endowment is part of the design being compared, separate from what the organization learns during the run.
      </p>
      <p><strong>Each period:</strong></p>
      <ol class="guide-steps">
        <li><strong>Turbulence.</strong> Under random turbulence a shock occurs in each period with probability <code>p</code>; under periodic turbulence shocks occur at periods <code>T</code>, <code>2T</code>, and so on. A shock redraws each choice's contribution table with probability 0.5 while the interaction structure stays fixed. All configurations are hit in the same periods with the same redraws, and what nodes learned earlier becomes partly obsolete (Posen and Levinthal 2012).</li>
        <li><strong>Exchange.</strong> Each node moves its representation toward the views of the nodes it learns from. Permeability <code>P</code> sets how far it moves; low commensurability <code>C</code> adds a mistranslation that is fixed for each pair of nodes, plus noise.</li>
        <li><strong>Strategy change.</strong> Every searcher tries a few candidate changes within its domain (one-bit changes, or distant jumps with probability <code>rho</code>; Levinthal 1997) and keeps each candidate only if its own representation predicts a gain over the current strategy. The decision rights set who searches: the coalition alone; units, whose proposals the coalition approves or rejects (Rivkin and Siggelkow 2003); or units in their own domains while the coalition searches the remaining choices (Siggelkow and Levinthal 2003). Candidates are judged on representations, never on true payoffs.</li>
        <li><strong>Market feedback.</strong> The market reveals a noisy payoff only for the adopted strategy (Denrell and March 2001). The DC observes the firm payoff; other nodes observe the firm payoff, only the payoff contribution of their own domain, or nothing, depending on who sees market results (Gaba and Joseph 2013; Tushman and Scanlan 1981). Nodes that observe results learn from them. Unit experiments, track-record learning, and the coalition's aspiration update also happen in this step (Greve 2003).</li>
        <li><strong>Metrics.</strong> Performance and representation metrics are recorded on the current landscape (see Metrics below).</li>
      </ol>
      <p>
        Nodes that observe nothing learn only from colleagues, much as individuals in March (1991) learn only from the organizational code.
      </p>
    </div>

    <div class="card">
      <div class="card-header">
        <h2 class="card-title">Using the Laboratory</h2>
      </div>
      <h3 class="guide-subhead">Configurations</h3>
      <p>
        A configuration is one organization design: its nodes, links, and organization settings. The bar above the editor holds up to 8 configurations. Click one to edit it, rename it with the pencil or by double-clicking its name, and use "+ Add configuration" to copy the active one. Run Simulation runs a single configuration on its own, or two to eight side by side on the same landscapes; the charts show each configuration in its color, and the paired table compares every configuration with the first one, firm by firm.
      </p>
      <h3 class="guide-subhead">Building an organization</h3>
      <ul class="guide-notes">
        <li>Start from a template and edit it. "+ Unit" adds a unit that reports to the selected manager, or to the DC; "+ Manager" adds a manager that reports to the DC; double-clicking empty canvas adds a unit at that spot.</li>
        <li>Drag from the dot on a node's right edge to another node to add a link; the arrow points from the source to the node that learns from it. Alternatively, select the source and Shift+click the learner.</li>
        <li>Click a node or a link to edit it in the inspector below the canvas (beside it in the expanded editor). For a link, set its permeability <code>P</code> and commensurability <code>C</code>, reverse it, or delete it; clicking a node or empty canvas returns to node editing. A node's Links list accepts exact values from the keyboard.</li>
        <li>Drag nodes to arrange them. Auto-layout puts the DC on top and every other node by its link distance from the DC; Fit shows the whole graph, and Expand opens a large editor for complex graphs.</li>
        <li>Delete or Backspace removes the selected link or node (the DC stays); Ctrl+Z (Cmd+Z on a Mac) undoes up to 30 changes.</li>
      </ul>
      <h3 class="guide-subhead">Files</h3>
      <p>
        Export JSON saves the market, every configuration with its canvas layout, and the latest results (file format 0.1.0). Import JSON loads a saved experiment file back into the laboratory. The command-line tool runs the same file without a browser: <code>org-neural-search run experiment.json --out results.json</code>.
      </p>
    </div>

    <div class="card" id="guide-options-dict">
      <div class="card-header">
        <h2 class="card-title">Settings Reference</h2>
      </div>
      <h3 class="guide-subhead">Market environment (shared by all configurations)</h3>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Setting</th><th>Default</th><th>What it does</th></tr>
          </thead>
          <tbody>
            <tr><td><strong>Choices N</strong></td><td class="mono">12</td><td>Number of binary choices in a firm's strategy (2 to 16); the chips set 8, 12, or 16. With <code>N</code> choices there are <code>2^N</code> strategies (from 4 at <code>N = 2</code> to 4,096 at <code>N = 12</code> and 65,536 at <code>N = 16</code>). Changing <code>N</code> rescales each node's domain proportionally and bounds <code>K</code> to at most <code>N - 1</code>.</td></tr>
            <tr><td><strong>Interdependence K</strong></td><td class="mono">3</td><td>Number of other choices that affect each choice's payoff contribution (0 to <code>N - 1</code>). <code>K = 0</code> gives a single peak; higher values add local peaks.</td></tr>
            <tr><td><strong>Interactions</strong></td><td class="mono">Random</td><td>Random: each choice interacts with <code>K</code> randomly drawn choices. Modular: choices interact within consecutive blocks of <code>K + 1</code>; domains that coincide with blocks make own-domain results informative (Rivkin and Siggelkow 2003).</td></tr>
            <tr><td><strong>Feedback noise</strong></td><td class="mono">0.65</td><td>Standard deviation of the external measurement noise in every observed market result (0 to 3); the chips set 0, 0.30, 0.65, or 1.00.</td></tr>
            <tr><td><strong>Turbulence</strong></td><td class="mono">None</td><td>None keeps the landscape fixed. Random: a shock in each period with the given probability (0 to 0.10). Periodic: a shock every <code>T</code> periods (1 to 20,000). A shock redraws each choice's contribution table with probability 0.5; nodes keep their representations (Posen and Levinthal 2012).</td></tr>
            <tr><td><strong>Iterations (firms)</strong></td><td class="mono">50</td><td>Number of independent Monte Carlo landscapes (1 to 2,048); the chips set 20, 50, or 100. Every configuration runs on the same set, so differences are paired; the standard error of a paired difference shrinks with the square root of the number of iterations, so use 100 or more for publication-grade comparisons. Run time grows with the number of iterations.</td></tr>
            <tr><td><strong>Horizon, Log every, Seed</strong></td><td class="mono">600, 25, 2026</td><td>Number of periods (10 to 20,000), the interval in periods between full-landscape evaluations of Global Accuracy, Complexity, and Heterogeneity (1 to 20,000, default 25), and the random seed shared by all configurations (0 to 4,294,967,295).</td></tr>
          </tbody>
        </table>
      </div>

      <h3 class="guide-subhead">Organization settings (one set per configuration)</h3>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Setting</th><th>Default</th><th>What it does</th></tr>
          </thead>
          <tbody>
            <tr><td><strong>Who sees market results</strong></td><td>DC sees firm results, units see their own domain</td><td>Units see the average payoff contribution of the choices in their domain; a unit whose domain covers every choice sees the firm result. "Only the DC sees results" leaves the other nodes to learn from colleagues, as in March (1991). A node's own setting in the inspector overrides this (Gaba and Joseph 2013; Tushman and Scanlan 1981).</td></tr>
            <tr><td><strong>Decision rights</strong></td><td>Coalition decides</td><td>"Units propose, coalition approves": each other node searches its own domain the way the coalition does, with its own search reach and steps, and proposes changes its model predicts will pay off; the coalition adopts the combination or single proposal it predicts is best without searching itself (Rivkin and Siggelkow 2003). "Units decide in their domains": each other node searches its domain the same way and makes those changes itself, and the coalition searches only over choices outside every other node's domain, so it does not search when the units cover every choice (Siggelkow and Levinthal 2003). Without other nodes, both work like "Coalition decides".</td></tr>
            <tr><td><strong>Initial accuracy of units</strong></td><td>Uninformed</td><td>Starting accuracy <code>a</code> (0 to 1) of every node other than the DC, as described under "Before period 1". A node's own setting in the inspector overrides it; the DC is set only in the inspector.</td></tr>
            <tr><td><strong>Unit experiments</strong> (Advanced)</td><td class="mono">0.50</td><td>Each period, each non-DC node that sees results tests, with this probability, one random one-choice change in its domain off-line and learns from the result; the firm's strategy does not change (Gavetti and Levinthal 2000; Knudsen and Levinthal 2007). 0.00 turns testing off.</td></tr>
            <tr><td><strong>Search after shortfalls</strong> (Advanced)</td><td class="mono">0.00</td><td>Problemistic search: the coalition keeps an aspiration level <code>A</code> (an exponentially weighted average of its past results, weight 0.1), and its distant-search probability becomes <code>min(1, rho + kappa * max(0, A - y))</code>, where <code>y</code> is its last observed result (Greve 2003).</td></tr>
            <tr><td><strong>Track-record learning</strong> (Advanced)</td><td class="mono">0.00</td><td>Rate <code>lambda</code> (0 to 5). Each node compares each source's forecast for the adopted strategy with its own, against the result it observes; permeability is multiplied by <code>exp(lambda * (own error - source error))</code>, bounded between 0.001 and 1, so sources more accurate than the node gain weight.</td></tr>
          </tbody>
        </table>
      </div>

      <h3 class="guide-subhead">Nodes and links (node inspector)</h3>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Setting</th><th>What it does</th></tr>
          </thead>
          <tbody>
            <tr><td><strong>Representation, capacity H</strong></td><td>Neural network with <code>H</code> hidden units, linear additive model (<code>H = 0</code>), or lookup table with one entry per combination of its own choices (one per strategy when its domain covers all <code>N</code>) and no generalization.</td></tr>
            <tr><td><strong>Domain</strong></td><td>The choices the node attends to. Its representation ignores all other choices, and its own-domain results cover these choices only.</td></tr>
            <tr><td><strong>Sees market results</strong></td><td>Per-node override of the organization setting: firm results, own-domain results, or none. The DC always sees firm results.</td></tr>
            <tr><td><strong>Initial accuracy</strong></td><td>Per-node starting accuracy: the organization default (other nodes only), uninformed, or a set value; the DC is uninformed unless set here. After a run the inspector shows the realized accuracy, measured as described under "Before period 1" and averaged over firms; it varies with the misleading draws and can be slightly above or below the set value (within about 0.03 on average).</td></tr>
            <tr><td><strong>Market learning rate</strong></td><td>Step size for updating this node's representation toward observed market payoffs and unit experiments in Step 4. Learning from colleagues during Step 2 exchange uses link Permeability <code>P</code> instead.</td></tr>
            <tr><td><strong>Distant search rho, search choices per period</strong></td><td>How a node searches: the probability <code>rho</code> that a candidate change is a distant jump (several choices in its domain at once) and how many candidate changes it tries each period (under Advanced node settings; default 1), kept only if the node's own model predicts a gain (Levinthal 1997). The DC uses them when it searches; other nodes use them when units propose or decide.</td></tr>
            <tr><td><strong>Learning noise, Headcount, Combining several sources</strong> (Advanced)</td><td><code>Learning noise</code> is internal cognitive noise added to market and experiment updates (not to colleague exchange). <code>Headcount</code> is the number of parallel representations in the node (always 1 for the DC). <code>Combining several sources</code> sets how incoming links are aggregated in Step 2: permeability-weighted mean (default), most optimistic source (max), or most skeptical source (min).</td></tr>
            <tr><td><strong>Link permeability P, commensurability C</strong></td><td><code>P</code> (0 to 1) is directional: how far the learner moves toward the source. <code>C</code> (0 to 1) is shared by both directions: how faithfully the two nodes translate each other's views. Permeabilities are normalized within each learner, and the exchange step stops growing once a node's total permeability reaches 0.75. Links may be reciprocal and may form cycles; graphs have at most three tiers.</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <h2 class="card-title">Templates</h2>
      </div>
      <p>Each template is a complete organization (nodes, links, and organization settings); every part can be edited after loading.</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Template</th><th>Key</th><th>Design</th></tr>
          </thead>
          <tbody id="guide-template-tbody">
            <tr><td colspan="3" style="color:var(--text-muted)">Loading templates...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <h2 class="card-title">Ready-Made Experiments</h2>
      </div>
      <p>Each experiment replaces the configurations with a complete set and clears the results; press Run Simulation to start it. Experiments set K = 3 with random interactions and no turbulence unless stated, and keep the current Choices N, noise, horizon, iterations (firms), and seed.</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Experiment</th><th>Design</th><th style="width:70px">Action</th></tr>
          </thead>
          <tbody id="guide-presets-tbody"></tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <h2 class="card-title">Metrics</h2>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Key</th><th>Metric</th><th>Definition</th></tr>
          </thead>
          <tbody>
            <tr><td class="mono">v_true</td><td><strong>Enacted payoff V(s_t)</strong></td><td>Payoff of the adopted strategy, standardized over the firm's <code>2^N</code> strategies (mean 0, SD 1; 4,096 strategies at default <code>N = 12</code>).</td></tr>
            <tr><td class="mono">v_pct_max</td><td><strong>% of landscape max</strong></td><td>Position of the adopted strategy between the worst (0 percent) and the best (100 percent) strategy.</td></tr>
            <tr><td class="mono">v_percentile</td><td><strong>Share of strategies beaten</strong></td><td>Share of the firm's <code>2^N</code> strategies with a lower true payoff than the adopted strategy, shown in percent.</td></tr>
            <tr><td class="mono">v_max_ref</td><td><strong>Best strategy</strong></td><td>Payoff of the landscape's best strategy (dashed gray line on the payoff chart); it changes only with shocks.</td></tr>
            <tr><td class="mono">v_local_ref</td><td><strong>Local search with perfect knowledge</strong></td><td>Average payoff reached by changing one choice at a time, always to the best one-choice change, while knowing the true payoffs, starting from every strategy (dotted gray line on the payoff chart); it changes only with shocks.</td></tr>
            <tr><td class="mono">dc_acc_global</td><td><strong>Global epistemic accuracy</strong></td><td>Correlation between the DC's representation and the true payoffs across all <code>2^N</code> strategies.</td></tr>
            <tr><td class="mono">dc_complexity</td><td><strong>Representational complexity</strong></td><td>Number of strategies that the DC's representation rates above all of their one-bit neighbors (its perceived local peaks).</td></tr>
            <tr><td class="mono">orthogonality</td><td><strong>Representational heterogeneity</strong></td><td>Mean of <code>1 - |corr|</code> between all pairs of representations, the DC's included, across all strategies; higher values mean more diverse views across units.</td></tr>
            <tr><td class="mono">orthogonality_linked</td><td><strong>Heterogeneity (linked pairs)</strong></td><td>The same <code>1 - |corr|</code> dissimilarity measure, restricted to pairs of nodes joined by a link in either direction.</td></tr>
            <tr><td class="mono">dc_r_local</td><td><strong>Local gradient correlation</strong></td><td>Correlation between the DC's predicted and the true payoff changes of the <code>N</code> one-bit moves from the current strategy.</td></tr>
            <tr><td class="mono">false_stall</td><td><strong>False belief stall rate</strong></td><td>Share of firms stuck on a suboptimal strategy because of a false belief: the Dominant Coalition believes no single-choice change can improve performance (so local search stalls), even though a better single-choice change actually exists on the true landscape.</td></tr>
            <tr><td class="mono">at_local_peak</td><td><strong>At a local peak (share of firms)</strong></td><td>Share of firms whose strategy is a local peak of the true landscape: no one-choice change would raise its payoff.</td></tr>
            <tr><td class="mono">emp_div_local_slope</td><td><strong>Subunit slope diversity</strong></td><td>Variance across non-DC nodes of their predicted payoff changes for one-bit moves from the current strategy.</td></tr>
            <tr><td class="mono">mean_permeability</td><td><strong>Mean permeability</strong></td><td>Average permeability of active links; it changes over time only with track-record learning.</td></tr>
          </tbody>
        </table>
      </div>
      <h3 class="guide-subhead">Reading payoffs in absolute terms</h3>
      <p>
        Payoffs are in standard-deviation units relative to all <code>2^N</code> strategies of each landscape (4,096 at default <code>N = 12</code>), so 0 is an average strategy. Typical values for <code>N = 12, K = 3</code> with random interactions: the best strategy is about 3.25, local search with perfect knowledge reaches about 2.5, an average local peak is about 2.2, and the top 5 percent of strategies lie above 1.65. The dashed reference lines on the payoff chart and the share of strategies beaten put an organization's payoff on this scale.
      </p>
      <h3 class="guide-subhead">Reading the metrics</h3>
      <ul class="guide-notes">
        <li>Accuracy, complexity, and the two heterogeneity measures evaluate all <code>2^N</code> strategies, so they are recomputed in period 1, every <code>Log every</code> periods (default 25), and in the last period; their curves move in steps when <code>Log every</code> is above 1.</li>
        <li>Complexity measures how rugged the landscape looks to the DC (the perceived ruggedness of its representation), not structural complexity in the sense of Csaszar and Ostler (2020).</li>
        <li><code>1 - |corr|</code> treats two opposed representations (correlation near -1) as similar rather than diverse.</li>
        <li>Learning only from adopted strategies can produce self-confirming beliefs: the coalition's model tends to peak where the firm has been, so it may stall even when a better neighbor exists (Denrell and March 2001); unit experiments, initial accuracy, and delegation are the levers that counter this.</li>
        <li>The misleading landscape behind initial accuracy is drawn once, not at every learning step: redrawing it would average the misleading part away, so starting beliefs would converge to a scaled-down copy of the truth and <code>a</code> would no longer set the accuracy. Results stay robust to a particular draw because every firm and node gets its own draw, results average over firms, and configurations are compared on the same draws.</li>
        <li>The DC is an epistemic leader that forms beliefs and adopts strategies, not a political coalition with interests of its own; Levinthal and Pham (2024) model coalition politics.</li>
        <li>Two-sided paired tests across firms compare each configuration with the first one on the same landscapes; the difference column is the other configuration minus the first.</li>
      </ul>
    </div>

    <div class="card">
      <div class="card-header">
        <h2 class="card-title">References</h2>
      </div>
      <ul class="guide-refs">
        <li>Csaszar and Ostler (2020). A contingency theory of representational complexity in organizations. <em>Organization Science</em>.</li>
        <li>Denrell and March (2001). Adaptation as information restriction: The hot stove effect. <em>Organization Science</em>.</li>
        <li>Gaba and Joseph (2013). Corporate structure and performance feedback: Aspirations and adaptation in M-form firms. <em>Organization Science</em>.</li>
        <li>Greve (2003). <em>Organizational Learning from Performance Feedback</em>. Cambridge University Press.</li>
        <li>Levinthal (1997). Adaptation on rugged landscapes. <em>Management Science</em>.</li>
        <li>Levinthal and Pham (2024). Bringing politics back in: The role of power and coalitions in organizational adaptation. <em>Organization Science</em>.</li>
        <li>March (1991). Exploration and exploitation in organizational learning. <em>Organization Science</em>.</li>
        <li>Posen and Levinthal (2012). Chasing a moving target: Exploitation and exploration in dynamic environments. <em>Management Science</em>.</li>
        <li>Puranam and Swamy (2016). How initial representations shape coupled learning processes. <em>Organization Science</em>.</li>
        <li>Rivkin and Siggelkow (2003). Balancing search and stability: Interdependencies among elements of organizational design. <em>Management Science</em>.</li>
        <li>Siggelkow and Levinthal (2003). Temporarily divide to conquer: Centralized, decentralized, and reintegrated organizational approaches to exploration and adaptation. <em>Organization Science</em>.</li>
        <li>Tushman and Scanlan (1981). Boundary spanning individuals: Their role in information transfer and their antecedents. <em>Academy of Management Journal</em>.</li>
      </ul>
    </div>
  </div>
</div>

<!-- Large organization editor; the editor block moves here while it is open -->
<div class="overlay" id="editor-overlay" hidden>
  <div class="overlay-panel" role="dialog" aria-modal="true" aria-labelledby="overlay-title">
    <div class="overlay-head">
      <strong id="overlay-title">Organization editor: <span id="overlay-config-name"></span></strong>
      <button type="button" class="btn" id="btn-close-overlay">Close</button>
    </div>
    <div class="overlay-slot" id="overlay-slot"></div>
  </div>
</div>

<!-- Shared help popover opened by every "?" button -->
<div class="help-pop" id="help-pop" role="tooltip" hidden></div>

<script>
(function() {
  "use strict";

  let CATALOG = null;
  let activeTopView = "lab";
  let abortCtrl = null;

  // File format written by Export JSON and read by Import JSON and `cli.py run`.
  const EXPORT_SCHEMA_VERSION = "0.1.0";
  // Matches engine.MAX_CONFIGURATIONS.
  const MAX_CONFIGS = 8;
  // Stable configuration keys: they fix each configuration's color and its slot id in comparison runs.
  const CONFIG_KEYS = ["a", "b", "c", "d", "e", "f", "g", "h"];
  const CONFIG_COLORS = {
    a: { stroke: "#1d4ed8", band: "rgba(29, 78, 216, 0.12)" },
    b: { stroke: "#d97706", band: "rgba(217, 119, 6, 0.12)" },
    c: { stroke: "#15803d", band: "rgba(21, 128, 61, 0.12)" },
    d: { stroke: "#be123c", band: "rgba(190, 18, 60, 0.12)" },
    e: { stroke: "#7c3aed", band: "rgba(124, 58, 237, 0.12)" },
    f: { stroke: "#0f766e", band: "rgba(15, 118, 110, 0.12)" },
    g: { stroke: "#92400e", band: "rgba(146, 64, 14, 0.12)" },
    h: { stroke: "#64748b", band: "rgba(100, 116, 139, 0.12)" }
  };

  // Template groups in the template selector; keys missing from the catalog are skipped.
  const TEMPLATE_GROUPS = [
    {
      label: "Two-level hierarchies",
      keys: ["coordinated_neural_hierarchy", "conformity_trap", "bundled_neural_hierarchy", "incommensurable_silos", "adaptive_meritocracy"]
    },
    {
      label: "Other structures",
      keys: ["m_form_3_tier", "matrix_dual_authority", "ambidextrous_org", "hybrid_human_ai", "boundary_spanners"]
    },
    {
      label: "Coalition alone (benchmarks)",
      keys: ["enacted_solo_dc", "active_solo_dc"]
    }
  ];

  // Market fields the UI controls; market_config sends exactly these keys.
  const MARKET_DEFAULTS = {
    n_bits: 12,
    nk_k: 3,
    nk_pattern: "random",
    sigma_obs: 0.65,
    turbulence: "none",
    p_shock: 0.02,
    shock_period: 100,
    t_steps: 600,
    eval_every: 25,
    b_firms: 50,
    seed: 2026
  };
  const MIN_N_BITS = 2;
  const MAX_N_BITS = 16;
  const MAX_FIRMS = 2048;
  // Input ranges of the market bar; imported files are held to the same ranges.
  const MIN_HORIZON = 10;
  const MAX_HORIZON = 20000;
  const MAX_SHOCK_PERIOD = 20000;
  const MAX_P_SHOCK = 0.1;
  const MAX_SEED = 4294967295;
  // Organization-level fields of OrgGraph and their defaults.
  const ORG_SETTING_DEFAULTS = {
    feedback_regime: "dc_firm_units_domain",
    decision_rights: "centralized",
    experiment_rate: 0.5,
    aspiration_sensitivity: 0.0,
    lambda_track: 0.0,
    unit_initial_accuracy: null
  };
  const ALLOWED_MODEL_TYPES = ["neural", "linear", "tabular"];
  // Explicit per-node value for random starting beliefs; null on a non-DC node follows the organization default.
  const NODE_UNINFORMED = "uninformed";

  const FEEDBACK_REGIME_LABELS = {
    dc_firm_units_domain: "DC sees firm results, units see their own domain",
    everyone_firm: "Everyone sees firm results",
    dc_only: "Only the DC sees results"
  };
  const FEEDBACK_REGIME_HINTS = {
    dc_firm_units_domain: "The coalition observes the firm payoff; every other node observes only the payoff contribution of the choices in its own domain.",
    everyone_firm: "Every node observes the same noisy firm payoff after each period.",
    dc_only: "Only the coalition observes market results; the other nodes learn only from colleagues."
  };
  // Signal a non-DC node observes when it follows the organization setting.
  const REGIME_UNIT_FEEDBACK = { dc_firm_units_domain: "domain", everyone_firm: "firm", dc_only: "none" };

  const DECISION_RIGHTS_LABELS = {
    centralized: "Coalition decides",
    propose_approve: "Units propose, coalition approves",
    delegated: "Units decide in their domains"
  };
  const DECISION_RIGHTS_HINTS = {
    centralized: "The coalition searches for and adopts strategy changes on its own.",
    propose_approve: "Each other node searches its own domain the way the coalition does (with its own reach and search steps) and proposes changes it predicts will pay off; the coalition adopts the proposal, or combination of proposals, it predicts is best.",
    delegated: "Each other node searches its own domain the way the coalition does (with its own reach and search steps) and makes the changes it predicts will pay off; the coalition searches only over choices that no other node covers."
  };

  const NODE_FEEDBACK_SHORT = { firm: "firm results", domain: "own-domain results", none: "no market results" };
  const FEEDBACK_BADGE = {
    firm: { letter: "F", color: "#1d4ed8" },
    domain: { letter: "D", color: "#15803d" },
    none: { letter: "", color: "#71717a" }
  };

  // Texts of the "?" help popovers: self-contained plain-language definitions, mechanism distinctions, and reference values.
  const HELP = {
    files: ["Experiment files", "Export JSON saves the market environment, every configuration with its canvas layout, and the latest simulation results (schema 0.1.0). Import JSON restores a saved file into the laboratory, and the command-line runner (python3 cli.py run file.json) executes the exact same file without a browser."],
    configurations: ["Configurations", "Each configuration is one organizational design (its nodes, communication links, and organization settings). You can compare up to 8 configurations side by side. All configurations in a run face the exact same NK landscapes, observation noise draws, and environmental shocks, enabling paired statistical comparisons across iterations (firms)."],
    experiments: ["Ready-made experiments", "Loads a pre-configured set of organizational designs that test a classic research question (such as feedback regimes, delegation on modular landscapes, or conformity traps) and clears previous results. Press Run Simulation after loading to compare them."],
    n_bits: ["Choices N", "Number of binary strategic choices (0 or 1) that define a firm's strategy (2 to 16, default 12). With N choices there are 2^N possible strategies (256 at N = 8, 4,096 at N = 12, and 65,536 at N = 16). Changing N proportionally rescales each node's domain and caps interdependence K at N - 1."],
    nk_k: ["Interdependence K", "Number of other choices that epistatically affect each choice's payoff contribution (0 to N - 1). K controls landscape ruggedness: K = 0 produces a smooth, single-peaked additive landscape where local search can find the global optimum; K = 2 to 4 produces moderate ruggedness with multiple local peaks; K = 6 or higher creates a highly rugged landscape with many local traps."],
    nk_pattern: ["Interactions", "How the K interdependent choices are wired on the NK landscape. Random: each choice depends on K randomly selected other choices across the whole firm. Modular: choices are grouped into self-contained blocks of size K + 1, so interdependencies stay inside modules unless K + 1 does not divide N evenly (Rivkin and Siggelkow 2003)."],
    sigma_obs: ["Feedback noise", "External measurement noise (standard deviation) added to the true landscape payoff when the market reveals results in Step 4. Because true payoffs are standardized to SD = 1.0 across strategies, 0.00 is noiseless feedback, 0.30 is low noise, 0.65 (default) is moderate noise, and 1.00+ means single-period noise is as large as the landscape's true payoff variation. Unlike Learning noise, Feedback noise is shared market ambiguity rather than internal cognitive error."],
    turbulence: ["Turbulence", "Whether and how the environment changes during a run (Step 1). None: the NK landscape remains static. Random: in each period, a shock hits with probability p. Periodic: a shock hits every T periods. Each shock redraws roughly half of the landscape's payoff contributions while preserving N and K, making previously learned mental models partially obsolete (Posen and Levinthal 2012)."],
    p_shock: ["Probability per period", "Under Random turbulence, the probability that an environmental shock hits in any given period (0.00 to 0.10). Reference values: 0.005 is rare change (roughly one shock every 200 periods), 0.02 (default) is moderate turbulence (roughly one shock every 50 periods), and 0.05+ is frequent turbulence."],
    shock_period: ["Every T periods", "Under Periodic turbulence, the exact interval between environmental shocks (1 to 20,000, default 100). Shocks hit at periods T, 2T, 3T, and so on. Shorter intervals (such as 25 or 50) prevent representations from fully settling before the landscape shifts; longer intervals (100 to 500) allow organizations to converge between shocks."],
    b_firms: ["Iterations (firms)", "Number of independent Monte Carlo simulation replications (1 to 2,048, default 50). Each iteration draws a fresh NK landscape and starting strategy, and runs all configurations in parallel on that exact same landscape. More iterations reduce the standard error of means and paired differences (which shrinks with the square root of the number of iterations): use 20 for quick exploration, 50 for standard checks, and 100 to 500 for publication-grade statistical power."],
    t_steps: ["Horizon", "Total number of simulation periods per run (10 to 20,000, default 600). Each period executes one full cycle of environmental shocks, colleague exchange, strategy search, and market feedback. Typical values: 80 to 150 periods capture short-run adaptation; 600 periods (default) capture medium-run convergence; 2,000 to 20,000 periods test long-run ergodic behavior or repeated turbulence cycles."],
    eval_every: ["Log every", "Interval in periods (1 to 20,000, default 25) at which full-landscape epistemic metrics (Global Accuracy, Representational Complexity, and Representational Heterogeneity) are evaluated across all 2^N strategies. Payoff V(s_t), Local Gradient Correlation, and False Belief Stall Rate are always computed every period. Use 25 (default) for fast runs, 1 to inspect every single period on short horizons, or 50 to 100 on very long horizons (5,000+ periods)."],
    seed: ["Seed", "Master pseudo-random number generator seed (0 to 4,294,967,295, default 2026) shared across all configurations. Holding the seed fixed guarantees identical NK landscapes, initial strategies, observation noise draws, and environmental shocks across runs."],
    template: ["Template", "Replaces the active configuration with a pre-built organizational architecture (nodes, communication links, and organization settings). After loading a template, you can freely modify any node, link, or setting."],
    feedback_regime: ["Who sees market results", "Controls which nodes directly observe market payoffs in Step 4 after the firm acts. 'DC sees firm results, units see their own domain' (default): the Dominant Coalition observes the whole-firm payoff while specialists observe the average payoff contribution of choices in their own domain. 'Everyone sees firm results': every node observes the whole-firm payoff. 'Only the DC sees results': specialists receive no direct market feedback and learn only from colleagues during Step 2 exchange (March 1991). Individual nodes can override this default."],
    decision_rights: ["Decision rights", "Defines how strategic choices are searched and enacted in Step 3. 'Coalition decides' (centralized, default): the Dominant Coalition searches and enacts strategy changes using its own mental model. 'Units propose, coalition approves': each unit searches its own domain and proposes improvements, and the DC screens and approves the combination its own model rates highest (Rivkin and Siggelkow 2003). 'Units decide in their domains' (delegated): units autonomously enact changes in their domains without DC screening, and the DC searches only unclaimed choices (Siggelkow and Levinthal 2003)."],
    initial_accuracy: ["Initial accuracy", "How accurate a node's prior mental model is before Period 1 (0.00 to 1.00, or Uninformed). 'Uninformed' (default) initializes weights randomly near zero. Setting an accuracy a fits the node's initial representation to a mixture of the true observable payoff and an independent misleading NK landscape so that its initial correlation with the explainable part of its observed payoff is approximately a (scaled to 0.5 SD so market experience can revise it). Reference values: 0.20 is poorly informed, 0.50 is moderately informed, and 0.70 to 0.90 represents strong domain expertise."],
    experiment_rate: ["Unit experiments", "Probability per period (0.00 to 1.00, default 0.50) that each unit that sees market results tests one random single-choice change in its domain off-line, like a pilot or prototype. The unit observes the noisy result of the test and updates its own mental model; the firm's strategy does not change. Testing lets units learn about changes the organization has not adopted. 0.00 turns testing off; 0.50 is about one test every other period; 1.00 is one test every period."],
    aspiration_sensitivity: ["Search after shortfalls", "Problemistic search intensity kappa (0.00 to 5.00, default 0.00; Greve 2003). The Dominant Coalition tracks a historical aspiration level A (exponentially weighted moving average of observed payoffs, weight 0.10). When the latest observed payoff y falls below A, the DC increases its distant-search probability to min(1, rho + kappa * max(0, A - y)). Reference values: 0.00 disables shortfall-triggered search, 0.25 is mild sensitivity, and 1.00+ triggers frequent distant jumps after performance drops."],
    lambda_track: ["Track-record learning", "Rate lambda (0.00 to 5.00, default 0.00) at which nodes dynamically reweight incoming communication links based on relative prediction accuracy in Step 4. After observing a market payoff, a learner compares its own squared prediction error against each linked colleague's squared prediction error and multiplies that link's Permeability P by exp(lambda * (own_error - source_error)), clamped to [0.001, 1.0]. Reference values: 0.00 keeps link weights fixed, 1.00 adapts weights gradually, and 2.50+ shifts attention rapidly toward accurate advisers."],
    count: ["Headcount", "Number of parallel agent representations with identical settings inside this node (1 to 32, default 1; fixed at 1 for the Dominant Coalition). When Headcount is above 1, each member learns with independent noise draws and their predictions are averaged when communicating or proposing."],
    model_type: ["Representation", "Cognitive architecture used by this node to approximate the payoff landscape. 'Neural network' (default): a two-layer ReLU network with H hidden units that can learn nonlinear interactions across choices while generalizing across similar strategies. 'Linear (additive)': a main-effects model (H = 0) that assumes choices do not interact. 'Lookup table': memorizes a separate value for every combination of choices in its domain with zero generalization to unvisited strategies."],
    h_agent: ["Capacity (H)", "Number of hidden units in a neural representation (4, 8, 16, 32, or 64; 0 for Linear). Higher capacity allows the node to represent more complex epistatic interactions (higher K), while smaller capacity forces cognitive simplification and smoother generalization. Reference values: H = 4 (Simplifier), H = 16 (Moderate, default), H = 64 (Complexifier)."],
    domain: ["Domain: choices it attends to", "Subset of strategic choices (bits 1 to N) that this node observes in its representation, searches over when proposing or deciding, and receives domain payoff feedback on. The node is blind to choices outside its domain and implicitly treats them as fixed."],
    node_feedback: ["Sees market results", "Per-node override for which market payoff signal this node observes in Step 4. 'Organization default' follows the organization-wide 'Who sees market results' setting; 'Firm results' observes the whole-firm payoff; 'Own-domain results' observes the average payoff contribution of choices in this node's domain; 'None' receives no market feedback (learns only from colleagues). The Dominant Coalition always observes firm results."],
    lr: ["Market learning rate", "Step size (0.01 to 1.00, default 0.14 to 0.16) for updating this node's representation toward observed market payoffs and unit experiments in Step 4 (normalized LMS gradient step). Importantly, this parameter scales learning from market results; learning from colleagues during Step 2 exchange is governed by incoming link Permeability P. Reference values: 0.05 is slow/cautious learning, 0.15 is balanced (default), and 0.35+ reacts aggressively to recent noisy payoffs."],
    sigma_learn: ["Learning noise", "Internal cognitive noise (standard deviation, 0.00 to 1.50, default 0.05) added inside this node when it updates its representation from market or experiment results in Step 4. Unlike Market Feedback noise (which is external ambiguity in the market signal shared by observers), Learning noise represents idiosyncratic human or algorithmic update imperfection inside a specific node. Reference values: 0.00 is noise-free updating (typical for AI analysts), 0.05 is standard low cognitive noise, and 0.20+ is noisy/erratic updating."],
    rho: ["Distant search (rho)", "Probability (0.00 to 1.00, default 0.00) that a candidate strategy change evaluated by this node in Step 3 flips multiple choices in its domain simultaneously (2 to 4 choices) rather than a single choice (local 1-bit search). Reference values: 0.00 is pure local hill-climbing, 0.10 to 0.20 mixes occasional long jumps with local refinement, and 0.50+ is predominantly distant exploration."],
    search_steps: ["Search choices per period", "Number of sequential candidate strategy changes this node evaluates per period when it searches in Step 3 (1 to N, default 1). At each step, the node samples a candidate change in its domain and adopts or proposes it only if its own mental model predicts a strictly higher payoff than the current baseline."],
    links: ["Links", "Directed communication channels connected to this node. 'Learns from' lists incoming links (sources whose views shape this node in Step 2); 'Sends its views to' lists outgoing links (learners influenced by this node). You can edit exact Permeability P and Commensurability C values here, or click/drag links directly on the canvas."],
    upward_agg: ["Combining several sources", "Rule this node uses in Step 2 to aggregate predictions when it learns from multiple incoming links. 'Permeability-weighted mean' (default) blends sources in proportion to their link Permeability P; 'Most optimistic source (max)' follows the highest predicted payoff; 'Most skeptical source (min)' follows the lowest predicted payoff."],
    permeability: ["Permeability P", "Directional social influence weight (0.00 to 1.00, default 0.25 to 0.40) on a link from source A to learner B. During Step 2 exchange, B updates its representation toward A's predictions on sampled strategies with effective step size proportional to P. Reference values: 0.00 means no influence, 0.15 is light advisory input, 0.35 is standard organizational coupling, and 0.60+ is strong socialization that rapidly aligns B's beliefs with A's."],
    commensurability: ["Commensurability C", "Shared cognitive translation fidelity (0.00 to 1.00, default 0.85) between a pair of linked nodes. When C = 1.00, views pass without distortion. Lower C introduces a persistent pairwise translation bias plus communication noise during Step 2 exchange, modeling jargon barriers or incompatible specialist languages. Links with C below 0.60 are drawn dashed. Reference values: 0.85 to 1.00 is high shared coding, 0.50 is moderate friction, and 0.20 to 0.35 represents incommensurable silos."],
    m_v: ["Enacted payoff V(s_t)", "True landscape payoff of the strategy enacted by the firm in period t, standardized across all 2^N strategies on each landscape (mean 0.0, standard deviation 1.0). Shaded bands show +/- 1 standard error across iterations (firms). Higher curves indicate superior organizational performance."],
    payoff_refs: ["Reference lines", "Two benchmark lines computed on the exact same landscapes: 'Best strategy' (dashed dark gray) is the global maximum payoff (100th percentile). 'Local search with perfect knowledge' (dotted light gray) is the average payoff of true-landscape 1-bit greedy local peaks; organizations rising above the dotted line are outperforming noiseless myopic local hill-climbing."],
    m_acc: ["Global epistemic accuracy", "Pearson correlation (-1.0 to +1.0) between the Dominant Coalition's mental model predictions and the true landscape payoffs across all 2^N possible strategies. Measures how well leadership understands the entire strategic landscape, including unvisited regions. Evaluated every 'Log every' periods."],
    m_comp: ["Representational complexity", "Number of strategies (out of 2^N) that the Dominant Coalition's representation perceives as local peaks (rated strictly higher than all N single-choice neighbors). Measures how rugged or multi-peaked the landscape looks inside the DC's mental model: 1.0 means a single-peaked linear/smooth belief surface, whereas higher values indicate a rugged internal representation with multiple perceived peaks."],
    m_orth: ["Representational heterogeneity", "Average cognitive dissimilarity (1 - |corr|) across all pairs of representations in the organization (including the DC), evaluated over all 2^N strategies. A value near 0.00 means all units hold virtually identical (or mirror-image) beliefs (high conformity); a value near 1.00 means units hold uncorrelated, diverse perspectives on how strategies pay off."],
    m_r: ["Local gradient correlation", "Pearson correlation (-1.0 to +1.0) between the Dominant Coalition's predicted payoff changes and the true payoff changes across the N single-choice neighbor moves from the firm's current strategy. Measures local directional accuracy right around the enacted strategy in every period."],
    m_c6: ["Sixth chart & False Belief Stall Rate", "Displays the metric selected in the dropdown. By default, shows False Belief Stall Rate: the share of firms stuck on a suboptimal strategy because of a false belief, where the Dominant Coalition believes no single-choice change can improve performance (so local search stalls) even though a better single-choice change actually exists on the true landscape. Use the dropdown to inspect Share of strategies beaten, Heterogeneity (linked pairs), Share of firms at a true local peak, Mean Permeability, Subunit Slope Diversity, or % of Landscape Max."]
  };

  function humanizeOption(val) {
    return String(val).replace(/_/g, " ").replace(/\\b\\w/g, c => c.toUpperCase());
  }

  // Ready-made experiments: each replaces the configurations with a complete set. Market fields not
  // listed reset to K = 3, random interactions, and no turbulence; noise, horizon, firms, and seed stay.
  const EXPERIMENTS = [
    {
      id: "feedback_regimes",
      title: "Who sees market results",
      text: "The coordinated hierarchy under the three feedback regimes: units see their own domain, everyone sees firm results, or only the DC sees results.",
      anchors: "March 1991; Gaba and Joseph 2013",
      market: {},
      configs: [
        { name: "Units see their own domain", template: "coordinated_neural_hierarchy", settings: { feedback_regime: "dc_firm_units_domain" } },
        { name: "Everyone sees firm results", template: "coordinated_neural_hierarchy", settings: { feedback_regime: "everyone_firm" } },
        { name: "Only the DC sees results", template: "coordinated_neural_hierarchy", settings: { feedback_regime: "dc_only" } }
      ]
    },
    {
      id: "decision_rights",
      title: "Decision rights",
      text: "The coordinated hierarchy under the three decision-rights modes on a modular NK landscape (K = 2), where each specialist's domain is a union of interaction blocks.",
      anchors: "Rivkin and Siggelkow 2003; Siggelkow and Levinthal 2003",
      market: { nk_k: 2, nk_pattern: "modular" },
      configs: [
        { name: "Coalition decides", template: "coordinated_neural_hierarchy", settings: { decision_rights: "centralized" } },
        { name: "Units propose", template: "coordinated_neural_hierarchy", settings: { decision_rights: "propose_approve" } },
        { name: "Units decide", template: "coordinated_neural_hierarchy", settings: { decision_rights: "delegated" } }
      ]
    },
    {
      id: "turbulence_reach",
      title: "Cognitive reach under turbulence",
      text: "The coordinated hierarchy with the coalition's search reach rho at 0, 0.15, 0.30, and 0.50 under random turbulence (probability 0.02 per period); set turbulence to None for the stable benchmark.",
      anchors: "Levinthal 1997; Posen and Levinthal 2012",
      market: { turbulence: "random", p_shock: 0.02 },
      configs: [0, 0.15, 0.3, 0.5].map(r => ({ name: "rho " + r.toFixed(2), template: "coordinated_neural_hierarchy", dcRho: r }))
    },
    {
      id: "accuracy_conformity",
      title: "Initial accuracy and conformity",
      text: "A 2 x 2 design: the coordinated hierarchy and the conformity trap, each with uninformed units and with units at initial accuracy 0.7.",
      anchors: "March 1991; Puranam and Swamy 2016",
      market: {},
      configs: [
        { name: "Coordinated, uninformed units", template: "coordinated_neural_hierarchy", settings: { unit_initial_accuracy: null } },
        { name: "Coordinated, units at 0.7", template: "coordinated_neural_hierarchy", settings: { unit_initial_accuracy: 0.7 } },
        { name: "Conformity trap, uninformed units", template: "conformity_trap", settings: { unit_initial_accuracy: null } },
        { name: "Conformity trap, units at 0.7", template: "conformity_trap", settings: { unit_initial_accuracy: 0.7 } }
      ]
    },
    {
      id: "boundary_spanners",
      title: "Boundary spanners versus everyone sees results",
      text: "Front-line units see their own domain and internal units see nothing, against the same structure in which every node sees firm results.",
      anchors: "Tushman and Scanlan 1981",
      market: {},
      configs: [
        { name: "Boundary spanners", template: "boundary_spanners" },
        { name: "Everyone sees firm results", template: "boundary_spanners", settings: { feedback_regime: "everyone_firm" }, clearFeedback: true }
      ]
    },
    {
      id: "track_record_periodic",
      title: "Track-record learning under periodic turbulence",
      text: "The adaptive meritocracy with track-record learning lambda at 0, 1, and 2.5 under periodic turbulence (a shock every 100 periods).",
      anchors: "Posen and Levinthal 2012",
      market: { turbulence: "periodic", shock_period: 100 },
      configs: [0, 1, 2.5].map(l => ({ name: "lambda " + l, template: "adaptive_meritocracy", settings: { lambda_track: l } }))
    }
  ];

  // Shared market environment of every configuration.
  const market = Object.assign({}, MARKET_DEFAULTS);

  // Configurations in display order; activeIdx points at the one shown in the editor.
  const configs = [];
  let activeIdx = 0;

  // Results of the last run in run order: [{key, uid, label, traj, summary, realized, paired}].
  // running: a run is streaming; stopped: the user pressed Stop before the run completed.
  const results = {
    t_steps: 600,
    entries: [],
    complete: null,
    running: false,
    stopped: false
  };
  // Results are stale once the market, a configuration's design, or the set of configurations changes after a run.
  let resultsStale = false;
  // Every configuration gets a unique id, so results never attach to a later configuration that reuses a key.
  let configUid = 0;

  function cloneObj(obj) {
    return obj ? JSON.parse(JSON.stringify(obj)) : null;
  }

  function makeConfig(key, name) {
    configUid += 1;
    return {
      key: key,
      uid: configUid,
      name: name,
      template_key: "",
      org_graph: null,
      layout: {},
      sel_node: "hq",
      sel_edge: null,
      undo: [],
      compiled: null,
      compiled_key: "",
      error: "",
      notes: [],
      view: null
    };
  }

  function activeConfig() {
    return configs[activeIdx] || configs[0];
  }

  function nextConfigKey() {
    return CONFIG_KEYS.find(k => !configs.some(c => c.key === k)) || null;
  }

  function nextConfigName() {
    let n = configs.length + 1;
    while (configs.some(c => c.name === "Configuration " + n)) n += 1;
    return "Configuration " + n;
  }

  function configColor(cfg) {
    return (CONFIG_COLORS[cfg.key] || CONFIG_COLORS.a).stroke;
  }

  // Trajectory arrays kept per configuration; chunk keys the engine does not send stay empty.
  function emptyTraj() {
    return {
      steps: [],
      v_true_mean: [],
      v_true_se: [],
      v_pct_max_mean: [],
      v_percentile_mean: [],
      v_max_ref_mean: [],
      v_local_ref_mean: [],
      dc_r_local_mean: [],
      false_stall_mean: [],
      at_local_peak_mean: [],
      emp_div_local_slope_mean: [],
      dc_acc_global_mean: [],
      dc_acc_global_se: [],
      dc_complexity_mean: [],
      dc_complexity_se: [],
      orthogonality_mean: [],
      orthogonality_se: [],
      orthogonality_linked_mean: [],
      orthogonality_linked_se: [],
      mean_permeability_mean: []
    };
  }

  function clampNum(raw, lo, hi, fallback) {
    const v = parseFloat(raw);
    if (!isFinite(v)) return fallback;
    return Math.max(lo, Math.min(hi, v));
  }

  function clampInt(raw, lo, hi, fallback) {
    const v = parseInt(raw, 10);
    if (!isFinite(v)) return fallback;
    return Math.max(lo, Math.min(hi, v));
  }

  function currentNBits() {
    return clampInt(market.n_bits, MIN_N_BITS, MAX_N_BITS, MARKET_DEFAULTS.n_bits);
  }

  function allBits(n) {
    const count = n === undefined ? currentNBits() : n;
    return Array.from({ length: count }, (_, i) => i);
  }

  function firstHalfBits(n) {
    const count = n === undefined ? currentNBits() : n;
    const h = Math.floor(count / 2);
    return Array.from({ length: h }, (_, i) => i);
  }

  function secondHalfBits(n) {
    const count = n === undefined ? currentNBits() : n;
    const h = Math.floor(count / 2);
    return Array.from({ length: count - h }, (_, i) => h + i);
  }

  function formatBitsRuns(bits) {
    const sorted = Array.from(new Set(bits)).sort((a, b) => a - b);
    if (sorted.length === 0) return "1";
    const runs = [];
    let start = sorted[0];
    let prev = sorted[0];
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i] === prev + 1) {
        prev = sorted[i];
      } else {
        runs.push(start === prev ? String(start + 1) : (start + 1) + "-" + (prev + 1));
        start = sorted[i];
        prev = sorted[i];
      }
    }
    runs.push(start === prev ? String(start + 1) : (start + 1) + "-" + (prev + 1));
    return runs.join(", ");
  }

  function rescaleJurisdictionBits(bits, oldN, newN) {
    const b = Array.isArray(bits)
      ? Array.from(new Set(bits.map(x => parseInt(x, 10)).filter(x => isFinite(x)))).sort((a, c) => a - c)
      : [];
    if (oldN === newN && b.length > 0 && b.every(x => x >= 0 && x < newN)) {
      return b;
    }
    if (b.length === 0 || (b.length === oldN && b.every((x, i) => x === i))) {
      return allBits(newN);
    }
    const isContig = b.every((x, i) => i === 0 || x === b[i - 1] + 1);
    if (isContig) {
      const lo = Math.max(0, Math.min(newN - 1, Math.round(b[0] * newN / oldN)));
      const hi = Math.max(lo + 1, Math.min(newN, Math.round((b[b.length - 1] + 1) * newN / oldN)));
      return Array.from({ length: hi - lo }, (_, i) => lo + i);
    }
    const denom = Math.max(1, oldN - 1);
    const mapped = Array.from(new Set(b.map(x => Math.max(0, Math.min(newN - 1, Math.round(x * (newN - 1) / denom)))))).sort((a, c) => a - c);
    return mapped.length > 0 ? mapped : allBits(newN);
  }

  function rescaleGraphBits(g, oldN, newN) {
    if (!g || !Array.isArray(g.nodes)) return g;
    for (const n of g.nodes) {
      n.jurisdiction_bits = rescaleJurisdictionBits(n.jurisdiction_bits, oldN, newN);
      if (typeof n.label === "string" && /\\(Bits [^)]+\\)/.test(n.label)) {
        n.label = n.label.replace(/\\(Bits [^)]+\\)/, "(Bits " + formatBitsRuns(n.jurisdiction_bits) + ")");
      }
      if (Number(n.search_steps) > newN) {
        n.search_steps = newN;
      }
    }
    if (typeof g.description === "string") {
      const oldH = Math.floor(oldN / 2);
      const newH = Math.floor(newN / 2);
      const oldFirst = oldH === 1 ? "1" : "1-" + oldH;
      const oldSecond = oldH + 1 === oldN ? String(oldN) : (oldH + 1) + "-" + oldN;
      const newFirst = newH === 1 ? "1" : "1-" + newH;
      const newSecond = newH + 1 === newN ? String(newN) : (newH + 1) + "-" + newN;
      g.description = g.description
        .split("Bits " + oldFirst).join("__BITS_FIRST__")
        .split("Bits " + oldSecond).join("Bits " + newSecond)
        .split("__BITS_FIRST__").join("Bits " + newFirst)
        .split(oldH + " decisions each").join(newH + " decisions each");
    }
    return g;
  }

  // Organization-level setting, or its default when the graph does not set it.
  function orgSetting(g, key) {
    const v = g ? g[key] : undefined;
    return (v === undefined || v === null || v === "") ? ORG_SETTING_DEFAULTS[key] : v;
  }

  function isFeedbackValue(v) {
    return v === "firm" || v === "domain" || v === "none";
  }

  // Signal a non-DC node observes under the organization's feedback regime.
  function regimeFeedback(g) {
    return REGIME_UNIT_FEEDBACK[orgSetting(g, "feedback_regime")] || "domain";
  }

  // Signal a node observes: the DC always sees firm results; other nodes follow their own setting or the regime.
  function effectiveFeedback(node, g) {
    if (!node || node.role === "hq") return "firm";
    return isFeedbackValue(node.feedback) ? node.feedback : regimeFeedback(g);
  }

  // True when a node's domain includes every choice (or is unset, which means all choices).
  function coversAllChoices(node) {
    const nBits = currentNBits();
    const bits = node ? node.jurisdiction_bits : null;
    if (!Array.isArray(bits) || bits.length === 0) return true;
    return new Set(bits).size >= nBits;
  }

  // Signal shown for a node: own-domain results over every choice are the firm result.
  function displayFeedback(node, fb) {
    return (fb === "domain" && coversAllChoices(node)) ? "firm" : fb;
  }

  // The engine's explanation from the last compile, or null when the design changed since then.
  function freshExplanation(cfg) {
    const exp = cfg && cfg.compiled && cfg.compiled.graph_explanation;
    return (exp && cfg.compiled_key === JSON.stringify(cfg.org_graph)) ? exp : null;
  }

  // Effective feedback per node: the engine's resolution when fresh, otherwise the same rule computed locally.
  function nodeFeedbackMap(cfg) {
    const out = {};
    const g = cfg && cfg.org_graph;
    if (!g || !Array.isArray(g.nodes)) return out;
    const exp = freshExplanation(cfg);
    for (const n of g.nodes) {
      const serverFb = exp && exp.node_feedback ? exp.node_feedback[n.id] : undefined;
      out[n.id] = isFeedbackValue(serverFb) ? serverFb : effectiveFeedback(n, g);
    }
    return out;
  }

  function isAccuracyValue(v) {
    return v !== null && v !== undefined && v !== "" && v !== NODE_UNINFORMED && isFinite(Number(v));
  }

  // Effective initial accuracy (null = uninformed): the DC uses its own field; other nodes their own
  // value, "uninformed", or (when null) the organization default.
  function effectiveAccuracy(node, g) {
    if (!node || node.initial_accuracy === NODE_UNINFORMED) return null;
    if (isAccuracyValue(node.initial_accuracy)) return Number(node.initial_accuracy);
    if (node.role === "hq") return null;
    const orgAcc = g ? g.unit_initial_accuracy : null;
    return isAccuracyValue(orgAcc) ? Number(orgAcc) : null;
  }

  // Fill organization defaults and normalize the optional node fields of a template or a loaded graph.
  function normalizeGraph(g) {
    if (!g || !Array.isArray(g.nodes)) return g;
    for (const [k, v] of Object.entries(ORG_SETTING_DEFAULTS)) {
      if (g[k] === undefined || g[k] === "") g[k] = v;
    }
    if (!Array.isArray(g.edges)) g.edges = [];
    for (const n of g.nodes) {
      if (!isFeedbackValue(n.feedback)) n.feedback = null;
      if (!isAccuracyValue(n.initial_accuracy) && n.initial_accuracy !== NODE_UNINFORMED) n.initial_accuracy = null;
    }
    return g;
  }

  function clearNodeFeedbackOverrides(cfg) {
    if (!cfg || !cfg.org_graph || !Array.isArray(cfg.org_graph.nodes)) return;
    for (const n of cfg.org_graph.nodes) {
      n.feedback = null;
    }
    cfg.template_key = "";
  }

  // Market environment shared by all configurations, as MarketConfig fields.
  function buildMarketConfig() {
    return {
      n_bits: currentNBits(),
      nk_k: Number(market.nk_k),
      nk_pattern: market.nk_pattern,
      sigma_obs: Number(market.sigma_obs),
      turbulence: market.turbulence,
      p_shock: Number(market.p_shock),
      shock_period: Number(market.shock_period),
      t_steps: Number(market.t_steps),
      eval_every: Number(market.eval_every),
      b_firms: Number(market.b_firms),
      seed: Number(market.seed)
    };
  }

  // Chart update interval: about 40 updates per run, capped at 250 periods per chunk.
  function chunkSizeFor(tSteps) {
    return Math.max(5, Math.min(250, Math.round(Number(tSteps) / 40)));
  }

  function appendChunk(target, chunk) {
    if (!target || !chunk) return;
    for (const k of Object.keys(target)) {
      if (Array.isArray(chunk[k])) {
        target[k].push(...chunk[k]);
      }
    }
  }

  async function initApp() {
    try {
      const res = await fetch("/api/catalog");
      CATALOG = await res.json();
    } catch (err) {
      document.getElementById("compile-error-area").innerHTML =
        '<div class="error-box">Could not load the catalog from the server: ' + escapeHtml(String(err)) + "</div>";
      return;
    }
    const backend = CATALOG && CATALOG.devices ? CATALOG.devices.default_backend : "";
    setElText("device-active-badge", backend ? "Device: auto (" + backend + ")" : "");

    populateSelects();
    renderGuideTemplates();
    renderGuideExperiments();
    const first = makeConfig("a", "Configuration 1");
    configs.push(first);
    activeIdx = 0;
    applyTemplate(first, defaultTemplateKey(), null);
    bindEvents();
    initHelp();
    renderAll();
    resetResults();
    await compileConfig(first);
  }

  function defaultTemplateKey() {
    const tpls = (CATALOG && CATALOG.graph_templates) || {};
    return tpls.coordinated_neural_hierarchy ? "coordinated_neural_hierarchy" : (Object.keys(tpls)[0] || "");
  }

  const PENCIL_SVG =
    '<svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true">' +
    '<path d="M11.2 1.8a1.6 1.6 0 0 1 2.3 0l0.7 0.7a1.6 1.6 0 0 1 0 2.3L5.4 13.6 1.8 14.2l0.6-3.6z" ' +
    'fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/></svg>';

  function iconButton(html, label) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip-icon";
    btn.innerHTML = html;
    btn.setAttribute("aria-label", label);
    return btn;
  }

  // Configurations bar: one chip per configuration (color, name, rename, remove) before the add button.
  // When only the active configuration changes, the chips are updated in place so a double-click survives.
  function renderConfigBar() {
    const bar = document.getElementById("config-bar");
    const addBtn = document.getElementById("btn-add-config");
    if (!bar || !addBtn) return;
    const existing = Array.from(bar.querySelectorAll(".config-chip"));
    const signature = configs.map(c => c.key + ":" + c.uid + ":" + c.name).join("|");
    if (bar.dataset.signature === signature && existing.length === configs.length && !bar.querySelector(".config-name-input")) {
      existing.forEach((chip, idx) => {
        chip.classList.toggle("active", idx === activeIdx);
        chip.setAttribute("aria-pressed", idx === activeIdx ? "true" : "false");
      });
    } else {
      for (const old of existing) {
        old.remove();
      }
      configs.forEach((cfg, idx) => bar.insertBefore(buildConfigChip(cfg, idx), addBtn));
      bar.dataset.signature = signature;
    }
    addBtn.disabled = configs.length >= MAX_CONFIGS;
    setElText("active-config-name", activeConfig() ? activeConfig().name : "");
    setElText("overlay-config-name", activeConfig() ? activeConfig().name : "");
  }

  function buildConfigChip(cfg, idx) {
    const isActive = idx === activeIdx;
    const chip = document.createElement("div");
    chip.className = "config-chip" + (isActive ? " active" : "");
    chip.style.setProperty("--chip-color", configColor(cfg));
    chip.setAttribute("role", "button");
    chip.setAttribute("aria-pressed", isActive ? "true" : "false");
    chip.tabIndex = 0;
    chip.dataset.key = cfg.key;

    const swatch = document.createElement("span");
    swatch.className = "config-swatch";
    swatch.style.background = configColor(cfg);
    chip.appendChild(swatch);

    const nameEl = document.createElement("span");
    nameEl.className = "config-name";
    nameEl.textContent = cfg.name;
    nameEl.addEventListener("dblclick", (ev) => {
      ev.stopPropagation();
      startRename(cfg, nameEl);
    });
    chip.appendChild(nameEl);

    const renameBtn = iconButton(PENCIL_SVG, "Rename " + cfg.name);
    renameBtn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      startRename(cfg, nameEl);
    });
    chip.appendChild(renameBtn);

    if (configs.length > 1) {
      const removeBtn = iconButton("&times;", "Remove " + cfg.name);
      removeBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        removeConfig(cfg);
      });
      chip.appendChild(removeBtn);
    }

    chip.addEventListener("click", () => {
      const i = configs.indexOf(cfg);
      if (i >= 0 && i !== activeIdx) setActiveConfig(i);
    });
    chip.addEventListener("keydown", (ev) => {
      if (ev.target === chip && (ev.key === "Enter" || ev.key === " ")) {
        ev.preventDefault();
        const i = configs.indexOf(cfg);
        if (i >= 0) setActiveConfig(i);
      }
    });
    return chip;
  }

  // Inline rename: Enter or leaving the field keeps the new name; Escape cancels.
  function startRename(cfg, nameEl) {
    const input = document.createElement("input");
    input.type = "text";
    input.className = "config-name-input";
    input.value = cfg.name;
    input.maxLength = 80;
    input.setAttribute("aria-label", "Configuration name");
    nameEl.replaceWith(input);
    input.focus();
    input.select();
    let finished = false;
    const finish = (commit) => {
      if (finished) return;
      finished = true;
      const v = input.value.trim();
      if (commit && v) cfg.name = v;
      renderConfigBar();
      renderResults();
      updateDesignReadout();
    };
    input.addEventListener("keydown", (ev) => {
      ev.stopPropagation();
      if (ev.key === "Enter") finish(true);
      if (ev.key === "Escape") finish(false);
    });
    input.addEventListener("blur", () => finish(true));
    input.addEventListener("click", (ev) => ev.stopPropagation());
  }

  // "+ Add configuration" copies the active configuration.
  function addConfig() {
    const key = nextConfigKey();
    if (!key || configs.length >= MAX_CONFIGS) return;
    const src = activeConfig();
    const cfg = makeConfig(key, nextConfigName());
    cfg.template_key = src.template_key;
    cfg.org_graph = cloneObj(src.org_graph);
    cfg.layout = cloneObj(src.layout) || {};
    cfg.sel_node = src.sel_node;
    configs.push(cfg);
    markStale();
    setActiveConfig(configs.length - 1);
    renderResults();
  }

  // Removing a configuration drops it from the charts and tables and marks the remaining results stale.
  function removeConfig(cfg) {
    const idx = configs.indexOf(cfg);
    if (idx < 0 || configs.length <= 1) return;
    configs.splice(idx, 1);
    if (idx < activeIdx || activeIdx >= configs.length) activeIdx = Math.max(0, activeIdx - 1);
    markStale();
    setActiveConfig(activeIdx);
    renderResults();
  }

  function setActiveConfig(idx) {
    activeIdx = Math.max(0, Math.min(configs.length - 1, idx));
    closeEdgeEditor();
    renderAll();
    const cfg = activeConfig();
    if (!freshExplanation(cfg)) scheduleCompile(cfg, 0);
  }

  function templateTitle(key) {
    const tpl = CATALOG && CATALOG.graph_templates ? CATALOG.graph_templates[key] : null;
    const meta = CATALOG && CATALOG.graph_preset_metadata ? CATALOG.graph_preset_metadata[key] : null;
    return (meta && meta.title) || (tpl && tpl.name) || humanizeOption(key);
  }

  // Template keys in display order: the configured groups first, then any other catalog templates.
  function orderedTemplateGroups() {
    const allKeys = CATALOG && CATALOG.graph_templates ? Object.keys(CATALOG.graph_templates) : [];
    const placed = new Set();
    const groups = [];
    for (const grp of TEMPLATE_GROUPS) {
      const keys = grp.keys.filter(k => allKeys.includes(k));
      keys.forEach(k => placed.add(k));
      if (keys.length > 0) groups.push({ label: grp.label, keys: keys });
    }
    const rest = allKeys.filter(k => !placed.has(k));
    if (rest.length > 0) groups.push({ label: "Other templates", keys: rest });
    return groups;
  }

  function populateSelects() {
    const tplSel = document.getElementById("ctrl-graph-template");
    if (tplSel && CATALOG && CATALOG.graph_templates) {
      tplSel.innerHTML = "";
      const customTplOpt = document.createElement("option");
      customTplOpt.value = "";
      customTplOpt.textContent = "Edited organization (custom)";
      tplSel.appendChild(customTplOpt);
      for (const grp of orderedTemplateGroups()) {
        const grpEl = document.createElement("optgroup");
        grpEl.label = grp.label;
        for (const k of grp.keys) {
          const opt = document.createElement("option");
          opt.value = k;
          opt.textContent = templateTitle(k);
          grpEl.appendChild(opt);
        }
        tplSel.appendChild(grpEl);
      }
    }

    const expSel = document.getElementById("ctrl-experiment");
    if (expSel) {
      expSel.innerHTML = "";
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = "Load a ready-made experiment...";
      expSel.appendChild(placeholder);
      for (const exp of EXPERIMENTS) {
        const opt = document.createElement("option");
        opt.value = exp.id;
        opt.textContent = exp.title;
        expSel.appendChild(opt);
      }
    }
  }

  function renderGuideTemplates() {
    const tbody = document.getElementById("guide-template-tbody");
    if (!tbody || !CATALOG || !CATALOG.graph_templates) return;
    const meta = CATALOG.graph_preset_metadata || {};
    const rows = [];
    for (const grp of orderedTemplateGroups()) {
      for (const k of grp.keys) {
        const tpl = CATALOG.graph_templates[k] || {};
        const m = meta[k] || {};
        const desc = tpl.description || m.description || m.subtitle || "";
        rows.push(
          "<tr><td><strong>" + escapeHtml(templateTitle(k)) + "</strong></td>" +
          '<td class="mono">' + escapeHtml(k) + "</td>" +
          "<td>" + escapeHtml(desc) + "</td></tr>"
        );
      }
    }
    tbody.innerHTML = rows.join("");
  }

  function renderGuideExperiments() {
    const tbody = document.getElementById("guide-presets-tbody");
    if (!tbody) return;
    tbody.innerHTML = EXPERIMENTS.map(exp =>
      "<tr><td><strong>" + escapeHtml(exp.title) + "</strong></td>" +
      "<td>" + escapeHtml(exp.text) +
      ' <span class="guide-anchor">(' + escapeHtml(exp.anchors) + ")</span>" +
      '<div class="field-hint">Configurations: ' + escapeHtml(exp.configs.map(c => c.name).join("; ")) + "</div></td>" +
      '<td><button type="button" class="btn exp-load-btn" data-exp="' + escapeHtml(exp.id) + '">Load</button></td></tr>'
    ).join("");
    for (const btn of tbody.querySelectorAll(".exp-load-btn")) {
      btn.addEventListener("click", () => loadExperiment(btn.dataset.exp));
    }
  }

  function formatDomainBits(bits) {
    const nBits = currentNBits();
    if (!Array.isArray(bits) || bits.length === 0 || bits.length === nBits) {
      return "B1-" + nBits;
    }
    const sorted = bits.slice().sort((a, b) => a - b);
    const isContig = sorted.every((v, idx) => idx === 0 || v === sorted[idx - 1] + 1);
    if (isContig) {
      return sorted.length === 1 ? "B" + (sorted[0] + 1) : "B" + (sorted[0] + 1) + "-" + (sorted[sorted.length - 1] + 1);
    }
    return "B:" + sorted.map(b => b + 1).join(",");
  }

  function detectJurisdictionPreset(bits, n) {
    const nBits = n === undefined ? currentNBits() : n;
    if (!Array.isArray(bits)) return "all";
    const s = bits.slice().sort((a, b) => a - b).join(",");
    if (s === allBits(nBits).join(",")) return "all";
    if (s === firstHalfBits(nBits).join(",")) return "first_half";
    if (s === secondHalfBits(nBits).join(",")) return "second_half";
    return "custom";
  }
  const jurisdictionPresetForBits = detectJurisdictionPreset;

  function bitsForJurisdictionPreset(preset, n) {
    const nBits = n === undefined ? currentNBits() : n;
    if (preset === "first_half") return firstHalfBits(nBits);
    if (preset === "second_half") return secondHalfBits(nBits);
    return allBits(nBits);
  }

  function syncJurisdictionPresetOptions() {
    const sel = document.getElementById("node-jurisdiction-preset");
    if (!sel) return;
    const n = currentNBits();
    const h = Math.floor(n / 2);
    const firstRange = h === 1 ? "1" : "1-" + h;
    const secondRange = h + 1 === n ? String(n) : (h + 1) + "-" + n;
    const labels = {
      all: "All " + n + " (1-" + n + ")",
      first_half: "First half (" + firstRange + ")",
      second_half: "Second half (" + secondRange + ")",
      custom: "Custom"
    };
    for (const opt of sel.options) {
      if (labels[opt.value]) opt.textContent = labels[opt.value];
    }
  }

  function selectedNode(cfg) {
    const g = cfg && cfg.org_graph;
    if (!g || !Array.isArray(g.nodes) || g.nodes.length === 0) {
      return null;
    }
    const found = g.nodes.find(n => n.id === cfg.sel_node);
    if (found) return found;
    const hq = g.nodes.find(n => n.role === "hq") || g.nodes[0];
    cfg.sel_node = hq.id;
    return hq;
  }

  // Organization editor: node cards and link geometry in canvas units (1 unit is 1 pixel at full size).
  const CARD_W = 140;
  const CARD_H = 56;
  const COL_GAP = 20;
  const ROW_GAP = 18;
  const FIT_PAD = 24;
  // Vertical offset of a new node below the node it reports to.
  const LAYER_GAP = 112;
  // Auto-layout wraps wide layers into rows so the graph shrinks at most to this scale in the canvas.
  const MIN_LAYOUT_SCALE = 0.8;
  const ALL_BITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
  const CARD_STYLE = {
    hq: { fill: "#1f2a44", stroke: "#1f2a44", text: "#ffffff", sub: "#c9d2e3", on: "#e2e8f0", off: "#3a4763" },
    manager: { fill: "#4f6286", stroke: "#4f6286", text: "#ffffff", sub: "#dde4f0", on: "#ffffff", off: "#6c7ea0" },
    unit: { fill: "#f4f6fa", stroke: "#b6c0d1", text: "#1f2937", sub: "#5b6474", on: "#4f6286", off: "#dfe4ec" }
  };
  const SELECT_COLOR = "#2563eb";
  const EDGE_COLOR = "#475569";
  const EDGE_LIGHT = "#94a3b8";
  const LOW_C = 0.6;
  const UNDO_LIMIT = 30;

  // Pointer interaction in progress: dragging a node or drawing a link from a handle.
  const editorState = { drag: null, sliderSnap: null };

  function nodeById(g, id) {
    return g && Array.isArray(g.nodes) ? g.nodes.find(n => n.id === id) || null : null;
  }

  function nodeLabel(cfg, id) {
    const n = nodeById(cfg.org_graph, id);
    return n ? (n.label || n.id) : id;
  }

  function fmt2(v) {
    return Number(v).toFixed(2);
  }

  // Layered layout: the DC on top and every other node by its link distance from the DC (links in
  // either direction), ordered within layers by the barycenter of their neighbors to reduce crossings.
  function computeAutoLayout(g) {
    const nodes = g.nodes;
    const adj = {};
    for (const n of nodes) adj[n.id] = new Set();
    for (const e of g.edges) {
      if (adj[e.source] && adj[e.target]) {
        adj[e.source].add(e.target);
        adj[e.target].add(e.source);
      }
    }
    const layerOf = {};
    const hq = nodes.find(n => n.role === "hq") || nodes[0];
    if (hq) {
      layerOf[hq.id] = 0;
      const queue = [hq.id];
      while (queue.length > 0) {
        const cur = queue.shift();
        for (const nb of adj[cur]) {
          if (layerOf[nb] === undefined) {
            layerOf[nb] = layerOf[cur] + 1;
            queue.push(nb);
          }
        }
      }
    }
    let maxLayer = 0;
    for (const n of nodes) {
      if (layerOf[n.id] !== undefined) maxLayer = Math.max(maxLayer, layerOf[n.id]);
    }
    // Nodes without a path to the DC go one layer below the others.
    for (const n of nodes) {
      if (layerOf[n.id] === undefined) layerOf[n.id] = maxLayer + 1;
    }
    const nLayers = Math.max(1, ...nodes.map(n => layerOf[n.id] + 1));
    const layers = Array.from({ length: nLayers }, () => []);
    const roleRank = { hq: 0, manager: 1, unit: 2 };
    nodes.forEach((n, i) => {
      layers[layerOf[n.id]].push({ id: n.id, i: i, rank: roleRank[n.role] !== undefined ? roleRank[n.role] : 2, bc: 0 });
    });
    const posIdx = {};
    for (const layer of layers) {
      layer.sort((a, b) => a.rank - b.rank || a.i - b.i);
      layer.forEach((it, k) => { posIdx[it.id] = k; });
    }
    const sweep = (from, to, step) => {
      for (let L = from; L !== to; L += step) {
        const ref = L - step;
        for (const it of layers[L]) {
          const nbs = Array.from(adj[it.id]).filter(nb => layerOf[nb] === ref);
          it.bc = nbs.length > 0 ? nbs.reduce((s, nb) => s + posIdx[nb], 0) / nbs.length : posIdx[it.id];
        }
        layers[L].sort((a, b) => a.bc - b.bc || a.i - b.i);
        layers[L].forEach((it, k) => { posIdx[it.id] = k; });
      }
    };
    for (let pass = 0; pass < 4 && nLayers > 1; pass++) {
      sweep(1, nLayers, 1);
      sweep(nLayers - 2, -1, -1);
    }
    // Rows: wide layers wrap so the graph stays legible; the space between layers grows to use the canvas height.
    const size = canvasSize();
    const maxPerRow = Math.max(2, Math.floor((size.w / MIN_LAYOUT_SCALE - 2 * FIT_PAD + COL_GAP) / (CARD_W + COL_GAP)));
    const rowsPerLayer = layers.map(layer => Math.max(1, Math.ceil(layer.length / maxPerRow)));
    const blockH = rowsPerLayer.reduce((s, r) => s + r * CARD_H + (r - 1) * ROW_GAP, 0);
    const interLayer = nLayers > 1 ? Math.max(56, Math.min(110, (size.h - 2 * FIT_PAD - blockH) / (nLayers - 1))) : 0;
    const pos = {};
    let y = 0;
    layers.forEach((layer, L) => {
      for (let r = 0; r < rowsPerLayer[L]; r++) {
        const row = layer.slice(r * maxPerRow, (r + 1) * maxPerRow);
        const width = row.length * CARD_W + Math.max(0, row.length - 1) * COL_GAP;
        row.forEach((it, k) => {
          pos[it.id] = { x: Math.round(-width / 2 + k * (CARD_W + COL_GAP)), y: Math.round(y) };
        });
        y += CARD_H + ROW_GAP;
      }
      y += interLayer - ROW_GAP;
    });
    return pos;
  }

  function validPos(p) {
    return Boolean(p) && isFinite(p.x) && isFinite(p.y);
  }

  function overlapsCard(layout, p, skipId) {
    return Object.entries(layout).some(([id, q]) =>
      id !== skipId && Math.abs(q.x - p.x) < CARD_W + 10 && Math.abs(q.y - p.y) < CARD_H + 10);
  }

  // First free spot at or to the right of a preferred position.
  function freeSpot(layout, p) {
    const spot = { x: Math.round(p.x), y: Math.round(p.y) };
    let guard = 0;
    while (overlapsCard(layout, spot) && guard < 60) {
      spot.x += CARD_W + COL_GAP;
      guard += 1;
    }
    return spot;
  }

  // Give every node a position: auto-layout for a new graph, a free spot for nodes added later.
  function ensureLayout(cfg) {
    const g = cfg.org_graph;
    if (!g || !Array.isArray(g.nodes)) return;
    if (!cfg.layout || typeof cfg.layout !== "object") cfg.layout = {};
    for (const id of Object.keys(cfg.layout)) {
      if (!nodeById(g, id) || !validPos(cfg.layout[id])) delete cfg.layout[id];
    }
    const missing = g.nodes.filter(n => !cfg.layout[n.id]);
    if (missing.length === 0) return;
    const auto = computeAutoLayout(g);
    if (missing.length === g.nodes.length) {
      cfg.layout = auto;
      return;
    }
    for (const n of missing) {
      cfg.layout[n.id] = freeSpot(cfg.layout, auto[n.id]);
    }
  }

  // Canvas size in pixels; while the canvas is hidden (for example in the Guide), the last visible size.
  let lastCanvasSize = { w: 565, h: 400 };
  function canvasSize() {
    const wrap = document.getElementById("editor-canvas-wrap");
    const r = wrap ? wrap.getBoundingClientRect() : null;
    if (r && r.width > 0 && r.height > 0) lastCanvasSize = { w: r.width, h: r.height };
    return lastCanvasSize;
  }

  // Show the whole graph, centered, never magnified beyond full size.
  function fitView(cfg) {
    const size = canvasSize();
    const pts = Object.values(cfg.layout || {});
    if (pts.length === 0) {
      cfg.view = { x: -size.w / 2, y: -40, w: size.w, h: size.h };
      return;
    }
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const p of pts) {
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x + CARD_W);
      maxY = Math.max(maxY, p.y + CARD_H);
    }
    const pad = FIT_PAD;
    minX -= pad;
    minY -= pad;
    let w = maxX - minX + pad;
    let h = maxY - minY + pad;
    if (w < size.w) {
      minX -= (size.w - w) / 2;
      w = size.w;
    }
    if (h < size.h) {
      minY -= (size.h - h) / 2;
      h = size.h;
    }
    const aspect = size.w / size.h;
    if (w / h > aspect) {
      const nh = w / aspect;
      minY -= (nh - h) / 2;
      h = nh;
    } else {
      const nw = h * aspect;
      minX -= (nw - w) / 2;
      w = nw;
    }
    cfg.view = { x: minX, y: minY, w: w, h: h };
  }

  function cardCenter(cfg, id) {
    const p = cfg.layout[id];
    return { x: p.x + CARD_W / 2, y: p.y + CARD_H / 2 };
  }

  // Point where the line from a card's center toward (tx, ty) crosses the card border, plus a gap.
  function cardExit(c, tx, ty, gap) {
    const dx = tx - c.x;
    const dy = ty - c.y;
    if (Math.abs(dx) < 1e-6 && Math.abs(dy) < 1e-6) return { x: c.x, y: c.y };
    const s = Math.min((CARD_W / 2 + gap) / Math.max(Math.abs(dx), 1e-6), (CARD_H / 2 + gap) / Math.max(Math.abs(dy), 1e-6));
    return { x: c.x + dx * s, y: c.y + dy * s };
  }

  function isReciprocal(g, e) {
    return g.edges.some(o => o.source === e.target && o.target === e.source);
  }

  // Curved link from source to learner. Reciprocal pairs bend to opposite sides; links between nodes
  // in the same row arc above the cards so they do not cross the cards in between.
  function edgeGeometry(cfg, e, reciprocal) {
    const a = cardCenter(cfg, e.source);
    const b = cardCenter(cfg, e.target);
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy) || 1;
    let ox = -dy / len;
    let oy = dx / len;
    let bend = reciprocal ? 20 : 0;
    const sameRow = Math.abs(dy) < CARD_H * 0.6;
    if (sameRow && Math.abs(dx) > CARD_W * 1.5) {
      bend = Math.max(bend, Math.min(90, 28 + Math.abs(dx) * 0.12));
      if (!reciprocal && oy > 0) {
        ox = -ox;
        oy = -oy;
      }
    }
    const mid = { x: (a.x + b.x) / 2 + ox * bend, y: (a.y + b.y) / 2 + oy * bend };
    const start = cardExit(a, mid.x, mid.y, 1);
    const end = cardExit(b, mid.x, mid.y, 3);
    const ctrl = { x: 2 * mid.x - (start.x + end.x) / 2, y: 2 * mid.y - (start.y + end.y) / 2 };
    return { start: start, end: end, ctrl: ctrl, mid: mid };
  }

  function edgeMarkup(cfg, e, selEdge) {
    if (!cfg.layout[e.source] || !cfg.layout[e.target]) return "";
    const geo = edgeGeometry(cfg, e, isReciprocal(cfg.org_graph, e));
    const p = clampNum(e.weight, 0, 1, 0.25);
    const c = clampNum(e.commensurability, 0, 1, 0.85);
    const isSel = Boolean(selEdge && selEdge.source === e.source && selEdge.target === e.target);
    const low = c < LOW_C;
    const color = isSel ? SELECT_COLOR : (low ? EDGE_LIGHT : EDGE_COLOR);
    const width = 1 + 4 * p;
    // Arrowhead at the learner; the line stops at the arrow's base so thick lines stay clean.
    const tx0 = geo.end.x - geo.ctrl.x;
    const ty0 = geo.end.y - geo.ctrl.y;
    const tl = Math.hypot(tx0, ty0) || 1;
    const tx = tx0 / tl;
    const ty = ty0 / tl;
    const head = 8 + 3 * p;
    const half = 3.6 + 1.6 * p;
    const bx = geo.end.x - tx * head;
    const by = geo.end.y - ty * head;
    const d = "M " + geo.start.x.toFixed(1) + " " + geo.start.y.toFixed(1) +
      " Q " + geo.ctrl.x.toFixed(1) + " " + geo.ctrl.y.toFixed(1) + " " + bx.toFixed(1) + " " + by.toFixed(1);
    const arrow = geo.end.x.toFixed(1) + "," + geo.end.y.toFixed(1) + " " +
      (bx - ty * half).toFixed(1) + "," + (by + tx * half).toFixed(1) + " " +
      (bx + ty * half).toFixed(1) + "," + (by - tx * half).toFixed(1);
    return '<g class="edge" data-src="' + escapeHtml(e.source) + '" data-tgt="' + escapeHtml(e.target) + '">' +
      '<path d="' + d + '" fill="none" stroke="' + color + '" stroke-width="' + width.toFixed(2) + '"' +
      (low ? ' stroke-dasharray="6 4"' : "") + "/>" +
      '<polygon points="' + arrow + '" fill="' + color + '"/>' +
      '<path class="edge-hit" d="' + d + '" fill="none" stroke="transparent" stroke-width="14"/>' +
      "</g>";
  }

  // Card label that fits the card: drop a trailing parenthetical first, then shorten with an ellipsis.
  function cardLabel(text, maxChars) {
    let s = String(text);
    if (s.length > maxChars) {
      const cut = s.indexOf(" (");
      if (cut > 0) s = s.slice(0, cut);
    }
    return s.length > maxChars ? s.slice(0, maxChars - 1).trim() + "\\u2026" : s;
  }

  // Feedback badge: F firm, D own domain, dash none; filled when set on the node itself.
  function badgeMarkup(cfg, n, fbMap) {
    const fbEngine = fbMap[n.id] || effectiveFeedback(n, cfg.org_graph);
    const fb = displayFeedback(n, fbEngine);
    const style = FEEDBACK_BADGE[fb] || FEEDBACK_BADGE.firm;
    const filled = n.role !== "hq" && isFeedbackValue(n.feedback);
    const bx = CARD_W - 27;
    const by = 23;
    const ink = filled ? "#ffffff" : style.color;
    let s = '<rect x="' + bx + '" y="' + by + '" width="17" height="14" rx="3" fill="' + (filled ? style.color : "#ffffff") +
      '" stroke="' + style.color + '" stroke-width="1.2"/>';
    if (fb === "none") {
      s += '<line x1="' + (bx + 5) + '" y1="' + (by + 7) + '" x2="' + (bx + 12) + '" y2="' + (by + 7) +
        '" stroke="' + ink + '" stroke-width="1.6"/>';
    } else {
      s += '<text x="' + (bx + 8.5) + '" y="' + (by + 10.5) + '" text-anchor="middle" font-size="9.5" font-weight="700" ' +
        'font-family="ui-monospace, Menlo, monospace" fill="' + ink + '">' + style.letter + "</text>";
    }
    return s;
  }

  // Node card: label, representation and headcount, accuracy tag, N-cell domain strip, feedback badge, link handle.
  function nodeMarkup(cfg, n, fbMap, isSel) {
    const p = cfg.layout[n.id];
    if (!p) return "";
    const st = CARD_STYLE[n.role === "hq" ? "hq" : (n.role === "manager" ? "manager" : "unit")];
    let s = '<g class="node' + (isSel ? " selected" : "") + '" data-id="' + escapeHtml(n.id) +
      '" transform="translate(' + p.x.toFixed(1) + " " + p.y.toFixed(1) + ')">';
    if (isSel) {
      s += '<rect x="-4" y="-4" width="' + (CARD_W + 8) + '" height="' + (CARD_H + 8) +
        '" rx="11" fill="none" stroke="' + SELECT_COLOR + '" stroke-width="2.4"/>';
    }
    s += '<rect class="card" width="' + CARD_W + '" height="' + CARD_H + '" rx="8" fill="' + st.fill +
      '" stroke="' + st.stroke + '" stroke-width="1.2"/>';
    s += '<text x="10" y="18" font-size="11.5" font-weight="600" fill="' + st.text + '">' +
      escapeHtml(cardLabel(n.label || n.id, 19)) + "</text>";
    const model = n.model_type === "linear" ? "linear" : (n.model_type === "tabular" ? "table" : "H" + (n.h_agent !== undefined ? n.h_agent : 16));
    const count = Number(n.count || 1) > 1 ? "  \\u00d7" + Number(n.count) : "";
    s += '<text x="10" y="34" font-size="10.5" fill="' + st.sub + '" font-family="ui-monospace, Menlo, monospace">' +
      escapeHtml(model + count) + "</text>";
    const acc = effectiveAccuracy(n, cfg.org_graph);
    if (acc !== null) {
      const accTxt = "acc " + (Math.round(acc * 100) / 100).toString();
      const tagW = 8 + accTxt.length * 6.1;
      const tagX = CARD_W - 31 - tagW;
      s += '<rect x="' + tagX.toFixed(1) + '" y="24" width="' + tagW.toFixed(1) +
        '" height="13" rx="3" fill="#ffffff" stroke="#c7d2e3" stroke-width="1"/>' +
        '<text x="' + (tagX + tagW / 2).toFixed(1) + '" y="34" text-anchor="middle" font-size="9.5" ' +
        'font-family="ui-monospace, Menlo, monospace" fill="#3f3f46">' + accTxt + "</text>";
    }
    const nBits = currentNBits();
    const bits = new Set(Array.isArray(n.jurisdiction_bits) && n.jurisdiction_bits.length > 0 ? n.jurisdiction_bits : allBits(nBits));
    const gap = nBits > 12 ? 1.5 : 2;
    const cellW = (CARD_W - 20 - (nBits - 1) * gap) / nBits;
    for (let b = 0; b < nBits; b++) {
      s += '<rect x="' + (10 + b * (cellW + gap)).toFixed(1) + '" y="' + (CARD_H - 12) + '" width="' + cellW.toFixed(1) +
        '" height="6" rx="1" fill="' + (bits.has(b) ? st.on : st.off) + '"/>';
    }
    s += badgeMarkup(cfg, n, fbMap);
    s += '<circle class="handle" cx="' + CARD_W + '" cy="' + (CARD_H / 2) + '" r="5.5" fill="#ffffff" stroke="' +
      SELECT_COLOR + '" stroke-width="1.6"/>';
    return s + "</g>";
  }

  function renderEditor() {
    const svg = document.getElementById("org-canvas");
    if (!svg) return;
    const cfg = activeConfig();
    const g = cfg && cfg.org_graph;
    if (!g || !Array.isArray(g.nodes) || g.nodes.length === 0) {
      svg.innerHTML = "";
      return;
    }
    ensureLayout(cfg);
    if (!cfg.view) fitView(cfg);
    const v = cfg.view;
    svg.setAttribute("viewBox", v.x.toFixed(1) + " " + v.y.toFixed(1) + " " + v.w.toFixed(1) + " " + v.h.toFixed(1));
    const fbMap = nodeFeedbackMap(cfg);
    const sel = selectedNode(cfg);
    const drag = editorState.drag;
    let html = "";
    for (const e of g.edges) {
      html += edgeMarkup(cfg, e, cfg.sel_edge);
    }
    for (const n of g.nodes) {
      html += nodeMarkup(cfg, n, fbMap, !cfg.sel_edge && sel && n.id === sel.id);
    }
    if (drag && drag.type === "link" && cfg.layout[drag.source]) {
      const sp = cfg.layout[drag.source];
      const x1 = sp.x + CARD_W;
      const y1 = sp.y + CARD_H / 2;
      const target = nodeAtPoint(cfg, drag.x, drag.y);
      if (target && target.id !== drag.source && cfg.layout[target.id]) {
        const tp = cfg.layout[target.id];
        const ok = !linkError(cfg, drag.source, target.id);
        html += '<rect x="' + (tp.x - 4) + '" y="' + (tp.y - 4) + '" width="' + (CARD_W + 8) + '" height="' + (CARD_H + 8) +
          '" rx="11" fill="none" stroke="' + (ok ? "#15803d" : "#b91c1c") + '" stroke-width="2" stroke-dasharray="5 3"/>';
      }
      html += '<line x1="' + x1.toFixed(1) + '" y1="' + y1.toFixed(1) + '" x2="' + drag.x.toFixed(1) + '" y2="' + drag.y.toFixed(1) +
        '" stroke="' + SELECT_COLOR + '" stroke-width="1.6" stroke-dasharray="5 4"/>' +
        '<circle cx="' + drag.x.toFixed(1) + '" cy="' + drag.y.toFixed(1) + '" r="3.5" fill="' + SELECT_COLOR + '"/>';
    }
    svg.innerHTML = html;
    svg.classList.toggle("dragging", Boolean(drag && drag.type === "node" && drag.moved));
    const delBtn = document.getElementById("btn-delete");
    if (delBtn) delBtn.disabled = !cfg.sel_edge && (!sel || sel.role === "hq");
  }

  function svgPoint(svg, clientX, clientY) {
    const m = svg.getScreenCTM();
    if (!m) return { x: 0, y: 0 };
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    const r = pt.matrixTransform(m.inverse());
    return { x: r.x, y: r.y };
  }

  // Topmost card under a canvas point.
  function nodeAtPoint(cfg, x, y) {
    const g = cfg.org_graph;
    for (let i = g.nodes.length - 1; i >= 0; i--) {
      const p = cfg.layout[g.nodes[i].id];
      if (p && x >= p.x - 6 && x <= p.x + CARD_W + 6 && y >= p.y - 6 && y <= p.y + CARD_H + 6) return g.nodes[i];
    }
    return null;
  }

  function onCanvasPointerDown(ev) {
    if (ev.button !== 0) return;
    const svg = ev.currentTarget;
    const cfg = activeConfig();
    if (!cfg || !cfg.org_graph) return;
    hideCanvasTip();
    svg.focus({ preventScroll: true });
    const pt = svgPoint(svg, ev.clientX, ev.clientY);
    const handleEl = ev.target.closest(".handle");
    const nodeEl = ev.target.closest(".node");
    const edgeEl = ev.target.closest(".edge");
    if (handleEl && nodeEl) {
      editorState.drag = { type: "link", source: nodeEl.dataset.id, x: pt.x, y: pt.y };
      svg.setPointerCapture(ev.pointerId);
      ev.preventDefault();
      renderEditor();
      return;
    }
    if (nodeEl) {
      const id = nodeEl.dataset.id;
      if (ev.shiftKey && cfg.sel_node && cfg.sel_node !== id && !cfg.sel_edge) {
        // Shift+click: the selected node becomes the source of a link to this node.
        tryAddLink(cfg, cfg.sel_node, id);
        ev.preventDefault();
        return;
      }
      const p = cfg.layout[id];
      editorState.drag = {
        type: "node", id: id, startX: pt.x, startY: pt.y, origX: p.x, origY: p.y, moved: false, snapshot: snapshotState(cfg)
      };
      svg.setPointerCapture(ev.pointerId);
      ev.preventDefault();
      if (cfg.sel_node !== id || cfg.sel_edge) {
        cfg.sel_node = id;
        cfg.sel_edge = null;
        closeEdgeEditor();
        renderEditor();
        renderNodeInspector();
        updateDesignReadout();
      }
      return;
    }
    if (edgeEl) {
      cfg.sel_edge = { source: edgeEl.dataset.src, target: edgeEl.dataset.tgt };
      renderEditor();
      openEdgeEditor(cfg, true);
      ev.preventDefault();
      return;
    }
    if (cfg.sel_edge) {
      cfg.sel_edge = null;
      closeEdgeEditor();
      renderEditor();
    }
  }

  function onCanvasPointerMove(ev) {
    const svg = ev.currentTarget;
    const cfg = activeConfig();
    const drag = editorState.drag;
    if (!cfg || !cfg.org_graph) return;
    if (!drag) {
      updateCanvasTip(ev, cfg);
      return;
    }
    const pt = svgPoint(svg, ev.clientX, ev.clientY);
    if (drag.type === "node") {
      const dx = pt.x - drag.startX;
      const dy = pt.y - drag.startY;
      if (!drag.moved && Math.hypot(dx, dy) < 3) return;
      drag.moved = true;
      cfg.layout[drag.id] = { x: Math.round(drag.origX + dx), y: Math.round(drag.origY + dy) };
      renderEditor();
    } else if (drag.type === "link") {
      drag.x = pt.x;
      drag.y = pt.y;
      renderEditor();
    }
  }

  function onCanvasPointerUp(ev) {
    const svg = ev.currentTarget;
    const cfg = activeConfig();
    const drag = editorState.drag;
    editorState.drag = null;
    if (svg.hasPointerCapture && svg.hasPointerCapture(ev.pointerId)) svg.releasePointerCapture(ev.pointerId);
    if (!drag || !cfg || !cfg.org_graph) return;
    if (drag.type === "node") {
      // Moving a node changes only the layout, so results stay valid.
      if (drag.moved) pushUndo(cfg, drag.snapshot);
      renderEditor();
    } else if (drag.type === "link") {
      const pt = svgPoint(svg, ev.clientX, ev.clientY);
      const target = nodeAtPoint(cfg, pt.x, pt.y);
      renderEditor();
      if (target && target.id !== drag.source) tryAddLink(cfg, drag.source, target.id);
      else if (target) showEditorMsg("A node cannot learn from itself.");
    }
  }

  function onCanvasDblClick(ev) {
    const cfg = activeConfig();
    if (!cfg || !cfg.org_graph) return;
    if (ev.target.closest(".node") || ev.target.closest(".edge")) return;
    const pt = svgPoint(ev.currentTarget, ev.clientX, ev.clientY);
    addUnit(cfg, { x: pt.x - CARD_W / 2, y: pt.y - CARD_H / 2 });
  }

  function hideCanvasTip() {
    const tip = document.getElementById("canvas-tip");
    if (tip) tip.hidden = true;
  }

  function nodeTipText(cfg, id) {
    const n = nodeById(cfg.org_graph, id);
    if (!n) return "";
    const fbEngine = nodeFeedbackMap(cfg)[n.id] || effectiveFeedback(n, cfg.org_graph);
    const fb = displayFeedback(n, fbEngine);
    if (n.role !== "hq" && fbEngine === "domain" && fb === "firm") {
      return "Own domain covers every choice, so this node sees the firm result.";
    }
    const how = n.role === "hq" ? " (always, as the coalition)" : (isFeedbackValue(n.feedback) ? " (set on this node)" : " (organization setting)");
    return (n.label || n.id) + " sees " + NODE_FEEDBACK_SHORT[fb] + how;
  }

  // Hover text for links, nodes, and handles, placed next to the pointer inside the canvas.
  function updateCanvasTip(ev, cfg) {
    const tip = document.getElementById("canvas-tip");
    const wrap = document.getElementById("editor-canvas-wrap");
    if (!tip || !wrap || !ev.target.closest) return;
    const t = ev.target;
    let text = "";
    if (t.closest(".handle")) {
      text = "Drag to another node to add a link";
    } else if (t.closest(".edge")) {
      const el = t.closest(".edge");
      const e = findEdge(cfg.org_graph, { source: el.dataset.src, target: el.dataset.tgt });
      if (e) {
        text = nodeLabel(cfg, e.target) + " learns from " + nodeLabel(cfg, e.source) + ": P " + fmt2(e.weight) + ", C " + fmt2(e.commensurability);
      }
    } else if (t.closest(".node")) {
      text = nodeTipText(cfg, t.closest(".node").dataset.id);
    }
    if (!text) {
      tip.hidden = true;
      return;
    }
    tip.textContent = text;
    tip.hidden = false;
    const wr = wrap.getBoundingClientRect();
    let left = ev.clientX - wr.left + 12;
    let top = ev.clientY - wr.top + 16;
    if (left + tip.offsetWidth > wr.width - 4) left = Math.max(4, ev.clientX - wr.left - tip.offsetWidth - 12);
    if (top + tip.offsetHeight > wr.height - 4) top = Math.max(4, ev.clientY - wr.top - tip.offsetHeight - 10);
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }

  let editorMsgTimer = null;
  // Message at the bottom of the canvas: editing errors disappear after a few seconds, engine errors stay.
  function showEditorMsg(text, persistent) {
    const box = document.getElementById("editor-msg");
    if (!box) return;
    clearTimeout(editorMsgTimer);
    box.textContent = text || "";
    box.hidden = !text;
    if (text && !persistent) {
      editorMsgTimer = setTimeout(renderErrors, 4000);
    }
  }

  function snapshotState(cfg) {
    return JSON.stringify({ g: cfg.org_graph, l: cfg.layout, t: cfg.template_key });
  }

  function pushUndo(cfg, snap) {
    cfg.undo.push(snap || snapshotState(cfg));
    if (cfg.undo.length > UNDO_LIMIT) cfg.undo.shift();
    updateUndoButton();
  }

  function updateUndoButton() {
    const btn = document.getElementById("btn-undo");
    const cfg = activeConfig();
    if (btn) btn.disabled = !cfg || cfg.undo.length === 0;
  }

  function undo() {
    const cfg = activeConfig();
    if (!cfg || cfg.undo.length === 0) return;
    const prev = JSON.parse(cfg.undo.pop());
    const graphChanged = JSON.stringify(prev.g) !== JSON.stringify(cfg.org_graph);
    cfg.org_graph = prev.g;
    cfg.layout = prev.l || {};
    cfg.template_key = prev.t || "";
    cfg.sel_edge = null;
    closeEdgeEditor();
    renderAll();
    if (graphChanged) {
      markStale();
      scheduleCompile(cfg);
    }
  }

  // Apply a change to a configuration's graph or layout: one undo step, then re-render and revalidate.
  // Structural and node edits mark the design as edited; organization settings keep the template name.
  function mutateGraph(cfg, fn, opts) {
    const snap = snapshotState(cfg);
    if (fn(cfg.org_graph, cfg) === false) return false;
    pushUndo(cfg, snap);
    if (!(opts && opts.keepTemplate)) cfg.template_key = "";
    markStale();
    renderAll();
    scheduleCompile(cfg);
    return true;
  }

  function findEdge(g, sel) {
    return sel && g ? (g.edges.find(e => e.source === sel.source && e.target === sel.target) || null) : null;
  }

  function linkError(cfg, src, tgt) {
    if (src === tgt) return "A node cannot learn from itself.";
    if (cfg.org_graph.edges.some(e => e.source === src && e.target === tgt)) {
      return "This link already exists; click it to change P or C.";
    }
    return "";
  }

  function tryAddLink(cfg, src, tgt) {
    const err = linkError(cfg, src, tgt);
    if (err) {
      showEditorMsg(err);
      return false;
    }
    // A reciprocal link shares the commensurability of the existing direction.
    const rev = findEdge(cfg.org_graph, { source: tgt, target: src });
    const c = rev ? clampNum(rev.commensurability, 0, 1, 0.85) : 0.85;
    mutateGraph(cfg, (g, cc) => {
      g.edges.push({ source: src, target: tgt, weight: 0.25, commensurability: c });
      cc.sel_edge = { source: src, target: tgt };
    });
    openEdgeEditor(cfg);
    return true;
  }

  function deleteEdge(cfg, sel) {
    if (!findEdge(cfg.org_graph, sel)) return;
    closeEdgeEditor();
    mutateGraph(cfg, (g, cc) => {
      g.edges = g.edges.filter(e => !(e.source === sel.source && e.target === sel.target));
      cc.sel_edge = null;
    });
  }

  function reverseEdge(cfg, sel) {
    const e = findEdge(cfg.org_graph, sel);
    if (!e) return;
    if (findEdge(cfg.org_graph, { source: e.target, target: e.source })) {
      showEditorMsg("The reverse link already exists.");
      return;
    }
    mutateGraph(cfg, (g, cc) => {
      const edge = findEdge(g, sel);
      const src = edge.source;
      edge.source = edge.target;
      edge.target = src;
      cc.sel_edge = { source: edge.source, target: edge.target };
    });
    openEdgeEditor(cfg);
  }

  function uniqueNodeId(g, prefix) {
    let i = 1;
    while (g.nodes.some(n => n.id === prefix + i)) i += 1;
    return prefix + i;
  }

  // Nearest manager or DC to a canvas point, preferring nodes above it.
  function nearestSupervisor(cfg, at) {
    let best = null;
    let bestD = Infinity;
    for (const n of cfg.org_graph.nodes) {
      const p = cfg.layout[n.id];
      if (n.role === "unit" || !p) continue;
      const d = Math.hypot(p.x - at.x, 1.5 * (p.y - at.y)) + (p.y > at.y ? 400 : 0);
      if (d < bestD) {
        bestD = d;
        best = n;
      }
    }
    return best;
  }

  function keepInView(cfg, id) {
    const p = cfg.layout[id];
    const v = cfg.view;
    if (!p || !v) return;
    if (p.x < v.x || p.y < v.y || p.x + CARD_W > v.x + v.w || p.y + CARD_H > v.y + v.h) {
      fitView(cfg);
      renderEditor();
    }
  }

  // New unit: reports to the selected manager or DC (or, when placed by double-click, to the nearest one above).
  function addUnit(cfg, at) {
    const g = cfg.org_graph;
    if (!g) return;
    const sel = selectedNode(cfg);
    const hq = g.nodes.find(n => n.role === "hq") || g.nodes[0];
    const parent = at ? (nearestSupervisor(cfg, at) || hq) : (sel && sel.role !== "unit" ? sel : hq);
    const id = uniqueNodeId(g, "unit_");
    const num = g.nodes.filter(n => n.role === "unit").length + 1;
    mutateGraph(cfg, (g2, cc) => {
      g2.nodes.push({
        id: id, label: "Unit " + num, role: "unit", count: 1, model_type: "neural", h_agent: 16,
        jurisdiction_bits: firstHalfBits(), lr: 0.16, sigma_learn: 0.05, rho: 0.0, search_steps: 1,
        upward_agg: "mean", feedback: null, initial_accuracy: null
      });
      if (parent) g2.edges.push({ source: id, target: parent.id, weight: 0.35, commensurability: 0.85 });
      const pp = parent && cc.layout[parent.id] ? cc.layout[parent.id] : { x: -CARD_W / 2, y: 0 };
      cc.layout[id] = at ? { x: Math.round(at.x), y: Math.round(at.y) } : freeSpot(cc.layout, { x: pp.x, y: pp.y + LAYER_GAP });
      cc.sel_node = id;
      cc.sel_edge = null;
    });
    keepInView(cfg, id);
  }

  function addManager(cfg) {
    const g = cfg.org_graph;
    if (!g) return;
    const hq = g.nodes.find(n => n.role === "hq") || g.nodes[0];
    const id = uniqueNodeId(g, "mgr_");
    const num = g.nodes.filter(n => n.role === "manager").length + 1;
    mutateGraph(cfg, (g2, cc) => {
      g2.nodes.push({
        id: id, label: "Manager " + num, role: "manager", count: 1, model_type: "neural", h_agent: 16,
        jurisdiction_bits: allBits(), lr: 0.14, sigma_learn: 0.05, rho: 0.0, search_steps: 1,
        upward_agg: "mean", feedback: null, initial_accuracy: null
      });
      if (hq) g2.edges.push({ source: id, target: hq.id, weight: 0.4, commensurability: 0.85 });
      const hp = hq && cc.layout[hq.id] ? cc.layout[hq.id] : { x: -CARD_W / 2, y: 0 };
      cc.layout[id] = freeSpot(cc.layout, { x: hp.x, y: hp.y + LAYER_GAP });
      cc.sel_node = id;
      cc.sel_edge = null;
    });
    keepInView(cfg, id);
  }

  function deleteNode(cfg, id) {
    const n = nodeById(cfg.org_graph, id);
    if (!n) return;
    if (n.role === "hq") {
      showEditorMsg("The dominant coalition cannot be deleted.");
      return;
    }
    mutateGraph(cfg, (g, cc) => {
      g.edges = g.edges.filter(e => e.source !== id && e.target !== id);
      g.nodes = g.nodes.filter(x => x.id !== id);
      delete cc.layout[id];
      const hq = g.nodes.find(x => x.role === "hq") || g.nodes[0];
      cc.sel_node = hq ? hq.id : null;
      cc.sel_edge = null;
    });
  }

  function deleteSelection(cfg) {
    if (cfg.sel_edge) {
      deleteEdge(cfg, cfg.sel_edge);
      return;
    }
    const n = selectedNode(cfg);
    if (n) deleteNode(cfg, n.id);
  }

  function autoLayout(cfg) {
    if (!cfg || !cfg.org_graph) return;
    pushUndo(cfg);
    cfg.layout = computeAutoLayout(cfg.org_graph);
    fitView(cfg);
    renderEditor();
  }

  // Show the selected link in the inspector; reveal scrolls the inspector into view after a click on the canvas.
  function openEdgeEditor(cfg, reveal) {
    editorState.sliderSnap = null;
    renderNodeInspector();
    const box = document.getElementById("edge-editor");
    if (reveal && box && !box.hidden && !isOverlayOpen()) box.scrollIntoView({ block: "nearest" });
  }

  // Fill the link editor in the inspector with the selected link.
  function renderLinkEditor(cfg, e) {
    setElText("edge-editor-title", nodeLabel(cfg, e.target) + " learns from " + nodeLabel(cfg, e.source));
    setElValue("edge-ctrl-p", fmt2(e.weight));
    setElText("edge-ctrl-p-value", fmt2(e.weight));
    setElValue("edge-ctrl-c", fmt2(e.commensurability));
    setElText("edge-ctrl-c-value", fmt2(e.commensurability));
  }

  // Leave link editing: deselect the link of the active configuration and show the node fields again.
  function closeEdgeEditor() {
    editorState.sliderSnap = null;
    const cfg = activeConfig();
    if (cfg) cfg.sel_edge = null;
    renderNodeInspector();
  }

  // The link sliders redraw the canvas while moving; releasing a slider records one undo step.
  function onEdgeSliderInput(field, value) {
    const cfg = activeConfig();
    const e = cfg ? findEdge(cfg.org_graph, cfg.sel_edge) : null;
    if (!e) return;
    if (!editorState.sliderSnap) editorState.sliderSnap = snapshotState(cfg);
    const v = clampNum(value, 0, 1, field === "weight" ? 0.25 : 0.85);
    e[field] = v;
    if (field === "commensurability") {
      const rev = findEdge(cfg.org_graph, { source: e.target, target: e.source });
      if (rev) rev.commensurability = v;
    }
    setElText(field === "weight" ? "edge-ctrl-p-value" : "edge-ctrl-c-value", fmt2(v));
    renderEditor();
  }

  function onEdgeSliderChange() {
    const cfg = activeConfig();
    if (!cfg || !editorState.sliderSnap) return;
    pushUndo(cfg, editorState.sliderSnap);
    editorState.sliderSnap = null;
    cfg.template_key = "";
    markStale();
    setElValue("ctrl-graph-template", "");
    updateTemplateHint();
    renderNodeInspector();
    scheduleCompile(cfg);
  }

  function isOverlayOpen() {
    const ov = document.getElementById("editor-overlay");
    return Boolean(ov && !ov.hidden);
  }

  // Expand moves the editor and the node inspector into a large overlay; closing moves them back.
  function openOverlay() {
    const ov = document.getElementById("editor-overlay");
    const slot = document.getElementById("overlay-slot");
    const ed = document.getElementById("editor");
    const insp = document.getElementById("node-inspector-panel");
    if (!ov || !slot || !ed || !insp || !ov.hidden) return;
    hideCanvasTip();
    ov.hidden = false;
    slot.appendChild(ed);
    slot.appendChild(insp);
    setElText("btn-expand", "Collapse");
    const cfg = activeConfig();
    if (cfg) fitView(cfg);
    renderEditor();
    document.getElementById("org-canvas").focus({ preventScroll: true });
  }

  function closeOverlay() {
    const ov = document.getElementById("editor-overlay");
    const home = document.getElementById("editor-home");
    const ed = document.getElementById("editor");
    const insp = document.getElementById("node-inspector-panel");
    if (!ov || !home || !ed || !insp || ov.hidden) return;
    hideCanvasTip();
    home.appendChild(ed);
    home.after(insp);
    ov.hidden = true;
    setElText("btn-expand", "Expand");
    const cfg = activeConfig();
    if (cfg) fitView(cfg);
    renderEditor();
  }

  // Keyboard: Ctrl or Cmd + Z undoes; Delete or Backspace removes the selection while the editor has focus.
  function onKeyDown(ev) {
    if (ev.defaultPrevented || activeTopView !== "lab") return;
    const el = ev.target;
    const tag = el && el.tagName ? el.tagName.toUpperCase() : "";
    const inField = tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || Boolean(el && el.isContentEditable);
    if (ev.key === "Escape" && isOverlayOpen()) {
      closeOverlay();
      return;
    }
    if (inField) return;
    const cfg = activeConfig();
    if (!cfg || !cfg.org_graph) return;
    if ((ev.ctrlKey || ev.metaKey) && !ev.shiftKey && !ev.altKey && String(ev.key).toLowerCase() === "z") {
      ev.preventDefault();
      undo();
      return;
    }
    const editorEl = document.getElementById("editor");
    if ((ev.key === "Delete" || ev.key === "Backspace") && editorEl && editorEl.contains(document.activeElement)) {
      ev.preventDefault();
      deleteSelection(cfg);
    }
  }

  function renderNodeInspector() {
    const cfg = activeConfig();
    const g = cfg && cfg.org_graph;
    const panel = document.getElementById("node-inspector-panel");
    // A selected link replaces the node fields with the link editor.
    const edge = g ? findEdge(g, cfg.sel_edge) : null;
    if (cfg && cfg.sel_edge && !edge) cfg.sel_edge = null;
    showElement("edge-editor", Boolean(edge));
    showElement("node-inspector-head", !edge);
    showElement("node-form", !edge);
    if (edge) {
      if (panel) panel.hidden = false;
      renderLinkEditor(cfg, edge);
      const delLinkBtn = document.getElementById("btn-delete");
      if (delLinkBtn) delLinkBtn.disabled = false;
      return;
    }
    const node = selectedNode(cfg);
    if (!g || !node) {
      if (panel) panel.hidden = true;
      return;
    }
    if (panel) panel.hidden = false;
    const isHq = node.role === "hq";

    setElText("node-inspector-title", node.label || node.id);
    setElText("node-inspector-role-pill", isHq ? "DC" : (node.role === "manager" ? "Manager" : "Unit"));
    setElValue("node-ctrl-label", node.label || node.id);
    const countEl = document.getElementById("node-ctrl-count");
    if (countEl) {
      countEl.value = isHq ? 1 : (node.count || 1);
      countEl.disabled = isHq;
    }
    setElValue("node-ctrl-model", ALLOWED_MODEL_TYPES.includes(node.model_type) ? node.model_type : "neural");
    setElValue("node-ctrl-h", String(node.h_agent !== undefined ? node.h_agent : 16));
    setElValue("node-ctrl-lr", fmt2(node.lr !== undefined ? node.lr : 0.15));
    setElValue("node-ctrl-sigma-learn", fmt2(node.sigma_learn !== undefined ? node.sigma_learn : 0.05));
    setElValue("node-ctrl-rho", fmt2(node.rho !== undefined ? node.rho : 0.0));
    const searchStepsEl = document.getElementById("node-ctrl-search-steps");
    if (searchStepsEl) {
      searchStepsEl.max = String(currentNBits());
      searchStepsEl.value = String(Math.min(currentNBits(), node.search_steps !== undefined ? node.search_steps : 1));
    }
    setElValue("node-ctrl-agg", node.upward_agg || "mean");

    // Market feedback: the DC always sees firm results; other nodes may override the organization setting.
    const fbSel = document.getElementById("node-ctrl-feedback");
    if (fbSel) {
      const defOpt = fbSel.querySelector('option[value=""]');
      if (defOpt) defOpt.textContent = "Organization default (" + NODE_FEEDBACK_SHORT[regimeFeedback(g)] + ")";
      fbSel.disabled = isHq;
      fbSel.value = isHq ? "firm" : (isFeedbackValue(node.feedback) ? node.feedback : "");
    }
    let fbNote = "";
    if (isHq) {
      fbNote = "The coalition always sees firm results.";
    } else if ((nodeFeedbackMap(cfg)[node.id] || effectiveFeedback(node, g)) === "domain" && coversAllChoices(node)) {
      fbNote = "Own domain covers every choice, so this node sees the firm result.";
    }
    const fbNoteEl = document.getElementById("node-feedback-note");
    if (fbNoteEl) {
      fbNoteEl.textContent = fbNote;
      fbNoteEl.hidden = !fbNote;
    }

    renderAccuracyControls(cfg, node);
    const searchNote = searchNoteFor(cfg, node);
    setElText("node-search-text", searchNote);
    showElement("node-search-note", Boolean(searchNote));
    const nodeAdv = document.getElementById("node-advanced-details");
    if (nodeAdv && (
      Number(node.count || 1) > 1 ||
      Number(node.search_steps || 1) > 1 ||
      (node.upward_agg && node.upward_agg !== "mean")
    )) {
      nodeAdv.open = true;
    }
    renderLinksList(cfg, node);
    renderBitPills(cfg, node);
    const delBtn = document.getElementById("btn-delete");
    if (delBtn) delBtn.disabled = !cfg.sel_edge && isHq;
  }

  // Inspector mode of a node's initial accuracy: "default", "uninformed", or "set".
  function accuracyMode(node) {
    if (isAccuracyValue(node.initial_accuracy)) return "set";
    if (node.initial_accuracy === NODE_UNINFORMED || node.role === "hq") return "uninformed";
    return "default";
  }

  // Initial accuracy controls, plus the realized accuracy of this node in the last run.
  function renderAccuracyControls(cfg, node) {
    const g = cfg.org_graph;
    const isHq = node.role === "hq";
    const mode = accuracyMode(node);
    const modeSel = document.getElementById("node-ctrl-acc-mode");
    if (modeSel) {
      const defOpt = modeSel.querySelector('option[value="default"]');
      if (defOpt) {
        const orgAcc = isAccuracyValue(g.unit_initial_accuracy) ? fmt2(g.unit_initial_accuracy) : "uninformed";
        defOpt.textContent = "Organization default (" + orgAcc + ")";
        defOpt.hidden = isHq;
        defOpt.disabled = isHq;
      }
      modeSel.value = mode;
    }
    showElement("wrap-node-acc", mode === "set");
    const accVal = mode === "set" ? Number(node.initial_accuracy) : 0.5;
    setElValue("node-ctrl-acc", String(accVal));
    setElText("node-ctrl-acc-value", fmt2(accVal));
    const parts = [];
    const eff = effectiveAccuracy(node, g);
    if (mode === "default") parts.push(eff === null ? "Starts uninformed, as set for the organization." : "Starts at accuracy " + fmt2(eff) + ", as set for the organization.");
    if (isHq && mode !== "set") parts.push("The organization default does not apply to the DC.");
    const res = resultFor(cfg);
    const realized = res && res.realized ? res.realized[node.id] : undefined;
    if (realized !== undefined && realized !== null && isFinite(Number(realized))) {
      parts.push("Realized initial accuracy in the last run: " + Number(realized).toFixed(2) + (resultsStale ? " (settings changed since)." : "."));
    }
    setElText("node-acc-note", parts.join(" "));
  }

  function searchNoteFor(cfg, node) {
    const g = cfg.org_graph;
    const isHq = node.role === "hq";
    const dr = orgSetting(g, "decision_rights");
    const hasOthers = g.nodes.some(n => n.role !== "hq");
    const exp = freshExplanation(cfg);
    if (isHq && hasOthers && dr === "propose_approve") {
      return "Under 'Units propose, coalition approves' the coalition only screens the units' proposals, so its own search reach and steps have no effect.";
    }
    if (isHq && hasOthers && dr === "delegated") {
      const residual = exp && Array.isArray(exp.dc_search_bits) ? exp.dc_search_bits : null;
      if (residual && residual.length === 0) {
        return "Under 'Units decide in their domains' every choice lies in another node's domain, so the coalition does not search and its search reach and steps have no effect.";
      }
      return "Under 'Units decide in their domains' the coalition searches only over choices outside every other node's domain" +
        (residual ? " (" + formatDomainBits(residual) + ")." : ".");
    }
    if (!isHq && dr === "centralized") {
      return "Search reach and steps matter only when units propose or decide (see Decision rights); while the coalition decides, this node does not search.";
    }
    if (!isHq && dr === "propose_approve") {
      return "This node searches its own domain with this reach and number of steps and proposes changes its model predicts will pay off.";
    }
    if (!isHq && dr === "delegated") {
      return "This node searches its own domain with this reach and number of steps and makes the changes its model predicts will pay off.";
    }
    return "";
  }

  // Links of the selected node: incoming (it learns from) and outgoing (others learn from it), with exact P and C.
  // The rows are rebuilt only when the set of links changes; value edits update the fields in place, so keyboard
  // focus (for example Tab from P to C) survives the re-render.
  function renderLinksList(cfg, node) {
    const box = document.getElementById("node-links-box");
    if (!box) return;
    const g = cfg.org_graph;
    const incoming = g.edges.filter(e => e.target === node.id);
    const outgoing = g.edges.filter(e => e.source === node.id);
    const signature = cfg.uid + "|" + node.id + "|" + incoming.concat(outgoing).map(e =>
      e.source + ">" + e.target + ":" + nodeLabel(cfg, e.source) + ":" + nodeLabel(cfg, e.target)).join("|");
    if (box.dataset.signature === signature) {
      for (const inp of box.querySelectorAll("input[data-field]")) {
        const e = findEdge(g, { source: inp.dataset.src, target: inp.dataset.tgt });
        if (e) inp.value = fmt2(e[inp.dataset.field]);
      }
      return;
    }
    box.dataset.signature = signature;
    box.innerHTML = "";
    const section = (title, list, otherEnd) => {
      const head = document.createElement("div");
      head.className = "links-subhead";
      head.textContent = title;
      box.appendChild(head);
      if (list.length === 0) {
        const empty = document.createElement("div");
        empty.className = "links-empty";
        empty.textContent = "None.";
        box.appendChild(empty);
        return;
      }
      for (const e of list) {
        box.appendChild(linkRow(cfg, e, otherEnd(e)));
      }
    };
    section("Learns from", incoming, e => e.source);
    section("Sends its views to", outgoing, e => e.target);
  }

  function linkRow(cfg, e, otherId) {
    const row = document.createElement("div");
    row.className = "link-row";
    const name = document.createElement("span");
    name.className = "link-name";
    name.textContent = nodeLabel(cfg, otherId);
    row.appendChild(name);
    const src = nodeLabel(cfg, e.source);
    const tgt = nodeLabel(cfg, e.target);
    const sel = { source: e.source, target: e.target };
    const numberField = (text, field, aria) => {
      const wrap = document.createElement("label");
      wrap.textContent = text;
      const inp = document.createElement("input");
      inp.type = "number";
      inp.min = "0";
      inp.max = "1";
      inp.step = "0.05";
      inp.value = fmt2(e[field]);
      inp.dataset.src = e.source;
      inp.dataset.tgt = e.target;
      inp.dataset.field = field;
      inp.setAttribute("aria-label", aria);
      inp.addEventListener("change", () => {
        const v = clampNum(inp.value, 0, 1, field === "weight" ? 0.25 : 0.85);
        mutateGraph(cfg, (g) => {
          const edge = findEdge(g, sel);
          if (!edge) return false;
          edge[field] = v;
          if (field === "commensurability") {
            const rev = findEdge(g, { source: sel.target, target: sel.source });
            if (rev) rev.commensurability = v;
          }
          return true;
        });
      });
      wrap.appendChild(inp);
      return wrap;
    };
    row.appendChild(numberField("P", "weight", "Permeability of the link from " + src + " to " + tgt));
    row.appendChild(numberField("C", "commensurability", "Commensurability of the link from " + src + " to " + tgt));
    const removeBtn = iconButton("&times;", "Remove the link from " + src + " to " + tgt);
    removeBtn.addEventListener("click", () => deleteEdge(cfg, sel));
    row.appendChild(removeBtn);
    return row;
  }

  // Toggles for the N choices in the node's domain; at least one stays selected. For the same node and N the
  // toggles are updated in place, so keyboard focus survives a toggle.
  function renderBitPills(cfg, node) {
    const nBits = currentNBits();
    const bits = Array.isArray(node.jurisdiction_bits) && node.jurisdiction_bits.length > 0 ? node.jurisdiction_bits.slice() : allBits(nBits);
    const bitSet = new Set(bits);
    syncJurisdictionPresetOptions();
    setElValue("node-jurisdiction-preset", detectJurisdictionPreset(bits, nBits));
    const row = document.getElementById("node-bit-pills");
    if (!row) return;
    row.style.gridTemplateColumns = "repeat(" + nBits + ", minmax(0, 1fr))";
    const signature = cfg.uid + "|" + node.id + "|" + nBits;
    const pills = row.querySelectorAll(".bit-pill");
    if (row.dataset.signature === signature && pills.length === nBits) {
      pills.forEach((btn, b) => {
        btn.classList.toggle("active", bitSet.has(b));
        btn.setAttribute("aria-pressed", bitSet.has(b) ? "true" : "false");
      });
      return;
    }
    row.dataset.signature = signature;
    row.innerHTML = "";
    for (let b = 0; b < nBits; b++) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "bit-pill" + (bitSet.has(b) ? " active" : "");
      btn.textContent = String(b + 1);
      btn.setAttribute("aria-pressed", bitSet.has(b) ? "true" : "false");
      btn.setAttribute("aria-label", "Choice " + (b + 1));
      btn.addEventListener("click", () => {
        const cur = nodeById(cfg.org_graph, node.id);
        if (!cur) return;
        const curN = currentNBits();
        const set = new Set(Array.isArray(cur.jurisdiction_bits) && cur.jurisdiction_bits.length > 0 ? cur.jurisdiction_bits : allBits(curN));
        if (set.has(b)) {
          if (set.size <= 1) return;
          set.delete(b);
        } else {
          set.add(b);
        }
        const next = Array.from(set).sort((x, y) => x - y);
        mutateGraph(cfg, (g) => {
          const n = nodeById(g, node.id);
          if (!n) return false;
          n.jurisdiction_bits = next;
          return true;
        });
      });
      row.appendChild(btn);
    }
  }

  function setElValue(id, val) {
    const el = document.getElementById(id);
    if (el) el.value = val;
  }

  function setElText(id, txt) {
    const el = document.getElementById(id);
    if (el) el.textContent = txt;
  }

  function showElement(id, cond) {
    const el = document.getElementById(id);
    if (el) el.hidden = !cond;
  }

  // Plain-language summary of the shared market environment, shown under the market bar.
  function describeMarket() {
    const parts = [];
    const nBits = currentNBits();
    const nStrats = (1 << nBits).toLocaleString("en-US");
    const k = Number(market.nk_k);
    const nkHead = "NK landscape with N = " + nBits + " (" + nStrats + " strategies), K = " + k;
    if (k === 0) {
      parts.push(nkHead + ": every choice pays off independently, so there is a single peak.");
    } else if (market.nk_pattern === "modular") {
      parts.push(nkHead + " in modular blocks: choices interact only within consecutive blocks of " + (k + 1) + ".");
    } else {
      parts.push(nkHead + ": each choice's payoff contribution depends on " + k + " randomly drawn other choice" + (k === 1 ? "." : "s."));
    }
    if (market.turbulence === "random") {
      parts.push("In each period, a shock redraws about half of the landscape with probability " + Number(market.p_shock).toFixed(3) + ".");
    } else if (market.turbulence === "periodic") {
      parts.push("A shock redraws about half of the landscape every " + market.shock_period + " periods.");
    } else {
      parts.push("The landscape does not change over time.");
    }
    parts.push(
      "Feedback noise SD " + fmt2(market.sigma_obs) + "; " + market.t_steps + " periods (logged every " +
      market.eval_every + ") across " + market.b_firms + " iterations (firms) shared by all configurations."
    );
    return parts.join(" ");
  }
  const marketSummaryText = describeMarket;

  // Update Choices N, clamp K to 0 .. N - 1, and proportionally rescale every node's domain across all configurations.
  function setNBits(rawVal) {
    const oldN = currentNBits();
    const newN = clampInt(rawVal, MIN_N_BITS, MAX_N_BITS, MARKET_DEFAULTS.n_bits);
    market.n_bits = newN;
    const kSlider = document.getElementById("ctrl-nk-k");
    if (kSlider) kSlider.max = String(newN - 1);
    market.nk_k = clampInt(market.nk_k, 0, newN - 1, Math.min(3, newN - 1));
    if (newN !== oldN) {
      for (const cfg of configs) {
        rescaleGraphBits(cfg.org_graph, oldN, newN);
        cfg.compiled = null;
        cfg.compiled_key = "";
        cfg.error = "";
      }
      markStale();
      renderAll();
      for (const cfg of configs) {
        scheduleCompile(cfg, 0);
      }
    } else {
      syncMarketControls();
    }
  }

  function syncMarketControls() {
    const nBits = currentNBits();
    setElValue("ctrl-n-bits", String(nBits));
    const kSlider = document.getElementById("ctrl-nk-k");
    if (kSlider) kSlider.max = String(nBits - 1);
    if (Number(market.nk_k) > nBits - 1) market.nk_k = nBits - 1;
    setElValue("ctrl-nk-k", String(market.nk_k));
    setElText("ctrl-nk-k-value", String(market.nk_k));
    setElValue("ctrl-nk-pattern", market.nk_pattern);
    setElValue("ctrl-noise", fmt2(market.sigma_obs));
    setElValue("ctrl-turbulence", market.turbulence);
    setElValue("ctrl-pshock", String(market.p_shock));
    setElValue("ctrl-shock-period", String(market.shock_period));
    setElValue("ctrl-firms", String(market.b_firms));
    setElValue("ctrl-tsteps", String(market.t_steps));
    setElValue("ctrl-eval-every", String(market.eval_every));
    setElValue("ctrl-seed", String(market.seed));
    showElement("wrap-pshock", market.turbulence === "random");
    showElement("wrap-shock-period", market.turbulence === "periodic");
    markChips("nbits-chips", nBits);
    markChips("noise-chips", Number(market.sigma_obs));
    markChips("firms-chips", Number(market.b_firms));
    setElText("market-summary-line", describeMarket());
  }

  // Highlight the preset chip that matches the current value.
  function markChips(containerId, value) {
    const box = document.getElementById(containerId);
    if (!box) return;
    for (const chip of box.querySelectorAll(".chip")) {
      const on = Math.abs(Number(chip.dataset.value) - value) < 1e-9;
      chip.classList.toggle("active", on);
      chip.setAttribute("aria-pressed", on ? "true" : "false");
    }
  }

  function updateTemplateHint() {
    const cfg = activeConfig();
    const key = cfg ? cfg.template_key || "" : "";
    const meta = CATALOG && CATALOG.graph_preset_metadata ? CATALOG.graph_preset_metadata[key] : null;
    let txt = "";
    if (!key) {
      txt = "Edited organization; choose a template to start again from a standard design.";
    } else if (meta && meta.subtitle) {
      txt = meta.subtitle;
    }
    setElText("template-hint", txt);
  }

  // Organization settings of the active configuration (the values live in its org_graph).
  function renderOrgSettings() {
    const cfg = activeConfig();
    const g = cfg && cfg.org_graph;
    const fr = orgSetting(g, "feedback_regime");
    const dr = orgSetting(g, "decision_rights");
    const expRate = Number(orgSetting(g, "experiment_rate"));
    const aspSens = Number(orgSetting(g, "aspiration_sensitivity"));
    const lamTrack = Number(orgSetting(g, "lambda_track"));
    setElValue("ctrl-feedback-regime", fr);
    setElValue("ctrl-decision-rights", dr);
    setElValue("ctrl-experiment-rate", fmt2(expRate));
    setElValue("ctrl-aspiration", fmt2(aspSens));
    setElValue("ctrl-org-lambda", fmt2(lamTrack));
    const orgAdv = document.getElementById("org-advanced-details");
    if (orgAdv && (Math.abs(expRate - 0.5) > 1e-9 || aspSens > 0 || lamTrack > 0)) {
      orgAdv.open = true;
    }
    const unitAcc = g ? g.unit_initial_accuracy : null;
    const informed = isAccuracyValue(unitAcc);
    setElValue("ctrl-unit-acc-mode", informed ? "set" : "uninformed");
    showElement("wrap-unit-acc", informed);
    setElValue("ctrl-unit-acc", String(informed ? Number(unitAcc) : 0.5));
    setElText("ctrl-unit-acc-value", fmt2(informed ? unitAcc : 0.5));

    let frHint = FEEDBACK_REGIME_HINTS[fr] || "";
    if (g && Array.isArray(g.nodes) && g.nodes.some(n => n.role !== "hq" && isFeedbackValue(n.feedback))) {
      frHint += " Nodes with a filled badge keep their own setting.";
    }
    setElText("hint-feedback-regime", frHint);
    setElText("hint-decision-rights", DECISION_RIGHTS_HINTS[dr] || "");
  }

  // Redraw everything that depends on the active configuration or the market.
  function renderAll() {
    syncMarketControls();
    renderConfigBar();
    const cfg = activeConfig();
    setElValue("ctrl-graph-template", cfg && cfg.template_key ? cfg.template_key : "");
    updateTemplateHint();
    renderOrgSettings();
    renderEditor();
    renderNodeInspector();
    renderNotes();
    renderErrors();
    renderLegend();
    updateDesignReadout();
    updateUndoButton();
    showElement("results-stale-note", resultsStale && results.entries.length > 0);
  }

  function jumpToOptionsGuide() {
    switchTopView("guide");
    const target = document.getElementById("guide-options-dict");
    if (target && target.scrollIntoView) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function feedbackSummary(cfg) {
    const g = cfg && cfg.org_graph;
    const fbMap = nodeFeedbackMap(cfg);
    const counts = { firm: 0, domain: 0, none: 0 };
    let others = 0;
    for (const n of (g && g.nodes) || []) {
      if (n.role === "hq") continue;
      others += 1;
      counts[displayFeedback(n, fbMap[n.id] || effectiveFeedback(n, g))] += 1;
    }
    if (others === 0) return "the DC sees firm results (no other nodes)";
    const parts = [];
    if (counts.firm) parts.push(counts.firm + " firm");
    if (counts.domain) parts.push(counts.domain + " own domain");
    if (counts.none) parts.push(counts.none + " none");
    return "DC firm; other nodes: " + parts.join(", ");
  }

  function accuracySummary(cfg) {
    const g = cfg.org_graph;
    const informed = g.nodes.filter(n => effectiveAccuracy(n, g) !== null).length;
    const unitDefault = isAccuracyValue(g.unit_initial_accuracy) ? fmt2(g.unit_initial_accuracy) : "uninformed";
    if (informed === 0) return "every node starts uninformed";
    return informed + " of " + g.nodes.length + " nodes start informed (units default: " + unitDefault + ")";
  }

  function summarizeConfig(cfg) {
    const g = cfg && cfg.org_graph;
    if (!g) return "No organization loaded.";
    const nodes = g.nodes || [];
    const totalReps = nodes.reduce((acc, n) => acc + Number(n.count || 1), 0);
    const dr = orgSetting(g, "decision_rights");
    return (
      nodes.length + " nodes (" + totalReps + " representations) and " + g.edges.length + " links. Decision rights: " +
      (DECISION_RIGHTS_LABELS[dr] || dr) + ". Market results: " + feedbackSummary(cfg) + ". Initial accuracy: " +
      accuracySummary(cfg) + ". Unit experiments " + fmt2(orgSetting(g, "experiment_rate")) +
      "; search after shortfalls " + fmt2(orgSetting(g, "aspiration_sensitivity")) +
      "; track-record learning " + fmt2(orgSetting(g, "lambda_track")) + "."
    );
  }

  function updateDesignReadout() {
    const box = document.getElementById("design-readout-box");
    if (!box) return;
    const cfg = activeConfig();
    const g = cfg && cfg.org_graph;
    let body = "Choose a template to start.";
    if (g) {
      body = g.description || "Custom organization.";
      const sel = selectedNode(cfg);
      if (sel) {
        body += " Selected node '" + (sel.label || sel.id) + "': " + (sel.model_type || "neural") + ", H = " + sel.h_agent + ", " +
          formatDomainBits(sel.jurisdiction_bits) + ", sees " +
          NODE_FEEDBACK_SHORT[displayFeedback(sel, nodeFeedbackMap(cfg)[sel.id] || effectiveFeedback(sel, g))] + ".";
      }
    }
    box.innerHTML =
      '<div class="design-readout-focus"><strong>' + escapeHtml(cfg ? cfg.name : "") + ":</strong> " + escapeHtml(body) + "</div>" +
      '<div class="design-readout-summary">' + escapeHtml(summarizeConfig(cfg)) + "</div>" +
      '<button type="button" class="design-readout-link" id="link-open-options-dict">See all settings and their definitions</button>';
    const openLink = document.getElementById("link-open-options-dict");
    if (openLink) {
      openLink.addEventListener("click", jumpToOptionsGuide);
    }
  }

  function resultFor(cfg) {
    return cfg ? (results.entries.find(e => e.key === cfg.key && e.uid === cfg.uid) || null) : null;
  }

  // Result entries whose configuration still exists; tables never show removed configurations.
  function liveEntries() {
    return results.entries.filter(e => configs.some(c => c.key === e.key && c.uid === e.uid));
  }

  // Label of a result entry: the configuration's current name, or its name at run time if it was removed.
  function resultLabel(entry) {
    const cfg = configs.find(c => c.key === entry.key && c.uid === entry.uid);
    return cfg ? cfg.name : entry.label;
  }

  // Chart legend: every configuration in its color; after a run, configurations added later are marked.
  function renderLegend() {
    const row = document.getElementById("chart-legend-row");
    if (!row) return;
    const ran = results.entries.length > 0;
    row.innerHTML = configs.map(cfg => {
      const has = !ran || Boolean(resultFor(cfg));
      return '<span class="legend-item"' + (has ? "" : ' style="opacity:0.55"') + ">" +
        '<span class="swatch" style="background:' + configColor(cfg) + '"></span>' +
        "<span>" + escapeHtml(cfg.name) + (has ? "" : " (not run)") + "</span></span>";
    }).join("");
  }

  const compileTimers = {};

  function scheduleCompile(cfg, delay) {
    if (!cfg) return;
    clearTimeout(compileTimers[cfg.uid]);
    compileTimers[cfg.uid] = setTimeout(() => { compileConfig(cfg); }, delay === undefined ? 250 : delay);
  }

  // Validate a configuration with the engine and keep its explanation (node feedback, accuracy, notes).
  async function compileConfig(cfg) {
    if (!cfg || !cfg.org_graph) return null;
    const key = JSON.stringify(cfg.org_graph);
    try {
      const res = await fetch("/api/compile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ org_graph: cfg.org_graph, market_config: buildMarketConfig() })
      });
      const data = await res.json();
      // The design changed while the request ran; a newer compile is scheduled.
      if (!configs.includes(cfg) || JSON.stringify(cfg.org_graph) !== key) return null;
      if (res.ok) {
        cfg.compiled = data;
        cfg.compiled_key = key;
        cfg.error = "";
      } else {
        cfg.compiled = null;
        cfg.compiled_key = "";
        cfg.error = data.error || "Invalid configuration";
      }
    } catch (err) {
      cfg.error = "Could not reach the server: " + String(err);
    }
    if (cfg === activeConfig()) {
      renderEditor();
      renderNodeInspector();
      renderNotes();
      renderErrors();
      updateDesignReadout();
    }
    return cfg.error ? null : cfg.compiled;
  }

  // Validate every configuration; returns the first one with an error, or null.
  async function compileAll() {
    for (const cfg of configs) {
      if (!(await compileConfig(cfg))) return cfg;
    }
    return null;
  }

  // Engine notes about the active configuration (for example, settings without effect) and file notes.
  function renderNotes() {
    const area = document.getElementById("constraint-notes-area");
    if (!area) return;
    const cfg = activeConfig();
    const exp = freshExplanation(cfg);
    const notes = (exp && Array.isArray(exp.notes) ? exp.notes : []).concat((cfg && cfg.notes) || []);
    area.innerHTML = notes.map(n => '<div class="notice-box">' + escapeHtml(String(n)) + "</div>").join("");
  }

  function renderErrors() {
    const cfg = activeConfig();
    const err = cfg && cfg.error ? cfg.error : "";
    const area = document.getElementById("compile-error-area");
    if (area) area.innerHTML = err ? '<div class="error-box">' + escapeHtml(err) + "</div>" : "";
    showEditorMsg(err, true);
  }

  function switchTopView(viewName) {
    activeTopView = viewName;
    document.getElementById("view-tab-lab").classList.toggle("active", viewName === "lab");
    document.getElementById("view-tab-guide").classList.toggle("active", viewName === "guide");
    document.getElementById("view-lab").hidden = viewName !== "lab";
    document.getElementById("view-guide").hidden = viewName !== "guide";
    if (viewName === "lab") {
      renderEditor();
      renderAllCharts();
    }
  }

  // Load a catalog template into a configuration; optional settings override organization-level fields.
  function applyTemplate(cfg, tplKey, settings) {
    if (!CATALOG || !CATALOG.graph_templates || !CATALOG.graph_templates[tplKey]) return false;
    cfg.template_key = tplKey;
    cfg.org_graph = normalizeGraph(cloneObj(CATALOG.graph_templates[tplKey]));
    if (currentNBits() !== 12) {
      rescaleGraphBits(cfg.org_graph, 12, currentNBits());
    }
    if (settings) {
      Object.assign(cfg.org_graph, settings);
    }
    cfg.layout = computeAutoLayout(cfg.org_graph);
    cfg.view = null;
    const hq = cfg.org_graph.nodes.find(n => n.role === "hq") || cfg.org_graph.nodes[0];
    cfg.sel_node = hq ? hq.id : null;
    cfg.sel_edge = null;
    cfg.compiled = null;
    cfg.compiled_key = "";
    cfg.error = "";
    cfg.notes = [];
    return true;
  }

  function markStale() {
    if (results.entries.length > 0 && !resultsStale) {
      resultsStale = true;
      showElement("results-stale-note", true);
    }
  }

  // Clear charts and result tables, for example after loading an experiment or a file.
  function resetResults() {
    results.entries = [];
    results.complete = null;
    results.t_steps = Number(market.t_steps);
    results.running = false;
    results.stopped = false;
    resultsStale = false;
    stopRunTimer();
    setPreparing(false);
    document.getElementById("progress-fill-bar").style.width = "0%";
    setElText("progress-status-text", "Ready");
    setElText("progress-time-text", "0.00s");
    renderResults();
  }

  // Replace the configurations with a ready-made experiment and clear the results.
  async function loadExperiment(expId) {
    const exp = EXPERIMENTS.find(x => x.id === expId);
    if (!exp) return;
    if (abortCtrl) {
      abortCtrl.abort();
      abortCtrl = null;
    }
    Object.assign(market, {
      nk_k: MARKET_DEFAULTS.nk_k,
      nk_pattern: MARKET_DEFAULTS.nk_pattern,
      turbulence: "none",
      p_shock: MARKET_DEFAULTS.p_shock,
      shock_period: MARKET_DEFAULTS.shock_period
    }, exp.market);
    market.nk_k = clampInt(market.nk_k, 0, currentNBits() - 1, Math.min(MARKET_DEFAULTS.nk_k, currentNBits() - 1));
    const missing = [];
    const loaded = [];
    exp.configs.forEach((spec, i) => {
      const cfg = makeConfig(CONFIG_KEYS[i], spec.name);
      if (!applyTemplate(cfg, spec.template, spec.settings || null)) {
        missing.push(spec.template);
        return;
      }
      if (spec.clearFeedback) clearNodeFeedbackOverrides(cfg);
      if (spec.dcRho !== undefined) {
        const hq = cfg.org_graph.nodes.find(n => n.role === "hq");
        if (hq) hq.rho = spec.dcRho;
      }
      loaded.push(cfg);
    });
    if (loaded.length > 0) {
      configs.splice(0, configs.length, ...loaded);
      activeIdx = 0;
    }
    switchTopView("lab");
    resetResults();
    renderAll();
    if (missing.length > 0) {
      document.getElementById("compile-error-area").innerHTML =
        '<div class="error-box">This experiment needs templates that are not available: ' + escapeHtml(missing.join(", ")) + ".</div>";
    }
    for (const cfg of loaded) {
      await compileConfig(cfg);
    }
  }

  function bindEvents() {
    document.getElementById("view-tab-lab").addEventListener("click", () => switchTopView("lab"));
    document.getElementById("view-tab-guide").addEventListener("click", () => switchTopView("guide"));
    document.getElementById("link-org-design-help").addEventListener("click", jumpToOptionsGuide);

    // Configurations and ready-made experiments.
    document.getElementById("btn-add-config").addEventListener("click", addConfig);
    document.getElementById("ctrl-experiment").addEventListener("change", async (e) => {
      const val = e.target.value;
      e.target.value = "";
      if (val) {
        await loadExperiment(val);
      }
    });

    document.getElementById("ctrl-graph-template").addEventListener("change", (e) => {
      const cfg = activeConfig();
      const val = e.target.value;
      if (!val || !cfg) return;
      const snap = snapshotState(cfg);
      if (applyTemplate(cfg, val, null)) {
        pushUndo(cfg, snap);
        markStale();
        closeEdgeEditor();
        renderAll();
        scheduleCompile(cfg, 0);
      }
    });

    // Organization settings live in the configuration's org_graph.
    const bindOrgSetting = (id, key, parse) => {
      document.getElementById(id).addEventListener("change", (e) => {
        const cfg = activeConfig();
        if (!cfg || !cfg.org_graph) return;
        const v = parse(e.target.value);
        mutateGraph(cfg, (g) => { g[key] = v; }, { keepTemplate: true });
      });
    };
    bindOrgSetting("ctrl-feedback-regime", "feedback_regime",
      v => (FEEDBACK_REGIME_LABELS[v] ? v : ORG_SETTING_DEFAULTS.feedback_regime));
    bindOrgSetting("ctrl-decision-rights", "decision_rights",
      v => (DECISION_RIGHTS_LABELS[v] ? v : ORG_SETTING_DEFAULTS.decision_rights));
    bindOrgSetting("ctrl-experiment-rate", "experiment_rate", v => clampNum(v, 0, 1, 0.5));
    bindOrgSetting("ctrl-aspiration", "aspiration_sensitivity", v => clampNum(v, 0, 5, 0));
    bindOrgSetting("ctrl-org-lambda", "lambda_track", v => clampNum(v, 0, 5, 0));
    bindOrgSetting("ctrl-unit-acc-mode", "unit_initial_accuracy",
      v => (v === "set" ? clampNum(document.getElementById("ctrl-unit-acc").value, 0, 1, 0.5) : null));
    const unitAcc = document.getElementById("ctrl-unit-acc");
    unitAcc.addEventListener("input", (e) => setElText("ctrl-unit-acc-value", fmt2(clampNum(e.target.value, 0, 1, 0.5))));
    bindOrgSetting("ctrl-unit-acc", "unit_initial_accuracy", v => clampNum(v, 0, 1, 0.5));

    // Editor toolbar, canvas, and keyboard.
    document.getElementById("btn-add-unit").addEventListener("click", () => addUnit(activeConfig(), null));
    document.getElementById("btn-add-manager").addEventListener("click", () => addManager(activeConfig()));
    document.getElementById("btn-auto-layout").addEventListener("click", () => autoLayout(activeConfig()));
    document.getElementById("btn-fit").addEventListener("click", () => {
      const cfg = activeConfig();
      if (!cfg) return;
      fitView(cfg);
      renderEditor();
    });
    document.getElementById("btn-undo").addEventListener("click", undo);
    document.getElementById("btn-delete").addEventListener("click", () => deleteSelection(activeConfig()));
    document.getElementById("btn-expand").addEventListener("click", () => {
      if (isOverlayOpen()) {
        closeOverlay();
      } else {
        openOverlay();
      }
    });
    document.getElementById("btn-close-overlay").addEventListener("click", closeOverlay);
    document.getElementById("editor-overlay").addEventListener("pointerdown", (e) => {
      if (e.target.id === "editor-overlay") closeOverlay();
    });
    const svg = document.getElementById("org-canvas");
    svg.addEventListener("pointerdown", onCanvasPointerDown);
    svg.addEventListener("pointermove", onCanvasPointerMove);
    svg.addEventListener("pointerup", onCanvasPointerUp);
    svg.addEventListener("pointercancel", onCanvasPointerUp);
    svg.addEventListener("pointerleave", hideCanvasTip);
    svg.addEventListener("dblclick", onCanvasDblClick);
    document.addEventListener("keydown", onKeyDown);

    // Link editor.
    document.getElementById("edge-editor-close").addEventListener("click", () => {
      const cfg = activeConfig();
      cfg.sel_edge = null;
      closeEdgeEditor();
      renderEditor();
    });
    const pSlider = document.getElementById("edge-ctrl-p");
    pSlider.addEventListener("input", (e) => onEdgeSliderInput("weight", e.target.value));
    pSlider.addEventListener("change", onEdgeSliderChange);
    const cSlider = document.getElementById("edge-ctrl-c");
    cSlider.addEventListener("input", (e) => onEdgeSliderInput("commensurability", e.target.value));
    cSlider.addEventListener("change", onEdgeSliderChange);
    document.getElementById("edge-btn-reverse").addEventListener("click", () => {
      const cfg = activeConfig();
      reverseEdge(cfg, cfg.sel_edge);
    });
    document.getElementById("edge-btn-delete").addEventListener("click", () => {
      const cfg = activeConfig();
      deleteEdge(cfg, cfg.sel_edge);
    });

    // Node inspector: every field writes to the selected node of the active configuration.
    const bindNodeField = (id, apply) => {
      document.getElementById(id).addEventListener("change", (e) => {
        const cfg = activeConfig();
        const node = selectedNode(cfg);
        if (!node) return;
        const nodeId = node.id;
        const value = e.target.value;
        mutateGraph(cfg, (g) => {
          const n = nodeById(g, nodeId);
          return n ? apply(n, value) : false;
        });
      });
    };
    bindNodeField("node-ctrl-label", (n, v) => { n.label = String(v).trim() || n.id; });
    bindNodeField("node-ctrl-count", (n, v) => {
      if (n.role === "hq") return false;
      n.count = clampInt(v, 1, 32, 1);
      return true;
    });
    bindNodeField("node-ctrl-model", (n, v) => {
      n.model_type = ALLOWED_MODEL_TYPES.includes(v) ? v : "neural";
      if (n.model_type === "linear") {
        n.h_agent = 0;
      } else if (n.model_type === "neural" && Number(n.h_agent) === 0) {
        n.h_agent = 16;
      }
      return true;
    });
    bindNodeField("node-ctrl-h", (n, v) => {
      n.h_agent = clampInt(v, 0, 64, 16);
      if (n.h_agent > 0 && n.model_type === "linear") {
        n.model_type = "neural";
      } else if (n.h_agent === 0 && n.model_type === "neural") {
        n.model_type = "linear";
      }
      return true;
    });
    bindNodeField("node-jurisdiction-preset", (n, v) => {
      if (v === "custom") return false;
      n.jurisdiction_bits = bitsForJurisdictionPreset(v);
      return true;
    });
    bindNodeField("node-ctrl-feedback", (n, v) => {
      if (n.role === "hq") return false;
      n.feedback = isFeedbackValue(v) ? v : null;
      return true;
    });
    bindNodeField("node-ctrl-acc-mode", (n, v) => {
      if (v === "set") {
        n.initial_accuracy = clampNum(document.getElementById("node-ctrl-acc").value, 0, 1, 0.5);
      } else if (v === "uninformed") {
        n.initial_accuracy = n.role === "hq" ? null : NODE_UNINFORMED;
      } else {
        n.initial_accuracy = null;
      }
      return true;
    });
    const nodeAcc = document.getElementById("node-ctrl-acc");
    nodeAcc.addEventListener("input", (e) => setElText("node-ctrl-acc-value", fmt2(clampNum(e.target.value, 0, 1, 0.5))));
    bindNodeField("node-ctrl-acc", (n, v) => { n.initial_accuracy = clampNum(v, 0, 1, 0.5); });
    bindNodeField("node-ctrl-lr", (n, v) => { n.lr = clampNum(v, 0.001, 2, 0.15); });
    bindNodeField("node-ctrl-sigma-learn", (n, v) => { n.sigma_learn = clampNum(v, 0, 2, 0.05); });
    bindNodeField("node-ctrl-rho", (n, v) => { n.rho = clampNum(v, 0, 1, 0); });
    bindNodeField("node-ctrl-search-steps", (n, v) => { n.search_steps = clampInt(v, 1, currentNBits(), 1); });
    bindNodeField("node-ctrl-agg", (n, v) => { n.upward_agg = ["mean", "max", "min"].includes(v) ? v : "mean"; });

    // Market environment shared by every configuration; changes affect results, not designs (except Choices N, which rescales domains).
    const onMarketChange = () => {
      markStale();
      syncMarketControls();
    };
    document.getElementById("ctrl-n-bits").addEventListener("change", (e) => {
      setNBits(e.target.value);
    });
    for (const chip of document.querySelectorAll("#nbits-chips .chip")) {
      chip.addEventListener("click", () => {
        setNBits(chip.dataset.value);
      });
    }
    const kSlider = document.getElementById("ctrl-nk-k");
    kSlider.addEventListener("input", (e) => setElText("ctrl-nk-k-value", String(clampInt(e.target.value, 0, currentNBits() - 1, 3))));
    kSlider.addEventListener("change", (e) => {
      market.nk_k = clampInt(e.target.value, 0, currentNBits() - 1, 3);
      onMarketChange();
    });
    document.getElementById("ctrl-nk-pattern").addEventListener("change", (e) => {
      market.nk_pattern = e.target.value === "modular" ? "modular" : "random";
      onMarketChange();
    });
    document.getElementById("ctrl-noise").addEventListener("change", (e) => {
      market.sigma_obs = Math.round(clampNum(e.target.value, 0, 3, 0.65) * 1000) / 1000;
      onMarketChange();
    });
    for (const chip of document.querySelectorAll("#noise-chips .chip")) {
      chip.addEventListener("click", () => {
        market.sigma_obs = Number(chip.dataset.value);
        onMarketChange();
      });
    }
    document.getElementById("ctrl-turbulence").addEventListener("change", (e) => {
      const v = e.target.value;
      market.turbulence = (v === "random" || v === "periodic") ? v : "none";
      if (market.turbulence === "random" && !(Number(market.p_shock) > 0)) market.p_shock = MARKET_DEFAULTS.p_shock;
      onMarketChange();
    });
    document.getElementById("ctrl-pshock").addEventListener("change", (e) => {
      market.p_shock = Math.round(clampNum(e.target.value, 0, MAX_P_SHOCK, MARKET_DEFAULTS.p_shock) * 10000) / 10000;
      onMarketChange();
    });
    document.getElementById("ctrl-shock-period").addEventListener("change", (e) => {
      market.shock_period = clampInt(e.target.value, 1, MAX_SHOCK_PERIOD, MARKET_DEFAULTS.shock_period);
      onMarketChange();
    });
    document.getElementById("ctrl-firms").addEventListener("change", (e) => {
      market.b_firms = clampInt(e.target.value, 1, MAX_FIRMS, MARKET_DEFAULTS.b_firms);
      onMarketChange();
    });
    for (const chip of document.querySelectorAll("#firms-chips .chip")) {
      chip.addEventListener("click", () => {
        market.b_firms = Number(chip.dataset.value);
        onMarketChange();
      });
    }
    document.getElementById("ctrl-tsteps").addEventListener("change", (e) => {
      market.t_steps = clampInt(e.target.value, MIN_HORIZON, MAX_HORIZON, MARKET_DEFAULTS.t_steps);
      onMarketChange();
    });
    document.getElementById("ctrl-eval-every").addEventListener("change", (e) => {
      market.eval_every = clampInt(e.target.value, 1, MAX_HORIZON, MARKET_DEFAULTS.eval_every);
      onMarketChange();
    });
    document.getElementById("ctrl-seed").addEventListener("change", (e) => {
      market.seed = clampInt(e.target.value, 0, MAX_SEED, MARKET_DEFAULTS.seed);
      onMarketChange();
    });

    document.getElementById("chart4-metric-select").addEventListener("change", (e) => {
      const titles = {
        false_stall_mean: "False Belief Stall Rate",
        v_percentile_mean: "Share of strategies beaten",
        orthogonality_linked_mean: "Heterogeneity (linked pairs)",
        at_local_peak_mean: "At a local peak (share of firms)",
        mean_permeability_mean: "Mean Permeability (P_ab)",
        emp_div_local_slope_mean: "Subunit Slope Diversity",
        v_pct_max_mean: "% of Landscape Max"
      };
      setElText("chart4-title", titles[e.target.value] || "Trajectory Metric");
      renderAllCharts();
    });

    document.getElementById("btn-run").addEventListener("click", startSimulationStream);
    document.getElementById("btn-stop").addEventListener("click", stopSimulationStream);
    document.getElementById("btn-download-json").addEventListener("click", exportJson);
    document.getElementById("btn-load-json").addEventListener("click", () => {
      document.getElementById("file-load-input").click();
    });
    document.getElementById("file-load-input").addEventListener("change", importJsonFile);
    window.addEventListener("resize", () => {
      const cfg = activeConfig();
      if (cfg && cfg.org_graph) fitView(cfg);
      renderEditor();
      renderAllCharts();
    });
  }

  // Elapsed-time display and preparation state of the current run.
  var runTimer = null;
  var runStartedAt = 0;
  var runPreparing = false;
  var PREPARING_TEXT = "Preparing the simulation (generating tasks and compiling the model)...";
  var STILL_PREPARING_TEXT = "Still preparing... this is normal for a new setup, many firms, or a long horizon";

  function startRunTimer() {
    stopRunTimer();
    runStartedAt = performance.now();
    runTimer = setInterval(() => {
      const sec = (performance.now() - runStartedAt) / 1000;
      setElText("progress-time-text", sec.toFixed(1) + "s");
      if (runPreparing && sec > 6) setElText("progress-status-text", STILL_PREPARING_TEXT);
    }, 200);
  }

  function stopRunTimer() {
    if (runTimer) {
      clearInterval(runTimer);
      runTimer = null;
    }
  }

  function setPreparing(on) {
    runPreparing = on;
    document.getElementById("progress-fill-bar").classList.toggle("indeterminate", on);
  }

  // One configuration streams as a single run; two to eight stream as a matched comparison.
  async function startSimulationStream() {
    if (abortCtrl) {
      abortCtrl.abort();
    }
    // Disable Run before the first await, so a double click cannot start two streams.
    document.getElementById("btn-run").disabled = true;
    setElText("progress-status-text", "Checking configurations...");
    const bad = await compileAll();
    if (bad) {
      document.getElementById("btn-run").disabled = false;
      const idx = configs.indexOf(bad);
      if (idx >= 0 && idx !== activeIdx) setActiveConfig(idx);
      setElText("progress-status-text", "Fix the error in " + bad.name + " first");
      return;
    }
    const runCtrl = new AbortController();
    abortCtrl = runCtrl;
    revealResults();
    const mc = buildMarketConfig();
    const runCfgs = configs.slice();
    results.entries = runCfgs.map(cfg => ({
      key: cfg.key, uid: cfg.uid, label: cfg.name, traj: emptyTraj(), summary: null, realized: null, paired: null
    }));
    results.complete = null;
    results.t_steps = mc.t_steps;
    results.running = true;
    results.stopped = false;
    resultsStale = false;
    renderResults();

    document.getElementById("btn-run").disabled = true;
    document.getElementById("btn-stop").disabled = false;
    document.getElementById("progress-fill-bar").style.width = "0%";
    setPreparing(true);
    setElText("progress-status-text", PREPARING_TEXT);
    setElText("progress-time-text", "0.0s");
    startRunTimer();

    const chunkSize = chunkSizeFor(mc.t_steps);
    const reqBody = runCfgs.length === 1
      ? { mode: "single", org_graph: runCfgs[0].org_graph, market_config: mc, chunk_size: chunkSize }
      : {
          mode: "comparison",
          market_config: mc,
          chunk_size: chunkSize,
          orgs: runCfgs.map(cfg => ({ slot_id: cfg.key, label: cfg.name, org_graph: cfg.org_graph }))
        };

    try {
      const response = await fetch("/api/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reqBody),
        signal: runCtrl.signal
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || ("HTTP " + response.status));
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\\n");
        buffer = lines.pop() || "";

        for (const rawLine of lines) {
          const line = rawLine.trim();
          if (!line) continue;
          const payloadStr = line.startsWith("data:") ? line.slice(5).trim() : line;
          if (!payloadStr) continue;
          handleStreamEvent(JSON.parse(payloadStr));
        }
      }
      if (buffer.trim()) {
        handleStreamEvent(JSON.parse(buffer.trim()));
      }
    } catch (err) {
      if (err.name === "AbortError") {
        // A run replaced by a loaded experiment or file keeps the status that the load set.
        if (abortCtrl === runCtrl) {
          setElText("progress-status-text", "Simulation stopped");
          results.stopped = true;
        }
      } else {
        setElText("progress-status-text", "Error: " + String(err.message || err));
        // Nothing arrived (for example a 400 from the server), so drop the empty entries and their "when the run completes" texts.
        if (!results.complete && results.entries.every(e => !e.traj || e.traj.steps.length === 0)) {
          results.entries = [];
          renderResults();
        }
      }
    } finally {
      document.getElementById("btn-run").disabled = false;
      document.getElementById("btn-stop").disabled = true;
      if (abortCtrl === runCtrl) {
        stopRunTimer();
        setPreparing(false);
      }
      // A load that replaced this run has already reset the results.
      if (abortCtrl === runCtrl) {
        results.running = false;
        renderResults();
      }
      abortCtrl = null;
    }
  }

  function stopSimulationStream() {
    if (abortCtrl) {
      abortCtrl.abort();
    }
  }

  // Once per run, scroll the Results section into view unless it already fills part of the window.
  function revealResults() {
    const sec = document.getElementById("results-section");
    if (!sec || isOverlayOpen() || activeTopView !== "lab") return;
    const r = sec.getBoundingClientRect();
    const visible = r.top < window.innerHeight - 160 && r.bottom > 160;
    if (!visible) sec.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function synthesizeTrajectoryFromSummary(summary) {
    if (!summary || !summary.v_true) return null;
    const ckKeys = Object.keys(summary.v_true).filter(
      k => k !== "t_final" && k.startsWith("t") && !isNaN(parseInt(k.slice(1), 10))
    );
    ckKeys.sort((a, b) => parseInt(a.slice(1), 10) - parseInt(b.slice(1), 10));
    if (ckKeys.length === 0) return null;
    const getMean = (obj, ck) => (obj && obj[ck] && obj[ck].mean !== undefined) ? Number(obj[ck].mean) : 0.0;
    const getSe = (obj, ck) => (obj && obj[ck] && obj[ck].se !== undefined) ? Number(obj[ck].se) : 0.0;
    // Metrics missing from the summary (for example the payoff references) stay empty, so nothing is drawn for them.
    const meansOf = (obj) => (obj ? ckKeys.map(ck => getMean(obj, ck)) : []);
    return {
      steps: ckKeys.map(ck => parseInt(ck.slice(1), 10)),
      v_true_mean: ckKeys.map(ck => getMean(summary.v_true, ck)),
      v_true_se: ckKeys.map(ck => getSe(summary.v_true, ck)),
      v_pct_max_mean: ckKeys.map(ck => getMean(summary.v_pct_max, ck)),
      v_percentile_mean: meansOf(summary.v_percentile),
      v_max_ref_mean: meansOf(summary.v_max_ref),
      v_local_ref_mean: meansOf(summary.v_local_ref),
      dc_r_local_mean: ckKeys.map(ck => getMean(summary.dc_r_local, ck)),
      false_stall_mean: ckKeys.map(ck => getMean(summary.false_stall, ck)),
      at_local_peak_mean: meansOf(summary.at_local_peak),
      emp_div_local_slope_mean: ckKeys.map(ck => getMean(summary.emp_div_local_slope, ck)),
      dc_acc_global_mean: ckKeys.map(ck => getMean(summary.dc_acc_global, ck)),
      dc_acc_global_se: ckKeys.map(ck => getSe(summary.dc_acc_global, ck)),
      dc_complexity_mean: ckKeys.map(ck => getMean(summary.dc_complexity, ck)),
      dc_complexity_se: ckKeys.map(ck => getSe(summary.dc_complexity, ck)),
      orthogonality_mean: ckKeys.map(ck => getMean(summary.orthogonality, ck)),
      orthogonality_se: ckKeys.map(ck => getSe(summary.orthogonality, ck)),
      orthogonality_linked_mean: ckKeys.map(ck => getMean(summary.orthogonality_linked, ck)),
      orthogonality_linked_se: ckKeys.map(ck => getSe(summary.orthogonality_linked, ck)),
      mean_permeability_mean: ckKeys.map(ck => getMean(summary.mean_permeability, ck))
    };
  }

  function entryForSlot(slotId) {
    return results.entries.find(e => e.key === slotId) || null;
  }

  function handleStreamEvent(ev) {
    if (ev.device_used) {
      setElText("device-active-badge", "Device: " + ev.device_used);
    }
    if (ev.event === "start") {
      results.t_steps = ev.t_steps || results.t_steps;
      // The server compiles the model and sets starting beliefs before the first steps arrive.
      if (runPreparing) setElText("progress-status-text", PREPARING_TEXT);
      return;
    }
    if (ev.event === "progress") {
      if (runPreparing) setPreparing(false);
      const pct = Math.round((ev.progress || 0) * 100);
      document.getElementById("progress-fill-bar").style.width = pct + "%";
      setElText("progress-status-text", "Step " + ev.step + " / " + ev.t_steps + " (" + pct + "%)");
      if (ev.chunks && typeof ev.chunks === "object") {
        for (const [slotId, chunk] of Object.entries(ev.chunks)) {
          const entry = entryForSlot(slotId);
          if (entry) appendChunk(entry.traj, chunk);
        }
      } else if (ev.chunk && results.entries[0]) {
        appendChunk(results.entries[0].traj, ev.chunk);
      }
      renderAllCharts();
      return;
    }
    if (ev.event === "complete") {
      stopRunTimer();
      setPreparing(false);
      document.getElementById("progress-fill-bar").style.width = "100%";
      setElText("progress-status-text", "Complete (" + results.t_steps + " / " + results.t_steps + " steps)");
      setElText("progress-time-text", Number(ev.wall_time_sec || 0).toFixed(2) + "s");
      applyCompleteEvent(ev);
      renderResults();
    }
  }

  // Keep the final summaries, trajectories, realized accuracies, and paired tests of a complete event.
  function applyCompleteEvent(ev) {
    results.complete = ev;
    if (Array.isArray(ev.orgs)) {
      ev.orgs.forEach((org, idx) => {
        const entry = entryForSlot(org.slot_id) || results.entries[idx];
        if (!entry) return;
        entry.traj = org.trajectories || synthesizeTrajectoryFromSummary(org.summary) || entry.traj;
        entry.summary = org.summary || null;
        entry.realized = org.initial_accuracy_realized || null;
        entry.paired = org.paired_vs_a || null;
      });
    } else if (results.entries[0]) {
      const entry = results.entries[0];
      entry.traj = ev.trajectories || synthesizeTrajectoryFromSummary(ev.summary) || entry.traj;
      entry.summary = ev.summary || null;
      entry.realized = ev.initial_accuracy_realized || null;
    }
    const first = results.entries[0];
    if (first && first.traj && first.traj.steps && first.traj.steps.length > 0) {
      results.t_steps = first.traj.steps[first.traj.steps.length - 1];
    }
  }

  // Redraw everything that shows results: legend, tables, charts, and the realized accuracy in the inspector.
  function renderResults() {
    renderLegend();
    renderPairedTable();
    renderCheckpointTable();
    renderAllCharts();
    showElement("results-stale-note", resultsStale && results.entries.length > 0);
    renderNodeInspector();
  }

  function renderAllCharts() {
    if (activeTopView !== "lab") return;
    const c4Key = document.getElementById("chart4-metric-select").value;
    drawTrajectoryChart("canvas-v", "v_true_mean", "v_true_se", "stat-badge-v", { refs: true, legendId: "payoff-ref-legend" });
    drawTrajectoryChart("canvas-acc", "dc_acc_global_mean", "dc_acc_global_se", "stat-badge-acc");
    drawTrajectoryChart("canvas-comp", "dc_complexity_mean", "dc_complexity_se", "stat-badge-comp");
    drawTrajectoryChart("canvas-orth", "orthogonality_mean", "orthogonality_se", "stat-badge-orth");
    drawTrajectoryChart("canvas-r", "dc_r_local_mean", null, "stat-badge-r");
    drawTrajectoryChart("canvas-fs", c4Key, null, "stat-badge-fs");
  }

  // Metrics stored as fractions and shown in percent.
  const PERCENT_METRICS = ["v_percentile_mean", "v_pct_max_mean"];
  // Payoff references, drawn once in gray from the first configuration because all configurations share the landscapes.
  const PAYOFF_REFS = [
    { key: "v_max_ref_mean", color: "#52525b", dash: [6, 3] },
    { key: "v_local_ref_mean", color: "#a1a1aa", dash: [2, 3] }
  ];

  function drawTrajectoryChart(canvasId, metricKey, seKey, badgeId, opts) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(260, Math.floor(rect.width));
    const height = Math.max(180, Math.floor(rect.height));
    canvas.width = width * dpr;
    canvas.height = height * dpr;

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const padLeft = 48;
    const padRight = 14;
    const padTop = 12;
    const padBottom = 28;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    // Series: every configuration with results from the last run, in display order.
    const series = [];
    for (const cfg of configs) {
      const entry = resultFor(cfg);
      const tr = entry && entry.traj;
      if (tr && tr.steps && tr.steps.length > 0 && Array.isArray(tr[metricKey]) && tr[metricKey].length > 0) {
        series.push({ traj: tr, color: CONFIG_COLORS[cfg.key] || CONFIG_COLORS.a });
      }
    }

    // Reference series of the payoff chart; absent keys (when absent) draw nothing and hide the legend.
    const refs = [];
    if (opts && opts.refs && series.length > 0) {
      const tr = series[0].traj;
      for (const ref of PAYOFF_REFS) {
        const arr = tr[ref.key];
        if (Array.isArray(arr) && arr.length > 0 && arr.length === tr.steps.length) {
          refs.push({ steps: tr.steps, vals: arr, color: ref.color, dash: ref.dash });
        }
      }
    }
    if (opts && opts.legendId) showElement(opts.legendId, refs.length > 0);

    const isPct = PERCENT_METRICS.includes(metricKey);
    const fmtVal = (v) => isPct
      ? (Number(v) * 100).toFixed(1) + "%"
      : (metricKey === "dc_complexity_mean" ? Number(v).toFixed(1) : Number(v).toFixed(3));

    // Badge: the latest value of every series, each in its configuration's color.
    const badgeEl = document.getElementById(badgeId);
    if (badgeEl) {
      if (series.length === 0) {
        badgeEl.innerHTML = "<span>t_final: --</span>";
      } else {
        const lastStep = series[0].traj.steps[series[0].traj.steps.length - 1];
        badgeEl.innerHTML = "<span>t=" + lastStep + ":</span> " + series.map(s =>
          '<span style="color:' + s.color.stroke + '">' + fmtVal(s.traj[metricKey][s.traj[metricKey].length - 1]) + "</span>"
        ).join(" ");
      }
    }

    // Y bounds over every series, including the standard-error bands.
    let yMin = Infinity;
    let yMax = -Infinity;
    const scanBounds = (tr) => {
      if (!tr || !tr[metricKey]) return;
      const arr = tr[metricKey];
      const seArr = seKey && tr[seKey] ? tr[seKey] : null;
      for (let i = 0; i < arr.length; i++) {
        const v = arr[i];
        const s = seArr ? seArr[i] : 0;
        if (v - s < yMin) yMin = v - s;
        if (v + s > yMax) yMax = v + s;
      }
    };
    for (const s of series) {
      scanBounds(s.traj);
    }
    // The y-axis always includes the reference lines.
    for (const ref of refs) {
      for (const v of ref.vals) {
        if (v < yMin) yMin = v;
        if (v > yMax) yMax = v;
      }
    }

    if (!isFinite(yMin) || !isFinite(yMax)) {
      yMin = 0.0;
      yMax = 1.0;
    } else if (Math.abs(yMax - yMin) < 1e-4) {
      yMin -= 0.1;
      yMax += 0.1;
    } else {
      const margin = 0.08 * (yMax - yMin);
      yMin -= margin;
      yMax += margin;
    }

    const maxStep = Math.max(10, results.t_steps || 600);
    const xOf = (step) => padLeft + ((step - 1) / Math.max(1, maxStep - 1)) * plotW;
    const yOf = (val) => padTop + (1 - (val - yMin) / (yMax - yMin)) * plotH;

    // Gridlines and Y ticks
    ctx.strokeStyle = "#e7e5df";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#71717a";
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    const nTicks = 4;
    const pctDigits = (yMax - yMin) * 100 < 8 ? 1 : 0;
    for (let i = 0; i <= nTicks; i++) {
      const val = yMin + (i / nTicks) * (yMax - yMin);
      const y = yOf(val);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(width - padRight, y);
      ctx.stroke();
      const label = isPct
        ? (val * 100).toFixed(pctDigits) + "%"
        : (metricKey === "dc_complexity_mean" && (yMax - yMin) >= 4 ? val.toFixed(0) : val.toFixed(2));
      ctx.fillText(label, padLeft - 6, y);
    }

    // X ticks
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const xTicks = [1, Math.round(maxStep * 0.25), Math.round(maxStep * 0.5), Math.round(maxStep * 0.75), maxStep];
    for (const xt of xTicks) {
      const x = xOf(xt);
      // The last label ends at the plot edge so it is not clipped.
      ctx.textAlign = xt === maxStep ? "right" : "center";
      ctx.fillText("t=" + xt, xt === maxStep ? x + 4 : x, padTop + plotH + 6);
    }

    // Axes border
    ctx.strokeStyle = "#c8c4b8";
    ctx.strokeRect(padLeft, padTop, plotW, plotH);

    const drawSeries = (tr, strokeColor, bandColor) => {
      if (!tr || !tr.steps || tr.steps.length === 0 || !tr[metricKey]) return;
      const steps = tr.steps;
      const vals = tr[metricKey];
      const ses = seKey && tr[seKey] ? tr[seKey] : null;

      if (ses && ses.length === vals.length) {
        ctx.fillStyle = bandColor;
        ctx.beginPath();
        for (let i = 0; i < steps.length; i++) {
          const x = xOf(steps[i]);
          const y = yOf(vals[i] + ses[i]);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        for (let i = steps.length - 1; i >= 0; i--) {
          const x = xOf(steps[i]);
          const y = yOf(vals[i] - ses[i]);
          ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.fill();
      }

      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      for (let i = 0; i < steps.length; i++) {
        const x = xOf(steps[i]);
        const y = yOf(vals[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    // Reference lines first, so the configurations' lines stay on top.
    for (const ref of refs) {
      ctx.strokeStyle = ref.color;
      ctx.lineWidth = 1.3;
      ctx.setLineDash(ref.dash);
      ctx.beginPath();
      for (let i = 0; i < ref.steps.length; i++) {
        const x = xOf(ref.steps[i]);
        const y = yOf(ref.vals[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    for (const s of series) {
      drawSeries(s.traj, s.color.stroke, s.color.band);
    }
  }

  function fmtNum(x, digits) {
    return (typeof x === "number" && isFinite(x)) ? x.toFixed(digits) : "--";
  }

  // One paired-test row; scale and digits format percent metrics (differences in percentage points).
  function appendPairedRow(tbody, r) {
    const scale = r.scale || 1;
    const digits = r.digits === undefined ? 4 : r.digits;
    const diff = Number(r.diff);
    const tVal = Number(r.t);
    const pVal = Number(r.p);
    const signStr = isFinite(diff) ? (diff >= 0 ? "+" : "") + (diff * scale).toFixed(digits) : "--";
    const tStr = isFinite(tVal) ? (tVal >= 0 ? "+" : "") + tVal.toFixed(2) : "--";
    // Highlight only real tests: with fewer than 2 firms the engine reports SE 0, t 0, and p 1.
    const cls = isFinite(pVal) && pVal < 0.05 && Number(r.se) > 0 ? (diff >= 0 ? "sig-pos" : "sig-neg") : "";
    const tr = document.createElement("tr");
    tr.innerHTML =
      "<td>" + escapeHtml(r.name) + "</td>" +
      '<td class="mono">' + escapeHtml(r.a) + "</td>" +
      '<td class="mono">' + escapeHtml(r.b) + "</td>" +
      '<td class="mono ' + cls + '">' + signStr + "</td>" +
      '<td class="mono">' + fmtNum(Number(r.se) * scale, digits) + "</td>" +
      '<td class="mono ' + cls + '">' + tStr + "</td>" +
      '<td class="mono">' + fmtNum(pVal, 6) + "</td>";
    tbody.appendChild(tr);
  }

  // Text of an empty results table: a run in progress, a stopped run, or nothing to show yet.
  function emptyTableText(kind, baseName) {
    if (results.running) {
      return kind === "paired" ? "Paired tests appear when the run completes." : "The table fills in when the run completes.";
    }
    if (results.stopped) return "Run stopped; press Run Simulation to start again.";
    return kind === "paired"
      ? "Press Run Simulation to compare each configuration with " + baseName + "."
      : "Press Run Simulation to stream trajectories.";
  }

  // Paired comparison: every configuration against the first one of the run, firm by firm (other minus first).
  function renderPairedTable() {
    const card = document.getElementById("paired-test-card");
    if (card) card.hidden = configs.length < 2;
    // The run's first configuration is the baseline; its comparisons are shown only while it still exists.
    const live = liveEntries();
    const base = results.entries.length > 0 && live.includes(results.entries[0]) ? results.entries[0] : null;
    const baseName = base ? resultLabel(base) : (configs[0] ? configs[0].name : "the first configuration");
    setElText("paired-test-title", "Paired comparison with " + baseName + " (same landscapes and noise draws)");
    document.getElementById("paired-test-thead").innerHTML =
      "<tr><th>Metric at t_final</th><th>" + escapeHtml(baseName) + " mean (SE)</th>" +
      "<th>Configuration mean (SE)</th><th>Difference (configuration minus " + escapeHtml(baseName) + ")</th>" +
      "<th>Paired SE</th><th>t</th><th>p</th></tr>";
    const tbody = document.getElementById("paired-test-tbody");
    tbody.innerHTML = "";
    const compared = base ? live.filter(e => e !== base && e.paired) : [];
    if (compared.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" style="color:var(--text-muted)">' + escapeHtml(emptyTableText("paired", baseName)) + "</td></tr>";
      return;
    }
    const fmtMeanSe = (obj, digits, scale) => (obj && obj.mean !== undefined)
      ? ((Number(obj.mean) * (scale || 1)).toFixed(digits) + " (" + (Number(obj.se) * (scale || 1)).toFixed(digits) + ")")
      : "--";
    const metricDefs = [
      { key: "v_true", pc: "v_true", name: "Enacted payoff V(t_final)", digits: 4 },
      { key: "v_percentile", pc: "v_percentile", name: "Share of strategies beaten (%)", digits: 1, scale: 100 },
      { key: "dc_acc_global", pc: "dc_acc_global", name: "Global accuracy (dc_acc_global)", digits: 4 },
      { key: "dc_complexity", pc: "dc_complexity", name: "Complexity (dc_complexity)", digits: 1 },
      { key: "orthogonality", pc: "orthogonality", name: "Heterogeneity (1 - |corr|)", digits: 4 },
      { key: "orthogonality_linked", pc: "orthogonality_linked", name: "Heterogeneity, linked pairs", digits: 4 },
      { key: "dc_r_local", pc: "r_local", name: "Local gradient correlation (r_local)", digits: 4 },
      { key: "false_stall", pc: "false_stall", name: "False belief stall rate", digits: 4 }
    ];
    const sumA = base.summary;
    for (const entry of compared) {
      const pc = entry.paired;
      const group = document.createElement("tr");
      group.className = "group-row";
      group.innerHTML = '<td colspan="7"><strong>' + escapeHtml(resultLabel(entry)) + "</strong> compared with " + escapeHtml(baseName) + "</td>";
      tbody.appendChild(group);
      for (const m of metricDefs) {
        if (pc[m.pc + "_diff"] === undefined) continue;
        appendPairedRow(tbody, {
          name: m.name,
          a: fmtMeanSe(sumA && sumA[m.key] && sumA[m.key].t_final, m.digits, m.scale),
          b: fmtMeanSe(entry.summary && entry.summary[m.key] && entry.summary[m.key].t_final, m.digits, m.scale),
          diff: pc[m.pc + "_diff"],
          se: pc[m.pc + "_se"],
          t: pc[m.pc + "_t_stat"],
          p: pc[m.pc + "_p_value"],
          scale: m.scale,
          digits: m.scale ? m.digits : undefined
        });
      }
    }
  }

  // Checkpoint table: every configuration of the run at each recorded checkpoint.
  function renderCheckpointTable() {
    const tbody = document.getElementById("checkpoint-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    const done = liveEntries().filter(e => e.summary);
    if (done.length === 0) {
      tbody.innerHTML = '<tr><td colspan="9" style="color:var(--text-muted)">' + escapeHtml(emptyTableText("checkpoints", "")) + "</td></tr>";
      return;
    }
    for (const entry of done) {
      appendCheckpointRows(tbody, resultLabel(entry), entry.summary);
    }
  }

  function appendCheckpointRows(tbody, label, summary) {
    if (!summary || !summary.v_true) return;
    const at = (key, ck) => (summary[key] && summary[key][ck]) || {};
    const pct = (x) => (typeof x === "number" && isFinite(x)) ? (x * 100).toFixed(1) + "%" : "--";
    for (const ck of Object.keys(summary.v_true).filter(k => k !== "t_final")) {
      const v = at("v_true", ck);
      const tr = document.createElement("tr");
      tr.innerHTML =
        "<td><strong>" + escapeHtml(label) + "</strong></td>" +
        '<td class="mono">' + escapeHtml(ck) + "</td>" +
        '<td class="mono">' + fmtNum(v.mean, 4) + " (" + fmtNum(v.se, 4) + ")</td>" +
        '<td class="mono">' + pct(at("v_percentile", ck).mean) + "</td>" +
        '<td class="mono">' + fmtNum(at("dc_acc_global", ck).mean, 4) + "</td>" +
        '<td class="mono">' + fmtNum(at("dc_complexity", ck).mean, 1) + "</td>" +
        '<td class="mono">' + fmtNum(at("orthogonality", ck).mean, 4) + "</td>" +
        '<td class="mono">' + fmtNum(at("dc_r_local", ck).mean, 4) + "</td>" +
        '<td class="mono">' + fmtNum(at("false_stall", ck).mean, 4) + "</td>";
      tbody.appendChild(tr);
    }
  }

  function roundedLayout(cfg) {
    const out = {};
    for (const [id, p] of Object.entries(cfg.layout || {})) {
      out[id] = { x: Math.round(p.x), y: Math.round(p.y) };
    }
    return out;
  }

  // Export JSON (schema 0.1.0): the market, every configuration with its canvas layout, and the latest results.
  // Results from an earlier setup are not saved, so a file never pairs results with a design that did not produce them.
  function exportJson() {
    const exportObj = {
      schema_version: EXPORT_SCHEMA_VERSION,
      market_config: buildMarketConfig(),
      configurations: configs.map(cfg => ({ name: cfg.name, org_graph: cfg.org_graph, layout: roundedLayout(cfg) })),
      simulation_output: resultsStale ? null : results.complete
    };
    const blob = new Blob([JSON.stringify(exportObj, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "org_neural_search_" + new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-") + ".json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // Hold market values from a file to the ranges of the market bar inputs.
  function sanitizeMarket() {
    market.n_bits = clampInt(market.n_bits, MIN_N_BITS, MAX_N_BITS, MARKET_DEFAULTS.n_bits);
    market.nk_k = clampInt(market.nk_k, 0, market.n_bits - 1, Math.min(MARKET_DEFAULTS.nk_k, market.n_bits - 1));
    if (market.nk_pattern !== "modular") market.nk_pattern = "random";
    market.sigma_obs = clampNum(market.sigma_obs, 0, 3, MARKET_DEFAULTS.sigma_obs);
    if (!["none", "random", "periodic"].includes(market.turbulence)) market.turbulence = "none";
    market.p_shock = clampNum(market.p_shock, 0, MAX_P_SHOCK, MARKET_DEFAULTS.p_shock);
    market.shock_period = clampInt(market.shock_period, 1, MAX_SHOCK_PERIOD, MARKET_DEFAULTS.shock_period);
    market.t_steps = clampInt(market.t_steps, MIN_HORIZON, MAX_HORIZON, MARKET_DEFAULTS.t_steps);
    market.eval_every = clampInt(market.eval_every, 1, MAX_HORIZON, MARKET_DEFAULTS.eval_every);
    market.b_firms = clampInt(market.b_firms, 1, MAX_FIRMS, MARKET_DEFAULTS.b_firms);
    market.seed = clampInt(market.seed, 0, MAX_SEED, MARKET_DEFAULTS.seed);
  }

  // Market bar labels, for notes about values adjusted on import.
  const MARKET_FIELD_LABELS = {
    n_bits: "Choices N",
    nk_k: "Interdependence K",
    nk_pattern: "Interactions",
    sigma_obs: "Feedback noise",
    turbulence: "Turbulence",
    p_shock: "Probability per period",
    shock_period: "Every T periods",
    t_steps: "Horizon",
    eval_every: "Log every",
    b_firms: "Iterations (firms)",
    seed: "Seed"
  };

  const OLD_FILE_MESSAGE = "Unsupported experiment file schema (expected 0.1.0).";

  async function importJsonFile(ev) {
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    try {
      let data = null;
      try {
        data = JSON.parse(await file.text());
      } catch (parseErr) {
        throw new Error("This file is not valid JSON.");
      }
      await loadExperimentFile(data);
    } catch (err) {
      showEditorMsg("Import failed: " + String(err.message || err), true);
      document.getElementById("compile-error-area").innerHTML =
        '<div class="error-box">Import failed: ' + escapeHtml(String(err.message || err)) + "</div>";
    } finally {
      ev.target.value = "";
    }
  }

  // Load an experiment file (schema 0.1.0): market, configurations with layouts, and saved results.
  async function loadExperimentFile(data) {
    if (!data || typeof data !== "object" || data.schema_version !== EXPORT_SCHEMA_VERSION || !Array.isArray(data.configurations)) {
      throw new Error(OLD_FILE_MESSAGE);
    }
    const n = data.configurations.length;
    if (n < 1 || n > MAX_CONFIGS) {
      throw new Error("An experiment file holds 1 to " + MAX_CONFIGS + " configurations; this one has " + n + ".");
    }
    const loaded = data.configurations.map((c, i) => {
      if (!c || !c.org_graph || !Array.isArray(c.org_graph.nodes) || c.org_graph.nodes.length === 0) {
        throw new Error("Configuration " + (i + 1) + " has no org_graph with nodes.");
      }
      const cfg = makeConfig(CONFIG_KEYS[i], String(c.name || "Configuration " + (i + 1)).slice(0, 80));
      cfg.org_graph = normalizeGraph(cloneObj(c.org_graph));
      cfg.layout = (c.layout && typeof c.layout === "object") ? cloneObj(c.layout) : {};
      const hq = cfg.org_graph.nodes.find(nd => nd.role === "hq") || cfg.org_graph.nodes[0];
      cfg.sel_node = hq.id;
      return cfg;
    });
    const mk = (data.market_config && typeof data.market_config === "object") ? data.market_config : {};
    Object.assign(market, MARKET_DEFAULTS);
    for (const k of Object.keys(MARKET_DEFAULTS)) {
      if (mk[k] !== undefined && mk[k] !== null) market[k] = mk[k];
    }
    sanitizeMarket();
    // Values outside the input ranges are adjusted; the saved results then belong to another market and are not shown.
    const adjusted = Object.keys(MARKET_DEFAULTS).filter(k => mk[k] !== undefined && mk[k] !== null && String(mk[k]) !== String(market[k]));
    if (adjusted.length > 0) {
      const note = "Market settings outside the input ranges were adjusted (" + adjusted.map(k => MARKET_FIELD_LABELS[k] || k).join(", ") +
        "), so the file's saved results were not loaded.";
      for (const cfg of loaded) cfg.notes = [note];
    }
    // A run in progress would otherwise overwrite the file's results when it completes.
    if (abortCtrl) {
      abortCtrl.abort();
      abortCtrl = null;
    }
    configs.splice(0, configs.length, ...loaded);
    activeIdx = 0;
    closeEdgeEditor();
    switchTopView("lab");
    resetResults();
    if (adjusted.length === 0) restoreResults(data.simulation_output);
    renderAll();
    for (const cfg of configs) {
      await compileConfig(cfg);
    }
  }

  // Show the results saved in a file when they match its configurations (same number, same order).
  function restoreResults(sim) {
    if (!sim || typeof sim !== "object") return;
    const orgs = Array.isArray(sim.orgs) ? sim.orgs : null;
    if ((orgs ? orgs.length : 1) !== configs.length) return;
    results.entries = configs.map((cfg, i) => ({
      key: cfg.key, uid: cfg.uid, label: cfg.name, traj: emptyTraj(), summary: null, realized: null, paired: null
    }));
    // Slot ids in the file can differ from the new configuration keys, so results map by position.
    applyCompleteEvent(orgs ? Object.assign({}, sim, { orgs: orgs.map((o, i) => Object.assign({}, o, { slot_id: configs[i].key })) }) : sim);
    results.complete = sim;
    setElText("progress-status-text", "Results loaded from the file");
    renderResults();
  }

  // Help popovers: every "?" button opens the shared popover on click (pinned) or on hover (preview).
  const helpState = { btn: null, pinned: false, hideTimer: null };

  function helpTextFor(btn) {
    return HELP[btn.dataset.help] || ["Help", btn.getAttribute("aria-label") || ""];
  }

  function showHelp(btn, pinned) {
    const pop = document.getElementById("help-pop");
    if (!pop) return;
    clearTimeout(helpState.hideTimer);
    if (helpState.btn && helpState.btn !== btn) helpState.btn.setAttribute("aria-expanded", "false");
    const [title, text] = helpTextFor(btn);
    pop.innerHTML = '<div class="help-pop-title">' + escapeHtml(title) + "</div>" + escapeHtml(text);
    pop.hidden = false;
    helpState.btn = btn;
    helpState.pinned = pinned;
    btn.setAttribute("aria-expanded", "true");
    btn.setAttribute("aria-describedby", "help-pop");
    // Below the button when there is room, otherwise above; always inside the window.
    const r = btn.getBoundingClientRect();
    const pw = pop.offsetWidth;
    const ph = pop.offsetHeight;
    let left = Math.min(Math.max(8, r.left + r.width / 2 - pw / 2), window.innerWidth - pw - 8);
    let top = r.bottom + 6;
    if (top + ph > window.innerHeight - 8) top = Math.max(8, r.top - ph - 6);
    pop.style.left = left + "px";
    pop.style.top = top + "px";
  }

  function hideHelp(returnFocus) {
    const pop = document.getElementById("help-pop");
    clearTimeout(helpState.hideTimer);
    if (pop) pop.hidden = true;
    const btn = helpState.btn;
    if (btn) {
      btn.setAttribute("aria-expanded", "false");
      btn.removeAttribute("aria-describedby");
      if (returnFocus) btn.focus();
    }
    helpState.btn = null;
    helpState.pinned = false;
  }

  function initHelp() {
    const pop = document.getElementById("help-pop");
    for (const btn of document.querySelectorAll(".help")) {
      btn.setAttribute("aria-expanded", "false");
      btn.setAttribute("aria-controls", "help-pop");
    }
    document.addEventListener("click", (ev) => {
      const btn = ev.target.closest ? ev.target.closest(".help") : null;
      if (btn) {
        ev.preventDefault();
        ev.stopPropagation();
        if (helpState.btn === btn && helpState.pinned) {
          hideHelp(false);
        } else {
          showHelp(btn, true);
        }
        return;
      }
      if (helpState.btn && !(pop && pop.contains(ev.target))) hideHelp(false);
    }, true);
    document.addEventListener("mouseover", (ev) => {
      const btn = ev.target.closest ? ev.target.closest(".help") : null;
      if (btn && !helpState.pinned) showHelp(btn, false);
      if (pop && (btn || pop.contains(ev.target))) clearTimeout(helpState.hideTimer);
    });
    document.addEventListener("mouseout", (ev) => {
      const from = ev.target.closest ? ev.target.closest(".help, .help-pop") : null;
      if (!from || helpState.pinned || !helpState.btn) return;
      helpState.hideTimer = setTimeout(() => {
        if (!helpState.pinned) hideHelp(false);
      }, 180);
    });
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape" && helpState.btn) {
        ev.preventDefault();
        hideHelp(true);
      }
    }, true);
    window.addEventListener("scroll", () => {
      if (helpState.btn) hideHelp(false);
    }, true);
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  initApp();
})();
</script>
</body>
</html>
"""


class OrgNeuralSearchRequestHandler(BaseHTTPRequestHandler):
  """HTTP request handler serving the single-page research UI and JSON/NDJSON APIs."""

  def log_message(self, format_str: str, *args: Any) -> None:
    # Keep standard server console quiet during streaming requests
    pass

  def _send_json(self, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json; charset=utf-8")
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", "no-cache")
    self.end_headers()
    self.wfile.write(body)

  def _read_json_body(self) -> dict[str, Any]:
    length = int(self.headers.get("Content-Length", "0"))
    if length <= 0:
      return {}
    raw = self.rfile.read(length).decode("utf-8")
    return json.loads(raw) if raw.strip() else {}

  def do_GET(self) -> None:
    parsed = urllib.parse.urlparse(self.path)
    path = parsed.path

    if path in ("/", "/index.html"):
      body = HTML_PAGE.encode("utf-8")
      self.send_response(200)
      self.send_header("Content-Type", "text/html; charset=utf-8")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)
      return

    if path == "/api/catalog":
      self._send_json(build_catalog_payload(), status=200)
      return

    self._send_json({"error": f"Not found: {path}"}, status=404)

  def do_POST(self) -> None:
    parsed = urllib.parse.urlparse(self.path)
    path = parsed.path

    try:
      data = self._read_json_body()
    except Exception as exc:
      self._send_json({"error": f"Invalid JSON request body: {exc}"}, status=400)
      return

    if path in ("/api/compile", "/api/stream") and not isinstance(data, dict):
      self._send_json({"error": "The request body must be a JSON object."}, status=400)
      return

    if path == "/api/compile":
      try:
        out = compile_config_payload(data)
        self._send_json(out, status=200)
      except Exception as exc:
        self._send_json({"error": str(exc)}, status=400)
      return

    if path == "/api/stream":
      self._handle_stream_request(data)
      return

    self._send_json({"error": f"Not found: {path}"}, status=404)

  def _handle_stream_request(self, data: dict[str, Any]) -> None:
    use_sse = (
        "text/event-stream" in self.headers.get("Accept", "")
        or str(data.get("format", "")).lower() == "sse"
    )

    try:
      # Parsed inside the try, so a bad value (for example "chunk_size": "abc") gets a 400 response.
      mode = str(data.get("mode", "single")).lower()
      chunk_size = int(data.get("chunk_size", 25))
      device = str(data.get("device", "auto"))
      if __package__:
        from .engine import stream_graph, stream_graph_sweep, stream_multi_graph_comparison
      else:
        from engine import stream_graph, stream_graph_sweep, stream_multi_graph_comparison  # type: ignore

      if mode == "comparison":
        # Two to eight configurations on one shared market_config, so every comparison is matched.
        raw_orgs = data.get("orgs")
        if not isinstance(raw_orgs, list):
          raise ValueError("A comparison needs an 'orgs' list of configurations, each with an 'org_graph'.")
        top_market = data.get("market_config")
        graphs_list: list[OrgGraph] = []
        labels_list: list[str] = []
        slots_list: list[str] = []
        shared_mkt: Optional[MarketConfig] = None
        for idx, item in enumerate(raw_orgs):
          if not isinstance(item, dict):
            raise ValueError(f"orgs[{idx}] must be an object with an 'org_graph'.")
          g_i, shared_mkt = resolve_graph_request({"org_graph": item.get("org_graph"), "market_config": top_market})
          s_id = str(item.get("slot_id") or chr(ord("a") + idx))
          graphs_list.append(g_i)
          labels_list.append(str(item.get("label") or g_i.name or f"Configuration {idx + 1}"))
          slots_list.append(s_id)
        event_iter = stream_multi_graph_comparison(
            graphs=graphs_list,
            market=shared_mkt,
            labels=labels_list,
            slot_ids=slots_list,
            chunk_size=chunk_size,
            device=device,
        )
      elif mode == "sweep":
        # Kept for API clients; the web laboratory compares configurations instead.
        graph_base, mkt_base = resolve_graph_request(data)
        raw_vals = data.get("sweep_values")
        if isinstance(raw_vals, str):
          sweep_values = [x.strip() for x in raw_vals.split(",") if x.strip()]
        elif isinstance(raw_vals, list):
          sweep_values = [x for x in raw_vals if x is not None and str(x).strip() != ""]
        else:
          raise ValueError("A sweep needs 'sweep_values' (a list or a comma-separated string).")
        event_iter = stream_graph_sweep(
            graph_base,
            mkt_base,
            sweep_param=str(data.get("sweep_param") or "rho").strip(),
            sweep_values=sweep_values,
            chunk_size=chunk_size,
            device=device,
        )
      elif mode == "single":
        graph_s, mkt_s = resolve_graph_request(data)
        event_iter = stream_graph(
            graph_s,
            market=mkt_s,
            chunk_size=chunk_size,
            device=device,
        )
      else:
        raise ValueError(f"Unknown mode {mode!r}; use single, comparison, or sweep.")

      first_event = next(event_iter)
    except (ImportError, ModuleNotFoundError):
      self._send_json(
          {
              "error": (
                  "JAX is required to run simulations. Install via: pip install"
                  " -r requirements.txt"
              )
          },
          status=400,
      )
      return
    except Exception as exc:
      self._send_json({"error": str(exc)}, status=400)
      return

    self.send_response(200)
    if use_sse:
      self.send_header("Content-Type", "text/event-stream; charset=utf-8")
    else:
      self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
    self.send_header("Cache-Control", "no-cache, no-transform")
    self.send_header("X-Accel-Buffering", "no")
    self.send_header("Connection", "close")
    self.end_headers()

    def write_event(ev_dict: dict[str, Any]) -> None:
      serialized = json.dumps(ev_dict)
      line = f"data: {serialized}\n\n" if use_sse else f"{serialized}\n"
      self.wfile.write(line.encode("utf-8"))
      self.wfile.flush()

    try:
      write_event(first_event)
      for ev in event_iter:
        write_event(ev)
    except (BrokenPipeError, ConnectionResetError):
      return


def format_server_url(server: ThreadingHTTPServer) -> str:
  """Format the base HTTP URL for a bound ThreadingHTTPServer (IPv4 or IPv6)."""
  addr = server.server_address
  bind_host = str(addr[0])
  bind_port = int(addr[1])
  host_str = f"[{bind_host}]" if ":" in bind_host else bind_host
  return f"http://{host_str}:{bind_port}"


def create_server(host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
  """Create and bind a ThreadingHTTPServer instance with automatic IPv4/IPv6 fallback."""
  candidates = [host]
  if host == "127.0.0.1":
    candidates.append("::1")
  elif host == "localhost":
    candidates.extend(["127.0.0.1", "::1"])

  last_err: Optional[Exception] = None
  for cand_host in candidates:
    family = socket.AF_INET6 if ":" in cand_host else socket.AF_INET
    try:
      server_cls = type(
          "BoundThreadingHTTPServer",
          (ThreadingHTTPServer,),
          {"address_family": family},
      )
      return server_cls((cand_host, port), OrgNeuralSearchRequestHandler)
    except OSError as exc:
      last_err = exc
  raise last_err or OSError(f"Failed to bind HTTP server on {host}:{port}")


def run_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    open_browser: bool = False,
) -> None:
  """Run the local interactive web server until interrupted."""
  server = create_server(host=host, port=port)
  url = format_server_url(server) + "/"
  print(f"Organizational Neural Search Laboratory running at: {url}")
  print("Press Ctrl+C to stop the server.")
  if open_browser:
    threading.Timer(0.25, lambda: webbrowser.open(url)).start()
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    print("\nShutting down server.")
  finally:
    server.server_close()


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="Launch the Organizational Neural Search local visual web interface.",
  )
  parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host address (default: 127.0.0.1).")
  parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080).")
  parser.add_argument("--no-browser", action="store_true", help="Do not open a browser window automatically.")
  return parser


def main(argv: Optional[list[str]] = None) -> int:
  parser = build_arg_parser()
  args = parser.parse_args(argv)
  run_server(host=args.host, port=args.port, open_browser=not args.no_browser)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
