"""Command line: launch the web laboratory, or run an exported experiment file headlessly."""

import argparse
import json
import sys
from typing import Any, Optional

try:
  from .org_graph import MarketConfig, OrgGraph
except ImportError:
  from org_graph import MarketConfig, OrgGraph  # type: ignore

EXPERIMENT_SCHEMA_VERSION = "0.1.0"


def load_experiment(path: str) -> tuple[MarketConfig, list[tuple[str, OrgGraph]], list[dict[str, Any]]]:
  """Read a web-laboratory export (schema 0.1.0).

  Returns the market, the 1 to 8 named configurations, and the raw configuration entries (which
  also carry web-laboratory fields such as the canvas layout).
  """
  with open(path, "r", encoding="utf-8") as f:
    try:
      data = json.load(f)
    except ValueError as exc:
      raise ValueError(f"{path}: not a valid JSON file ({exc})") from exc
  version = data.get("schema_version") if isinstance(data, dict) else None
  if version != EXPERIMENT_SCHEMA_VERSION:
    raise ValueError(f"{path}: schema_version {version!r} is not supported (expected {EXPERIMENT_SCHEMA_VERSION!r})")
  configs = data.get("configurations")
  if not isinstance(configs, list) or not 1 <= len(configs) <= 8:
    count = len(configs) if isinstance(configs, list) else 0
    raise ValueError(f"{path}: expected a list of 1 to 8 configurations, found {count}")
  try:
    market = MarketConfig.from_dict(data.get("market_config") or {})
  except (TypeError, ValueError) as exc:
    raise ValueError(f"{path}: {exc}") from exc
  named = []
  for i, entry in enumerate(configs):
    if not isinstance(entry, dict) or not isinstance(entry.get("org_graph"), dict):
      raise ValueError(f"{path}: configuration {i + 1} must be an object with an 'org_graph' object")
    name = str(entry.get("name") or f"Configuration {i + 1}")
    try:
      graph = OrgGraph.from_dict(entry["org_graph"])
    except (TypeError, ValueError) as exc:
      raise ValueError(f"{path}: configuration {i + 1} ({name}): {exc}") from exc
    named.append((name, graph))
  return market, named, configs


def run_experiment(market: MarketConfig, named: list[tuple[str, OrgGraph]]) -> dict[str, Any]:
  """Run every configuration on the same firms and return the final event of the run.

  The result has the form the web laboratory stores as simulation_output: the complete event of a
  single run (one configuration) or of a paired comparison (2 to 8 configurations).
  """
  try:
    from . import engine
  except ImportError:
    import engine  # type: ignore

  names = [name for name, _ in named]
  graphs = [graph for _, graph in named]
  # One chunk covers the whole horizon: no progress display is needed without a browser
  if len(graphs) == 1:
    events = engine.stream_graph(graphs[0], market, chunk_size=market.t_steps)
  else:
    events = engine.stream_multi_graph_comparison(graphs, market=market, labels=names, chunk_size=market.t_steps)
  complete: dict[str, Any] = {}
  for event in events:
    complete = event
  return complete


def format_summary(results: dict[str, Any], names: list[str]) -> str:
  """Markdown table: terminal payoff per configuration and paired tests vs the first one."""
  orgs = results.get("orgs") or [{"summary": results["summary"], "paired_vs_a": None}]
  lines = [
      f"| Configuration | Terminal V (SE) | Paired diff vs {names[0]} (SE) | t | p |",
      "| :--- | :--- | :--- | :--- | :--- |",
  ]
  for name, org in zip(names, orgs):
    v_final = org["summary"]["v_true"]["t_final"]
    paired = org["paired_vs_a"]
    if paired is None:
      diff_cell, t_cell, p_cell = "-", "-", "-"
    else:
      diff_cell = f"{paired['v_true_diff']:+.3f} ({paired['v_true_se']:.3f})"
      t_cell = f"{paired['v_true_t_stat']:.2f}"
      p_cell = f"{paired['v_true_p_value']:.4f}"
    lines.append(f"| {name} | {v_final['mean']:.3f} ({v_final['se']:.3f}) | {diff_cell} | {t_cell} | {p_cell} |")
  market = results["market_config"]
  lines.append("")
  lines.append(
      f"{market['b_firms']} firms x {market['t_steps']} periods, NK K={market['nk_k']} ({market['nk_pattern']}), "
      f"{results['wall_time_sec']:.1f} s"
  )
  return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog="org-neural-search",
      description="Organizational neural search laboratory: web interface and headless experiment runs.",
  )
  sub = parser.add_subparsers(dest="command", required=True)
  ui = sub.add_parser("ui", help="launch the web laboratory")
  ui.add_argument("--port", type=int, default=8080, help="port to listen on (default 8080)")
  ui.add_argument("--no-browser", action="store_true", help="do not open a browser window")
  run = sub.add_parser("run", help="run an experiment file exported from the web laboratory")
  run.add_argument("experiment", help="experiment JSON (schema 0.1.0)")
  run.add_argument("--out", default=None, help="write the full results JSON to this path")
  return parser


def main(argv: Optional[list[str]] = None) -> int:
  args = build_arg_parser().parse_args(argv)
  if args.command == "ui":
    try:
      from .webapp import run_server
    except ImportError:
      from webapp import run_server  # type: ignore
    run_server(host="127.0.0.1", port=args.port, open_browser=not args.no_browser)
    return 0
  try:
    market, named, entries = load_experiment(args.experiment)
    results = run_experiment(market, named)
  except (OSError, ValueError, KeyError, RuntimeError) as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  print(format_summary(results, [name for name, _ in named]))
  if not args.out:
    return 0
  # The same file format as Export JSON in the web laboratory: configurations keep their canvas
  # layout, and simulation_output holds the run's complete event
  payload = {
      "schema_version": EXPERIMENT_SCHEMA_VERSION,
      "market_config": results["market_config"],
      "configurations": [dict(entry, name=name, org_graph=graph.to_dict()) for entry, (name, graph) in zip(entries, named)],
      "simulation_output": results,
  }
  try:
    with open(args.out, "w", encoding="utf-8") as f:
      json.dump(payload, f, indent=2)
  except OSError as exc:
    print(f"error: could not write {args.out}: {exc}", file=sys.stderr)
    return 2
  print(f"\nResults written to {args.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
