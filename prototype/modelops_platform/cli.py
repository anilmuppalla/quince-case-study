"""Command-line entry point for the ModelOps Platform prototype.

Usage:

    python -m modelops_platform run scenarios/01_normal_load.json
    python -m modelops_platform run-all
    python -m modelops_platform list
    python -m modelops_platform report-html --output simulation-report.html

The CLI is deterministic: same scenario file, same output. That makes it
easy to demo, easy to test, and easy to verify in code review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from .autoscaling import compute_signals
from .canary import evaluate_canary
from .html_report import write_html_report
from .reporting import (
    build_cost_report,
    render_autoscale_signals,
    render_canary_decision,
    render_capacity_summary,
    render_cost_report,
    render_route_trace,
    render_scenario_list,
)
from .routing import apply_routing
from .scenarios import Scenario, load_scenario


SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"


def _print_section(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def _run_snapshot(
    *,
    name: str,
    description: str,
    models: dict,
    endpoints: list,
    requests: list,
    planned_events: dict,
    canary: dict | None = None,
    display_title: str | None = None,
    verbose: bool = True,
) -> dict:
    """Run one scenario snapshot and return a structured result dict."""
    if verbose:
        _print_section(display_title or name)
        if description:
            print(description)
        _print_section("Capacity And Targets")
        print(render_capacity_summary(endpoints, models))

    decisions = apply_routing(requests, models, endpoints)

    if verbose:
        _print_section("Route Trace")
        print(render_route_trace(decisions))

    cost_rows = build_cost_report(decisions, models)
    if verbose:
        _print_section("Cost-Per-Model Report")
        print(render_cost_report(cost_rows))

    signals = compute_signals(
        endpoints,
        models,
        planned_events=planned_events,
    )
    if verbose and signals:
        _print_section("Autoscaler Signals")
        print(render_autoscale_signals(signals))

    canary_decision = None
    if canary:
        canary_decision = evaluate_canary(
            candidate_version=canary["candidate_version"],
            candidate_metrics=canary["candidate_metrics"],
            control_metrics=canary["control_metrics"],
            guardrails=canary["guardrails"],
            rollback_to_version=canary.get("rollback_to_version"),
        )
        if verbose:
            _print_section("Canary Evaluation")
            print(render_canary_decision(canary_decision))

    return {
        "name": name,
        "decisions": [d.to_dict() for d in decisions],
        "cost_report": [c.__dict__ for c in cost_rows],
        "autoscale_signals": [s.__dict__ for s in signals],
        "canary_decision": canary_decision.__dict__ if canary_decision else None,
    }


def run_scenario(scenario: Scenario, *, verbose: bool = True) -> dict:
    """Run one scenario and return a structured result dict."""
    if scenario.phases:
        if verbose:
            _print_section(f"SCENARIO: {scenario.name}")
            if scenario.description:
                print(scenario.description)
        phase_results = []
        for phase in scenario.phases:
            phase_results.append(
                _run_snapshot(
                    name=phase.name,
                    description=phase.description,
                    models=scenario.models,
                    endpoints=phase.endpoints,
                    requests=phase.requests,
                    planned_events=phase.planned_events,
                    display_title=f"PHASE: {phase.name}",
                    verbose=verbose,
                )
            )
        return {
            "scenario": scenario.name,
            "phases": phase_results,
        }

    result = _run_snapshot(
        name=scenario.name,
        description=scenario.description,
        models=scenario.models,
        endpoints=scenario.endpoints,
        requests=scenario.requests,
        planned_events=scenario.planned_events,
        canary=scenario.canary,
        display_title=f"SCENARIO: {scenario.name}",
        verbose=verbose,
    )
    result["scenario"] = scenario.name
    return result


def cmd_run(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.path)
    result = run_scenario(scenario, verbose=not args.json_only)
    if args.json_only:
        print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    files = sorted(SCENARIOS_DIR.glob("*.json"))
    if not files:
        print(f"No scenarios found in {SCENARIOS_DIR}", file=sys.stderr)
        return 1
    for f in files:
        scenario = load_scenario(f)
        run_scenario(scenario)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    files = sorted(SCENARIOS_DIR.glob("*.json"))
    scenarios = [(f.name, load_scenario(f)) for f in files]
    print(render_scenario_list(scenarios))
    return 0


def cmd_report_html(args: argparse.Namespace) -> int:
    files = sorted(SCENARIOS_DIR.glob("*.json"))
    if not files:
        print(f"No scenarios found in {SCENARIOS_DIR}", file=sys.stderr)
        return 1
    write_html_report(files, args.output)
    abs_output = args.output.resolve()
    pages_dir = abs_output.with_name(f"{abs_output.stem}-scenarios")
    print(f"Report index : {abs_output}")
    print(f"Scenario pages: {pages_dir}")
    print(f"\nOpen in browser: file://{abs_output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modelops_platform",
        description="ModelOps Platform routing prototype.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a single scenario file.")
    p_run.add_argument("path", type=Path, help="Path to scenario JSON file.")
    p_run.add_argument(
        "--json-only", action="store_true", help="Emit JSON results only."
    )
    p_run.set_defaults(func=cmd_run)

    p_all = sub.add_parser("run-all", help="Run every scenario in scenarios/.")
    p_all.set_defaults(func=cmd_run_all)

    p_list = sub.add_parser("list", help="List bundled scenarios.")
    p_list.set_defaults(func=cmd_list)

    p_report = sub.add_parser(
        "report-html", help="Generate a standalone HTML simulation report."
    )
    p_report.add_argument(
        "--output",
        type=Path,
        default=Path("simulation-report.html"),
        help="Path to write the generated HTML report.",
    )
    p_report.set_defaults(func=cmd_report_html)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
