"""Cost-per-model report and route-trace formatting.

The cost-per-model report is the primary deliverable from the prompt:
"Include a simple cost report that shows cost-per-model after routing
decisions are made." Cloud providers bill the underlying GPU/instance-hours;
any per-request or per-1K value here is an allocation estimate derived from
that hourly cost and assumed safe throughput. This module shapes the report
and renders the human readable view.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import textwrap
from typing import Iterable

from .domain import AutoscaleSignal, CanaryDecision, Endpoint, Model, RouteDecision
from .policy import utilization_targets_for_model


@dataclass
class ModelCostRow:
    model_id: str
    owner_team: str
    routed_count: int = 0
    overflow_count: int = 0
    rejected_count: int = 0
    total_cost_usd: float = 0.0
    avg_estimated_latency_ms: float = 0.0
    _latency_sum: float = 0.0
    _latency_n: int = 0


def build_cost_report(
    decisions: Iterable[RouteDecision],
    models: dict[str, Model],
) -> list[ModelCostRow]:
    """Aggregate routing decisions into a per-model cost summary."""
    rows: dict[str, ModelCostRow] = {}
    for d in decisions:
        if d.model_id not in rows:
            owner = models[d.model_id].owner_team if d.model_id in models else "unknown"
            rows[d.model_id] = ModelCostRow(model_id=d.model_id, owner_team=owner)
        row = rows[d.model_id]
        if d.status == "routed":
            row.routed_count += 1
        elif d.status == "overflow_to_premium":
            row.overflow_count += 1
        else:
            row.rejected_count += 1

        if d.estimated_cost_usd is not None:
            row.total_cost_usd += d.estimated_cost_usd
        if d.estimated_latency_ms is not None:
            row._latency_sum += d.estimated_latency_ms
            row._latency_n += 1

    for row in rows.values():
        if row._latency_n:
            row.avg_estimated_latency_ms = round(row._latency_sum / row._latency_n, 2)
        row.total_cost_usd = round(row.total_cost_usd, 6)

    out = list(rows.values())
    out.sort(key=lambda r: r.model_id)
    return out


def _wrap_text(
    text: str,
    *,
    width: int = 88,
    initial_indent: str = "",
    subsequent_indent: str | None = None,
    break_on_underscores: bool = False,
) -> list[str]:
    """Wrap text deterministically for stable CLI output and tests."""
    if subsequent_indent is None:
        subsequent_indent = initial_indent
    if break_on_underscores and " " not in text and "_" in text:
        lines: list[str] = []
        current = ""
        tokens = text.split("_")
        for idx, token in enumerate(tokens):
            piece = token if idx == len(tokens) - 1 else f"{token}_"
            if current and len(current) + len(piece) > width:
                lines.append(f"{subsequent_indent if lines else initial_indent}{current}")
                current = piece
            else:
                current += piece
        if current:
            lines.append(f"{subsequent_indent if lines else initial_indent}{current}")
        return lines or [initial_indent.rstrip()]
    wrapped = textwrap.wrap(
        text,
        width=width,
        initial_indent=initial_indent,
        subsequent_indent=subsequent_indent,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped or [initial_indent.rstrip()]


def render_scenario_list(scenarios: Iterable[tuple[str, object]], *, width: int = 88) -> str:
    """Render bundled scenarios as readable cards with copyable run commands."""
    lines = ["Available Scenarios", "===================", ""]
    for filename, scenario in scenarios:
        name = getattr(scenario, "name", "unnamed")
        description = getattr(scenario, "description", "")
        lines.append(filename)
        lines.append(f"  Name: {name}")
        if description:
            lines.append("  Summary:")
            lines.extend(
                _wrap_text(
                    description,
                    width=max(20, width - 4),
                    initial_indent="    ",
                    subsequent_indent="    ",
                )
            )
        lines.append("  Run:")
        lines.append(f"    python -m modelops_platform run scenarios/{filename}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _fmt_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    wrap_columns: dict[str, int] | None = None,
) -> str:
    wrap_columns = wrap_columns or {}
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            header = headers[i]
            if header in wrap_columns:
                widths[i] = max(widths[i], min(wrap_columns[header], len(cell)))
            else:
                widths[i] = max(widths[i], len(cell))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    out = [line, sep]
    for row in rows:
        wrapped_cells = []
        row_height = 1
        for i, cell in enumerate(row):
            header = headers[i]
            if header in wrap_columns:
                wrapped = _wrap_text(
                    cell,
                    width=wrap_columns[header],
                    break_on_underscores=True,
                )
            else:
                wrapped = [cell]
            wrapped_cells.append(wrapped)
            row_height = max(row_height, len(wrapped))
        for line_idx in range(row_height):
            out.append(
                "  ".join(
                    (
                        wrapped_cells[i][line_idx]
                        if line_idx < len(wrapped_cells[i])
                        else ""
                    ).ljust(widths[i])
                    for i in range(len(headers))
                ).rstrip()
            )
    return "\n".join(out)


def render_capacity_summary(
    endpoints: Iterable[Endpoint],
    models: dict[str, Model],
) -> str:
    """Render current capacity and policy targets by model/backend deployment."""
    grouped: dict[tuple[str, str], list[Endpoint]] = defaultdict(list)
    for endpoint in endpoints:
        grouped[(endpoint.model_id, endpoint.serving_backend)].append(endpoint)

    headers = [
        "model",
        "backend",
        "replicas",
        "healthy",
        "qps/cap",
        "p99_target_ms",
        "max_gpu%",
        "gpu_target%",
        "max_cpu%",
        "cpu_target%",
    ]
    rows: list[list[str]] = []
    for (model_id, backend), eps in sorted(grouped.items()):
        model = models.get(model_id)
        targets = utilization_targets_for_model(model)
        healthy = [e for e in eps if e.healthy]
        total_qps = sum(e.current_qps for e in healthy)
        total_capacity = sum(e.capacity_qps for e in healthy)
        rows.append(
            [
                model_id,
                backend,
                str(len(eps)),
                str(len(healthy)),
                f"{total_qps}/{total_capacity}",
                str(model.sla_ms) if model is not None else "-",
                f"{max((e.gpu_util_pct for e in healthy), default=0):.1f}",
                str(targets["gpu"]),
                f"{max((e.cpu_util_pct for e in healthy), default=0):.1f}",
                str(targets["cpu"]),
            ]
        )
    return _fmt_table(headers, rows)


def render_route_trace(decisions: Iterable[RouteDecision]) -> str:
    headers = [
        "req",
        "model",
        "status",
        "endpoint",
        "p99",
        "gpu_cost_per_1k",
        "reason",
    ]
    rows: list[list[str]] = []
    for d in decisions:
        cost_per_1k = (
            f"{d.estimated_cost_usd * 1000:.4f}"
            if d.estimated_cost_usd is not None
            else "-"
        )
        rows.append(
            [
                d.request_id,
                d.model_id,
                d.status,
                d.chosen_endpoint or "-",
                f"{d.estimated_latency_ms:.2f}" if d.estimated_latency_ms else "-",
                cost_per_1k,
                d.route_reason,
            ]
        )
    note = (
        "Cost note: gpu_cost_per_1k = GPU-hour price / (safe allocation "
        "RPS * 3600 / 1000); cloud billing remains GPU/instance-hours."
    )
    return "\n".join(_wrap_text(note, width=100)) + "\n" + _fmt_table(
        headers, rows, wrap_columns={"reason": 39}
    )


def render_cost_report(rows: Iterable[ModelCostRow]) -> str:
    headers = [
        "model",
        "owner",
        "routed",
        "overflow",
        "rejected",
        "avg_latency_ms",
        "total_cost_usd",
    ]
    out_rows: list[list[str]] = []
    total_cost = 0.0
    for r in rows:
        total_cost += r.total_cost_usd
        out_rows.append(
            [
                r.model_id,
                r.owner_team,
                str(r.routed_count),
                str(r.overflow_count),
                str(r.rejected_count),
                f"{r.avg_estimated_latency_ms:.2f}",
                f"{r.total_cost_usd:.6f}",
            ]
        )
    table = _fmt_table(headers, out_rows)
    return f"{table}\n\nTOTAL cost (USD): {total_cost:.6f}"


def render_autoscale_signals(signals: Iterable[AutoscaleSignal]) -> str:
    headers = [
        "model",
        "backend",
        "current",
        "recommended",
        "gpu_util%",
        "gpu_target%",
        "cpu_target%",
        "load%",
        "reason",
    ]
    rows: list[list[str]] = []
    for s in signals:
        rows.append(
            [
                s.model_id,
                s.serving_backend,
                str(s.current_replicas),
                str(s.recommended_replicas),
                f"{s.aggregate_gpu_util_pct:.1f}",
                str(s.target_gpu_util_pct),
                str(s.target_cpu_util_pct),
                f"{s.aggregate_load_pct:.1f}",
                s.reason,
            ]
        )
    return _fmt_table(headers, rows, wrap_columns={"reason": 36})


def render_canary_decision(decision: CanaryDecision) -> str:
    lines = [
        f"candidate_version : {decision.candidate_version}",
        f"decision          : {decision.decision}",
        f"reason            : {decision.reason}",
    ]
    if decision.rollback_to_version:
        lines.append(f"rollback_to       : {decision.rollback_to_version}")
    if decision.effective_guardrails:
        lines.append("effective_guardrails :")
        for k, v in sorted(decision.effective_guardrails.items()):
            lines.append(f"  {k} = {v}")
    if decision.failed_guardrails:
        lines.append("failed_guardrails :")
        for f in decision.failed_guardrails:
            lines.append(f"  - {f}")
    return "\n".join(lines)
