"""Standalone HTML report renderer for the prototype scenarios."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Iterable

from .autoscaling import compute_signals
from .canary import evaluate_canary
from .domain import AutoscaleSignal, CanaryDecision, Endpoint, Model, RouteDecision
from .policy import utilization_targets_for_model
from .reporting import ModelCostRow, build_cost_report
from .routing import apply_routing
from .scenarios import Scenario, ScenarioPhase, load_scenario


@dataclass
class SnapshotResult:
    name: str
    description: str
    endpoints: list[Endpoint]
    decisions: list[RouteDecision]
    cost_rows: list[ModelCostRow]
    autoscale_signals: list[AutoscaleSignal]
    canary_decision: CanaryDecision | None = None


@dataclass
class MetricTimestep:
    """One step on the metric x-axis used by scenario charts."""

    label: str
    rps: int
    after_replicas: int
    before_replicas: int
    after_hourly_cost: float
    before_hourly_cost: float


@dataclass
class ModelTrafficProfile:
    """Per-model capacity and cost profile used by the 24-hour simulation."""

    model_id: str
    owner_team: str
    capacity_qps: float
    target_util_pct: int
    base_replicas: int
    per_replica_hourly_cost: float
    raw_capacity_qps: float
    sla_ms: int


@dataclass(frozen=True)
class InstanceCostAssumption:
    """Reviewer-facing infrastructure and pricing context for a mock instance."""

    instance_type: str
    accelerator: str
    public_hourly_usd: float
    pricing_basis: str


@dataclass
class TrafficWindowResult:
    """One aggregate time window in the reviewer-facing 24-hour simulation."""

    label: str
    start_hour: float
    end_hour: float
    duration_hours: float
    event: str
    total_rps: int
    fixed_capacity_rps: int
    platform_capacity_rps: int
    manual_capacity_rps: int
    fixed_replicas: int
    platform_replicas: int
    recommended_replicas: int
    manual_replicas: int
    fixed_hourly_cost: float
    platform_hourly_cost: float
    manual_hourly_cost: float
    fixed_cumulative_cost: float
    platform_cumulative_cost: float
    manual_cumulative_cost: float
    fixed_at_risk_requests: int
    platform_served_requests: int
    platform_overflow_requests: int
    platform_rejected_requests: int
    fixed_p99_ms: float
    platform_p99_ms: float
    sla_target_ms: int
    scale_action: str
    rps_by_model: dict[str, int]
    platform_replicas_by_model: dict[str, int]
    recommended_replicas_by_model: dict[str, int]
    manual_replicas_by_model: dict[str, int]
    trigger_util_pct_by_model: dict[str, float]
    post_scale_util_pct_by_model: dict[str, float]


_BEFORE_OVER_PROVISION_FACTOR = 1.6
"""Static over-provisioning multiplier used for the ``Before platform`` baseline.

Pre-platform serving teams typically provision for peak with a safety buffer
because there is no shared autoscaling/right-sizing surface. 1.6x of peak
adaptive replicas is a defensible illustrative ratio.
"""

_SECONDS_PER_HOUR = 3600

_INSTANCE_COST_ASSUMPTIONS = {
    "gpu_a10": InstanceCostAssumption(
        instance_type="AWS g5.xlarge",
        accelerator="NVIDIA A10G 24GB",
        public_hourly_usd=1.006,
        pricing_basis="us-east-1 Linux on-demand",
    )
}


def _pct(value: float, target: float) -> int:
    if target <= 0:
        return 0
    return max(0, min(100, int(round(100 * value / target))))


def _bar(value: float, target: float, label: str) -> str:
    width = _pct(value, target)
    return (
        '<div class="metric-bar" aria-label="'
        + escape(label)
        + '"><div class="bar-fill" style="width: '
        + str(width)
        + '%"></div></div>'
    )


def _slug(text: str, index: int) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")
    return f"scenario-{index}-{clean or 'unnamed'}"


def _scenario_page_filename(scenario: Scenario, index: int) -> str:
    return f"{_slug(scenario.name, index)}.html"


def _scenario_pages_dir(output_path: Path) -> Path:
    stem = output_path.stem if output_path.suffix else output_path.name
    return output_path.with_name(f"{stem}-scenarios")


def _money(value: float) -> str:
    return f"${value:.4f}"


def _money2(value: float) -> str:
    return f"${value:.2f}"


def _compact_int(value: float) -> str:
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(round(value)))


def _hour_label(value: float) -> str:
    hours = int(value)
    minutes = int(round((value - hours) * 60))
    if minutes == 60:
        hours += 1
        minutes = 0
    return f"{hours:02d}:{minutes:02d}"


def _window_range_label(start_hour: float, end_hour: float) -> str:
    return f"{_hour_label(start_hour)}-{_hour_label(end_hour)}"


def _compact_window_axis_label(start_hour: float, end_hour: float) -> str:
    duration = end_hour - start_hour
    if duration <= 0.3:
        return _hour_label(start_hour)
    if start_hour.is_integer() and end_hour.is_integer():
        return f"{int(start_hour)}-{int(end_hour)}h"
    if end_hour.is_integer():
        return f"{_hour_label(start_hour)}-{int(end_hour)}h"
    return _window_range_label(start_hour, end_hour)


def _portfolio_axis_label(name: str, index: int) -> str:
    lower = name.lower()
    if "normal pilot" in lower:
        suffix = "Normal"
    elif "organic" in lower:
        suffix = "Spike"
    elif "planned promotion" in lower:
        suffix = "Promo"
    elif "endpoint unavailability" in lower:
        suffix = "Unavailable"
    elif "auto-promote" in lower:
        suffix = "Canary pass"
    elif "owner review" in lower:
        suffix = "Review"
    elif "cpu-backed" in lower:
        suffix = "CPU"
    else:
        suffix = name.split()[0] if name.split() else "Scenario"
    return f"S{index}: {suffix}"


_SCENARIO_TYPE_DETAILS = {
    "traffic-cost": {
        "label": "Traffic / capacity / cost",
        "heading": "Traffic / Capacity / Cost Scenarios",
        "description": (
            "These scenarios are valid cost-attribution cases because the "
            "core evidence is demand, safe serving capacity, autoscaling, "
            "service risk, and replica-hour cost."
        ),
        "focus": "24-hour demand, autoscaling, service risk, and attributed serving cost",
        "cost_coupled": True,
    },
    "release-confidence": {
        "label": "Release confidence",
        "heading": "Release Confidence Scenarios",
        "description": (
            "These scenarios validate the developer rollout process. Cost is "
            "not the success metric; the proof is whether the platform makes "
            "the right canary promotion or review decision."
        ),
        "focus": "canary guardrails, owner review, and auditable rollout decisions",
        "cost_coupled": False,
    },
}


def _scenario_type_key(scenario: Scenario) -> str:
    if scenario.canary is not None:
        return "release-confidence"
    return "traffic-cost"


def _scenario_type_details(scenario: Scenario) -> dict[str, Any]:
    return _SCENARIO_TYPE_DETAILS[_scenario_type_key(scenario)]


def _scenario_cost_coupled(scenario: Scenario) -> bool:
    return bool(_scenario_type_details(scenario)["cost_coupled"])


def _scenario_short_title(name: str) -> str:
    return (
        name.replace(" (strict_realtime defaults)", "")
        .replace("Canary Evaluation - ", "Canary: ")
    )


def _model_display_name(model_id: str) -> str:
    if model_id.startswith("search-ranking"):
        return "Search Ranking"
    if model_id.startswith("recommendations"):
        return "Recommendations"
    if model_id.startswith("fraud-detection"):
        return "Fraud Detection"
    return model_id.replace("-v1", "").replace("-", " ").title()


def _scenario_model_label(scenario: Scenario) -> str:
    if not scenario.models:
        return "Model"
    labels = [
        _model_display_name(model.model_id)
        for model in sorted(scenario.models.values(), key=lambda item: item.model_id)
    ]
    return " + ".join(labels)


def _scenario_header_subtitle(scenario: Scenario) -> str:
    label = _scenario_model_label(scenario)
    if _scenario_type_key(scenario) == "release-confidence":
        return (
            f"{label} release-confidence walkthrough. Operational, statistical, "
            "and business guardrails are the decision evidence."
        )
    return (
        f"{label} serving walkthrough. Demand is context; GPU/CPU utilization, "
        "latency, health, and planned events are the decision evidence."
    )


def _scenario_review_summary(scenario: Scenario, index: int) -> dict[str, str]:
    name = scenario.name.lower()
    if "normal load" in name:
        value = "Right-size the normal daily Search curve instead of holding excess GPU capacity all day."
        evidence = "24h demand, GPU-target scale up/down, replica-hours, and cumulative cost."
        requirement = "Cost + autoscaling"
    elif "organic traffic spike" in name:
        value = "React to an unplanned spike before fixed capacity creates SLA risk."
        evidence = "GPU target breach, fixed-fleet risk, scale-out, and post-scale cost."
        requirement = "Spike handling"
    elif "planned promotion" in name:
        value = "Pre-warm Recommendations for a known flash-sale event, then scale down after demand tapers."
        evidence = "Planned-event trigger, peak serving capacity, and avoided all-day peak cost."
        requirement = "Planned events"
    elif "endpoint unavailability" in name:
        value = "Apply the rolling replacement pattern to runtime healing: replace the unhealthy pod rather than waiting for it to recover, and preserve Search SLA on healthy or premium capacity during the readiness gap."
        evidence = "Readiness signal, `replace_unhealthy_replicas` autoscaler recommendation, SLA-safe overflow routing, and incident cost."
        requirement = "Health-driven rolling replacement"
    elif "auto-promote" in name:
        value = "Let a Fraud Detection candidate promote automatically when strict realtime guardrails pass."
        evidence = "Operational, quality, business, and sample-size guardrails within bands."
        requirement = "Safe rollout"
    elif "owner review" in name:
        value = "Stop automatic Fraud promotion when statistical quality or business guardrails regress."
        evidence = "Strict realtime canary decision with failed quality/business retention checks."
        requirement = "Owner review"
    else:
        value = "Exercise platform serving policy with an auditable scenario-specific decision."
        evidence = _scenario_type_details(scenario)["focus"]
        requirement = _scenario_type_details(scenario)["label"]
    return {
        "axis": _portfolio_axis_label(scenario.name, index),
        "title": _scenario_short_title(scenario.name),
        "type": _scenario_type_details(scenario)["label"],
        "value": value,
        "evidence": evidence,
        "requirement": requirement,
    }


def _cost_per_1k(decision: RouteDecision) -> str:
    if decision.estimated_cost_usd is None:
        return "-"
    return _money(decision.estimated_cost_usd * 1000)


def _group_endpoints(endpoints: Iterable[Endpoint]) -> dict[tuple[str, str], list[Endpoint]]:
    grouped: dict[tuple[str, str], list[Endpoint]] = defaultdict(list)
    for endpoint in endpoints:
        grouped[(endpoint.model_id, endpoint.serving_backend)].append(endpoint)
    return grouped


def _run_snapshot(
    *,
    name: str,
    description: str,
    models: dict[str, Model],
    endpoints: list[Endpoint],
    requests,
    planned_events: dict[str, dict],
    canary: dict | None = None,
) -> SnapshotResult:
    decisions = apply_routing(requests, models, endpoints)
    cost_rows = build_cost_report(decisions, models)
    autoscale_signals = compute_signals(endpoints, models, planned_events=planned_events)

    canary_decision = None
    if canary:
        canary_decision = evaluate_canary(
            candidate_version=canary["candidate_version"],
            candidate_metrics=canary["candidate_metrics"],
            control_metrics=canary["control_metrics"],
            guardrails=canary["guardrails"],
            rollback_to_version=canary.get("rollback_to_version"),
        )

    return SnapshotResult(
        name=name,
        description=description,
        endpoints=endpoints,
        decisions=decisions,
        cost_rows=cost_rows,
        autoscale_signals=autoscale_signals,
        canary_decision=canary_decision,
    )


def _scenario_snapshots(scenario: Scenario) -> list[SnapshotResult]:
    if scenario.phases:
        return [
            _run_snapshot(
                name=phase.name,
                description=phase.description,
                models=scenario.models,
                endpoints=phase.endpoints,
                requests=phase.requests,
                planned_events=phase.planned_events,
            )
            for phase in scenario.phases
        ]
    return [
        _run_snapshot(
            name=scenario.name,
            description=scenario.description,
            models=scenario.models,
            endpoints=scenario.endpoints,
            requests=scenario.requests,
            planned_events=scenario.planned_events,
            canary=scenario.canary,
        )
    ]


def _render_summary_card(label: str, value: str, note: str | None = None) -> str:
    return (
        '<div class="summary-card"><span>'
        + escape(label)
        + "</span><strong>"
        + escape(value)
        + "</strong>"
        + (f"<small>{escape(note)}</small>" if note else "")
        + "</div>"
    )


def _model_trigger_axis(model: Model | None) -> str:
    if model is not None and not model.requires_gpu:
        return "CPU"
    return "GPU"


def _trigger_axis_for_event(event: str, model: Model | None) -> str:
    lower = event.lower()
    if "planned" in lower or "pre-warm" in lower or "prewarm" in lower:
        return "planned event"
    if "latency" in lower or "failover" in lower or "sla" in lower or "p99" in lower:
        return "Latency/SLA"
    if (
        "unhealthy" in lower
        or "readiness" in lower
        or "drain" in lower
        or "replacement" in lower
        or "recovered" in lower
    ):
        return "Health/readiness"
    return _model_trigger_axis(model)


def _safe_capacity_for_model(profile: ModelTrafficProfile, replicas: int) -> int:
    return int(round(replicas * profile.capacity_qps * (profile.target_util_pct / 100.0)))


def _estimated_util_pct(profile: ModelTrafficProfile, rps: int, replicas: int) -> float:
    if replicas <= 0 or profile.raw_capacity_qps <= 0:
        return 0.0
    return round(max(0.0, min(100.0, 100.0 * rps / (replicas * profile.raw_capacity_qps))), 1)


def _util_pct_text(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}%"


def _render_platform_impact() -> str:
    return """
<section class="impact">
  <h2>Before vs After Platform</h2>
  <div class="impact-grid">
    <article>
      <h3>Before shared platform</h3>
      <ul>
        <li>Teams run separate serving stacks with limited shared cost visibility.</li>
        <li>GPU fleets can stay over- or under-provisioned because utilization and ownership are fragmented.</li>
        <li>Training and inference compete for capacity without one common priority policy.</li>
      </ul>
    </article>
    <article>
      <h3>After ModelOps platform</h3>
      <ul>
        <li><strong>Cost transparency and optimization:</strong> real-time cost visibility at model and team level across inference and training workloads.</li>
        <li><strong>Automated right-sizing and instance recommendations:</strong> GPU vs CPU, spot vs on-demand, and stale provisioned fleet risk are visible as recommendations.</li>
        <li>Autoscaling and capacity planning protect online inference while pushing non-urgent training to cheaper or lower-risk capacity.</li>
      </ul>
    </article>
  </div>
</section>
"""


def _render_models(models: dict[str, Model]) -> str:
    if not models:
        return "<p>No models registered for this scenario.</p>"
    rows = []
    for model in sorted(models.values(), key=lambda m: m.model_id):
        targets = utilization_targets_for_model(model)
        rows.append(
            "<tr><td>"
            + escape(model.model_id)
            + "</td><td>"
            + escape(model.owner_team)
            + "</td><td>"
            + escape(model.serving_backend)
            + "</td><td>"
            + str(model.sla_ms)
            + "</td><td>"
            + str(targets["gpu"])
            + "</td><td>"
            + str(targets["cpu"])
            + "</td></tr>"
        )
    return (
        '<table><thead><tr><th>model</th><th>owner</th><th>backend</th>'
        '<th>p99_target_ms</th><th>gpu_target%</th>'
        '<th>cpu_target%</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )


def _primary_endpoint_for_model(scenario: Scenario, model_id: str) -> Endpoint | None:
    endpoints = [
        endpoint
        for endpoint in _profile_source_endpoints(scenario)
        if endpoint.model_id == model_id and endpoint.healthy
    ]
    if not endpoints:
        endpoints = [
            endpoint
            for endpoint in _profile_source_endpoints(scenario)
            if endpoint.model_id == model_id
        ]
    if not endpoints:
        return None
    return min(endpoints, key=lambda endpoint: endpoint.cost_per_1k_predictions)


def _derived_replica_hourly_cost(endpoint: Endpoint) -> float:
    return endpoint.capacity_qps * 3.6 * endpoint.cost_per_1k_predictions


def _endpoint_infra_parts(endpoint: Endpoint) -> list[str]:
    assumption = _INSTANCE_COST_ASSUMPTIONS.get(endpoint.instance_class)
    parts = []
    if assumption is not None:
        parts.append(f"{assumption.instance_type} / {assumption.accelerator}")
        parts.append(f"${assumption.public_hourly_usd:.3f}/GPU-hr")
    else:
        parts.append(endpoint.instance_class)
        parts.append(f"${_derived_replica_hourly_cost(endpoint):.3f}/replica-hr")
    return parts


def _scenario_cost_basis(scenario: Scenario) -> str:
    endpoint = next(
        (
            candidate
            for model in sorted(scenario.models.values(), key=lambda m: m.model_id)
            if (candidate := _primary_endpoint_for_model(scenario, model.model_id)) is not None
        ),
        None,
    )
    if endpoint is None:
        return "model-owned replica-hours"
    assumption = _INSTANCE_COST_ASSUMPTIONS.get(endpoint.instance_class)
    if assumption is not None:
        return (
            f"{assumption.accelerator.split(' 24GB')[0]} replica-hours "
            f"@ ${assumption.public_hourly_usd:.3f}/GPU-hr"
        )
    return f"{endpoint.instance_class} replica-hours"


def _scenario_cost_math(scenario: Scenario) -> str:
    endpoint = next(
        (
            candidate
            for model in sorted(scenario.models.values(), key=lambda m: m.model_id)
            if (candidate := _primary_endpoint_for_model(scenario, model.model_id)) is not None
        ),
        None,
    )
    if endpoint is None or endpoint.capacity_qps <= 0:
        return ""

    assumption = _INSTANCE_COST_ASSUMPTIONS.get(endpoint.instance_class)
    hourly_cost = (
        assumption.public_hourly_usd
        if assumption is not None
        else _derived_replica_hourly_cost(endpoint)
    )
    denominator = endpoint.capacity_qps * _SECONDS_PER_HOUR / 1000
    if denominator <= 0:
        return ""
    allocated_per_1k = hourly_cost / denominator
    capacity_float = float(endpoint.capacity_qps)
    capacity = int(capacity_float) if capacity_float.is_integer() else capacity_float
    return (
        '<p class="cost-math"><strong>Cost math:</strong> 24h spend = active '
        f"replica-hours x ${hourly_cost:.3f}/GPU-hr. "
        "<code>gpu_cost_per_1k</code> = "
        f"${hourly_cost:.3f} / ({capacity} safe allocation RPS x 3600 / 1000) = "
        f"{_money(allocated_per_1k)} per 1K requests. Cloud billing remains "
        "GPU/instance-hours; per-1K is an internal routing allocation.</p>"
    )


def _render_model_setup_strip(scenario: Scenario) -> str:
    models = scenario.models
    if not models:
        return ""
    items = []
    for model in sorted(models.values(), key=lambda m: m.model_id):
        targets = utilization_targets_for_model(model)
        endpoint = _primary_endpoint_for_model(scenario, model.model_id)
        infra_parts = _endpoint_infra_parts(endpoint) if endpoint is not None else []
        items.append(
            '<div class="setup-pill">'
            '<span class="setup-model">'
            + escape(model.model_id)
            + "</span>"
            + f"<span>{escape(model.model_type)} on {escape(model.serving_backend)}</span>"
            + f"<span>P99 {model.sla_ms}ms</span>"
            + f"<span>GPU target {targets['gpu']}%</span>"
            + f"<span>CPU target {targets['cpu']}%</span>"
            + "".join("<span>" + escape(part) + "</span>" for part in infra_parts)
            + "</div>"
        )
    return '<div class="scenario-setup">' + "".join(items) + "</div>"


def _hourly_cost_for_snapshot(snapshot: SnapshotResult) -> float:
    """Approximate full-utilization hourly run-rate cost across all endpoints."""
    total = 0.0
    for endpoint in snapshot.endpoints:
        total += endpoint.capacity_qps * 3.6 * endpoint.cost_per_1k_predictions
    return total


def _total_replicas(snapshot: SnapshotResult, *, use_recommended: bool = False) -> int:
    if snapshot.autoscale_signals:
        if use_recommended:
            return sum(signal.recommended_replicas for signal in snapshot.autoscale_signals)
        return sum(signal.current_replicas for signal in snapshot.autoscale_signals)
    return len(snapshot.endpoints)


def _total_rps(snapshot: SnapshotResult) -> int:
    return sum(endpoint.current_qps for endpoint in snapshot.endpoints)


def _profile_source_endpoints(scenario: Scenario) -> list[Endpoint]:
    if scenario.phases:
        return scenario.phases[0].endpoints
    return scenario.endpoints


def _profiles_by_model(scenario: Scenario) -> dict[str, ModelTrafficProfile]:
    endpoints = _profile_source_endpoints(scenario)
    grouped = _group_endpoints(endpoints)
    profiles: dict[str, ModelTrafficProfile] = {}
    for model_id, model in sorted(scenario.models.items()):
        candidates = [
            endpoint
            for (endpoint_model_id, _backend), eps in grouped.items()
            if endpoint_model_id == model_id
            for endpoint in eps
        ]
        healthy = [endpoint for endpoint in candidates if endpoint.healthy] or candidates
        if healthy:
            min_cost = min(endpoint.cost_per_1k_predictions for endpoint in healthy)
            primary = [
                endpoint
                for endpoint in healthy
                if endpoint.cost_per_1k_predictions <= min_cost * 1.2
            ]
            capacity_qps = sum(endpoint.capacity_qps for endpoint in primary) / len(primary)
            cost_per_1k = (
                sum(endpoint.cost_per_1k_predictions for endpoint in primary) / len(primary)
            )
            base_replicas = len(
                [
                    endpoint
                    for endpoint in candidates
                    if endpoint.cost_per_1k_predictions <= min_cost * 1.2
                ]
            ) or len(primary)
        else:
            # Canary-only scenarios do not always need a serving fleet. Keep
            # the chart contract valid without inventing meaningful volume.
            capacity_qps = 1.0
            cost_per_1k = 0.0
            base_replicas = 0

        targets = utilization_targets_for_model(model)
        axis = "gpu" if model.requires_gpu else "cpu"
        target_util_pct = targets[axis]
        profiles[model_id] = ModelTrafficProfile(
            model_id=model_id,
            owner_team=model.owner_team,
            capacity_qps=capacity_qps,
            target_util_pct=target_util_pct,
            base_replicas=base_replicas,
            per_replica_hourly_cost=capacity_qps * 3.6 * cost_per_1k,
            raw_capacity_qps=capacity_qps,
            sla_ms=model.sla_ms,
        )
    return profiles


def _snapshot_rps_by_name(scenario: Scenario) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    if scenario.phases:
        sources: Iterable[ScenarioPhase] = scenario.phases
    else:
        return {
            scenario.name: {
                model_id: int(sum(e.current_qps for e in endpoints if e.healthy))
                for (model_id, _backend), endpoints in _group_endpoints(
                    scenario.endpoints
                ).items()
            }
        }
    for phase in sources:
        out[phase.name] = {
            model_id: int(sum(e.current_qps for e in endpoints if e.healthy))
            for (model_id, _backend), endpoints in _group_endpoints(
                phase.endpoints
            ).items()
        }
    return out


def _fallback_traffic_windows(scenario: Scenario, snapshots: list[SnapshotResult]) -> list[dict[str, Any]]:
    if scenario.traffic_windows:
        return scenario.traffic_windows
    if not snapshots:
        return []

    duration = 24 / len(snapshots)
    windows = []
    for idx, snapshot in enumerate(snapshots):
        windows.append(
            {
                "label": snapshot.name,
                "start_hour": round(idx * duration, 2),
                "end_hour": round((idx + 1) * duration, 2),
                "source": snapshot.name,
                "event": "single-snapshot audit sample" if len(snapshots) == 1 else "",
            }
        )
    return windows


def _window_rps(
    window: dict[str, Any],
    *,
    scenario: Scenario,
    profiles: dict[str, ModelTrafficProfile],
    rps_sources: dict[str, dict[str, int]],
) -> dict[str, int]:
    raw = window.get("rps_by_model")
    if raw:
        return {model_id: int(value) for model_id, value in raw.items()}

    source_name = window.get("source") or scenario.name
    base = rps_sources.get(source_name, {})
    multiplier = float(window.get("rps_multiplier", 1.0))
    return {
        model_id: int(round(base.get(model_id, 0) * multiplier))
        for model_id in profiles
    }


def _needed_replicas(profile: ModelTrafficProfile, rps: int) -> int:
    if rps <= 0:
        return max(0, profile.base_replicas)
    target_capacity = profile.capacity_qps * (profile.target_util_pct / 100.0)
    if target_capacity <= 0:
        return max(1, profile.base_replicas)
    return max(profile.base_replicas, int(math.ceil(rps / target_capacity)))


def _replicas_for_window(
    window: dict[str, Any],
    key: str,
    *,
    profiles: dict[str, ModelTrafficProfile],
    rps_by_model: dict[str, int],
) -> dict[str, int]:
    raw = window.get(key)
    if raw:
        return {model_id: int(raw.get(model_id, 0)) for model_id in profiles}
    if key == "fixed_replicas":
        return {model_id: profile.base_replicas for model_id, profile in profiles.items()}
    return {
        model_id: _needed_replicas(profile, rps_by_model.get(model_id, 0))
        for model_id, profile in profiles.items()
    }


def _capacity_at_target(
    profiles: dict[str, ModelTrafficProfile],
    replicas_by_model: dict[str, int],
) -> float:
    total = 0.0
    for model_id, replicas in replicas_by_model.items():
        profile = profiles[model_id]
        total += replicas * profile.capacity_qps * (profile.target_util_pct / 100.0)
    return total


def _raw_capacity(
    profiles: dict[str, ModelTrafficProfile],
    replicas_by_model: dict[str, int],
) -> float:
    total = 0.0
    for model_id, replicas in replicas_by_model.items():
        profile = profiles[model_id]
        total += replicas * profile.raw_capacity_qps
    return total


def _hourly_cost(
    profiles: dict[str, ModelTrafficProfile],
    replicas_by_model: dict[str, int],
) -> float:
    total = 0.0
    for model_id, replicas in replicas_by_model.items():
        total += replicas * profiles[model_id].per_replica_hourly_cost
    return total


def _pressure_p99(total_rps: int, target_capacity: float, raw_capacity: float, sla_ms: int) -> float:
    if total_rps <= 0 or sla_ms <= 0:
        return 0.0
    if target_capacity <= 0:
        return float(sla_ms * 2)
    if total_rps <= target_capacity:
        return round(max(sla_ms * 0.55, sla_ms - 18), 1)
    if raw_capacity > 0 and total_rps <= raw_capacity:
        pressure = total_rps / target_capacity
        return round(min(sla_ms * 1.05, sla_ms * (0.72 + 0.18 * pressure)), 1)
    pressure = total_rps / max(raw_capacity, target_capacity, 1)
    return round(sla_ms * (1.15 + pressure), 1)


def _scale_action(previous: dict[str, int] | None, current: dict[str, int]) -> str:
    if previous is None:
        return "initial platform posture"
    ups = []
    downs = []
    for model_id, replicas in current.items():
        prior = previous.get(model_id, 0)
        if replicas > prior:
            ups.append(f"{model_id} {prior}->{replicas}")
        elif replicas < prior:
            downs.append(f"{model_id} {prior}->{replicas}")
    if ups:
        return "scale out: " + ", ".join(ups)
    if downs:
        return "scale down: " + ", ".join(downs)
    return "hold steady"


def _build_traffic_windows(
    scenario: Scenario,
    snapshots: list[SnapshotResult],
) -> list[TrafficWindowResult]:
    profiles = _profiles_by_model(scenario)
    if not profiles:
        return []

    raw_windows = _fallback_traffic_windows(scenario, snapshots)
    if not raw_windows:
        return []

    rps_sources = _snapshot_rps_by_name(scenario)
    prepared = []
    for raw in raw_windows:
        rps_by_model = _window_rps(
            raw,
            scenario=scenario,
            profiles=profiles,
            rps_sources=rps_sources,
        )
        platform_replicas = _replicas_for_window(
            raw,
            "platform_replicas",
            profiles=profiles,
            rps_by_model=rps_by_model,
        )
        recommended_replicas = _replicas_for_window(
            raw,
            "recommended_replicas",
            profiles=profiles,
            rps_by_model=rps_by_model,
        )
        fixed_replicas = _replicas_for_window(
            raw,
            "fixed_replicas",
            profiles=profiles,
            rps_by_model=rps_by_model,
        )
        prepared.append(
            {
                "raw": raw,
                "rps_by_model": rps_by_model,
                "platform_replicas": platform_replicas,
                "recommended_replicas": recommended_replicas,
                "fixed_replicas": fixed_replicas,
            }
        )

    peak_platform_by_model = {
        model_id: max(
            max(item["platform_replicas"].get(model_id, 0) for item in prepared),
            max(item["recommended_replicas"].get(model_id, 0) for item in prepared),
            profile.base_replicas,
        )
        for model_id, profile in profiles.items()
    }

    cumulative_fixed = 0.0
    cumulative_platform = 0.0
    cumulative_manual = 0.0
    previous_platform: dict[str, int] | None = None
    results: list[TrafficWindowResult] = []

    for item in prepared:
        raw = item["raw"]
        start_hour = float(raw.get("start_hour", 0))
        end_hour = float(raw.get("end_hour", start_hour))
        duration_hours = max(0.01, end_hour - start_hour)
        rps_by_model = item["rps_by_model"]
        platform_replicas = item["platform_replicas"]
        recommended_replicas = item["recommended_replicas"]
        fixed_replicas = item["fixed_replicas"]
        explicit_manual = raw.get("manual_peak_replicas") or raw.get("static_replicas")
        if explicit_manual:
            manual_replicas = {
                model_id: int(explicit_manual.get(model_id, 0)) for model_id in profiles
            }
        else:
            manual_replicas = {
                model_id: max(
                    profiles[model_id].base_replicas,
                    int(math.ceil(peak_platform_by_model[model_id] * _BEFORE_OVER_PROVISION_FACTOR)),
                )
                for model_id in profiles
            }

        trigger_util_pct_by_model: dict[str, float] = {}
        post_scale_util_pct_by_model: dict[str, float] = {}
        for model_id, profile in profiles.items():
            model = scenario.models.get(model_id)
            axis = _model_trigger_axis(model)
            trigger_key = (
                "trigger_gpu_util_pct_by_model"
                if axis == "GPU"
                else "trigger_cpu_util_pct_by_model"
            )
            after_key = (
                "post_scale_gpu_util_pct_by_model"
                if axis == "GPU"
                else "post_scale_cpu_util_pct_by_model"
            )
            prior_replicas = platform_replicas.get(model_id, 0)
            if previous_platform is not None:
                prior_replicas = previous_platform.get(model_id, prior_replicas)
            current_replicas = platform_replicas.get(model_id, 0)
            demand = rps_by_model.get(model_id, 0)
            trigger_util_pct_by_model[model_id] = float(
                raw.get(trigger_key, {}).get(
                    model_id, _estimated_util_pct(profile, demand, prior_replicas)
                )
            )
            post_scale_util_pct_by_model[model_id] = float(
                raw.get(after_key, {}).get(
                    model_id, _estimated_util_pct(profile, demand, current_replicas)
                )
            )

        fixed_capacity = _capacity_at_target(profiles, fixed_replicas)
        platform_capacity = _capacity_at_target(profiles, platform_replicas)
        manual_capacity = _capacity_at_target(profiles, manual_replicas)
        platform_raw_capacity = _raw_capacity(profiles, platform_replicas)
        total_rps = sum(rps_by_model.values())
        total_requests = int(round(total_rps * duration_hours * _SECONDS_PER_HOUR))

        fixed_capacity_requests = fixed_capacity * duration_hours * _SECONDS_PER_HOUR
        platform_target_requests = platform_capacity * duration_hours * _SECONDS_PER_HOUR
        platform_raw_requests = platform_raw_capacity * duration_hours * _SECONDS_PER_HOUR

        fixed_at_risk = int(round(max(0.0, total_requests - fixed_capacity_requests)))
        platform_served = int(round(min(total_requests, platform_target_requests)))
        platform_overflow = int(
            round(max(0.0, min(total_requests, platform_raw_requests) - platform_target_requests))
        )
        platform_rejected = int(round(max(0.0, total_requests - platform_raw_requests)))

        fixed_hourly_cost = _hourly_cost(profiles, fixed_replicas)
        platform_hourly_cost = _hourly_cost(profiles, platform_replicas)
        manual_hourly_cost = _hourly_cost(profiles, manual_replicas)
        cumulative_fixed += fixed_hourly_cost * duration_hours
        cumulative_platform += platform_hourly_cost * duration_hours
        cumulative_manual += manual_hourly_cost * duration_hours

        strictest_sla = min((profile.sla_ms for profile in profiles.values()), default=0)
        scale_action = _scale_action(previous_platform, platform_replicas)
        previous_platform = platform_replicas
        label = raw.get("label") or _window_range_label(start_hour, end_hour)

        results.append(
            TrafficWindowResult(
                label=label,
                start_hour=start_hour,
                end_hour=end_hour,
                duration_hours=duration_hours,
                event=raw.get("event", ""),
                total_rps=total_rps,
                fixed_capacity_rps=int(round(fixed_capacity)),
                platform_capacity_rps=int(round(platform_capacity)),
                manual_capacity_rps=int(round(manual_capacity)),
                fixed_replicas=sum(fixed_replicas.values()),
                platform_replicas=sum(platform_replicas.values()),
                recommended_replicas=sum(recommended_replicas.values()),
                manual_replicas=sum(manual_replicas.values()),
                fixed_hourly_cost=round(fixed_hourly_cost, 4),
                platform_hourly_cost=round(platform_hourly_cost, 4),
                manual_hourly_cost=round(manual_hourly_cost, 4),
                fixed_cumulative_cost=round(cumulative_fixed, 4),
                platform_cumulative_cost=round(cumulative_platform, 4),
                manual_cumulative_cost=round(cumulative_manual, 4),
                fixed_at_risk_requests=fixed_at_risk,
                platform_served_requests=platform_served,
                platform_overflow_requests=platform_overflow,
                platform_rejected_requests=platform_rejected,
                fixed_p99_ms=_pressure_p99(total_rps, fixed_capacity, _raw_capacity(profiles, fixed_replicas), strictest_sla),
                platform_p99_ms=_pressure_p99(total_rps, platform_capacity, platform_raw_capacity, strictest_sla),
                sla_target_ms=strictest_sla,
                scale_action=scale_action,
                rps_by_model=rps_by_model,
                platform_replicas_by_model=platform_replicas,
                recommended_replicas_by_model=recommended_replicas,
                manual_replicas_by_model=manual_replicas,
                trigger_util_pct_by_model=trigger_util_pct_by_model,
                post_scale_util_pct_by_model=post_scale_util_pct_by_model,
            )
        )

    return results


def _traffic_chart_data(windows: list[TrafficWindowResult]) -> dict:
    if not windows:
        return {}
    manual_total = windows[-1].manual_cumulative_cost
    platform_total = windows[-1].platform_cumulative_cost
    fixed_total = windows[-1].fixed_cumulative_cost
    savings_pct = (
        round((1 - platform_total / manual_total) * 100, 1)
        if manual_total > 0
        else 0.0
    )
    platform_rejected = sum(window.platform_rejected_requests for window in windows)
    fixed_at_risk = sum(window.fixed_at_risk_requests for window in windows)
    return {
        "labels": [
            f"{_window_range_label(window.start_hour, window.end_hour)} {window.label}"
            for window in windows
        ],
        "shortLabels": [_window_range_label(window.start_hour, window.end_hour) for window in windows],
        "axisLabels": [
            _compact_window_axis_label(window.start_hour, window.end_hour)
            for window in windows
        ],
        "events": [window.event for window in windows],
        "scaleActions": [window.scale_action for window in windows],
        "totalRps": [window.total_rps for window in windows],
        "fixedCapacityRps": [window.fixed_capacity_rps for window in windows],
        "platformCapacityRps": [window.platform_capacity_rps for window in windows],
        "manualCapacityRps": [window.manual_capacity_rps for window in windows],
        "fixedReplicas": [window.fixed_replicas for window in windows],
        "platformReplicas": [window.platform_replicas for window in windows],
        "recommendedReplicas": [window.recommended_replicas for window in windows],
        "manualReplicas": [window.manual_replicas for window in windows],
        "fixedHourlyCost": [window.fixed_hourly_cost for window in windows],
        "platformHourlyCost": [window.platform_hourly_cost for window in windows],
        "manualHourlyCost": [window.manual_hourly_cost for window in windows],
        "fixedCumulativeCost": [window.fixed_cumulative_cost for window in windows],
        "platformCumulativeCost": [window.platform_cumulative_cost for window in windows],
        "manualCumulativeCost": [window.manual_cumulative_cost for window in windows],
        "fixedAtRisk": [window.fixed_at_risk_requests for window in windows],
        "platformServed": [window.platform_served_requests for window in windows],
        "platformOverflow": [window.platform_overflow_requests for window in windows],
        "platformRejected": [window.platform_rejected_requests for window in windows],
        "fixedP99": [window.fixed_p99_ms for window in windows],
        "platformP99": [window.platform_p99_ms for window in windows],
        "slaTarget": [window.sla_target_ms for window in windows],
        "summary": {
            "peakRps": max(window.total_rps for window in windows),
            "replicaRange": f"{min(window.platform_replicas for window in windows)} -> {max(window.platform_replicas for window in windows)}",
            "manualCostTotal": round(manual_total, 2),
            "platformCostTotal": round(platform_total, 2),
            "fixedCostTotal": round(fixed_total, 2),
            "savingsPct": savings_pct,
            "fixedAtRisk": fixed_at_risk,
            "platformRejected": platform_rejected,
            "peakFixedP99": max(window.fixed_p99_ms for window in windows),
            "peakPlatformP99": max(window.platform_p99_ms for window in windows),
        },
    }


def _build_metric_timesteps(snapshots: list[SnapshotResult]) -> list[MetricTimestep]:
    """Build before/after time series (RPS, replicas, hourly cost) for one scenario."""
    if not snapshots:
        return []

    if len(snapshots) >= 2:
        raw = [
            (
                f"t{idx + 1}",
                _total_rps(snapshot),
                _total_replicas(snapshot),
                _hourly_cost_for_snapshot(snapshot),
            )
            for idx, snapshot in enumerate(snapshots)
        ]
    else:
        snapshot = snapshots[0]
        rps = _total_rps(snapshot)
        cost_now = _hourly_cost_for_snapshot(snapshot)
        replicas_now = _total_replicas(snapshot)
        replicas_after = _total_replicas(snapshot, use_recommended=True)
        ratio = (replicas_after / replicas_now) if replicas_now else 1.0
        cost_after = cost_now * ratio
        raw = [
            ("now", rps, replicas_now, cost_now),
            ("after HPA", rps, replicas_after, cost_after),
        ]

    peak_after_replicas = max((step[2] for step in raw), default=0) or 1
    before_replicas = max(
        1, int(round(peak_after_replicas * _BEFORE_OVER_PROVISION_FACTOR))
    )

    peak_step = max(raw, key=lambda step: step[2])
    per_replica_cost = (peak_step[3] / peak_step[2]) if peak_step[2] > 0 else 0.0
    before_cost = before_replicas * per_replica_cost

    return [
        MetricTimestep(
            label=label,
            rps=rps,
            after_replicas=replicas,
            before_replicas=before_replicas,
            after_hourly_cost=cost,
            before_hourly_cost=before_cost,
        )
        for label, rps, replicas, cost in raw
    ]


def _serving_counts(decisions: Iterable[RouteDecision]) -> dict[str, int]:
    counts = {"served": 0, "overflow": 0, "rejected": 0}
    for decision in decisions:
        if decision.status == "routed":
            counts["served"] += 1
        elif decision.status == "overflow_to_premium":
            counts["overflow"] += 1
        else:
            counts["rejected"] += 1
    return counts


def _max_observed_p99(decisions: Iterable[RouteDecision]) -> float:
    return max((d.estimated_latency_ms or 0.0 for d in decisions), default=0.0)


def _scenario_chart_data(
    scenario: Scenario,
    snapshots: list[SnapshotResult],
    timesteps: list[MetricTimestep],
    *,
    scenario_index: int,
) -> dict:
    scenario_type = _scenario_type_details(scenario)
    traffic_windows = (
        _build_traffic_windows(scenario, snapshots)
        if scenario_type["cost_coupled"]
        else []
    )
    traffic = _traffic_chart_data(traffic_windows)
    labels = [step.label for step in timesteps]

    before_peak = max((step.before_hourly_cost for step in timesteps), default=0.0)
    after_peak = max((step.after_hourly_cost for step in timesteps), default=0.0)
    savings_pct = (
        round((1 - after_peak / before_peak) * 100, 1)
        if before_peak > 0
        else 0.0
    )

    return {
        "name": scenario.name,
        "scenarioIndex": scenario_index,
        "axisLabel": _portfolio_axis_label(scenario.name, scenario_index),
        "scenarioType": _scenario_type_key(scenario),
        "scenarioTypeLabel": scenario_type["label"],
        "evidenceFocus": scenario_type["focus"],
        "costPortfolioEligible": scenario_type["cost_coupled"],
        "labels": labels,
        "traffic": traffic,
        "baselineLabel": "before: peak-provisioned",
        "rps": [step.rps for step in timesteps],
        "beforeReplicas": [step.before_replicas for step in timesteps],
        "afterReplicas": [step.after_replicas for step in timesteps],
        "beforeHourlyCost": [round(step.before_hourly_cost, 4) for step in timesteps],
        "afterHourlyCost": [round(step.after_hourly_cost, 4) for step in timesteps],
        "beforePeakCost": round(before_peak, 4),
        "afterPeakCost": round(after_peak, 4),
        "savingsPct": savings_pct,
        "canaryDecision": next(
            (
                snapshot.canary_decision.decision
                for snapshot in snapshots
                if snapshot.canary_decision is not None
            ),
            None,
        ),
    }


def _build_report_data(scenarios: list[Scenario]) -> dict:
    chart_scenarios = []
    for index, scenario in enumerate(scenarios, start=1):
        snapshots = _scenario_snapshots(scenario)
        chart_scenarios.append(
            _scenario_chart_data(
                scenario,
                snapshots,
                _build_metric_timesteps(snapshots),
                scenario_index=index,
            )
        )
    portfolio_before = []
    portfolio_after = []
    portfolio_savings = []
    portfolio_scenarios = [
        scenario for scenario in chart_scenarios if scenario["costPortfolioEligible"]
    ]
    for scenario in portfolio_scenarios:
        summary = scenario.get("traffic", {}).get("summary", {})
        if summary:
            portfolio_before.append(summary["manualCostTotal"])
            portfolio_after.append(summary["platformCostTotal"])
            portfolio_savings.append(summary["savingsPct"])
        else:
            portfolio_before.append(scenario["beforePeakCost"])
            portfolio_after.append(scenario["afterPeakCost"])
            portfolio_savings.append(scenario["savingsPct"])

    return {
        "portfolio": {
            "labels": [scenario["name"] for scenario in portfolio_scenarios],
            "axisLabels": [scenario["axisLabel"] for scenario in portfolio_scenarios],
            "beforePeakCost": [scenario["beforePeakCost"] for scenario in portfolio_scenarios],
            "afterPeakCost": [scenario["afterPeakCost"] for scenario in portfolio_scenarios],
            "savingsPct": [scenario["savingsPct"] for scenario in portfolio_scenarios],
            "beforeTotalCost": portfolio_before,
            "afterTotalCost": portfolio_after,
            "totalSavingsPct": portfolio_savings,
            "excluded": [
                {
                    "axisLabel": scenario["axisLabel"],
                    "name": scenario["name"],
                    "scenarioTypeLabel": scenario["scenarioTypeLabel"],
                    "evidenceFocus": scenario["evidenceFocus"],
                }
                for scenario in chart_scenarios
                if not scenario["costPortfolioEligible"]
            ],
        },
        "scenarios": chart_scenarios,
    }


def _json_script(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    payload = (
        payload.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    return f'<script type="application/json" id="report-data">{payload}</script>'


def _render_portfolio_chart(portfolio: dict) -> str:
    return (
        '<section class="portfolio-charts">'
        "<h2>Portfolio Cost Snapshot</h2>"
        '<p class="chart-note"><strong>What this shows:</strong> traffic/capacity scenarios only. Release-confidence scenarios are excluded because their proof is guardrail quality, not cost savings.</p>'
        '<div class="chart-panel chart-panel-wide">'
        '<h3>Before vs After 24-Hour Cost: Traffic Scenarios</h3>'
        '<div class="chart-canvas-wrap"><canvas id="portfolio-impact-chart" aria-label="Before vs after 24-hour attributed serving cost by traffic scenario"></canvas></div>'
        "</div>"
        '<p class="chart-note"><strong>Cost attribution:</strong> each bar sums model-owned serving replica-hours. <strong>Before</strong> holds peak-provisioned capacity all day; <strong>After</strong> follows the platform replica plan.</p>'
        "</section>"
    )


def _render_metric_charts(
    timesteps: list[MetricTimestep],
    *,
    index: int,
    traffic_windows: list[TrafficWindowResult],
) -> str:
    if not timesteps and not traffic_windows:
        return ""
    return (
        '<section class="scenario-charts">'
        '<h4>Evidence Charts</h4>'
        '<p class="chart-note"><strong>Capacity chart:</strong> RPS demand and safe capacity use the left axis; replica counts use the right axis. The fixed fleet stays flat while platform replicas change with the scaling policy.</p>'
        f'<div class="chart-panel chart-panel-wide"><h5>Before Fixed Fleet vs After Autoscaling</h5><div class="chart-canvas-wrap"><canvas id="scenario-simulation-{index}" aria-label="24-hour demand, safe capacity, and replica count chart"></canvas></div></div>'
        f'<div class="chart-panel chart-panel-wide"><h5>Cumulative Cost: Before vs After</h5><div class="chart-canvas-wrap"><canvas id="scenario-cost-{index}" aria-label="24-hour cumulative before versus after cost chart"></canvas></div></div>'
        + "</section>"
    )


def _render_scenario_type_note(scenario: Scenario) -> str:
    details = _scenario_type_details(scenario)
    return (
        '<section class="scenario-type-note"><h4>Scenario Type: '
        + escape(details["label"])
        + "</h4><p>"
        + escape(details["description"])
        + "</p></section>"
    )


def _replica_range(values: Iterable[int]) -> str:
    replica_values = list(values)
    if not replica_values:
        return "0"
    low = min(replica_values)
    high = max(replica_values)
    return str(low) if low == high else f"{low} -> {high}"


def _replica_label(count: int) -> str:
    return f"{count} replica" if count == 1 else f"{count} replicas"


def _scenario_value_points(scenario: Scenario) -> tuple[str, str]:
    name = scenario.name.lower()
    if "normal load" in name:
        return (
            "Before this platform: the pilot fleet stays pinned to daytime capacity even when overnight traffic is much lower.",
            "After this platform: replicas follow the smooth daily curve, scaling up for normal business-hour demand and back down overnight while staying SLA-safe.",
        )
    if "organic traffic spike" in name:
        return (
            "Before this platform: organic spikes require manual over-provisioning, late capacity requests, or ad hoc traffic shedding.",
            "After this platform: HPA-style signals add pods before inference traffic is rejected, while the report shows load, targets, replicas, and cost by phase.",
        )
    if "planned promotion" in name:
        return (
            "Before this platform: merchandising events depend on manual reminders and blanket over-provisioning for Recommendations.",
            "After this platform: planned-event signals pre-warm Recommendations capacity ahead of demand and make the incremental serving cost visible by model and team.",
        )
    if "endpoint unavailability" in name:
        return (
            "Before this platform: unhealthy capacity is discovered late, fallback routing is unclear, and operators wait for the bad pod to recover instead of replacing it.",
            "After this platform: a rolling replacement runs the moment readiness fails — bad pod is excluded from routing, a fresh pod comes up healthy, traffic shifts, the bad pod drains and terminates. High-priority traffic uses healthy or premium capacity for the readiness gap.",
        )
    if "canary evaluation - auto-promote" in name:
        return (
            "Before this platform: Fraud releases depend on manual metric inspection and uneven launch discipline.",
            "After this platform: passing operational, fraud-quality, business, and sample-size guardrails allow safe auto-promotion with an auditable decision.",
        )
    if "canary evaluation - requires owner review" in name or "fraud canary - requires owner review" in name:
        return (
            "Before this platform: Fraud quality regressions can be missed if operational metrics look clean.",
            "After this platform: fraud-quality and business guardrails force owner review even when latency, errors, and timeouts are healthy.",
        )
    if "fraud canary - auto-promote" in name:
        return (
            "Before this platform: Fraud releases depend on manual metric inspection and uneven launch discipline.",
            "After this platform: passing operational, fraud-quality, business, and sample-size guardrails allow safe auto-promotion with an auditable decision.",
        )
    return (
        "Before this platform: cost is hard to attribute at model/team level, and routing choices are difficult to audit.",
        "After this platform: cost-per-1K and service risk are visible per model, with SLA-aware routing and team-level cost attribution.",
    )


def _render_scenario_before_after(scenario: Scenario) -> str:
    before, after = _scenario_value_points(scenario)
    return (
        '<div class="scenario-before-after"><h4>Scenario Value: Before vs After</h4>'
        '<div class="impact-grid"><article><h5>Before</h5><p>'
        + escape(before)
        + '</p></article><article><h5>After</h5><p>'
        + escape(after)
        + "</p></article></div></div>"
    )


def _scenario_thesis(
    scenario: Scenario,
    windows: list[TrafficWindowResult],
    snapshots: list[SnapshotResult],
) -> str:
    name = scenario.name.lower()
    if windows:
        data = _traffic_chart_data(windows)
        summary = data.get("summary", {})
        cost_before = _money2(summary.get("manualCostTotal", 0.0))
        cost_after = _money2(summary.get("platformCostTotal", 0.0))
        savings = summary.get("savingsPct", 0.0)
        platform_range = summary.get("replicaRange", _replica_range(w.platform_replicas for w in windows))
        fixed_at_risk = _compact_int(summary.get("fixedAtRisk", 0))
        if "normal load" in name:
            text = (
                f"For normal daily Search traffic, the platform lowers 24-hour serving cost "
                f"from {cost_before} to {cost_after} ({savings}% lower) by running "
                f"{platform_range} replicas against the GPU utilization target as demand "
                f"rises and tapers."
            )
        elif "organic traffic spike" in name:
            text = (
                f"For an unplanned Search spike, the platform scales replicas with observed "
                f"GPU pressure, avoids {_compact_int(summary.get('fixedAtRisk', 0))} "
                f"fixed-fleet at-risk requests, and lands 24-hour cost at {cost_after} "
                f"versus {cost_before} for peak-provisioned capacity."
            )
        elif "planned promotion" in name:
            text = (
                f"For a known demand event, the platform pre-warms Recommendations capacity before "
                f"traffic arrives, then scales back down; cost lands at {cost_after} versus "
                f"{cost_before} for holding the peak fleet all day."
            )
        elif "endpoint unavailability" in name:
            text = (
                f"For endpoint unavailability, the platform applies the rolling replacement pattern to "
                f"runtime healing: bad pod stops receiving traffic, a fresh pod comes up healthy, traffic "
                f"shifts to it, the bad pod drains. High-priority Search overflows to healthy / premium "
                f"capacity during the readiness gap. Incident-day cost: {cost_after} versus {cost_before} "
                f"for peak-provisioned coverage."
            )
        elif "unavailable" in name or "unhealthy" in name:
            text = (
                "For serving failure modes, the platform uses health/readiness and latency "
                "signals to stop sending Search traffic to bad capacity and records the "
                "operational decision."
            )
        else:
            text = (
                f"The platform compares fixed capacity with an autoscaled Search fleet, "
                f"showing serving outcome, scale trigger, and attributed 24-hour cost "
                f"({cost_before} -> {cost_after})."
            )
    else:
        decision = _primary_canary_decision(snapshots)
        if decision is not None:
            text = (
                f"The release guardrail decision is {decision.decision}: "
                f"{decision.reason}"
            )
        else:
            text = scenario.description
    return '<p class="scenario-thesis">' + escape(text) + "</p>"


def _render_traffic_summary(scenario: Scenario, windows: list[TrafficWindowResult]) -> str:
    data = _traffic_chart_data(windows)
    if not data:
        return ""
    summary = data["summary"]
    fixed_range = _replica_range(window.manual_replicas for window in windows)
    platform_range = summary["replicaRange"]
    trigger = _scale_trigger_summary(scenario, windows)
    cards = [
        (
            f"24h cost ({summary['savingsPct']}% lower)",
            f"{_money2(summary['manualCostTotal'])} -> {_money2(summary['platformCostTotal'])}",
            _scenario_cost_basis(scenario),
        ),
        (
            "Replica policy",
            f"fixed {fixed_range}; platform {platform_range}",
            f"peak demand {_compact_int(summary['peakRps'])} RPS",
        ),
        (
            "Autoscale trigger",
            trigger,
            "utilization is the trigger; RPS is demand context",
        ),
    ]
    return (
        '<section class="traffic-summary"><h4>Impact Summary</h4>'
        '<div class="summary summary-traffic">'
        + "".join(_render_summary_card(label, value, note) for label, value, note in cards)
        + "</div>"
        + _scenario_cost_math(scenario)
        + "</section>"
    )


def _scale_trigger_summary(scenario: Scenario, windows: list[TrafficWindowResult]) -> str:
    axes = set()
    previous: dict[str, int] | None = None
    for window in windows:
        if previous is not None:
            for model_id, replicas in window.platform_replicas_by_model.items():
                if replicas != previous.get(model_id, replicas):
                    axes.add(
                        _trigger_axis_for_event(
                            window.event, scenario.models.get(model_id)
                        )
                    )
        previous = window.platform_replicas_by_model
    if not axes:
        return "No scale event"
    if axes == {"planned event"}:
        return "planned event pre-warm"
    if axes == {"GPU"}:
        return "GPU utilization target"
    if axes == {"CPU"}:
        return "CPU utilization target"
    labels = []
    for axis in sorted(axes):
        if axis in {"GPU", "CPU"}:
            labels.append(f"{axis} utilization target")
        else:
            labels.append(axis)
    return " + ".join(labels) + " trigger"


def _render_scaling_decisions(scenario: Scenario, windows: list[TrafficWindowResult]) -> str:
    if not windows:
        return ""
    profiles = _profiles_by_model(scenario)
    previous: dict[str, int] | None = None
    rows = []
    for window in windows:
        if previous is None:
            previous = window.platform_replicas_by_model
            continue
        for model_id in sorted(window.platform_replicas_by_model):
            prior = previous.get(model_id, 0)
            current = window.platform_replicas_by_model[model_id]
            if current == prior:
                continue
            profile = profiles.get(model_id)
            if profile is None:
                continue
            model = scenario.models.get(model_id)
            axis = _trigger_axis_for_event(window.event, model)
            target = profile.target_util_pct
            target_text = f"{target}%"
            if axis == "Latency/SLA":
                target_text = f"{profile.sla_ms} ms"
            elif axis == "Health/readiness":
                target_text = "healthy"
            elif axis == "planned event":
                target_text = "planned"
            demand = window.rps_by_model.get(model_id, 0)
            observed_util = window.trigger_util_pct_by_model.get(model_id)
            after_util = window.post_scale_util_pct_by_model.get(model_id)
            prior_safe = _safe_capacity_for_model(profile, prior)
            current_safe = _safe_capacity_for_model(profile, current)
            if current > prior:
                if axis == "planned event":
                    signal = "pre-warm before the planned demand window"
                elif axis in {"Latency/SLA", "Health/readiness"}:
                    signal = f"{axis} event: {window.event}"
                else:
                    signal = (
                        f"{axis} utilization crossed the {target}% target; add capacity "
                        "before SLA risk appears."
                    )
                action = f"scale out {prior}->{current}"
            else:
                if axis in {"Latency/SLA", "Health/readiness"}:
                    signal = f"{axis} event: {window.event}"
                else:
                    signal = (
                        f"{axis} utilization stayed below the {target}% target after "
                        "traffic tapered; remove idle capacity."
                    )
                action = f"scale down {prior}->{current}"
            trigger_text = (
                f"{axis} target {target_text}; demand {_compact_int(demand)} RPS"
                if axis in {"GPU", "CPU"}
                else f"{target_text}; demand {_compact_int(demand)} RPS"
            )
            before_text = (
                f"{_replica_label(prior)}, {_util_pct_text(observed_util)}, "
                f"{_compact_int(prior_safe)} RPS safe cap"
                if axis in {"GPU", "CPU"}
                else f"{_replica_label(prior)}, {_compact_int(prior_safe)} RPS safe cap"
            )
            after_text = (
                f"{_replica_label(current)}, {_util_pct_text(after_util)}, "
                f"{_compact_int(current_safe)} RPS safe cap"
                if axis in {"GPU", "CPU"}
                else f"{_replica_label(current)}, {_compact_int(current_safe)} RPS safe cap"
            )
            rows.append(
                "<tr><td>"
                + escape(_hour_label(window.start_hour))
                + "</td><td>"
                + escape(action)
                + "</td><td>"
                + escape(trigger_text)
                + "</td><td>"
                + escape(before_text)
                + "</td><td>"
                + escape(after_text)
                + "</td><td>"
                + escape(signal)
                + "</td></tr>"
            )
        previous = window.platform_replicas_by_model

    if not rows:
        return ""
    return (
        '<section class="scaling-decisions"><h4>Autoscaling Proof</h4>'
        '<p>Replica changes are explained by the trigger signal first. For GPU/CPU events, the proof is observed max-pod utilization compared with the model target; RPS is shown only as demand context.</p>'
        '<table><thead><tr><th>time</th><th>decision</th><th>trigger</th>'
        '<th>before</th><th>after</th><th>proof</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></section>"
    )


def _derived_key_events(windows: list[TrafficWindowResult]) -> list[dict[str, str]]:
    events = []
    previous_replicas: int | None = None
    for window in windows:
        changed = previous_replicas is not None and window.platform_replicas != previous_replicas
        has_event = bool(window.event)
        if changed or has_event or window.fixed_at_risk_requests:
            events.append(
                {
                    "time": _hour_label(window.start_hour),
                    "decision": window.scale_action,
                    "evidence": (
                        f"{_compact_int(window.total_rps)} RPS, "
                        f"current fleet safe capacity {_compact_int(window.fixed_capacity_rps)} RPS, "
                        f"platform safe capacity {_compact_int(window.platform_capacity_rps)} RPS"
                    ),
                    "outcome": (
                        f"{_compact_int(window.fixed_at_risk_requests)} current-fleet requests at risk; "
                        f"{_compact_int(window.platform_rejected_requests)} platform rejects"
                    ),
                }
            )
        previous_replicas = window.platform_replicas
    return events


def _render_key_events(scenario: Scenario, windows: list[TrafficWindowResult]) -> str:
    events = scenario.key_events or _derived_key_events(windows)
    if not events:
        return ""
    rows = []
    for event in events:
        rows.append(
            "<tr><td>"
            + escape(event.get("time", "-"))
            + "</td><td>"
            + escape(event.get("decision", "-"))
            + "</td><td>"
            + escape(event.get("evidence", "-"))
            + "</td><td>"
            + escape(event.get("outcome", "-"))
            + "</td></tr>"
        )
    return (
        '<details class="key-events"><summary>Scenario event audit</summary>'
        '<p>Scenario-level decisions preserved for audit: what signal fired, what the platform did, and what changed.</p>'
        '<table><thead><tr><th>time</th><th>platform decision</th><th>evidence</th><th>outcome</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></details>"
    )


def _primary_canary_decision(snapshots: list[SnapshotResult]) -> CanaryDecision | None:
    return next(
        (
            snapshot.canary_decision
            for snapshot in snapshots
            if snapshot.canary_decision is not None
        ),
        None,
    )


def _render_canary_summary(snapshots: list[SnapshotResult]) -> str:
    decision = _primary_canary_decision(snapshots)
    if decision is None:
        return ""
    failures = "".join(
        "<li>" + escape(failure) + "</li>" for failure in decision.failed_guardrails
    )
    guardrails = decision.effective_guardrails
    def _row(label: str, value: str) -> str:
        return "<tr><td>" + escape(label) + "</td><td>" + escape(value) + "</td></tr>"

    defined_guardrails = [
        ("P99 regression", "max_latency_regression_pct", "<=", "%"),
        ("Error rate regression", "max_error_regression_pct", "<=", "%"),
        ("Timeout rate regression", "max_timeout_regression_pct", "<=", "%"),
        ("FPR increase", "max_fpr_increase_pct", "<=", "%"),
        ("Precision drop", "max_precision_drop_pct", "<=", "%"),
        ("Recall drop", "max_recall_drop_pct", "<=", "%"),
        ("Min observations", "min_observations", ">=", ""),
    ]
    guardrail_rows = "".join(
        _row(label, f"{op} {guardrails[key]}{unit}")
        for label, key, op, unit in defined_guardrails
        if key in guardrails
    )

    if decision.decision == "auto_promote":
        proof = (
            "All operational and quality guardrails are within their pre-approved bands. "
            "The platform promotes without manual review."
        )
    elif decision.decision == "requires_owner_review":
        proof = (
            "Operational metrics are clean, but one or more quality guardrails "
            "(precision or recall) regressed. A fraud analyst must review before promotion."
        )
    elif decision.decision == "rollback":
        proof = (
            "An operational guardrail breached (latency, error rate, timeout, or FPR). "
            "The platform rolls back to the previous stable version."
        )
    else:
        proof = "The observation window has not reached the minimum threshold. Holding canary."
    return (
        '<section class="canary-summary"><h4>Release Guardrail Decision</h4>'
        + f"<p><strong>{escape(decision.decision)}</strong>: {escape(decision.reason)}</p>"
        + "<p>"
        + escape(proof)
        + "</p>"
        + '<dl class="metrics">'
        + f"<dt>candidate</dt><dd>{escape(decision.candidate_version)}</dd>"
        + f"<dt>rollback</dt><dd>{escape(decision.rollback_to_version or '-')}</dd>"
        + f"<dt>decision</dt><dd>{escape(decision.decision)}</dd>"
        + "</dl>"
        + '<h5>Guardrails</h5><table><thead><tr><th>guardrail</th><th>threshold</th></tr></thead><tbody>'
        + guardrail_rows
        + "</tbody></table>"
        + ("<h5>Failed guardrails</h5><ul>" + failures + "</ul>" if failures else "")
        + "</section>"
    )


def _render_window_summary(windows: list[TrafficWindowResult]) -> str:
    if not windows:
        return ""
    fixed_rows = []
    platform_rows = []
    for window in windows:
        outcome_cell = (
            f"{_compact_int(window.platform_served_requests)} served; "
            f"{_compact_int(window.platform_overflow_requests)} overflow; "
            f"{_compact_int(window.platform_rejected_requests)} rejected"
        )
        fixed_rows.append(
            "<tr><td>"
            + escape(_compact_window_axis_label(window.start_hour, window.end_hour))
            + "</td><td>"
            + escape(_window_range_label(window.start_hour, window.end_hour))
            + "</td><td>"
            + escape(window.label)
            + "</td><td>"
            + _compact_int(window.total_rps)
            + "</td><td>"
            + _compact_int(window.fixed_capacity_rps)
            + "</td><td>"
            + _compact_int(window.fixed_replicas)
            + "</td><td>"
            + f"{window.fixed_p99_ms:.1f}"
            + "</td><td>"
            + _compact_int(window.fixed_at_risk_requests)
            + "</td></tr>"
        )
        platform_rows.append(
            "<tr><td>"
            + escape(_compact_window_axis_label(window.start_hour, window.end_hour))
            + "</td><td>"
            + escape(_window_range_label(window.start_hour, window.end_hour))
            + "</td><td>"
            + escape(window.label)
            + "</td><td>"
            + _compact_int(window.total_rps)
            + "</td><td>"
            + _compact_int(window.platform_capacity_rps)
            + "</td><td>"
            + _compact_int(window.platform_replicas)
            + "</td><td>"
            + f"{window.platform_p99_ms:.1f}"
            + "</td><td>"
            + escape(outcome_cell)
            + "</td></tr>"
        )
    return (
        '<details class="window-summary"><summary>Traffic Bucket Details</summary>'
        '<p>Bucket-level validation split by operating model.</p>'
        '<h4>Before: Fixed Fleet Capacity</h4>'
        '<table><thead><tr><th>bucket</th><th>window</th><th>label</th><th>incoming RPS</th>'
        '<th>fixed safe capacity</th><th>fixed replicas</th><th>fixed P99 ms</th>'
        '<th>at-risk requests</th>'
        "</tr></thead><tbody>"
        + "".join(fixed_rows)
        + "</tbody></table>"
        '<h4>After: Autoscaled Platform Capacity</h4>'
        '<table><thead><tr><th>bucket</th><th>window</th><th>label</th><th>incoming RPS</th>'
        '<th>autoscaled safe capacity</th><th>active replicas</th><th>platform P99 ms</th>'
        '<th>platform outcome</th>'
        "</tr></thead><tbody>"
        + "".join(platform_rows)
        + "</tbody></table></details>"
    )


def _signal_by_deployment(
    signals: Iterable[AutoscaleSignal],
) -> dict[tuple[str, str], AutoscaleSignal]:
    return {(s.model_id, s.serving_backend): s for s in signals}


def _render_capacity(snapshot: SnapshotResult, models: dict[str, Model]) -> str:
    signals = _signal_by_deployment(snapshot.autoscale_signals)
    cards = []
    for (model_id, backend), endpoints in sorted(_group_endpoints(snapshot.endpoints).items()):
        model = models.get(model_id)
        signal = signals.get((model_id, backend))
        healthy = [e for e in endpoints if e.healthy]
        qps = sum(e.current_qps for e in healthy)
        capacity = sum(e.capacity_qps for e in healthy)
        max_gpu = max((e.gpu_util_pct for e in healthy), default=0.0)
        max_cpu = max((e.cpu_util_pct for e in healthy), default=0.0)
        gpu_target = signal.target_gpu_util_pct if signal else 0
        cpu_target = signal.target_cpu_util_pct if signal else 0
        replicas = len(endpoints)
        recommended = signal.recommended_replicas if signal else replicas
        p99 = str(model.sla_ms) if model else "-"
        cards.append(
            '<div class="capacity-card"><h5>'
            + escape(model_id)
            + "</h5><p>"
            + escape(backend)
            + "</p>"
            + '<dl class="metrics">'
            + f"<dt>qps/cap</dt><dd>{qps}/{capacity}</dd>"
            + f"<dt>p99_target_ms</dt><dd>{p99}</dd>"
            + f"<dt>replicas</dt><dd>{replicas} -> {recommended} recommended replicas</dd>"
            + f"<dt>max_gpu%</dt><dd>{max_gpu:.1f} / {gpu_target}</dd>"
            + "</dl>"
            + _bar(max_gpu, gpu_target or 100, f"{model_id} GPU utilization")
            + '<dl class="metrics">'
            + f"<dt>max_cpu%</dt><dd>{max_cpu:.1f} / {cpu_target}</dd>"
            + "</dl>"
            + _bar(max_cpu, cpu_target or 100, f"{model_id} CPU utilization")
            + "</div>"
        )
    return '<div class="capacity-grid">' + "".join(cards) + "</div>"


def _serving_outcome_label(status: str) -> str:
    if status == "routed":
        return "served"
    if status == "overflow_to_premium":
        return "overflow"
    return "rejected"


def _render_routes(decisions: Iterable[RouteDecision]) -> str:
    rows = []
    for decision in decisions:
        status_class = "ok" if decision.status in {"routed", "overflow_to_premium"} else "bad"
        rows.append(
            "<tr><td>"
            + escape(decision.request_id)
            + "</td><td>"
            + escape(decision.model_id)
            + '</td><td><span class="badge '
            + status_class
            + '">'
            + escape(_serving_outcome_label(decision.status))
            + "</span></td><td>"
            + escape(decision.chosen_endpoint or "-")
            + "</td><td>"
            + (f"{decision.estimated_latency_ms:.2f}" if decision.estimated_latency_ms else "-")
            + "</td><td>"
            + _cost_per_1k(decision)
            + "</td><td>"
            + escape(decision.route_reason)
            + "</td></tr>"
        )
    return (
        '<table><thead><tr><th>request</th><th>model</th><th>outcome</th>'
        '<th>endpoint</th><th>p99_ms</th><th>gpu_cost_per_1k</th><th>reason</th>'
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_costs(cost_rows: Iterable[ModelCostRow]) -> str:
    rows = []
    total = 0.0
    for row in cost_rows:
        total += row.total_cost_usd
        rows.append(
            "<tr><td>"
            + escape(row.model_id)
            + "</td><td>"
            + escape(row.owner_team)
            + f"</td><td>{row.routed_count}</td><td>{row.overflow_count}</td>"
            + f"<td>{row.rejected_count}</td><td>{row.avg_estimated_latency_ms:.2f}</td>"
            + f"<td>${row.total_cost_usd:.6f}</td></tr>"
        )
    return (
        '<table><thead><tr><th>model</th><th>owner</th><th>served</th>'
        '<th>overflow</th><th>rejected</th><th>avg_p99_ms</th><th>total_cost_usd</th>'
        "</tr></thead><tbody>"
        + "".join(rows)
        + f"</tbody></table><p><strong>Total:</strong> ${total:.6f}</p>"
    )


def _render_signals(signals: Iterable[AutoscaleSignal]) -> str:
    rows = []
    for signal in signals:
        rows.append(
            "<tr><td>"
            + escape(signal.model_id)
            + "</td><td>"
            + escape(signal.serving_backend)
            + f"</td><td>{signal.current_replicas} -> {signal.recommended_replicas}</td>"
            + f"<td>{signal.aggregate_gpu_util_pct:.1f}</td><td>{signal.target_gpu_util_pct}</td>"
            + f"<td>{signal.target_cpu_util_pct}</td><td>{signal.aggregate_load_pct:.1f}</td><td>"
            + escape(signal.reason)
            + "</td></tr>"
        )
    return (
        '<table><thead><tr><th>model</th><th>backend</th><th>recommended replicas</th>'
        '<th>gpu_util%</th><th>gpu_target%</th><th>cpu_target%</th><th>load%</th>'
        "<th>reason</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_canary(decision: CanaryDecision | None) -> str:
    if decision is None:
        return ""
    failures = "".join(
        "<li>" + escape(failure) + "</li>" for failure in decision.failed_guardrails
    )
    return (
        '<div class="canary"><h4>Canary Decision</h4>'
        + f"<p><strong>{escape(decision.decision)}</strong>: {escape(decision.reason)}</p>"
        + f"<p>Candidate: {escape(decision.candidate_version)}</p>"
        + ("<ul>" + failures + "</ul>" if failures else "")
        + "</div>"
    )


def _render_snapshot(snapshot: SnapshotResult, models: dict[str, Model]) -> str:
    return (
        '<details class="raw-audit"><summary>Raw CLI audit: '
        + escape(snapshot.name)
        + '</summary><article class="phase"><h3>'
        + escape(snapshot.name)
        + "</h3><p>"
        + escape(snapshot.description)
        + '</p><details class="audit-subdetail"><summary>Capacity And Targets</summary>'
        + _render_capacity(snapshot, models)
        + "</details>"
        + '<details class="audit-subdetail"><summary>Autoscaler Signals</summary>'
        + _render_signals(snapshot.autoscale_signals)
        + "</details>"
        + '<details class="audit-subdetail"><summary>Serving Decision Trace</summary>'
        + _render_routes(snapshot.decisions)
        + "</details>"
        + '<details class="audit-subdetail"><summary>Cost Summary</summary>'
        + _render_costs(snapshot.cost_rows)
        + "</details>"
        + _render_canary(snapshot.canary_decision)
        + "</article></details>"
    )


def render_scenario_section(scenario: Scenario, *, index: int) -> str:
    slug = _slug(scenario.name, index)
    snapshots = _scenario_snapshots(scenario)
    show_traffic_cost = _scenario_cost_coupled(scenario)
    traffic_windows = (
        _build_traffic_windows(scenario, snapshots) if show_traffic_cost else []
    )
    timeline = ""
    if len(snapshots) > 1:
        timeline = (
            '<div class="timeline"><h4>Phase Timeline</h4>'
            + "".join("<span>" + escape(s.name) + "</span>" for s in snapshots)
            + "</div>"
        )
    return (
        '<section class="scenario" id="'
        + escape(slug)
        + '"><h2>'
        + escape(scenario.name)
        + "</h2>"
        + _scenario_thesis(scenario, traffic_windows, snapshots)
        + _render_model_setup_strip(scenario)
        + _render_canary_summary(snapshots)
        + (timeline if show_traffic_cost else "")
        + (_render_traffic_summary(scenario, traffic_windows) if show_traffic_cost else "")
        + (_render_scaling_decisions(scenario, traffic_windows) if show_traffic_cost else "")
        + (
            _render_metric_charts(
                _build_metric_timesteps(snapshots),
                index=index,
                traffic_windows=traffic_windows,
            )
            if show_traffic_cost
            else ""
        )
        + (_render_window_summary(traffic_windows) if show_traffic_cost else "")
        + ("" if show_traffic_cost else _render_key_events(scenario, traffic_windows))
        + "</section>"
    )


def _render_nav(scenarios: list[Scenario], *, hrefs: list[str] | None = None) -> str:
    return _render_scenario_nav(scenarios, hrefs=hrefs)


def _render_scenario_nav(
    scenarios: list[Scenario],
    *,
    hrefs: list[str] | None = None,
    active_index: int | None = None,
    index_href: str | None = None,
) -> str:
    links = []
    if index_href is not None:
        links.append(
            '<a class="nav-index-link" href="'
            + escape(index_href)
            + '">Portfolio index</a>'
        )
    for idx, scenario in enumerate(scenarios, start=1):
        href = hrefs[idx - 1] if hrefs else "#" + _slug(scenario.name, idx)
        is_active = idx == active_index
        active_attrs = ' class="active" aria-current="page"' if is_active else ""
        summary = _scenario_review_summary(scenario, idx)
        links.append(
            '<a href="'
            + escape(href)
            + '"'
            + active_attrs
            + '><span class="nav-scenario-number">'
            + escape(summary["axis"].split(":")[0])
            + '</span><span class="nav-scenario-title">'
            + escape(summary["title"])
            + "</span>"
            + "</a>"
        )
    return '<nav class="scenario-nav"><h2>Scenarios</h2>' + "".join(links) + "</nav>"


def _render_index_overview(scenarios: list[Scenario], model_count: int) -> str:
    traffic_count = sum(1 for scenario in scenarios if _scenario_cost_coupled(scenario))
    release_count = len(scenarios) - traffic_count
    return (
        '<section class="index-overview"><h2>Scenario Review Map</h2>'
        '<p>Use this page as a launcher. Each scenario page carries the detailed proof: thesis, model policy, impact cards, decision evidence, charts, and optional audit details.</p>'
        '<div class="summary index-summary">'
        + _render_summary_card("Traffic / cost scenarios", str(traffic_count), "demand, capacity, replicas, cost")
        + _render_summary_card("Release confidence", str(release_count), "Fraud canary guardrails")
        + _render_summary_card("Models represented", str(model_count), "Search, Recommendations, Fraud")
        + _render_summary_card("Verification", "53 tests", "python -m unittest discover -s tests -t .")
        + "</div></section>"
    )


def _render_scenario_index(scenarios: list[Scenario], hrefs: list[str]) -> str:
    grouped_rows: dict[str, list[str]] = {
        key: [] for key in _SCENARIO_TYPE_DETAILS
    }
    for index, (scenario, href) in enumerate(zip(scenarios, hrefs), start=1):
        scenario_href = escape(href)
        summary = _scenario_review_summary(scenario, index)
        grouped_rows[_scenario_type_key(scenario)].append(
            "<tr><td><strong>"
            + escape(summary["axis"])
            + "</strong><br><a href=\""
            + scenario_href
            + '">'
            + escape(summary["title"])
            + "</a></td><td>"
            + escape(summary["value"])
            + "</td><td>"
            + escape(summary["evidence"])
            + "</td><td><span class=\"scenario-type-badge\">"
            + escape(summary["requirement"])
            + "</span></td></tr>"
        )
    groups = []
    for key, details in _SCENARIO_TYPE_DETAILS.items():
        rows = grouped_rows.get(key, [])
        if not rows:
            continue
        groups.append(
            '<div class="scenario-index-group"><h3>'
            + escape(details["heading"])
            + '</h3><table class="scenario-table"><thead><tr><th>scenario</th><th>platform value</th><th>primary evidence</th><th>requirement</th></tr></thead><tbody>'
            + "".join(rows)
            + "</tbody></table></div>"
        )
    return (
        '<section class="scenario-index"><h2>Scenario Dashboards</h2>'
        '<p>Pick the scenario that matches the review question. The index stays compact; the scenario page carries the evidence.</p>'
        + "".join(groups)
        + "</section>"
    )


def _css() -> str:
    return """
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #f6f8fb; }
* { box-sizing: border-box; }
header { padding: 32px 40px; background: #172033; color: white; }
header a { color: #dbe7ff; }
.layout { display: grid; grid-template-columns: 280px minmax(0, 1fr); gap: 24px; padding: 24px; }
.index-layout { grid-template-columns: minmax(0, 1fr); max-width: 1180px; margin: 0 auto; }
.scenario-page-layout { grid-template-columns: 280px minmax(0, 1fr); max-width: 1280px; margin: 0 auto; }
nav { position: sticky; top: 16px; align-self: start; background: white; border: 1px solid #d8dee9; border-radius: 14px; padding: 16px; }
nav a { display: flex; gap: 8px; align-items: flex-start; color: #1f5eff; text-decoration: none; padding: 9px 8px; border-bottom: 1px solid #edf0f5; border-radius: 8px; }
nav a:last-child { border-bottom: 0; }
nav a:hover { background: #f2f6ff; }
nav a.active { background: #eaf1ff; color: #173f9f; font-weight: 750; box-shadow: inset 3px 0 0 #1f5eff; }
.nav-index-link { display: block; margin-bottom: 8px; font-weight: 700; color: #40516a; }
.nav-scenario-number { flex: 0 0 auto; color: #607086; font-size: 12px; font-weight: 700; min-width: 24px; padding-top: 2px; }
.nav-scenario-title { min-width: 0; }
main { display: grid; gap: 24px; min-width: 0; }
.summary { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.summary-card, .scenario, .phase, .impact, .portfolio-charts, .scenario-index, .scenario-link-card, .traffic-summary, .scaling-decisions, .key-events, .window-summary, .raw-audit, .scenario-type-note, .canary-summary, .deployment-regions, .index-overview { background: white; border: 1px solid #d8dee9; border-radius: 14px; padding: 18px; box-shadow: 0 1px 4px rgba(10, 20, 40, 0.04); }
.summary-card, .scenario, .phase, .impact, .portfolio-charts, .scenario-index, .scenario-link-card, .traffic-summary, .scaling-decisions, .key-events, .window-summary, .raw-audit, .scenario-type-note, .canary-summary, .deployment-regions, .impact article, .index-overview { min-width: 0; }
.summary-card span { display: block; color: #607086; font-size: 13px; }
.summary-card strong { font-size: 24px; overflow-wrap: anywhere; }
.summary-card small { display: block; color: #607086; font-size: 12px; line-height: 1.35; margin-top: 6px; }
.summary-traffic { grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }
.index-summary { margin-top: 12px; }
.cost-math { margin: 14px 0 0; padding: 10px 12px; border: 1px solid #d8dee9; border-radius: 8px; background: #fbfcff; color: #40516a; font-size: 13px; line-height: 1.45; }
.cost-math code { color: #172033; font-weight: 700; }
.traffic-summary, .scaling-decisions, .key-events, .window-summary, .raw-audit, .scenario-type-note, .canary-summary, .deployment-regions { margin: 16px 0; box-shadow: none; }
.traffic-summary p, .scaling-decisions p, .key-events p, .window-summary p, .scenario-type-note p, .canary-summary p, .deployment-regions p { color: #40516a; }
.key-events summary, .window-summary summary { cursor: pointer; font-weight: 700; color: #172033; }
.scenario-thesis { margin: 0 0 14px; color: #142033; font-size: 18px; line-height: 1.45; max-width: 960px; }
.scenario-setup { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 4px; }
.setup-pill { display: inline-flex; flex-wrap: wrap; gap: 6px; align-items: center; padding: 8px 10px; border: 1px solid #d8dee9; border-radius: 8px; background: #fbfcff; color: #40516a; font-size: 13px; }
.setup-pill span { padding-right: 6px; border-right: 1px solid #d8dee9; }
.setup-pill span:last-child { border-right: 0; padding-right: 0; }
.setup-pill .setup-model { color: #172033; font-weight: 700; }
.raw-audit { background: #fbfcff; padding: 0; overflow: hidden; }
.raw-audit summary { cursor: pointer; padding: 14px 18px; font-weight: 700; color: #172033; }
.raw-audit summary:hover { background: #f2f6ff; }
.raw-audit .phase { border: 0; border-top: 1px solid #d8dee9; border-radius: 0; box-shadow: none; margin: 0; }
.audit-subdetail { border: 1px solid #d8dee9; border-radius: 10px; margin: 10px 0; background: white; }
.audit-subdetail summary { padding: 10px 12px; font-size: 14px; }
.audit-subdetail table, .audit-subdetail .capacity-grid, .audit-subdetail .metrics { margin-left: 12px; margin-right: 12px; }
.impact-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.impact article { border: 1px solid #d8dee9; border-radius: 12px; padding: 14px; background: #fbfcff; }
.scenario-card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
.scenario-index-group { margin-top: 18px; }
.scenario-index-group > h3 { margin-bottom: 6px; }
.scenario-index-group > p { color: #40516a; }
.scenario-table td:first-child { min-width: 170px; }
.scenario-table td:nth-child(2) { min-width: 250px; }
.scenario-table td:nth-child(3) { min-width: 260px; }
.scenario-link-card h3 { margin-top: 0; }
.scenario-link-card a { color: #1f5eff; text-decoration: none; }
.scenario-link-card a:hover { text-decoration: underline; }
.scenario-type-badge { display: inline-flex; align-items: center; min-height: 24px; padding: 3px 8px; border-radius: 999px; background: #edf3ff; color: #1f4fbf; font-size: 12px; font-weight: 700; margin-bottom: 10px; }
.scenario-before-after { border: 1px solid #d8dee9; border-radius: 14px; padding: 14px; margin: 16px 0; background: #fffdf8; }
.scenario-before-after h5 { margin: 0 0 8px; color: #607086; text-transform: uppercase; letter-spacing: .04em; font-size: 12px; }
.scenario-charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; margin: 16px 0; padding: 14px; background: #fbfcff; border: 1px solid #d8dee9; border-radius: 14px; }
.scenario-charts > h4, .scenario-charts > p { grid-column: 1 / -1; margin: 0 0 4px; }
.portfolio-charts p, .scenario-charts p { color: #40516a; }
.chart-note { margin-top: -6px; font-size: 14px; line-height: 1.5; }
.scenario-key { margin-top: 14px; border: 1px solid #d8dee9; border-radius: 10px; padding: 12px; background: #fbfcff; }
.scenario-key summary { cursor: pointer; font-weight: 700; color: #142033; }
.scenario-key table { margin-top: 10px; }
.window-summary h4 { margin: 14px 0 6px; color: #172033; font-size: 14px; }
.window-summary h4:first-of-type { margin-top: 8px; }
.chart-panel { min-width: 0; padding: 14px; background: white; border: 1px solid #d8dee9; border-radius: 12px; }
.chart-panel-wide { grid-column: 1 / -1; }
.chart-panel h3, .chart-panel h5 { margin: 0 0 10px; color: #172033; }
.chart-canvas-wrap { position: relative; height: 280px; width: 100%; }
.chart-canvas-wrap canvas { display: block; width: 100% !important; max-width: 100%; }
.portfolio-charts .chart-canvas-wrap { height: 340px; }
.chart-fallback { grid-column: 1 / -1; border: 1px solid #f0c36d; background: #fff8e8; color: #715000; border-radius: 8px; padding: 10px 12px; font-size: 13px; }
.timeline { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.timeline h4 { width: 100%; margin-bottom: 0; }
.timeline span { background: #e9efff; color: #173f9f; border-radius: 999px; padding: 6px 10px; font-weight: 600; }
.capacity-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
.capacity-card { border: 1px solid #d8dee9; border-radius: 12px; padding: 14px; background: #fbfcff; }
.capacity-card h5 { margin: 0; font-size: 16px; }
.capacity-card p { margin: 4px 0 12px; color: #607086; }
.metrics { display: grid; grid-template-columns: 120px 1fr; gap: 4px 8px; margin: 8px 0; }
.metrics dt { color: #607086; }
.metrics dd { margin: 0; font-weight: 650; }
.metric-bar { height: 10px; background: #e6eaf2; border-radius: 999px; overflow: hidden; margin: 8px 0 12px; }
.bar-fill { height: 100%; background: linear-gradient(90deg, #3d7eff, #21a67a); border-radius: 999px; }
table { display: block; width: 100%; overflow-x: auto; border-collapse: collapse; margin: 10px 0 18px; font-size: 13px; }
th, td { text-align: left; border-bottom: 1px solid #e4e8f0; padding: 8px; vertical-align: top; }
th { color: #607086; font-weight: 700; background: #f8faff; }
.badge { border-radius: 999px; padding: 3px 8px; font-weight: 700; white-space: nowrap; }
.badge.ok { background: #e8f7ef; color: #137344; }
.badge.bad { background: #fdecec; color: #a02727; }
.canary { border-left: 4px solid #3d7eff; padding-left: 12px; }
@media (max-width: 900px) { .layout { grid-template-columns: 1fr; } nav { position: static; } .summary { grid-template-columns: 1fr 1fr; } .impact-grid { grid-template-columns: 1fr; } }
@media (max-width: 640px) { header { padding: 24px; } .layout { padding: 14px; } .summary { grid-template-columns: 1fr; } .scenario-charts { grid-template-columns: 1fr; } .chart-canvas-wrap, .portfolio-charts .chart-canvas-wrap { height: 260px; } }
"""


def _chart_script() -> str:
    return """
<script>
(function () {
  const colors = {
    before: '#c14545',
    after: '#21a67a',
    demand: '#3d7eff',
    replicas: '#7c5cff',
    fixedReplicas: '#7d8794',
    served: '#21a67a',
    overflow: '#d99018',
    rejected: '#c14545',
    gpu: '#3d7eff',
    gpuTarget: '#83a9ff',
    cpu: '#7c5cff',
    cpuTarget: '#b8a8ff'
  };

  function readReportData() {
    const node = document.getElementById('report-data');
    if (!node) return { portfolio: {}, scenarios: [] };
    return JSON.parse(node.textContent);
  }

  function money(value) {
    return '$' + Number(value || 0).toFixed(2);
  }

  function compact(value) {
    const n = Number(value || 0);
    const a = Math.abs(n);
    if (a >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (a >= 1000) return (n / 1000).toFixed(1) + 'K';
    return Math.round(n).toString();
  }

  function baseOptions(extra) {
    return Object.assign({
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 12, usePointStyle: true } },
        tooltip: { callbacks: {} }
      },
      scales: {
        x: { ticks: { color: '#607086', maxRotation: 0, autoSkip: false }, grid: { display: false } }
      }
    }, extra || {});
  }

  function renderPortfolio(data) {
    const canvas = document.getElementById('portfolio-impact-chart');
    if (!canvas || !data.portfolio) return;
    const scenarioLabels = (data.portfolio.axisLabels || data.portfolio.labels).map(function (label) {
      const parts = String(label).split(': ');
      return parts.length === 2 ? [parts[0], parts[1]] : label;
    });
    if (!scenarioLabels.length) return;
    const before = data.portfolio.beforeTotalCost || data.portfolio.beforePeakCost;
    const after = data.portfolio.afterTotalCost || data.portfolio.afterPeakCost;
    const savings = data.portfolio.totalSavingsPct || data.portfolio.savingsPct;
    new Chart(canvas, {
      type: 'bar',
      data: {
        labels: scenarioLabels,
        datasets: [
          { label: 'Before: peak-provisioned', data: before, backgroundColor: colors.before, borderRadius: 4 },
          { label: 'After platform', data: after, backgroundColor: colors.after, borderRadius: 4 }
        ]
      },
      options: baseOptions({
        plugins: {
          legend: { position: 'bottom', labels: { boxWidth: 12, usePointStyle: true } },
          tooltip: {
            callbacks: {
              title: function (items) {
                return data.portfolio.labels[items[0].dataIndex];
              },
              afterBody: function (items) {
                const index = items[0].dataIndex;
                return [
                  'Savings: ' + savings[index] + '%',
                  'Attributed to model-owned serving replica-hours in this scenario'
                ];
              },
              label: function (context) {
                return context.dataset.label + ': ' + money(context.parsed.y);
              }
            }
          }
        },
        scales: {
          x: { title: { display: true, text: 'Scenario' }, ticks: { color: '#607086', maxRotation: 0, autoSkip: false }, grid: { display: false } },
          y: { beginAtZero: true, title: { display: true, text: 'Attributed 24-hour serving cost' }, ticks: { callback: money } }
        }
      })
    });
  }

  function renderSimulation(index, scenario) {
    const canvas = document.getElementById('scenario-simulation-' + index);
    if (!canvas) return;
    const traffic = scenario.traffic || {};
    const hasTraffic = Array.isArray(traffic.shortLabels) && traffic.shortLabels.length;
    if (hasTraffic) {
      new Chart(canvas, {
        data: {
          labels: traffic.axisLabels || traffic.shortLabels,
          datasets: [
            { type: 'line', label: 'Incoming demand (RPS)', data: traffic.totalRps, yAxisID: 'yRps', borderColor: colors.demand, backgroundColor: colors.demand, tension: 0.2, pointRadius: 3 },
            { type: 'line', label: 'Before: fixed-fleet safe capacity (RPS)', data: traffic.fixedCapacityRps, yAxisID: 'yRps', borderColor: colors.before, backgroundColor: colors.before, borderDash: [5, 4], tension: 0.15, pointRadius: 2 },
            { type: 'line', label: 'After: autoscaled safe capacity (RPS)', data: traffic.platformCapacityRps, yAxisID: 'yRps', borderColor: colors.after, backgroundColor: colors.after, tension: 0.15, pointRadius: 2 },
            { type: 'line', label: 'Before: fixed fleet replicas', data: traffic.fixedReplicas, yAxisID: 'yReplicas', borderColor: colors.fixedReplicas, backgroundColor: colors.fixedReplicas, borderDash: [5, 4], tension: 0.15, pointRadius: 2 },
            { type: 'line', label: 'After: active platform replicas', data: traffic.platformReplicas, yAxisID: 'yReplicas', borderColor: colors.replicas, backgroundColor: colors.replicas, tension: 0.15, pointRadius: 3 }
          ]
        },
        options: baseOptions({
          plugins: {
            legend: { position: 'bottom', labels: { boxWidth: 12, usePointStyle: true } },
            tooltip: {
              callbacks: {
                title: function (items) {
                  return traffic.labels[items[0].dataIndex] || items[0].label;
                },
                afterBody: function (items) {
                  const i = items[0].dataIndex;
                  return [
                    traffic.events[i] ? 'Event: ' + traffic.events[i] : 'Event: none',
                    'Decision: ' + traffic.scaleActions[i],
                    'Before fixed fleet: ' + traffic.fixedReplicas[i] + ' replicas',
                    'After active platform: ' + traffic.platformReplicas[i] + ' replicas',
                    'Autoscaler recommendation: ' + traffic.recommendedReplicas[i] + ' replicas',
                    'Current P99 / Platform P99: ' + traffic.fixedP99[i] + ' ms / ' + traffic.platformP99[i] + ' ms',
                    'SLA target: ' + traffic.slaTarget[i] + ' ms'
                  ];
                },
                label: function (context) {
                  const unit = context.dataset.yAxisID === 'yReplicas' ? ' replicas' : ' RPS';
                  return context.dataset.label + ': ' + compact(context.parsed.y) + unit;
                }
              }
            }
          },
          scales: {
            x: { ticks: { color: '#607086', maxRotation: 0, autoSkip: false }, grid: { display: false } },
            yRps: { beginAtZero: true, position: 'left', title: { display: true, text: 'RPS / safe capacity' }, ticks: { callback: compact } },
            yReplicas: { beginAtZero: true, position: 'right', title: { display: true, text: 'Replicas' }, grid: { drawOnChartArea: false }, ticks: { precision: 0 } }
          }
        })
      });
      return;
    }
    new Chart(canvas, {
      data: {
        labels: scenario.labels,
        datasets: [
          { type: 'bar', label: 'Static baseline cost', data: scenario.beforeHourlyCost, yAxisID: 'yCost', backgroundColor: 'rgba(193,69,69,.72)', borderRadius: 4 },
          { type: 'bar', label: 'After platform cost', data: scenario.afterHourlyCost, yAxisID: 'yCost', backgroundColor: 'rgba(33,166,122,.72)', borderRadius: 4 },
          { type: 'line', label: 'Incoming RPS', data: scenario.rps, yAxisID: 'yRps', borderColor: colors.demand, backgroundColor: colors.demand, tension: 0.25, pointRadius: 3 },
          { type: 'line', label: 'Adaptive replicas', data: scenario.afterReplicas, yAxisID: 'yReplicas', borderColor: colors.replicas, backgroundColor: colors.replicas, borderDash: [5, 4], tension: 0.25, pointRadius: 3 }
        ]
      },
      options: baseOptions({
        plugins: {
          legend: { position: 'bottom', labels: { boxWidth: 12, usePointStyle: true } },
          tooltip: {
            callbacks: {
              afterBody: function (items) {
                const i = items[0].dataIndex;
                return [
                  'Before fixed fleet: ' + scenario.beforeReplicas[i] + ' replicas',
                  'After active platform: ' + scenario.afterReplicas[i] + ' replicas'
                ];
              },
              label: function (context) {
                if (context.dataset.yAxisID === 'yCost') return context.dataset.label + ': ' + money(context.parsed.y);
                return context.dataset.label + ': ' + context.parsed.y;
              }
            }
          }
        },
        scales: {
          x: { ticks: { color: '#607086', maxRotation: 0, autoSkip: false }, grid: { display: false } },
          yCost: { beginAtZero: true, position: 'left', title: { display: true, text: 'Hourly cost' }, ticks: { callback: money } },
          yRps: { beginAtZero: true, position: 'right', title: { display: true, text: 'RPS' }, grid: { drawOnChartArea: false } },
          yReplicas: { beginAtZero: true, position: 'right', display: false, grid: { drawOnChartArea: false } }
        }
      })
    });
  }

  function renderCost(index, scenario) {
    const canvas = document.getElementById('scenario-cost-' + index);
    if (!canvas) return;
    const traffic = scenario.traffic || {};
    const hasTraffic = Array.isArray(traffic.shortLabels) && traffic.shortLabels.length;
    if (!hasTraffic) return;
    new Chart(canvas, {
      type: 'line',
      data: {
        labels: traffic.axisLabels || traffic.shortLabels,
        datasets: [
          { label: 'Before: peak-provisioned', data: traffic.manualCumulativeCost, borderColor: colors.before, backgroundColor: colors.before, tension: 0.2, pointRadius: 3 },
          { label: 'After: platform', data: traffic.platformCumulativeCost, borderColor: colors.after, backgroundColor: colors.after, tension: 0.2, pointRadius: 3 }
        ]
      },
      options: baseOptions({
        plugins: {
          legend: { position: 'bottom', labels: { boxWidth: 12, usePointStyle: true } },
          tooltip: {
            callbacks: {
              title: function (items) {
                return traffic.labels[items[0].dataIndex] || items[0].label;
              },
              afterBody: function (items) {
                const i = items[0].dataIndex;
                return [
                  'Hourly cost: before peak ' + money(traffic.manualHourlyCost[i]) + ', platform ' + money(traffic.platformHourlyCost[i]),
                  'Decision: ' + traffic.scaleActions[i]
                ];
              },
              label: function (context) {
                return context.dataset.label + ': ' + money(context.parsed.y);
              }
            }
          }
        },
        scales: {
          x: { ticks: { color: '#607086', maxRotation: 0, autoSkip: false }, grid: { display: false } },
          y: { beginAtZero: true, title: { display: true, text: 'Cumulative 24-hour cost' }, ticks: { callback: money } }
        }
      })
    });
  }

  function showFallback(message) {
    document.querySelectorAll('.scenario-charts, .portfolio-charts').forEach(section => {
      const node = document.createElement('div');
      node.className = 'chart-fallback';
      node.textContent = message;
      section.appendChild(node);
    });
  }

  window.addEventListener('DOMContentLoaded', function () {
    const data = readReportData();
    if (typeof Chart === 'undefined') {
      showFallback('Chart.js did not load. The deterministic tables below still show the simulation output.');
      return;
    }
    renderPortfolio(data);
    data.scenarios.forEach(function (scenario, i) {
      const index = scenario.scenarioIndex || i + 1;
      renderSimulation(index, scenario);
      renderCost(index, scenario);
    });
  });
})();
</script>
"""


def render_html_report(
    scenario_paths: Iterable[Path], *, scenario_dir_name: str = "simulation-report-scenarios"
) -> str:
    scenarios = [load_scenario(path) for path in scenario_paths]
    report_data = _build_report_data(scenarios)
    model_ids = {
        model_id for scenario in scenarios for model_id in scenario.models.keys()
    }
    hrefs = [
        f"{scenario_dir_name}/{_scenario_page_filename(scenario, idx)}"
        for idx, scenario in enumerate(scenarios, start=1)
    ]
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>ModelOps Platform Simulation Report</title><style>"
        + _css()
        + "</style>"
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.5.0/chart.umd.min.js"></script>'
        + _json_script({"portfolio": report_data["portfolio"], "scenarios": []})
        + "</head><body><header><h1>ModelOps Platform Simulation Report</h1>"
        "<p>Generated from the bundled prototype scenarios. Use this as a lightweight launchpad: traffic pages prove autoscaling and cost, failure pages prove operational safety, and Fraud canary pages prove release confidence.</p></header>"
        '<div class="layout index-layout"><main>'
        + _render_index_overview(scenarios, len(model_ids))
        + _render_scenario_index(scenarios, hrefs)
        + _render_portfolio_chart(report_data["portfolio"])
        + "</main></div>"
        + _chart_script()
        + "</body></html>"
    )


def render_scenario_html_page(
    scenario: Scenario,
    *,
    all_scenarios: list[Scenario],
    scenario_index: int,
    index_href: str,
) -> str:
    report_data = _build_report_data([scenario])
    if report_data["scenarios"]:
        report_data["scenarios"][0]["scenarioIndex"] = scenario_index
        report_data["scenarios"][0]["axisLabel"] = _portfolio_axis_label(
            scenario.name, scenario_index
        )
    hrefs = [
        _scenario_page_filename(item, idx)
        for idx, item in enumerate(all_scenarios, start=1)
    ]
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>"
        + escape(scenario.name)
        + " - ModelOps Scenario</title><style>"
        + _css()
        + "</style>"
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.5.0/chart.umd.min.js"></script>'
        + _json_script(report_data)
        + '</head><body><header><h1>'
        + escape(scenario.name)
        + "</h1><p>"
        + escape(_scenario_header_subtitle(scenario))
        + "</p></header>"
        '<div class="layout scenario-page-layout">'
        + _render_scenario_nav(
            all_scenarios,
            hrefs=hrefs,
            active_index=scenario_index,
            index_href=index_href,
        )
        + "<main>"
        + render_scenario_section(scenario, index=scenario_index)
        + "</main></div>"
        + _chart_script()
        + "</body></html>"
    )


def write_html_report(scenario_paths: Iterable[Path], output_path: Path) -> None:
    paths = list(scenario_paths)
    scenarios = [load_scenario(path) for path in paths]
    pages_dir = _scenario_pages_dir(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    for stale_page in pages_dir.glob("scenario-*.html"):
        stale_page.unlink()
    output_path.write_text(
        render_html_report(
            paths,
            scenario_dir_name=pages_dir.name,
        ),
        encoding="utf-8",
    )
    for idx, scenario in enumerate(scenarios, start=1):
        page_path = pages_dir / _scenario_page_filename(scenario, idx)
        page_path.write_text(
            render_scenario_html_page(
                scenario,
                all_scenarios=scenarios,
                scenario_index=idx,
                index_href=f"../{output_path.name}",
            ),
            encoding="utf-8",
        )
