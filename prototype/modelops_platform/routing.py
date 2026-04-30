"""Cost-aware SLA-respecting routing policy.

Vocabulary: each `Endpoint` dataclass instance is one serving pod. The
narrative below uses "pod" for the physical unit; the type / variable
names retain the `Endpoint` / `endpoints` naming for code-history reasons.

Decision order (this is the platform contract, not an algorithm choice):

  1. Resolve model and find candidate pods by model_id and serving_backend.
  2. Exclude unhealthy pods.
  3. Exclude pods that have no remaining capacity.
  4. Estimate latency including base + load penalty + batching window.
  5. Filter to SLA-feasible candidates.
  6. For high-priority requests, restrict to pods below the model's
     utilization target so a latency-sensitive workload does not land on
     a near-saturated pod. The target comes from `target_gpu_util_pct`
     (GPU-backed models) or `target_cpu_util_pct` (CPU-backed) on the
     model manifest.
  7. Choose the lowest cost-per-1k-predictions among the remaining
     feasible candidates.
  8. If no feasible candidate exists:
     - high priority: overflow to the fastest healthy pod with capacity
       (`overflow_to_premium`). This costs more but preserves the SLA tier.
     - normal/low priority: reject with an explainable reason
       (`rejected_sla`).

The headroom target in step 6 is the same utilization target the
autoscaler uses (see `policy.utilization_targets_for_model`). That keeps
the router and autoscaler aligned: whatever utilization triggers a scale
event for a model is also the load ceiling the router protects.
"""

from __future__ import annotations

from typing import Iterable, Optional

from .domain import Endpoint, Model, Request, RouteDecision
from .policy import utilization_targets_for_model


FALLBACK_HIGH_PRIORITY_HEADROOM = 0.6
"""Fallback headroom ceiling when the model is unknown.

Used only when the routing call cannot resolve the model's manifest
(which should never happen in practice; the registry is the source of
truth). Matches the default `target_gpu_util_pct` on the Model dataclass
so the fallback is consistent with what an untuned model would get.
"""


def headroom_target_for(model: Optional[Model]) -> float:
    """Headroom ceiling for high-priority routing, in [0, 1].

    Reads `target_gpu_util_pct` (GPU-backed) or `target_cpu_util_pct`
    (CPU-backed) directly from the model manifest via
    `policy.utilization_targets_for_model`. Same value the autoscaler
    uses — one definition of "too hot" per model, shared across both.
    """
    if model is None:
        return FALLBACK_HIGH_PRIORITY_HEADROOM
    targets = utilization_targets_for_model(model)
    axis = "gpu" if model.requires_gpu else "cpu"
    return targets[axis] / 100.0


def estimate_latency_ms(endpoint: Endpoint, model: Model) -> float:
    """Estimate served **P99 latency** for `model` on `endpoint` at current load.

    Three contributions:
    - base latency at the instance/runtime level (`Endpoint.base_latency_ms`,
      defined as P99 at low load — same percentile the SLA budget and the
      canary guardrail use).
    - load penalty: P99 rises sharply once an endpoint is past 60% util,
      and even more sharply past 80%. This is a stand-in for queue
      buildup and contention.
    - batching window: half the configured window is added on average for
      models that opt into batching.

    The result is compared 1:1 with `Model.sla_ms`, which is the per-request
    P99 budget from the manifest.
    """

    util = 0.0
    if endpoint.capacity_qps > 0:
        util = endpoint.current_qps / endpoint.capacity_qps

    base = endpoint.base_latency_ms
    if util >= 1.0:
        # over capacity; assume hard backpressure
        load_penalty = base * 5.0
    elif util > 0.8:
        load_penalty = base * (1.0 + (util - 0.8) * 5.0)
    elif util > 0.6:
        load_penalty = base * (1.0 + (util - 0.6))
    else:
        load_penalty = 0.0

    batching = (model.batching_window_ms / 2.0) if model.batching_window_ms else 0.0
    return round(base + load_penalty + batching, 2)


def _candidates_for(model: Model, endpoints: Iterable[Endpoint]) -> list[Endpoint]:
    """Find endpoints that could serve this model based on the platform contract."""
    return [
        e
        for e in endpoints
        if e.model_id == model.model_id and e.serving_backend == model.serving_backend
    ]


def route_request(
    request: Request,
    model: Model,
    endpoints: Iterable[Endpoint],
    *,
    headroom_target: Optional[float] = None,
) -> RouteDecision:
    """Apply the routing policy and return an auditable decision.

    `headroom_target` defaults to the class-driven GPU target for the
    model (e.g., 0.5 for `strict_realtime`). Tests can pass an explicit
    override.
    """
    sla_ms = request.sla_ms_override or model.sla_ms
    if headroom_target is None:
        headroom_target = headroom_target_for(model)
    candidates = _candidates_for(model, endpoints)

    if not candidates:
        return RouteDecision(
            request_id=request.request_id,
            model_id=model.model_id,
            status="rejected_no_capacity",
            chosen_endpoint=None,
            route_reason="no_endpoint_for_model_and_backend",
            sla_ms=sla_ms,
        )

    healthy = [e for e in candidates if e.healthy]
    if not healthy:
        return RouteDecision(
            request_id=request.request_id,
            model_id=model.model_id,
            status="rejected_unhealthy",
            chosen_endpoint=None,
            route_reason="all_endpoints_unhealthy",
            sla_ms=sla_ms,
        )

    with_capacity = [e for e in healthy if e.current_qps < e.capacity_qps]
    if not with_capacity:
        # All endpoints are at or above capacity. Treat as overload.
        return RouteDecision(
            request_id=request.request_id,
            model_id=model.model_id,
            status="rejected_no_capacity",
            chosen_endpoint=None,
            route_reason="all_endpoints_saturated",
            sla_ms=sla_ms,
        )

    # Latency estimate per candidate; partition into SLA-feasible vs not.
    estimates: list[tuple[Endpoint, float]] = [
        (e, estimate_latency_ms(e, model)) for e in with_capacity
    ]
    sla_feasible = [(e, lat) for (e, lat) in estimates if lat <= sla_ms]

    if not sla_feasible:
        # No endpoint can hit SLA. High-priority workloads can overflow to the
        # fastest available capacity (premium routing); others are rejected.
        if request.priority == "high":
            estimates.sort(key=lambda pair: pair[1])
            ep, lat = estimates[0]
            return RouteDecision(
                request_id=request.request_id,
                model_id=model.model_id,
                status="overflow_to_premium",
                chosen_endpoint=ep.endpoint_id,
                route_reason="high_priority_overflow_no_sla_feasible",
                estimated_latency_ms=lat,
                estimated_cost_usd=round(ep.cost_per_1k_predictions / 1000.0, 9),
                sla_ms=sla_ms,
            )
        return RouteDecision(
            request_id=request.request_id,
            model_id=model.model_id,
            status="rejected_sla",
            chosen_endpoint=None,
            route_reason="no_sla_feasible_capacity",
            sla_ms=sla_ms,
        )

    pool = sla_feasible
    reason_suffix = "lowest_cost"

    if request.priority == "high":
        with_headroom = [
            (e, lat)
            for (e, lat) in sla_feasible
            if (e.current_qps / max(e.capacity_qps, 1)) <= headroom_target
        ]
        if with_headroom:
            pool = with_headroom
            reason_suffix = "headroom_preserved"
        else:
            reason_suffix = "no_headroom_lowest_cost"

    # Choose lowest cost; tie-break on lowest estimated latency.
    pool.sort(key=lambda pair: (pair[0].cost_per_1k_predictions, pair[1]))
    ep, lat = pool[0]
    return RouteDecision(
        request_id=request.request_id,
        model_id=model.model_id,
        status="routed",
        chosen_endpoint=ep.endpoint_id,
        route_reason=f"sla_feasible_{reason_suffix}",
        estimated_latency_ms=lat,
        estimated_cost_usd=round(ep.cost_per_1k_predictions / 1000.0, 9),
        sla_ms=sla_ms,
    )


def apply_routing(
    requests: Iterable[Request],
    models: dict[str, Model],
    endpoints: list[Endpoint],
) -> list[RouteDecision]:
    """Route a batch of requests, mutating endpoint load along the way.

    The mutation is intentional: routing N requests should reflect that
    earlier routes consumed capacity. Each routed request increments the
    chosen endpoint's `current_qps` by 1.
    """
    endpoints_by_id = {e.endpoint_id: e for e in endpoints}
    decisions: list[RouteDecision] = []
    for request in requests:
        model = models.get(request.model_id)
        if model is None:
            decisions.append(
                RouteDecision(
                    request_id=request.request_id,
                    model_id=request.model_id,
                    status="rejected_no_capacity",
                    chosen_endpoint=None,
                    route_reason="model_id_not_in_registry",
                )
            )
            continue
        decision = route_request(request, model, endpoints)
        if decision.chosen_endpoint:
            endpoints_by_id[decision.chosen_endpoint].current_qps += 1
        decisions.append(decision)
    return decisions
