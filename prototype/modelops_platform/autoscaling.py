"""Online inference autoscaler signals.

Vocabulary: each `Endpoint` dataclass instance is one pod (one replica).
The narrative below uses "pod" for the physical unit; the type names
retain the `Endpoint` / `endpoints` naming for code-history reasons.

This module does not actually scale Kubernetes deployments. It computes the
autoscaler recommendation that an HPA-style controller would consume,
using the same metrics described in the architecture: GPU utilization,
CPU utilization, load percentage, planned event signal, and unhealthy
pod count.

Scaling target comes from the model's manifest. Each model carries
explicit `target_gpu_util_pct` and `target_cpu_util_pct` fields — a
latency-sensitive model can run cool at 50% while a throughput-bound
model can deliberately run hot at 85%.

Policy mirrors the attack plan:

- Both scale-up and scale-down triggers use the **maximum pod
  utilization**, so a cool reserve pod cannot mask a saturated
  workhorse (scale-up), and a hot workhorse cannot be averaged away
  by an idle reserve (scale-down).
- The aggregate is still reported for capacity-planning context, and
  it includes every GPU-backed pod (including cold reserves at 0%).
- A pod is treated as GPU-backed when the registered model has
  `requires_gpu=True`, falling back to the `instance_class` prefix
  when the model is missing from the registry.
- A planned event (flash sale, promotion, launch) triggers pre-warming
  up to a configured burst capacity even before utilization rises.
- Unhealthy pods are treated as missing capacity for the purpose
  of the recommendation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Optional

from .domain import AutoscaleSignal, Endpoint, Model
from .policy import utilization_targets_for_model


def aggregate_by_deployment(
    endpoints: Iterable[Endpoint],
) -> dict[tuple[str, str], list[Endpoint]]:
    """Group endpoints by (model_id, serving_backend) deployment key."""
    grouped: dict[tuple[str, str], list[Endpoint]] = defaultdict(list)
    for e in endpoints:
        grouped[(e.model_id, e.serving_backend)].append(e)
    return grouped


def _is_gpu_backed(endpoint: Endpoint, model: Optional[Model]) -> bool:
    """Decide whether an endpoint is GPU-backed.

    Prefer the model manifest's `requires_gpu` flag; fall back to a
    heuristic on `instance_class` when the model is missing from the
    registry. We deliberately do NOT use `gpu_util_pct > 0`, because a
    cold premium reserve pod sits at 0% util and would otherwise be
    excluded from the GPU aggregate — exactly the masking failure mode
    `DECISIONS.md` warned against.
    """
    if model is not None:
        return bool(model.requires_gpu)
    return endpoint.instance_class.startswith("gpu_")


def compute_signals(
    endpoints: Iterable[Endpoint],
    models: Optional[dict[str, Model]] = None,
    *,
    planned_events: Optional[dict[str, dict]] = None,
    min_replicas: int = 1,
    max_replicas: int = 50,
    burst_factor: float = 1.5,
) -> list[AutoscaleSignal]:
    """Compute scaling recommendations per (model_id, backend) deployment.

    `models` is the model registry; the autoscaler looks up each
    deployment's model to read its effective utilization targets. If the
    model is missing from the registry, fallback targets are used.

    `planned_events` maps `model_id` -> {"reason": str, "expected_load_x": float}
    where expected_load_x is the multiplier on current load (e.g., 2.0 means
    expect 2x the current QPS).
    """

    models = models or {}
    planned_events = planned_events or {}
    signals: list[AutoscaleSignal] = []

    for (model_id, backend), eps in aggregate_by_deployment(endpoints).items():
        model = models.get(model_id)
        healthy = [e for e in eps if e.healthy]
        current_replicas = len(eps)
        healthy_replicas = len(healthy)

        total_capacity = sum(e.capacity_qps for e in healthy) or 1
        total_qps = sum(e.current_qps for e in healthy)
        load_pct = round(100.0 * total_qps / total_capacity, 1)

        # Detect GPU-backed pods from the manifest (or instance_class as a
        # fallback) so a cold reserve pod at 0% util is still counted.
        gpu_endpoints = [e for e in healthy if _is_gpu_backed(e, model)]
        cpu_endpoints = [e for e in healthy if not _is_gpu_backed(e, model)]

        if gpu_endpoints:
            gpu_util_avg = round(
                sum(e.gpu_util_pct for e in gpu_endpoints) / len(gpu_endpoints), 1
            )
            gpu_util_max = max(e.gpu_util_pct for e in gpu_endpoints)
        else:
            gpu_util_avg = 0.0
            gpu_util_max = 0.0

        if cpu_endpoints:
            cpu_util_max = max(e.cpu_util_pct for e in cpu_endpoints)
        else:
            cpu_util_max = 0.0

        targets = utilization_targets_for_model(model)
        gpu_target = targets["gpu"]
        cpu_target = targets["cpu"]

        recommended = current_replicas
        reason = "stable"

        # Planned event: pre-warm up to burst_factor * current capacity.
        event = planned_events.get(model_id)
        if event:
            target_replicas = max(
                current_replicas,
                int(round(current_replicas * event.get("expected_load_x", burst_factor))),
            )
            recommended = max(recommended, target_replicas)
            reason = f"planned_event_prewarm:{event.get('reason', 'unspecified')}"

        # Unhealthy endpoints reduce effective replicas; recover at least to
        # the previous count so the deployment is not under-provisioned.
        if healthy_replicas < current_replicas:
            recommended = max(recommended, current_replicas)
            reason = (
                "replace_unhealthy_replicas"
                if reason == "stable"
                else f"{reason}+replace_unhealthy"
            )

        # Reactive scale uses MAX pod utilization in both directions so a
        # cold reserve cannot mask a hot workhorse (scale-up) and a hot
        # workhorse cannot be averaged away by an idle reserve
        # (scale-down). Use GPU for GPU-backed deployments; CPU otherwise.
        # Ceiling on scale-up so even small deviations add at least one
        # replica from a low base (real HPA behaves the same way).
        if gpu_endpoints and gpu_util_max > gpu_target:
            scale = gpu_util_max / gpu_target
            target_replicas = max(recommended, math.ceil(current_replicas * scale))
            if target_replicas > recommended:
                recommended = target_replicas
                hot_reason = (
                    f"gpu_util_above_target:max_pod_{gpu_util_max}%>target_{gpu_target}%"
                )
                reason = hot_reason if reason == "stable" else f"{reason}+{hot_reason}"
        elif cpu_endpoints and cpu_util_max > cpu_target:
            scale = cpu_util_max / cpu_target
            target_replicas = max(recommended, math.ceil(current_replicas * scale))
            if target_replicas > recommended:
                recommended = target_replicas
                hot_reason = (
                    f"cpu_util_above_target:max_pod_{cpu_util_max}%>target_{cpu_target}%"
                )
                reason = hot_reason if reason == "stable" else f"{reason}+{hot_reason}"
        elif gpu_endpoints and gpu_util_max < gpu_target * 0.5:
            # Even the hottest pod is well under target -> safe to scale
            # down by one (but keep min). Using max-pod (not average)
            # prevents an idle reserve from forcing scale-down while a
            # workhorse is still busy.
            target_replicas = max(min_replicas, current_replicas - 1)
            if target_replicas < recommended and reason == "stable":
                recommended = target_replicas
                reason = (
                    f"gpu_util_well_below_target:max_pod_{gpu_util_max}%<half_of_{gpu_target}%"
                )
        elif cpu_endpoints and cpu_util_max < cpu_target * 0.5:
            target_replicas = max(min_replicas, current_replicas - 1)
            if target_replicas < recommended and reason == "stable":
                recommended = target_replicas
                reason = (
                    f"cpu_util_well_below_target:max_pod_{cpu_util_max}%<half_of_{cpu_target}%"
                )

        recommended = max(min_replicas, min(recommended, max_replicas))

        signals.append(
            AutoscaleSignal(
                model_id=model_id,
                serving_backend=backend,
                current_replicas=current_replicas,
                recommended_replicas=recommended,
                aggregate_gpu_util_pct=gpu_util_avg,
                aggregate_load_pct=load_pct,
                reason=reason,
                target_gpu_util_pct=gpu_target,
                target_cpu_util_pct=cpu_target,
            )
        )

    signals.sort(key=lambda s: (s.model_id, s.serving_backend))
    return signals
