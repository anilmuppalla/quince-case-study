"""Utilization policy helpers.

Each model carries explicit `target_gpu_util_pct` and `target_cpu_util_pct`
fields on its manifest. This module reads those fields directly — no
lookup table, no named class. The same targets propagate to the
autoscaler and the router's high-priority headroom check so both
subsystems agree on what "too hot" means for a given model.

Default values (60% GPU / 70% CPU) live on the Model dataclass and are
a reasonable starting point for most online workloads. Teams set their
own targets based on their latency headroom needs and cost tolerance.
"""

from __future__ import annotations

from typing import Optional

from .domain import Model


def utilization_targets_for_model(model: Optional[Model]) -> dict[str, int]:
    """Return effective GPU/CPU utilization targets for a model.

    Reads `target_gpu_util_pct` and `target_cpu_util_pct` directly from
    the model manifest. If `model` is None (e.g., model missing from the
    registry), returns a safe fallback so the autoscaler still has
    something to compare against.
    """
    if model is None:
        return {"gpu": 60, "cpu": 70}
    return {
        "gpu": model.target_gpu_util_pct,
        "cpu": model.target_cpu_util_pct,
    }
