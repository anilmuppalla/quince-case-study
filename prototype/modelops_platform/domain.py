"""Core data primitives for the ModelOps Platform prototype.

These mirror the platform contracts described in the case study:
- Model: a deployable model with type, serving backend, SLA, and priority.
- Endpoint: one serving pod for a model+version+backend. The dataclass
  keeps its routing-domain name; in real Kubernetes a Service would
  front many Pods, but the prototype collapses them 1:1 for simplicity
  (see `Endpoint` docstring for the full vocabulary note).
- Request: an inference request from a client or backend service.
- RouteDecision: the platform's structured response for a single request.
- AutoscaleSignal: an autoscaler recommendation for a deployment.
- CanaryDecision: a deployment controller decision for a canary rollout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


PRIORITY_TIERS = {"high", "normal", "low"}


@dataclass(frozen=True)
class Model:
    """A model registered on the platform.

    Mirrors the model deployment manifest primitive: model_type and
    serving_backend together let the deployment controller pick the right
    runtime image (Triton, TF Serving, vLLM, ONNX Runtime, custom).

    Three SLA-shaped policy fields, all expressed at the **P99** percentile
    so router, autoscaler, and canary share one vocabulary:

    - sla_ms: per-request P99 latency budget. The router rejects or
      overflows when a candidate's estimated P99 latency exceeds this.
    - target_gpu_util_pct: per-deployment GPU saturation ceiling. Used by
      the autoscaler and the router's high-priority headroom check. Set
      this to match the headroom you want — e.g., 50 keeps a cool fleet
      with lots of burst capacity, 85 runs hot to maximise utilisation.
    - target_cpu_util_pct: per-deployment CPU saturation ceiling. Used by
      the autoscaler when the deployment is CPU-backed. Same semantics as
      target_gpu_util_pct.

    Both util targets default to 60/70 (a reasonable starting point for
    most online workloads). Set them explicitly on the manifest; the
    defaults are a convenience for scenarios that do not need to
    differentiate, not a policy prescription.
    """

    model_id: str
    owner_team: str
    model_type: str  # pytorch, tensorflow, onnx, llm, xgboost_ltr, sklearn, custom
    serving_backend: str  # triton, tf_serving, onnx_runtime, vllm, custom
    version: str
    sla_ms: int
    priority_tier: str  # one of PRIORITY_TIERS
    batching_window_ms: int = 0
    requires_gpu: bool = False
    target_gpu_util_pct: int = 60
    target_cpu_util_pct: int = 70

    def __post_init__(self) -> None:
        if self.priority_tier not in PRIORITY_TIERS:
            raise ValueError(
                f"priority_tier={self.priority_tier!r} not in {sorted(PRIORITY_TIERS)}"
            )


@dataclass
class Endpoint:
    """One serving pod for a model+version+backend.

    Vocabulary note: in real Kubernetes a Service fronts many Pods, and
    HPA scales the pods. For prototype simplicity we collapse Service
    and Pod into a single addressable unit: each `Endpoint` instance
    is one pod (one replica). The autoscaler's "max-pod" reason codes
    therefore iterate over `Endpoint` objects with the same model_id +
    serving_backend deployment key. A real implementation would split
    these into `Service` (routing target) and `Pod` (scaling unit).

    capacity_qps is steady-state safe throughput for this pod.
    base_latency_ms is the measured **P99** at low load (matches the
    SLA vocabulary used by `Model.sla_ms` and the canary p99 guardrail).
    The router's `estimate_latency_ms` adds a load-induced penalty plus
    half the batching window to model how P99 grows under contention.
    cost_per_1k_predictions is the fixture field rendered as
    gpu_cost_per_1k in CLI/reporting output. It is an allocation estimate
    derived from the underlying GPU/instance-hour price and assumed safe
    throughput, not a cloud-provider billing unit.
    """

    endpoint_id: str
    model_id: str
    version: str
    serving_backend: str
    instance_class: str  # e.g., gpu_a10, gpu_t4, cpu_med
    base_latency_ms: float  # measured P99 at low load
    capacity_qps: int
    current_qps: int
    gpu_util_pct: float  # 0-100; meaningless for CPU-only pods
    healthy: bool
    cost_per_1k_predictions: float
    spot: bool = False
    cpu_util_pct: float = 0.0  # 0-100; primary signal for CPU-only pods


@dataclass(frozen=True)
class Request:
    """An inference request from a backend caller."""

    request_id: str
    model_id: str
    priority: str  # one of PRIORITY_TIERS
    sla_ms_override: Optional[int] = None  # else use model's SLA

    def __post_init__(self) -> None:
        if self.priority not in PRIORITY_TIERS:
            raise ValueError(
                f"priority={self.priority!r} not in {sorted(PRIORITY_TIERS)}"
            )


@dataclass
class RouteDecision:
    """Structured output for one routing decision.

    Every field is auditable, so an agent or human can inspect why a
    request landed where it did. status + route_reason together form
    the explainable reason code referenced in the architecture.
    """

    request_id: str
    model_id: str
    status: str  # routed, overflow_to_premium, rejected_unhealthy, rejected_sla, rejected_no_capacity
    chosen_endpoint: Optional[str]
    route_reason: str
    estimated_latency_ms: Optional[float] = None
    estimated_cost_usd: Optional[float] = None
    sla_ms: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AutoscaleSignal:
    """A recommendation from the autoscaler for one model deployment.

    The autoscaler does not directly mutate Kubernetes in this prototype;
    it emits a recommendation and an explainable reason. Real HPA would
    consume the same signal.

    `target_gpu_util_pct` and `target_cpu_util_pct` are the effective
    utilization targets used to make this scaling decision, read directly
    from the model manifest.
    """

    model_id: str
    serving_backend: str
    current_replicas: int
    recommended_replicas: int
    aggregate_gpu_util_pct: float
    aggregate_load_pct: float
    reason: str  # gpu_util_above_target, planned_event_prewarm, latency_breach, etc.
    target_gpu_util_pct: int = 0
    target_cpu_util_pct: int = 0


@dataclass
class CanaryDecision:
    """Outcome of evaluating a canary deployment against guardrails."""

    candidate_version: str
    decision: str  # auto_promote, hold_canary, requires_owner_review, rollback
    reason: str
    rollback_to_version: Optional[str] = None
    failed_guardrails: list[str] = field(default_factory=list)
    effective_guardrails: dict[str, Any] = field(default_factory=dict)
