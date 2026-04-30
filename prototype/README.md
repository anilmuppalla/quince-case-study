# ModelOps Platform Prototype

A small, deterministic Python CLI that demonstrates the routing and
deployment policies of the proposed Quince ModelOps Platform. It is
intentionally lightweight: no Kubernetes, no real models, no cloud APIs.
It exists to make the platform's decision logic inspectable end-to-end.

This prototype accompanies the written memo. The behaviors it demonstrates
are as follows:

- Cost-aware, SLA-respecting routing with per-model high-priority
  headroom driven by explicit `target_gpu_util_pct` and
  `target_cpu_util_pct` manifest fields.
- HPA-style autoscaler signals on the same per-model utilization
  targets, planned-event pre-warm, and unhealthy-replica replacement.
- Canary auto-promotion, owner review, hold-on-small-window, and
  rollback for operational breaches. Operational hard gates (latency,
  error rate, timeout, FPR) trigger rollback; quality soft gates
  (precision, recall) trigger owner review.
- Endpoint unavailability handling with high-priority overflow to
  healthy capacity.

## Requirements

- Python 3.10 or newer.
- Standard library only. No `pip install` step.

## Vocabulary

The narrative below uses **pod** as the single term for one serving instance. The `Endpoint` dataclass and `endpoints[]` JSON arrays each represent one pod; the dataclass keeps its routing-domain name for code-history reasons. **Replica** is used only as a count word ("scaled from 2 to 3 replicas"). In real Kubernetes a Service would front many Pods; this prototype collapses Service and Pod 1:1 for simplicity.

## Layout

```
prototype/
├── README.md                     # this file
├── modelops_platform/
│   ├── __init__.py
│   ├── __main__.py               # `python -m modelops_platform`
│   ├── cli.py                    # CLI entry: list, run, run-all
│   ├── domain.py                 # Model, Endpoint, Request, decisions
│   ├── policy.py                 # reads per-model GPU/CPU util targets
│   ├── routing.py                # cost-aware SLA-respecting policy
│   ├── autoscaling.py            # HPA-style signal generator
│   ├── canary.py                 # canary auto-promotion / review
│   ├── html_report.py            # standalone browser report renderer
│   ├── reporting.py              # cost-per-model report + tables
│   └── scenarios.py              # JSON scenario loader
├── scenarios/
│   ├── 01_normal_load.json
│   ├── 02_organic_spike.json
│   ├── 03_planned_promotion.json
│   ├── 04_instance_unavailable.json
│   ├── 05_canary_auto_promote.json
│   └── 06_canary_owner_review.json
└── tests/
    ├── test_canary.py
    ├── test_html_report.py
    ├── test_policy.py
    ├── test_reporting.py
    └── test_routing.py
```

## Running

From the `prototype/` directory:

```bash
# List bundled scenarios with their descriptions.
python -m modelops_platform list

# Run one scenario with full human-readable output.
python -m modelops_platform run scenarios/01_normal_load.json

# Run every bundled scenario.
python -m modelops_platform run-all

# Generate the browser-friendly report index and per-scenario pages.
python -m modelops_platform report-html --output simulation-report.html

# Emit machine-readable JSON instead of tables.
python -m modelops_platform run scenarios/02_organic_spike.json --json-only
```

## HTML Simulation Report

The recommended reviewer surface is the generated HTML report index plus
per-scenario pages:

```bash
python -m modelops_platform report-html --output simulation-report.html
open simulation-report.html
```

This writes `simulation-report.html` as the portfolio index and
`simulation-report-scenarios/*.html` as one page per scenario.


## Tests

```bash
python -m unittest discover -s tests
```

## What The Scenarios Show

Each scenario is one self-contained simulation. Some scenarios are phased
to show state over time: baseline, threshold breach, and post-scale-out.
The CLI prints capacity and P99 latency targets, a serving decision trace, a
cost-per-model report, autoscaler signals, and (when present) a canary
deployment decision.


| Scenario                       | What it demonstrates                                                                                                                                                                                                                                                |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_normal_load`               | Smooth normal Search day with no surprise spike: late-night baseline, early ramp, morning ramp, midday scale-up, business-hours plateau, afternoon taper, evening browse, and night scale-down. The fixed before fleet stays at two replicas; the platform right-sizes from one to two active replicas and back down while staying SLA-safe. |
| `02_organic_spike`             | Sudden Search load surge shown in phases and in a 24-hour traffic window: baseline, ramp, spike detection, scale-out serving peak, sustained elevated load, and cooldown. The autoscaler reacts when observed Search max-pod GPU exceeds the 50% target; new pods join and inference traffic continues serving. |
| `03_planned_promotion`         | Known marketing event. The 24-hour window shows Recommendations pre-event baseline, pre-warm, flash-sale peak, taper, and scale-down. The autoscaler pre-warms Recommendations capacity from a `planned_events` signal before traffic arrives. |
| `04_instance_unavailable`      | One Search GPU pod is unhealthy. The 24-hour window shows fixed-fleet risk, unhealthy-pod exclusion via routing, the rolling-replacement signal, and incident recovery. High-priority Search traffic can move to premium capacity when the healthy A10 pod has no headroom. |
| `05_canary_auto_promote`       | Fraud Detection canary metrics are within all guardrails and sample size is sufficient. The deployment controller auto-promotes without human review. |
| `06_canary_owner_review`       | Fraud canary operational metrics are clean, but precision and recall drop below configured thresholds. The controller refuses to auto-promote and escalates to the model owner for review. |


## Cost Assumptions

Scenario costs use researched AWS `us-east-1` Linux on-demand instance
prices as illustrative raw-compute anchors. Cloud providers bill the
underlying GPU/instance-hours, not individual predictions. The HTML dashboard
presents the cost basis as GPU-hours because the before/after story is
replica count over a 24-hour traffic window.

The terminal route trace still shows an allocated `gpu_cost_per_1k` estimate
for debugging routing choices. That estimate is still GPU-hour based:

```text
gpu_cost_per_1k = GPU-hour price / (safe allocation RPS * 3600 / 1000)
```

For example, `$1.006/GPU-hr` at 200 RPS safe capacity yields
`$1.006 / (200 * 3600 / 1000) = $0.001397`, rounded to `$0.0014` per
1K predictions for allocation only.

The internal route-trace values are intentionally rounded:

| prototype class | public pricing anchor | assumed capacity | allocated cost / 1K predictions |
|---|---|---:|---:|
| `gpu_a10` | AWS `g5.xlarge`, 1x NVIDIA A10G, about $1.006/hour | 200 QPS | $0.0014 |
| `gpu_h100` | AWS `p5.48xlarge`, 8x NVIDIA H100, about $55.04/hour or $6.88/GPU-hour | 100 QPS | $0.0191 |


## Routing Policy (Summary)

The routing decision in `modelops_platform/routing.py` is deliberately
explicit so it is easy to audit. For each request:

1. Find candidate pods by `model_id` and `serving_backend`.
2. If every candidate is unhealthy, return `rejected_unhealthy` with
   reason `all_endpoints_unhealthy`. Otherwise drop unhealthy pods.
3. Exclude pods with no remaining capacity.
4. Estimate **P99 latency** = base + load penalty + half batching window.
   `Endpoint.base_latency_ms` is the measured P99 at low load, so the
   estimate, the SLA budget, and the canary guardrail all speak the
   same percentile.
5. Filter to SLA-feasible pods (`estimated_latency_ms <= sla_ms`).
6. For high-priority requests, restrict to pods below the model's
   utilization target on the right axis (GPU for GPU-backed, CPU for
   CPU-backed). The target is set explicitly per model on the manifest
   (`target_gpu_util_pct` / `target_cpu_util_pct`); the same value drives
   the autoscaler so router and autoscaler agree on what "too hot" means.
7. Choose the lowest cost-per-1k-predictions among the remaining set,
   breaking ties on lowest estimated latency.
8. If no SLA-feasible candidate exists:
   - high priority overflows to the fastest healthy pod with capacity
     (`overflow_to_premium`). Costs more, but preserves the SLA tier.
   - normal/low priority returns `rejected_sla` with an explainable
     reason.

Every decision carries a `route_reason` so cost spikes, rejections,
and overflows can be debugged from the trace alone.

## Autoscaling Signal (Summary)

`modelops_platform/autoscaling.py` does not mutate Kubernetes. It emits
one recommendation per `(model_id, serving_backend)` deployment:

- A pod is treated as **GPU-backed** when the registered model has
  `requires_gpu=True`, falling back to an `instance_class` prefix
  check when the model is missing from the registry. This means a
  cold reserve pod at 0% GPU util is still counted as GPU-backed and
  contributes to the aggregate (instead of being silently dropped).
- The **maximum pod utilization** is the trigger in *both* directions:
  scale-up fires when max pod > target; scale-down fires only when
  max pod < half the target. A cold reserve cannot mask a hot
  workhorse, and a hot workhorse cannot be averaged away by an idle
  reserve.
- The aggregate is also reported for capacity-planning context.
- The **per-model target** is what the max pod is compared against,
  not a global constant. Targets come from the model's manifest
  (`target_gpu_util_pct`, `target_cpu_util_pct`) with class-driven
  defaults (see below). `strict_realtime` runs cool with a 50% GPU
  target; `batch` deliberately runs hot at 85%.
- CPU-backed deployments use the CPU branch on the same contract
  (`max_pod_cpu_util_pct` vs `target_cpu_util_pct`). The Fraud canary
  scenarios use a CPU-backed ONNX Runtime deployment; CPU scaling behavior is
  also covered by unit tests.
- The autoscaler uses ceiling math for scale-up so even a small
  deviation from target adds at least one replica from a low base.
  This mirrors how Kubernetes HPA behaves.
- A `planned_events[model_id]` entry pre-warms capacity by an
  `expected_load_x` multiplier even if current load is normal.
- Unhealthy replicas trigger a `replace_unhealthy_replicas`
  recommendation. It composes with utilization-based reasons, e.g.,
  `replace_unhealthy_replicas+gpu_util_above_target:max_pod_78%>target_50%`.

### Per-Model Utilization Targets

`target_gpu_util_pct` and `target_cpu_util_pct` are explicit fields on
each model manifest. They default to 60% GPU / 70% CPU on the `Model`
dataclass — a reasonable starting point for most online workloads —
and teams set their own values based on the headroom they need vs. the
cost they're willing to pay. A latency-sensitive Search workload runs
at 50% to keep room for tail-latency bursts; a throughput-bound batch
job runs at 85% to maximise utilization.

The CLI prints the effective `gpu_target%` and `cpu_target%` next to
the measured utilization so the demo shows the policy at work. In the
organic-spike scenario, the pre-scale phase shows Search Ranking (50%
GPU target) crossing its target. The next phase models the new pods
after scale-out, bringing load and latency back toward target without
rejecting normal inference traffic.

In production, total GPU exhaustion is handled as an operations and
capacity-planning event before it becomes a router failure: online
inference is prioritized ahead of training/batch workloads, non-urgent
GPU-heavy training can be deferred, and provider capacity or reservations
are requested based on month-over-month utilization trends, launch
forecasts, scale-out events, and overflow/rejection signals.

## Canary Policy (Summary)

`modelops_platform/canary.py` evaluates a candidate version against a
control. Guardrails are passed explicitly per evaluation — there is no
named class lookup; every model's rollout policy is visible in its own
config block.

**Observation gate** (precondition, not a metric comparison): if labeled
observations are below `min_observations`, return `hold_canary` and
wait. Acting on too-small samples produces noisy decisions regardless of
what the metrics say.

**Operational hard gates** (any breach -> `rollback`):

- P99 latency regression vs control
- Error rate regression vs control
- Timeout rate regression vs control
- False positive rate (FPR) increase vs control. FPR going up means
  legitimate transactions get blocked — immediate user-facing harm,
  treated the same as a latency spike.

**Quality soft gates** (any breach -> `requires_owner_review`):

- Precision drop vs control. Lower precision = more false alarms per
  flag; requires fraud analyst judgment because the trade-off may be
  intentional.
- Recall drop vs control. Lower recall = more fraud slipping through;
  same reasoning.

**Business metrics excluded from the canary window** by design.
Chargeback rate and revenue impact are lagged by days to weeks and
cannot be attributed to the candidate during a short canary. The
real-time business signal for fraud is approval rate, which is the
inverse of FPR — already captured by the FPR hard gate. Business impact
review happens at the 30/60/90-day post-release reviews, not in the
deployment controller.

The CLI prints `effective_guardrails` and `failed_guardrails` on every
canary decision so the policy is auditable end-to-end.

## Limitations (On Purpose)

- No real Kubernetes interaction.
- No real metrics pipeline; scenarios encode point-in-time state.
- No streaming traffic; routing is one decision per request.
- Backend-specific details (Triton config, vLLM serving, etc.) are
represented through `serving_backend` metadata only.
- Training/batch policy is described in the memo as a cost/scheduling
  overlay on the same substrate, but it is not implemented in this CLI.

These limitations are deliberate. The point of the prototype is to
make the routing and deployment decisions visible in code, not to
reproduce the production platform.
