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

Each scenario is one self-contained simulation. The CLI prints capacity
and P99 latency targets, a serving decision trace, a cost-per-model
report, autoscaler signals, and (when present) a canary deployment
decision.

| Scenario                  | What it demonstrates                                                                                                                                                              |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_normal_load`          | Normal Search day: the platform right-sizes replicas with the diurnal curve while staying SLA-safe. Fixed fleet holds peak capacity all day; the platform scales down at night.   |
| `02_organic_spike`        | Unplanned Search QPS surge: autoscaler reacts when max-pod GPU exceeds the 50% target, scales out, then scales back down after the spike.                                         |
| `03_planned_promotion`    | Known flash-sale event: autoscaler pre-warms Recommendations capacity from a `planned_events` signal before traffic arrives, then scales down after demand tapers.                |
| `04_instance_unavailable` | One Search GPU pod is unhealthy: autoscaler recommends `replace_unhealthy_replicas`; router overflows high-priority traffic to healthy capacity so the SLA holds during the gap.  |
| `05_canary_auto_promote`  | Fraud Detection canary: all operational and quality guardrails pass, sample size is sufficient — auto-promote without owner intervention.                                          |
| `06_canary_owner_review`  | Fraud Detection canary: operational metrics are clean but precision and recall drop below thresholds — controller escalates to owner review rather than auto-promoting.            |

## Cost Assumptions

Scenario costs use AWS `us-east-1` Linux on-demand prices as illustrative
anchors. The HTML dashboard presents cost as GPU-hours; the terminal route
trace shows an allocated `gpu_cost_per_1k` estimate for debugging routing
choices. These are not Quince production costs.

| instance class | pricing anchor                                      | assumed capacity |
| -------------- | --------------------------------------------------- | ---------------: |
| `gpu_a10`      | AWS `g5.xlarge`, 1x NVIDIA A10G, ~$1.006/GPU-hr     | 200 QPS          |
| `gpu_h100`     | AWS `p5.48xlarge`, 8x NVIDIA H100, ~$6.88/GPU-hr    | 100 QPS          |

## Limitations (On Purpose)

- No real Kubernetes interaction.
- No real metrics pipeline; scenarios encode point-in-time state.
- No streaming traffic; routing is one decision per request.
- Backend-specific details (Triton config, vLLM serving, etc.) are
  represented through `serving_backend` metadata only.
- Training/batch policy is described in the memo but not implemented in
  this CLI.

These limitations are deliberate. The point of the prototype is to
make the routing and deployment decisions visible in code, not to
reproduce the production platform.
