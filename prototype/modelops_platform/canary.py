"""Canary deployment evaluation policy.

Mirrors the deployment controller flow:

- Hold canary when the observation window is too small to act on
  (fewer than `min_observations` labeled examples have accumulated).
- Roll back when any operational guardrail breaches its hard threshold.
- Require owner review when a quality guardrail fails but operational
  metrics are clean — the decision requires domain judgment.
- Auto-promote when every guardrail passes within its pre-approved band.

Operational hard gates (lower is better; regression = candidate worse than control):
  - P99 latency regression
  - Error rate regression
  - Timeout rate regression
  - False positive rate increase: FPR going up means more legitimate
    transactions are being blocked — immediate user-facing harm, treated
    as a hard gate the same as a latency spike.

Quality soft gates (require analyst / owner review on breach):
  - Precision drop: fraction of flagged transactions that are actual fraud
    decreasing vs control. Lower precision = more false alarms per flag.
  - Recall drop: fraction of actual fraud caught decreasing vs control.
    Lower recall = more fraud slipping through.

Precondition gate (not a guardrail — a confidence threshold):
  `min_observations` must be met before any guardrail is evaluated.
  Acting on too-small samples produces noisy, unreliable decisions.
  This is separate from the guardrails dict conceptually; it sits in the
  same config block for convenience but is evaluated before any metric
  comparison takes place.

Business metrics (chargeback rate, revenue impact) are deliberately
excluded from the canary window: they are lagged by days to weeks and
cannot be attributed to the candidate model during a short canary. The
real-time business signal for fraud is approval rate, which is the
inverse of FPR — captured by the `max_fpr_increase_pct` hard gate.

All guardrails are explicit per evaluation call. No defaults are derived
from a model class; every model's rollout policy is visible in its own
config block.

Expected guardrails shape:
    {
      # Operational hard gates
      "max_latency_regression_pct": float,
      "max_error_regression_pct": float,
      "max_timeout_regression_pct": float,
      "max_fpr_increase_pct": float,       # false positive rate increase

      # Quality soft gates (owner review on breach)
      "max_precision_drop_pct": float,     # fraction of flagged txns that are fraud
      "max_recall_drop_pct": float,        # fraction of fraud caught

      # Observation window precondition (not a metric comparison)
      "min_observations": int,
    }

Expected metrics shape (both candidate and control):
    {
      "p99_latency_ms": float,
      "error_rate_pct": float,
      "timeout_rate_pct": float,
      "false_positive_rate": float,   # fraction of legitimate txns incorrectly flagged
      "precision": float,             # of flagged txns, fraction that are actual fraud
      "recall": float,                # of actual fraud, fraction caught by the model
      "sample_size": int,             # number of labeled observations in the window
    }
"""

from __future__ import annotations

from .domain import CanaryDecision


def _regression_pct(candidate: float, control: float) -> float:
    """Percentage increase of candidate vs control (positive = candidate worse).

    Used for metrics where lower is better: latency, error rate, FPR.
    """
    if control == 0:
        return 0.0 if candidate == 0 else float("inf")
    return round(100.0 * (candidate - control) / control, 2)


def _drop_pct(candidate: float, control: float) -> float:
    """Percentage drop of candidate vs control (positive = candidate worse).

    Used for metrics where higher is better: precision, recall.
    Positive return value means candidate is lower (worse) than control.
    """
    if control == 0:
        return 0.0 if candidate >= control else float("inf")
    return round(100.0 * (control - candidate) / control, 2)


def evaluate_canary(
    candidate_version: str,
    candidate_metrics: dict,
    control_metrics: dict,
    *,
    guardrails: dict,
    rollback_to_version: str | None = None,
) -> CanaryDecision:
    """Evaluate a candidate version against control using explicit guardrails.

    `guardrails` must be provided by the caller — the scenario canary block
    or the deployment controller rollout config. No defaults are inferred
    from a model class; the policy is exactly what the caller passes.
    """
    failed: list[str] = []
    operational_breach = False

    # --- Precondition gate -------------------------------------------
    # Hold if the observation window is too small to act on. This is not
    # a guardrail comparison; it short-circuits before any metric is
    # evaluated so noisy early signals do not trigger rollback or review.
    observations = candidate_metrics.get("sample_size", 0)
    min_observations = guardrails.get("min_observations", 0)
    if observations < min_observations:
        return CanaryDecision(
            candidate_version=candidate_version,
            decision="hold_canary",
            reason=f"observations={observations}_below_min={min_observations}",
            rollback_to_version=rollback_to_version,
            failed_guardrails=[],
            effective_guardrails=guardrails,
        )

    # --- Operational hard gates (lower is better) --------------------
    latency_delta = _regression_pct(
        candidate_metrics.get("p99_latency_ms", 0.0),
        control_metrics.get("p99_latency_ms", 0.0),
    )
    error_delta = _regression_pct(
        candidate_metrics.get("error_rate_pct", 0.0),
        control_metrics.get("error_rate_pct", 0.0),
    )
    timeout_delta = _regression_pct(
        candidate_metrics.get("timeout_rate_pct", 0.0),
        control_metrics.get("timeout_rate_pct", 0.0),
    )
    fpr_delta = _regression_pct(
        candidate_metrics.get("false_positive_rate", 0.0),
        control_metrics.get("false_positive_rate", 0.0),
    )

    if latency_delta > guardrails.get("max_latency_regression_pct", float("inf")):
        failed.append(f"P99 latency regression={latency_delta:.1f}%")
        operational_breach = True
    if error_delta > guardrails.get("max_error_regression_pct", float("inf")):
        failed.append(f"Error-rate regression={error_delta:.1f}%")
        operational_breach = True
    if timeout_delta > guardrails.get("max_timeout_regression_pct", float("inf")):
        failed.append(f"Timeout-rate regression={timeout_delta:.1f}%")
        operational_breach = True
    if fpr_delta > guardrails.get("max_fpr_increase_pct", float("inf")):
        failed.append(f"False positive rate increase={fpr_delta:.1f}%")
        operational_breach = True

    # --- Quality soft gates (higher is better; drop = candidate worse) ---
    precision_drop = _drop_pct(
        candidate_metrics.get("precision", 1.0),
        control_metrics.get("precision", 1.0),
    )
    recall_drop = _drop_pct(
        candidate_metrics.get("recall", 1.0),
        control_metrics.get("recall", 1.0),
    )

    precision_breach = precision_drop > guardrails.get("max_precision_drop_pct", float("inf"))
    recall_breach = recall_drop > guardrails.get("max_recall_drop_pct", float("inf"))

    if precision_breach:
        failed.append(f"Precision drop={precision_drop:.2f}% (candidate {candidate_metrics.get('precision', '?'):.3f} vs control {control_metrics.get('precision', '?'):.3f})")
    if recall_breach:
        failed.append(f"Recall drop={recall_drop:.2f}% (candidate {candidate_metrics.get('recall', '?'):.3f} vs control {control_metrics.get('recall', '?'):.3f})")

    if operational_breach:
        return CanaryDecision(
            candidate_version=candidate_version,
            decision="rollback",
            reason="operational_guardrail_breach",
            rollback_to_version=rollback_to_version,
            failed_guardrails=failed,
            effective_guardrails=guardrails,
        )

    if precision_breach or recall_breach:
        return CanaryDecision(
            candidate_version=candidate_version,
            decision="requires_owner_review",
            reason="quality_guardrail_breach",
            rollback_to_version=rollback_to_version,
            failed_guardrails=failed,
            effective_guardrails=guardrails,
        )

    return CanaryDecision(
        candidate_version=candidate_version,
        decision="auto_promote",
        reason="all_guardrails_within_bands",
        rollback_to_version=rollback_to_version,
        failed_guardrails=[],
        effective_guardrails=guardrails,
    )
