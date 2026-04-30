"""Canary evaluation policy tests."""

from __future__ import annotations

import unittest

from modelops_platform.canary import evaluate_canary


# Guardrails for a fraud detection model: tight FPR and latency gates
# (hard), precision and recall gates requiring analyst review (soft).
FRAUD_GUARDRAILS = {
    "max_latency_regression_pct": 5.0,
    "max_error_regression_pct": 10.0,
    "max_timeout_regression_pct": 5.0,
    "max_fpr_increase_pct": 10.0,
    "max_precision_drop_pct": 2.0,
    "max_recall_drop_pct": 3.0,
    "min_observations": 5000,
}

# Looser guardrails — illustrates that the same metrics yield different
# decisions when different explicit guardrails are passed.
LOOSE_GUARDRAILS = {
    "max_latency_regression_pct": 20.0,
    "max_error_regression_pct": 20.0,
    "max_timeout_regression_pct": 15.0,
    "max_fpr_increase_pct": 25.0,
    "max_precision_drop_pct": 8.0,
    "max_recall_drop_pct": 8.0,
    "min_observations": 1000,
}


def _control() -> dict:
    return {
        "p99_latency_ms": 50.0,
        "error_rate_pct": 0.1,
        "timeout_rate_pct": 0.05,
        "false_positive_rate": 0.022,
        "precision": 0.960,
        "recall": 0.871,
        "sample_size": 6000,
    }


class CanaryEvaluationTests(unittest.TestCase):
    def test_auto_promote_when_all_guardrails_pass(self) -> None:
        candidate = dict(_control())
        candidate["p99_latency_ms"] = 49.0   # latency improved
        candidate["false_positive_rate"] = 0.021  # FPR improved
        candidate["precision"] = 0.964        # precision improved
        candidate["recall"] = 0.875           # recall improved
        decision = evaluate_canary(
            "v2",
            candidate,
            _control(),
            guardrails=FRAUD_GUARDRAILS,
            rollback_to_version="v1",
        )
        self.assertEqual(decision.decision, "auto_promote")
        self.assertEqual(decision.failed_guardrails, [])
        self.assertEqual(decision.effective_guardrails, FRAUD_GUARDRAILS)

    def test_rollback_on_latency_breach(self) -> None:
        candidate = dict(_control())
        candidate["p99_latency_ms"] = 80.0  # 60% regression, limit 5%
        decision = evaluate_canary(
            "v2", candidate, _control(),
            guardrails=FRAUD_GUARDRAILS, rollback_to_version="v1",
        )
        self.assertEqual(decision.decision, "rollback")
        self.assertEqual(decision.rollback_to_version, "v1")
        self.assertTrue(any("P99 latency regression" in r for r in decision.failed_guardrails))

    def test_rollback_on_error_rate_breach(self) -> None:
        candidate = dict(_control())
        candidate["error_rate_pct"] = 0.5  # 5x error rate
        decision = evaluate_canary(
            "v2", candidate, _control(),
            guardrails=FRAUD_GUARDRAILS, rollback_to_version="v1",
        )
        self.assertEqual(decision.decision, "rollback")
        self.assertTrue(any("Error-rate regression" in r for r in decision.failed_guardrails))

    def test_rollback_on_fpr_increase(self) -> None:
        # FPR increase = more legitimate customers blocked. Hard gate.
        candidate = dict(_control())
        candidate["false_positive_rate"] = 0.040  # 82% increase, limit 10%
        decision = evaluate_canary(
            "v2", candidate, _control(),
            guardrails=FRAUD_GUARDRAILS, rollback_to_version="v1",
        )
        self.assertEqual(decision.decision, "rollback")
        self.assertTrue(
            any("False positive rate increase" in r for r in decision.failed_guardrails)
        )

    def test_owner_review_on_precision_drop(self) -> None:
        # Precision drop: more false alarms per fraud flag. Soft gate.
        candidate = dict(_control())
        candidate["precision"] = 0.925  # drop from 0.960 = 3.6%, limit 2%
        decision = evaluate_canary(
            "v2", candidate, _control(),
            guardrails=FRAUD_GUARDRAILS, rollback_to_version="v1",
        )
        self.assertEqual(decision.decision, "requires_owner_review")
        self.assertEqual(decision.reason, "quality_guardrail_breach")
        self.assertTrue(any("Precision drop" in r for r in decision.failed_guardrails))

    def test_owner_review_on_recall_drop(self) -> None:
        # Recall drop: more fraud slipping through. Soft gate.
        candidate = dict(_control())
        candidate["recall"] = 0.840  # drop from 0.871 = 3.6%, limit 3%
        decision = evaluate_canary(
            "v2", candidate, _control(),
            guardrails=FRAUD_GUARDRAILS, rollback_to_version="v1",
        )
        self.assertEqual(decision.decision, "requires_owner_review")
        self.assertTrue(any("Recall drop" in r for r in decision.failed_guardrails))

    def test_operational_breach_takes_precedence_over_quality_breach(self) -> None:
        # Both FPR (hard) and precision (soft) fail. Should rollback, not owner review.
        candidate = dict(_control())
        candidate["false_positive_rate"] = 0.040  # FPR hard breach
        candidate["precision"] = 0.925            # precision soft breach
        decision = evaluate_canary(
            "v2", candidate, _control(), guardrails=FRAUD_GUARDRAILS,
        )
        self.assertEqual(decision.decision, "rollback")
        self.assertIn("False positive rate increase", " ".join(decision.failed_guardrails))

    def test_hold_when_observation_window_too_small(self) -> None:
        candidate = dict(_control())
        candidate["sample_size"] = 200  # below min_observations=5000
        decision = evaluate_canary(
            "v2", candidate, _control(),
            guardrails=FRAUD_GUARDRAILS, rollback_to_version="v1",
        )
        self.assertEqual(decision.decision, "hold_canary")
        self.assertIn("observations", decision.reason)

    def test_explicit_guardrails_drive_decisions(self) -> None:
        # Precision drop of 3.6% (0.960 -> 0.925).
        # Tight guardrails: max 2% drop -> owner review.
        # Loose guardrails: max 8% drop -> auto promote.
        candidate = dict(_control())
        candidate["precision"] = 0.925

        tight = evaluate_canary("v2", candidate, _control(), guardrails=FRAUD_GUARDRAILS)
        self.assertEqual(tight.decision, "requires_owner_review")

        loose = evaluate_canary("v2", candidate, _control(), guardrails=LOOSE_GUARDRAILS)
        self.assertEqual(loose.decision, "auto_promote")

    def test_guardrails_visible_in_effective_guardrails(self) -> None:
        candidate = dict(_control())
        candidate["precision"] = 0.925
        decision = evaluate_canary(
            "v2", candidate, _control(),
            guardrails={"max_precision_drop_pct": 5.0, "min_observations": 100},
        )
        # 3.6% drop < 5% limit -> passes
        self.assertEqual(decision.decision, "auto_promote")
        self.assertEqual(decision.effective_guardrails["max_precision_drop_pct"], 5.0)

    def test_must_provide_guardrails(self) -> None:
        with self.assertRaises(TypeError):
            evaluate_canary("v2", _control(), _control())  # type: ignore[call-arg]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
