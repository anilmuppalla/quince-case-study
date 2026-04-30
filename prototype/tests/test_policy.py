"""Utilization policy tests."""

from __future__ import annotations

import unittest

from modelops_platform.domain import Endpoint, Model
from modelops_platform.policy import utilization_targets_for_model
from modelops_platform.autoscaling import compute_signals


def _model(**overrides) -> Model:
    base = dict(
        model_id="m1",
        owner_team="team-a",
        model_type="pytorch",
        serving_backend="triton",
        version="1",
        sla_ms=50,
        priority_tier="high",
        batching_window_ms=0,
        requires_gpu=True,
        target_gpu_util_pct=50,
        target_cpu_util_pct=60,
    )
    base.update(overrides)
    return Model(**base)


def _ep(**overrides) -> Endpoint:
    base = dict(
        endpoint_id="e1",
        model_id="m1",
        version="1",
        serving_backend="triton",
        instance_class="gpu_a10",
        base_latency_ms=20.0,
        capacity_qps=100,
        current_qps=10,
        gpu_util_pct=20.0,
        healthy=True,
        cost_per_1k_predictions=0.40,
    )
    base.update(overrides)
    return Endpoint(**base)


class UtilizationPolicyTests(unittest.TestCase):
    def test_model_uses_explicit_gpu_target(self) -> None:
        model = _model(target_gpu_util_pct=50)
        targets = utilization_targets_for_model(model)
        self.assertEqual(targets["gpu"], 50)

    def test_model_uses_explicit_cpu_target(self) -> None:
        model = _model(requires_gpu=False, target_cpu_util_pct=80)
        targets = utilization_targets_for_model(model)
        self.assertEqual(targets["cpu"], 80)

    def test_model_defaults_when_not_set(self) -> None:
        # Model dataclass defaults: 60% GPU / 70% CPU
        model = Model(
            model_id="m1",
            owner_team="t",
            model_type="pytorch",
            serving_backend="triton",
            version="1",
            sla_ms=100,
            priority_tier="normal",
        )
        targets = utilization_targets_for_model(model)
        self.assertEqual(targets["gpu"], 60)
        self.assertEqual(targets["cpu"], 70)

    def test_none_model_returns_safe_fallback(self) -> None:
        targets = utilization_targets_for_model(None)
        self.assertIn("gpu", targets)
        self.assertIn("cpu", targets)


class AutoscaleUsesExplicitTargetsTests(unittest.TestCase):
    def test_lower_target_scales_up_at_lower_utilization(self) -> None:
        # Model A: 50% GPU target. Model B: 85% GPU target.
        # Both see 55% GPU — A should scale up, B should stay stable.
        model_a = _model(model_id="tight", target_gpu_util_pct=50)
        model_b = _model(model_id="hot", target_gpu_util_pct=85)
        ep_a = _ep(model_id="tight", endpoint_id="ep_tight", gpu_util_pct=55.0)
        ep_b = _ep(model_id="hot", endpoint_id="ep_hot", gpu_util_pct=55.0)
        models = {"tight": model_a, "hot": model_b}
        signals = compute_signals([ep_a, ep_b], models)
        by_model = {s.model_id: s for s in signals}
        # 55% > 50% -> scale up
        self.assertGreater(
            by_model["tight"].recommended_replicas,
            by_model["tight"].current_replicas,
        )
        # 55% < 85% -> stable
        self.assertEqual(
            by_model["hot"].recommended_replicas,
            by_model["hot"].current_replicas,
        )

    def test_cpu_target_drives_scaling_for_cpu_only_deployment(self) -> None:
        cpu_model = _model(
            model_id="cpu_m",
            requires_gpu=False,
            target_gpu_util_pct=60,
            target_cpu_util_pct=70,
        )
        cpu_ep = _ep(
            model_id="cpu_m",
            endpoint_id="cpu_ep",
            instance_class="cpu_med",
            gpu_util_pct=0.0,
            cpu_util_pct=85.0,
        )
        signals = compute_signals([cpu_ep], {"cpu_m": cpu_model})
        self.assertEqual(len(signals), 1)
        self.assertGreater(
            signals[0].recommended_replicas, signals[0].current_replicas
        )
        self.assertIn("cpu_util_above_target", signals[0].reason)


class AutoscaleAggregateAndScaleDownTests(unittest.TestCase):
    """Cold reserve pods at 0% util must still count as GPU-backed.
    Scale-down must use max-pod so an idle reserve cannot mask a hot workhorse."""

    def test_cold_reserve_pod_included_in_gpu_aggregate(self) -> None:
        model = _model(model_id="m1", requires_gpu=True, target_gpu_util_pct=60)
        workhorse = _ep(model_id="m1", endpoint_id="workhorse", gpu_util_pct=80.0)
        reserve = _ep(model_id="m1", endpoint_id="reserve", gpu_util_pct=0.0)
        signals = compute_signals([workhorse, reserve], {"m1": model})
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].aggregate_gpu_util_pct, 40.0)

    def test_scale_up_triggers_on_max_not_average(self) -> None:
        # max-pod 80% > target 60% -> scale up, even though average is 40%.
        model = _model(model_id="m1", requires_gpu=True, target_gpu_util_pct=60)
        workhorse = _ep(model_id="m1", endpoint_id="workhorse", gpu_util_pct=80.0)
        reserve = _ep(model_id="m1", endpoint_id="reserve", gpu_util_pct=0.0)
        signals = compute_signals([workhorse, reserve], {"m1": model})
        self.assertGreater(
            signals[0].recommended_replicas, signals[0].current_replicas
        )
        self.assertIn("gpu_util_above_target", signals[0].reason)

    def test_scale_down_blocked_when_max_pod_is_busy(self) -> None:
        model = _model(model_id="m1", requires_gpu=True, target_gpu_util_pct=60)
        workhorse = _ep(
            model_id="m1",
            endpoint_id="workhorse",
            gpu_util_pct=80.0,
            current_qps=80,
            capacity_qps=100,
        )
        reserve = _ep(
            model_id="m1",
            endpoint_id="reserve",
            gpu_util_pct=0.0,
            current_qps=0,
            capacity_qps=100,
        )
        signals = compute_signals([workhorse, reserve], {"m1": model})
        self.assertGreaterEqual(
            signals[0].recommended_replicas, signals[0].current_replicas
        )

    def test_scale_down_when_every_pod_is_cool(self) -> None:
        model = _model(model_id="m1", requires_gpu=True, target_gpu_util_pct=60)
        eps = [
            _ep(
                model_id="m1",
                endpoint_id=f"ep{i}",
                gpu_util_pct=10.0,
                current_qps=5,
                capacity_qps=100,
            )
            for i in range(3)
        ]
        signals = compute_signals(eps, {"m1": model})
        self.assertLess(
            signals[0].recommended_replicas, signals[0].current_replicas
        )
        self.assertIn("gpu_util_well_below_target", signals[0].reason)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
