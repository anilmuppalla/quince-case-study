"""Routing policy tests.

These tests pin the platform contract: SLA feasibility before cost,
high-priority headroom preservation driven by each model's explicit
utilization target, unhealthy exclusion, and the overflow-to-premium
escape hatch for high-priority requests.
"""

from __future__ import annotations

import unittest

from modelops_platform.domain import Endpoint, Model, Request
from modelops_platform.routing import (
    FALLBACK_HIGH_PRIORITY_HEADROOM,
    headroom_target_for,
    route_request,
)


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


class RoutingPolicyTests(unittest.TestCase):
    def test_routes_to_lowest_cost_feasible_for_normal_priority(self) -> None:
        cheap = _ep(endpoint_id="cheap", cost_per_1k_predictions=0.10)
        pricey = _ep(endpoint_id="pricey", cost_per_1k_predictions=0.80)
        request = Request(request_id="r1", model_id="m1", priority="normal")
        decision = route_request(request, _model(priority_tier="normal"), [cheap, pricey])
        self.assertEqual(decision.status, "routed")
        self.assertEqual(decision.chosen_endpoint, "cheap")
        self.assertIn("lowest_cost", decision.route_reason)

    def test_excludes_unhealthy_endpoints(self) -> None:
        unhealthy = _ep(endpoint_id="unhealthy", healthy=False, cost_per_1k_predictions=0.05)
        healthy = _ep(endpoint_id="healthy", cost_per_1k_predictions=0.40)
        request = Request(request_id="r1", model_id="m1", priority="normal")
        decision = route_request(request, _model(), [unhealthy, healthy])
        self.assertEqual(decision.status, "routed")
        self.assertEqual(decision.chosen_endpoint, "healthy")

    def test_premium_endpoint_takes_high_priority_when_cheap_misses_sla(self) -> None:
        cheap = _ep(
            endpoint_id="cheap",
            cost_per_1k_predictions=0.10,
            base_latency_ms=20.0,
            capacity_qps=100,
            current_qps=99,
        )
        cheap_b = _ep(
            endpoint_id="cheap_b",
            cost_per_1k_predictions=0.10,
            base_latency_ms=20.0,
            capacity_qps=100,
            current_qps=99,
        )
        premium = _ep(
            endpoint_id="premium",
            cost_per_1k_predictions=0.95,
            base_latency_ms=14.0,
            capacity_qps=100,
            current_qps=5,
        )
        request = Request(request_id="r1", model_id="m1", priority="high")
        decision = route_request(
            request, _model(sla_ms=25), [cheap, cheap_b, premium]
        )
        self.assertEqual(decision.status, "routed")
        self.assertEqual(decision.chosen_endpoint, "premium")

    def test_high_priority_overflow_when_no_endpoint_under_sla(self) -> None:
        slow = _ep(
            endpoint_id="slow",
            cost_per_1k_predictions=0.10,
            base_latency_ms=80.0,
            capacity_qps=100,
            current_qps=50,
        )
        request = Request(request_id="r1", model_id="m1", priority="high")
        decision = route_request(request, _model(sla_ms=25), [slow])
        self.assertEqual(decision.status, "overflow_to_premium")
        self.assertEqual(decision.chosen_endpoint, "slow")

    def test_normal_priority_rejected_when_no_sla_feasible(self) -> None:
        slow = _ep(
            endpoint_id="slow",
            cost_per_1k_predictions=0.10,
            base_latency_ms=80.0,
            capacity_qps=100,
            current_qps=50,
        )
        request = Request(request_id="r1", model_id="m1", priority="normal")
        decision = route_request(request, _model(sla_ms=25), [slow])
        self.assertEqual(decision.status, "rejected_sla")

    def test_high_priority_prefers_headroom_when_available(self) -> None:
        loaded_cheap = _ep(
            endpoint_id="loaded_cheap",
            cost_per_1k_predictions=0.10,
            capacity_qps=100,
            current_qps=80,
            gpu_util_pct=80.0,
        )
        spacious_pricier = _ep(
            endpoint_id="spacious_pricier",
            cost_per_1k_predictions=0.30,
            capacity_qps=100,
            current_qps=30,
            gpu_util_pct=25.0,
        )
        request = Request(request_id="r1", model_id="m1", priority="high")
        decision = route_request(request, _model(), [loaded_cheap, spacious_pricier])
        self.assertEqual(decision.status, "routed")
        self.assertEqual(decision.chosen_endpoint, "spacious_pricier")
        self.assertIn("headroom_preserved", decision.route_reason)

    def test_no_endpoints_for_model_returns_no_capacity(self) -> None:
        wrong_backend = _ep(serving_backend="vllm")
        request = Request(request_id="r1", model_id="m1", priority="normal")
        decision = route_request(request, _model(), [wrong_backend])
        self.assertEqual(decision.status, "rejected_no_capacity")
        self.assertEqual(decision.route_reason, "no_endpoint_for_model_and_backend")

    def test_all_endpoints_unhealthy_returns_rejected_unhealthy(self) -> None:
        a = _ep(endpoint_id="a", healthy=False)
        b = _ep(endpoint_id="b", healthy=False)
        request = Request(request_id="r1", model_id="m1", priority="high")
        decision = route_request(request, _model(), [a, b])
        self.assertEqual(decision.status, "rejected_unhealthy")
        self.assertEqual(decision.route_reason, "all_endpoints_unhealthy")


class HeadroomTargetTests(unittest.TestCase):
    """Routing headroom comes from the model's explicit utilization target."""

    def test_explicit_gpu_target_50_gives_50_percent_headroom(self) -> None:
        m = _model(target_gpu_util_pct=50, requires_gpu=True)
        self.assertAlmostEqual(headroom_target_for(m), 0.50)

    def test_explicit_gpu_target_85_gives_85_percent_headroom(self) -> None:
        m = _model(target_gpu_util_pct=85, requires_gpu=True)
        self.assertAlmostEqual(headroom_target_for(m), 0.85)

    def test_cpu_only_model_uses_cpu_target(self) -> None:
        m = _model(requires_gpu=False, target_cpu_util_pct=80)
        self.assertAlmostEqual(headroom_target_for(m), 0.80)

    def test_unknown_model_falls_back_to_default(self) -> None:
        self.assertAlmostEqual(
            headroom_target_for(None), FALLBACK_HIGH_PRIORITY_HEADROOM
        )

    def test_explicit_target_propagates_to_routing(self) -> None:
        m = _model(requires_gpu=True, target_gpu_util_pct=40)
        self.assertAlmostEqual(headroom_target_for(m), 0.40)

    def test_explicit_targets_drive_routing_headroom(self) -> None:
        # Same endpoints, same load. Model with 50% GPU target avoids the
        # 65%-loaded endpoint; model with 85% target accepts it.
        loaded = _ep(
            endpoint_id="loaded",
            cost_per_1k_predictions=0.10,
            capacity_qps=100,
            current_qps=65,
            gpu_util_pct=65.0,
        )
        cool_pricier = _ep(
            endpoint_id="cool_pricier",
            cost_per_1k_predictions=0.30,
            capacity_qps=100,
            current_qps=20,
            gpu_util_pct=20.0,
        )
        request = Request(request_id="r1", model_id="m1", priority="high")

        tight_decision = route_request(
            request, _model(target_gpu_util_pct=50), [loaded, cool_pricier]
        )
        self.assertEqual(tight_decision.chosen_endpoint, "cool_pricier")
        self.assertIn("headroom_preserved", tight_decision.route_reason)

        hot_decision = route_request(
            request, _model(target_gpu_util_pct=85), [loaded, cool_pricier]
        )
        self.assertEqual(hot_decision.chosen_endpoint, "loaded")
        self.assertIn("headroom_preserved", hot_decision.route_reason)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
