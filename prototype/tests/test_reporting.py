"""CLI reporting and scenario-shape tests."""

from __future__ import annotations

import unittest
from pathlib import Path

from modelops_platform import reporting
from modelops_platform.autoscaling import compute_signals
from modelops_platform.cli import run_scenario
from modelops_platform.domain import Endpoint, Model, Request, RouteDecision
from modelops_platform.routing import apply_routing, route_request
from modelops_platform.scenarios import load_scenario


SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"


def _model(**overrides) -> Model:
    base = dict(
        model_id="search-ranking-v1",
        owner_team="search",
        model_type="pytorch",
        serving_backend="triton",
        version="1.0.0",
        sla_ms=50,
        priority_tier="high",
        batching_window_ms=0,
        requires_gpu=True,
        target_gpu_util_pct=50,
        target_cpu_util_pct=60,
    )
    base.update(overrides)
    return Model(**base)


def _endpoint(**overrides) -> Endpoint:
    base = dict(
        endpoint_id="search-ep-a",
        model_id="search-ranking-v1",
        version="1.0.0",
        serving_backend="triton",
        instance_class="gpu_a10",
        base_latency_ms=22.0,
        capacity_qps=200,
        current_qps=100,
        gpu_util_pct=50.0,
        healthy=True,
        cost_per_1k_predictions=0.0014,
    )
    base.update(overrides)
    return Endpoint(**base)


class ReportingOutputTests(unittest.TestCase):
    def test_scenario_list_renders_wrapped_cards_with_run_commands(self) -> None:
        renderer = getattr(reporting, "render_scenario_list", None)
        self.assertIsNotNone(renderer)

        scenario = load_scenario(SCENARIOS_DIR / "01_normal_load.json")
        output = renderer([("01_normal_load.json", scenario)], width=72)

        self.assertIn("Available Scenarios", output)
        self.assertIn("01_normal_load.json", output)
        self.assertIn("Name: Normal Load", output)
        self.assertIn("Summary:", output)
        self.assertIn(
            "python -m modelops_platform run scenarios/01_normal_load.json",
            output,
        )
        summary_lines = [
            line for line in output.splitlines() if line.startswith("    ")
        ]
        self.assertGreater(len(summary_lines), 1)
        self.assertTrue(all(len(line) <= 72 for line in summary_lines))

    def test_capacity_summary_shows_targets_and_current_capacity(self) -> None:
        renderer = getattr(reporting, "render_capacity_summary", None)
        self.assertIsNotNone(renderer)

        models = {
            "search-ranking-v1": _model(),
            "personalization-v1": _model(
                model_id="personalization-v1",
                owner_team="personalization",
                serving_backend="tf_serving",
                sla_ms=150,
                target_gpu_util_pct=60,
                target_cpu_util_pct=70,
            ),
        }
        endpoints = [
            _endpoint(endpoint_id="search-ep-a", current_qps=100, gpu_util_pct=50),
            _endpoint(
                endpoint_id="pers-ep-a",
                model_id="personalization-v1",
                serving_backend="tf_serving",
                instance_class="gpu_t4",
                capacity_qps=150,
                current_qps=90,
                gpu_util_pct=60,
                cost_per_1k_predictions=0.001,
            ),
        ]

        output = renderer(endpoints, models)

        self.assertIn("search-ranking-v1", output)
        self.assertIn("100/200", output)
        self.assertIn("p99_target_ms", output)
        self.assertIn("50", output)
        self.assertIn("50", output)
        self.assertIn("personalization-v1", output)
        self.assertIn("90/150", output)
        self.assertIn("150", output)
        self.assertIn("60", output)

    def test_route_trace_wraps_long_reasons_and_explains_gpu_cost_allocation(self) -> None:
        decision = RouteDecision(
            request_id="req-1",
            model_id="search-ranking-v1",
            status="routed",
            chosen_endpoint="search-ep-a",
            route_reason=(
                "sla_feasible_headroom_preserved_because_high_priority_traffic_"
                "must_stay_below_the_strict_realtime_target"
            ),
            estimated_latency_ms=24.0,
            estimated_cost_usd=0.000001,
            sla_ms=50,
        )

        output = reporting.render_route_trace([decision])

        self.assertIn("gpu_cost_per_1k", output)
        self.assertIn("GPU-hour price / (safe allocation", output)
        self.assertIn("RPS * 3600 / 1000)", output)
        self.assertIn("cloud billing", output)
        self.assertIn("remains GPU/instance-hours", output)
        self.assertIn("0.0010", output)
        self.assertIn("strict_realtime_target", output)
        reason_lines = [
            line for line in output.splitlines() if "strict_realtime_target" in line
        ]
        self.assertTrue(reason_lines)
        self.assertTrue(all(len(line) <= 110 for line in output.splitlines()))


class OrganicSpikeScenarioTests(unittest.TestCase):
    def test_organic_spike_has_baseline_spike_and_after_scaleout_phases(self) -> None:
        scenario = load_scenario(SCENARIOS_DIR / "02_organic_spike.json")
        phases = getattr(scenario, "phases", [])

        self.assertGreaterEqual(len(phases), 3)
        self.assertEqual(
            [phase.name for phase in phases[:3]],
            [
                "Baseline at target",
                "Organic spike before scale-out",
                "After scale-out",
            ],
        )

    def test_organic_spike_scales_when_gpu_targets_are_exceeded(self) -> None:
        scenario = load_scenario(SCENARIOS_DIR / "02_organic_spike.json")
        phases = getattr(scenario, "phases", [])
        self.assertGreaterEqual(len(phases), 2)
        spike = phases[1]

        signals = compute_signals(
            spike.endpoints,
            scenario.models,
            planned_events=spike.planned_events,
        )
        by_model = {signal.model_id: signal for signal in signals}

        self.assertGreater(
            by_model["search-ranking-v1"].recommended_replicas,
            by_model["search-ranking-v1"].current_replicas,
        )

    def test_organic_spike_routes_after_scaleout_without_normal_rejection(self) -> None:
        scenario = load_scenario(SCENARIOS_DIR / "02_organic_spike.json")
        phases = getattr(scenario, "phases", [])
        self.assertGreaterEqual(len(phases), 3)
        after_scaleout = phases[2]

        decisions = apply_routing(
            after_scaleout.requests,
            scenario.models,
            after_scaleout.endpoints,
        )

        rejected = [decision for decision in decisions if decision.status.startswith("rejected")]
        self.assertEqual(rejected, [])
        self.assertTrue(
            any(
                request.model_id == "search-ranking-v1"
                and request.priority == "high"
                for request in after_scaleout.requests
            )
        )


class JsonContractTests(unittest.TestCase):
    def test_single_snapshot_json_uses_raw_scenario_name(self) -> None:
        scenario = load_scenario(SCENARIOS_DIR / "01_normal_load.json")
        result = run_scenario(scenario, verbose=False)

        self.assertEqual(result["scenario"], "Normal Load")
        self.assertEqual(result["name"], "Normal Load")

    def test_phased_json_uses_raw_phase_names(self) -> None:
        scenario = load_scenario(SCENARIOS_DIR / "02_organic_spike.json")
        result = run_scenario(scenario, verbose=False)

        self.assertEqual(result["scenario"], "Organic Traffic Spike")
        self.assertEqual(
            [phase["name"] for phase in result["phases"]],
            ["Baseline at target", "Organic spike before scale-out", "After scale-out"],
        )


class ScenarioCostTests(unittest.TestCase):
    def test_route_decision_preserves_small_realistic_cost_precision(self) -> None:
        model = _model()
        endpoint = _endpoint(cost_per_1k_predictions=0.0014)
        decision = route_request(
            Request(request_id="req-cost", model_id="search-ranking-v1", priority="high"),
            model,
            [endpoint],
        )

        self.assertAlmostEqual(decision.estimated_cost_usd or 0, 0.0000014, places=10)
        self.assertIn("0.0014", reporting.render_route_trace([decision]))

    def test_representative_instance_costs_are_research_backed_per_1k_estimates(self) -> None:
        expected = {
            ("gpu_a10", 200): 0.0014,
            ("gpu_h100", 100): 0.0191,
        }
        observed: dict[tuple[str, int], float] = {}
        for path in SCENARIOS_DIR.glob("*.json"):
            scenario = load_scenario(path)
            phase_endpoints = []
            phases = getattr(scenario, "phases", [])
            if phases:
                for phase in phases:
                    phase_endpoints.extend(phase.endpoints)
            else:
                phase_endpoints.extend(scenario.endpoints)
            for endpoint in phase_endpoints:
                observed.setdefault(
                    (endpoint.instance_class, endpoint.capacity_qps),
                    endpoint.cost_per_1k_predictions,
                )
                self.assertGreater(endpoint.cost_per_1k_predictions, 0)

        for key, expected_cost in expected.items():
            self.assertIn(key, observed)
            self.assertAlmostEqual(observed[key], expected_cost, places=4)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
