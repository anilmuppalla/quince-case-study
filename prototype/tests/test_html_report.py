"""HTML simulation report tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from modelops_platform.scenarios import Scenario


SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"


class HtmlReportTests(unittest.TestCase):
    def test_report_includes_all_scenarios_and_navigation(self) -> None:
        from modelops_platform.html_report import render_html_report

        html = render_html_report(sorted(SCENARIOS_DIR.glob("*.json")))

        self.assertIn("ModelOps Platform Simulation Report", html)
        self.assertIn("Scenario Review Map", html)
        self.assertIn("Scenario Dashboards", html)
        for scenario_name in [
            "Normal Load",
            "Organic Traffic Spike",
            "Planned Promotion / Flash-Sale Pre-Warm",
            "Endpoint Unavailability",
            "Fraud Canary - Auto-Promote",
            "Fraud Canary - Requires Owner Review",
        ]:
            self.assertIn(scenario_name, html)
        self.assertNotIn("All Search Endpoints Unhealthy", html)
        self.assertNotIn("Regional Latency", html)
        self.assertNotIn("Pod Linking", html)

    def test_organic_spike_renders_phases_scaleout_and_capacity_visuals(self) -> None:
        from modelops_platform.html_report import render_scenario_html_page
        from modelops_platform.scenarios import load_scenario

        scenario = load_scenario(SCENARIOS_DIR / "02_organic_spike.json")
        html = render_scenario_html_page(
            scenario,
            all_scenarios=[scenario],
            scenario_index=1,
            index_href="../simulation-report.html",
        )

        self.assertNotIn("Traffic Flow Over Time", html)
        self.assertNotIn("traffic-pulse", html)
        self.assertNotIn("animation-play-state", html)
        self.assertIn("Baseline at target", html)
        self.assertIn("Organic spike before scale-out", html)
        self.assertIn("After scale-out", html)
        self.assertIn("Phase Timeline", html)
        self.assertIn("P99 50ms", html)
        self.assertIn("xgboost_ltr on triton", html)
        self.assertIn("AWS g5.xlarge / NVIDIA A10G 24GB", html)
        self.assertIn("$1.006/GPU-hr", html)
        self.assertNotIn("$0.0014/1K @ 200 RPS", html)
        self.assertNotIn("per-1K cost is derived", html)
        self.assertIn("GPU target 50%", html)
        self.assertIn("CPU target 60%", html)
        self.assertIn("Autoscaling Proof", html)
        self.assertIn("scale out 2-&gt;3", html)
        self.assertIn("GPU target 50%; demand 300 RPS", html)
        self.assertIn("Before: Fixed Fleet Capacity", html)
        self.assertIn("After: Autoscaled Platform Capacity", html)
        self.assertNotIn("Deterministic Audit Summary", html)
        self.assertNotIn("Raw CLI audit", html)
        self.assertNotIn("Serving Decision Trace", html)
        self.assertIn("<details", html)

    def test_report_summarizes_scenario_review_map(self) -> None:
        from modelops_platform.html_report import render_html_report

        html = render_html_report(sorted(SCENARIOS_DIR.glob("*.json")))

        self.assertIn("Scenario Review Map", html)
        self.assertIn("Traffic / cost scenarios", html)
        self.assertIn("Release confidence", html)
        self.assertIn("Fraud canary guardrails", html)
        self.assertIn("Models represented", html)
        self.assertIn("Search, Recommendations, Fraud", html)

    def test_each_scenario_has_local_before_after_value_summary(self) -> None:
        from modelops_platform.html_report import render_scenario_html_page
        from modelops_platform.scenarios import load_scenario

        scenarios = [load_scenario(path) for path in sorted(SCENARIOS_DIR.glob("*.json"))]
        html = "".join(
            render_scenario_html_page(
                scenario,
                all_scenarios=scenarios,
                scenario_index=index,
                index_href="../simulation-report.html",
            )
            for index, scenario in enumerate(scenarios, start=1)
        )

        self.assertEqual(html.count('class="scenario-thesis"'), 6)
        self.assertIn("For normal daily Search traffic", html)
        self.assertIn("the platform lowers 24-hour serving cost", html)
        self.assertIn("For an unplanned Search spike", html)
        self.assertIn("the platform scales replicas with observed GPU pressure", html)
        self.assertIn("For a known demand event, the platform pre-warms Recommendations capacity", html)
        self.assertIn("Fraud Detection release-confidence walkthrough", html)
        self.assertIn("class=\"scenario-setup\"", html)
        self.assertIn('class="active" aria-current="page"', html)

    def test_each_scenario_has_before_after_metric_charts(self) -> None:
        from modelops_platform.html_report import render_html_report, render_scenario_html_page
        from modelops_platform.scenarios import load_scenario

        paths = sorted(SCENARIOS_DIR.glob("*.json"))
        scenarios = [load_scenario(path) for path in paths]
        index_html = render_html_report(paths)
        scenario_html = "".join(
            render_scenario_html_page(
                scenario,
                all_scenarios=scenarios,
                scenario_index=index,
                index_href="../simulation-report.html",
            )
            for index, scenario in enumerate(scenarios, start=1)
        )
        html = index_html + scenario_html

        self.assertIn("Chart.js/4.5.0/chart.umd.min.js", html)
        self.assertIn('id="report-data"', html)
        self.assertIn("const index = scenario.scenarioIndex || i + 1", html)
        self.assertIn("Portfolio Cost Snapshot", html)
        self.assertIn("Scenario Dashboards", index_html)
        self.assertIn("simulation-report-scenarios/scenario-1-normal-load.html", index_html)
        planned = load_scenario(SCENARIOS_DIR / "03_planned_promotion.json")
        planned_html = render_scenario_html_page(
            planned,
            all_scenarios=scenarios,
            scenario_index=3,
            index_href="../simulation-report.html",
        )
        self.assertIn('"scenarioIndex":3', planned_html)
        self.assertIn('id="scenario-simulation-3"', planned_html)
        self.assertIn('id="scenario-cost-3"', planned_html)
        self.assertNotIn("<th>view</th>", index_html)
        self.assertNotIn(">Open</a>", index_html)
        self.assertEqual(index_html.count('class="dashboard-link"'), 0)
        self.assertIn("Traffic / Capacity / Cost Scenarios", index_html)
        self.assertIn("Release Confidence Scenarios", index_html)
        self.assertIn("Release Guardrail Decision", html)
        self.assertIn("Before vs After 24-Hour Cost: Traffic Scenarios", html)
        self.assertIn("Cost attribution:", html)
        self.assertIn("S1: Normal", html)
        self.assertIn("Attributed 24-hour serving cost", html)
        self.assertIn("recommendations-v1", html)
        self.assertIn("tensorflow on tf_serving", html)
        self.assertIn("fraud-detection-v1", html)
        self.assertIn("onnx on onnx_runtime", html)
        self.assertIn("Precision drop", html)
        self.assertIn("Recall drop", html)
        self.assertNotIn("Fraud-quality retention", html)
        self.assertNotIn("Business retention", html)
        self.assertEqual(html.count('id="scenario-simulation-'), 4)
        self.assertEqual(html.count('id="scenario-cost-'), 4)
        self.assertNotIn('id="routing-outcomes-', html)
        self.assertNotIn('id="utilization-targets-', html)
        self.assertNotIn("chart-panel-wide chart-panel-tall", html)
        self.assertIn("Before Fixed Fleet vs After Autoscaling", html)
        self.assertNotIn("Cost attribution uses model-owned GPU replica-hours", html)
        self.assertNotIn("Request volume is demand context, not the cost unit shown here.", html)
        self.assertIn("NVIDIA A10G replica-hours @ $1.006/GPU-hr", html)
        self.assertIn("Cost math:", html)
        self.assertIn("<code>gpu_cost_per_1k</code>", html)
        self.assertIn(
            "$1.006 / (200 safe allocation RPS x 3600 / 1000) = $0.0014 per 1K requests",
            html,
        )
        self.assertIn("Cloud billing remains GPU/instance-hours", html)
        self.assertIn("Before: fixed-fleet safe capacity (RPS)", html)
        self.assertIn("After: autoscaled safe capacity (RPS)", html)
        self.assertIn("Before: fixed fleet replicas", html)
        self.assertIn("After: active platform replicas", html)
        self.assertIn("Traffic Bucket Details", html)
        self.assertIn("Before: Fixed Fleet Capacity", html)
        self.assertIn("After: Autoscaled Platform Capacity", html)
        self.assertIn("fixed safe capacity", html)
        self.assertIn("at-risk requests", html)
        self.assertIn("autoscaled safe capacity", html)
        self.assertIn("active replicas", html)
        self.assertIn("platform outcome", html)
        self.assertIn("Cumulative Cost: Before vs After", html)
        self.assertNotIn("Serving Outcomes", html)
        self.assertIn("served", html)
        self.assertNotIn("Active Utilization Vs Target", html)
        self.assertIn("Autoscale trigger", html)
        self.assertNotIn("Risk avoided", html)
        self.assertNotIn("Service protection", html)
        self.assertNotIn("Service posture", html)
        self.assertIn("Autoscaling Proof", html)
        self.assertIn("<th>trigger</th>", html)
        self.assertIn("<th>before</th>", html)
        self.assertIn("<th>after</th>", html)
        self.assertIn("demand", html)
        self.assertIn("Health/readiness", html)
        self.assertIn("Traffic Bucket Details", html)
        self.assertIn("Before: peak-provisioned", html)
        self.assertIn("After: platform", html)
        self.assertNotIn("Deterministic Audit Summary", html)
        self.assertNotIn("audit-summary", html)
        self.assertNotIn("Raw CLI audit", html)
        self.assertNotIn("Serving Decision Trace", html)
        self.assertNotIn("Fixed current fleet", html)
        self.assertNotIn("Static peak capacity", html)
        self.assertNotIn("<svg", html)
        self.assertNotIn("<polyline", html)

    def test_report_data_includes_simulation_dimensions_for_charts(self) -> None:
        from modelops_platform.html_report import (
            _build_report_data,
        )
        from modelops_platform.scenarios import load_scenario

        scenarios = [
            load_scenario(SCENARIOS_DIR / "02_organic_spike.json"),
            load_scenario(SCENARIOS_DIR / "05_canary_auto_promote.json"),
        ]
        data = _build_report_data(scenarios)

        self.assertEqual(data["portfolio"]["axisLabels"], ["S1: Spike"])
        self.assertEqual(data["portfolio"]["excluded"][0]["axisLabel"], "S2: Canary pass")
        self.assertEqual(
            data["portfolio"]["excluded"][0]["scenarioTypeLabel"],
            "Release confidence",
        )

        organic = data["scenarios"][0]
        self.assertEqual(organic["baselineLabel"], "before: peak-provisioned")
        for key in [
            "rps",
            "beforeReplicas",
            "afterReplicas",
            "beforeHourlyCost",
            "afterHourlyCost",
        ]:
            self.assertIn(key, organic)
        self.assertIn("traffic", organic)
        self.assertIn("summary", organic["traffic"])
        self.assertEqual(organic["traffic"]["summary"]["peakRps"], 1400)
        self.assertEqual(organic["traffic"]["summary"]["platformRejected"], 0)
        self.assertGreater(organic["traffic"]["summary"]["fixedAtRisk"], 1_000_000)
        self.assertEqual(len(organic["traffic"]["shortLabels"]), 7)
        self.assertEqual(len(organic["traffic"]["axisLabels"]), 7)
        self.assertIn("scale out", "; ".join(organic["traffic"]["scaleActions"]))
        self.assertGreaterEqual(len(organic["labels"]), 3)
        for key in ["served", "overflow", "rejected", "utilization"]:
            self.assertNotIn(key, organic)

        canary = data["scenarios"][1]
        self.assertEqual(canary["canaryDecision"], "auto_promote")
        self.assertEqual(canary["scenarioType"], "release-confidence")
        self.assertFalse(canary["costPortfolioEligible"])
        self.assertEqual(canary["traffic"], {})

    def test_traffic_cost_scenarios_define_real_traffic_windows(self) -> None:
        from modelops_platform.html_report import _build_report_data
        from modelops_platform.scenarios import load_scenario

        scenarios = [load_scenario(path) for path in sorted(SCENARIOS_DIR.glob("*.json"))]
        for scenario in scenarios:
            if scenario.canary is None:
                self.assertGreater(
                    len(scenario.traffic_windows),
                    1,
                    f"{scenario.name} should not use the one-bucket fallback",
                )

        normal = load_scenario(SCENARIOS_DIR / "01_normal_load.json")
        normal_data = _build_report_data([normal])["scenarios"][0]["traffic"]

        self.assertEqual(
            normal_data["axisLabels"],
            ["0-4h", "4-7h", "7-10h", "10-13h", "13-16h", "16-19h", "19-22h", "22-24h"],
        )
        self.assertEqual(normal_data["totalRps"], [45, 65, 95, 125, 145, 135, 105, 65])
        self.assertEqual(normal_data["fixedReplicas"], [2, 2, 2, 2, 2, 2, 2, 2])
        self.assertEqual(normal_data["platformReplicas"], [1, 1, 1, 2, 2, 2, 2, 1])
        self.assertEqual(
            normal_data["scaleActions"],
            [
                "initial platform posture",
                "hold steady",
                "hold steady",
                "scale out: search-ranking-v1 1->2",
                "hold steady",
                "hold steady",
                "hold steady",
                "scale down: search-ranking-v1 2->1",
            ],
        )
        self.assertEqual(normal_data["summary"]["fixedAtRisk"], 0)
        self.assertEqual(normal_data["summary"]["platformRejected"], 0)

        from modelops_platform.html_report import render_scenario_html_page

        normal_html = render_scenario_html_page(
            normal,
            all_scenarios=[normal],
            scenario_index=1,
            index_href="../simulation-report.html",
        )
        self.assertIn("Autoscaling Proof", normal_html)
        self.assertIn("GPU utilization target", normal_html)
        self.assertIn("GPU target 50%; demand 125 RPS", normal_html)
        self.assertIn("1 replica, 62.5%, 100 RPS safe cap", normal_html)
        self.assertIn("2 replicas, 31.2%, 200 RPS safe cap", normal_html)
        self.assertIn("GPU utilization crossed the 50% target", normal_html)
        self.assertIn("2 replicas, 16.2%, 200 RPS safe cap", normal_html)
        self.assertIn("1 replica, 32.5%, 100 RPS safe cap", normal_html)

    def test_metric_timesteps_make_after_cost_below_before(self) -> None:
        from modelops_platform.html_report import (
            _build_metric_timesteps,
            _scenario_snapshots,
        )
        from modelops_platform.scenarios import load_scenario

        scenario = load_scenario(SCENARIOS_DIR / "02_organic_spike.json")
        snapshots = _scenario_snapshots(scenario)
        timesteps = _build_metric_timesteps(snapshots)

        self.assertGreaterEqual(len(timesteps), 3)
        self.assertTrue(
            all(
                step.after_hourly_cost <= step.before_hourly_cost + 1e-9
                for step in timesteps
            )
        )
        peak = max(timesteps, key=lambda step: step.after_replicas)
        self.assertLess(peak.after_hourly_cost, peak.before_hourly_cost)
        self.assertGreater(peak.before_replicas, peak.after_replicas)

    def test_report_escapes_dynamic_scenario_text(self) -> None:
        from modelops_platform.html_report import render_scenario_section

        scenario = Scenario(
            name="<script>alert('x')</script>",
            description="5 < 6 & route safely",
            models={},
            endpoints=[],
            requests=[],
        )

        html = render_scenario_section(scenario, index=1)

        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;", html)
        self.assertIn("5 &lt; 6 &amp; route safely", html)

    def test_write_html_report_creates_output_file(self) -> None:
        from modelops_platform.html_report import write_html_report

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "simulation.html"
            stale_dir = Path(tmp) / "simulation-scenarios"
            stale_dir.mkdir()
            stale_page = stale_dir / "scenario-99-stale.html"
            stale_page.write_text("stale", encoding="utf-8")
            write_html_report([SCENARIOS_DIR / "01_normal_load.json"], output)
            scenario_page = (
                Path(tmp)
                / "simulation-scenarios"
                / "scenario-1-normal-load.html"
            )

            self.assertTrue(output.exists())
            self.assertTrue(scenario_page.exists())
            self.assertFalse(stale_page.exists())
            self.assertIn("Scenario Dashboards", output.read_text(encoding="utf-8"))
            self.assertIn("Normal Load", scenario_page.read_text(encoding="utf-8"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
