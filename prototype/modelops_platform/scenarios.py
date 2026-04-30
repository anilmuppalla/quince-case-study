"""Scenario file loading.

A scenario JSON file describes a self-contained simulation: the model
catalog, the deployed pods (each held by an `Endpoint` dataclass instance)
with their current load, the inference requests to route, optional planned
events for the autoscaler, and an optional canary evaluation case.

Schema (informal):

    {
      "name": str,
      "description": str,
      "models": [Model dict, ...],
      "endpoints": [Endpoint dict, ...],
      "requests": [Request dict, ...],
      "phases": [
        {
          "name": str,
          "description": str,
          "endpoints": [Endpoint dict, ...],
          "requests": [Request dict, ...],
          "planned_events": {...}
        }
      ],
      "traffic_windows": [
        {
          "label": str,
          "start_hour": float,
          "end_hour": float,
          "rps_by_model": {"<model_id>": int},
          "platform_replicas": {"<model_id>": int} (optional),
          "recommended_replicas": {"<model_id>": int} (optional),
          "event": str (optional)
        }
      ],
      "key_events": [
        {
          "time": str,
          "decision": str,
          "evidence": str,
          "outcome": str
        }
      ],
      "planned_events": {"<model_id>": {"reason": str, "expected_load_x": float}},
      "canary": {
        "candidate_version": str,
        "rollback_to_version": str (optional),
        "candidate_metrics": {...},
        "control_metrics": {...},
        "guardrails": {...}
      }
    }

All fields except `name` and `models` are optional. A scenario can use either
top-level `endpoints` / `requests` or phased snapshots.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .domain import Endpoint, Model, Request


@dataclass
class ScenarioPhase:
    name: str
    description: str
    endpoints: list[Endpoint]
    requests: list[Request]
    planned_events: dict[str, dict] = field(default_factory=dict)


@dataclass
class Scenario:
    name: str
    description: str
    models: dict[str, Model]
    endpoints: list[Endpoint]
    requests: list[Request]
    planned_events: dict[str, dict] = field(default_factory=dict)
    canary: Optional[dict[str, Any]] = None
    phases: list[ScenarioPhase] = field(default_factory=list)
    traffic_windows: list[dict[str, Any]] = field(default_factory=list)
    key_events: list[dict[str, str]] = field(default_factory=list)


def _load_endpoints(raw_endpoints: list[dict[str, Any]]) -> list[Endpoint]:
    return [Endpoint(**e) for e in raw_endpoints]


def _load_requests(raw_requests: list[dict[str, Any]]) -> list[Request]:
    return [Request(**r) for r in raw_requests]


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a scenario from a JSON file."""
    text = Path(path).read_text(encoding="utf-8")
    raw = json.loads(text)

    models = {m["model_id"]: Model(**m) for m in raw.get("models", [])}
    endpoints = _load_endpoints(raw.get("endpoints", []))
    requests = _load_requests(raw.get("requests", []))
    phases = [
        ScenarioPhase(
            name=p.get("name", f"phase-{idx + 1}"),
            description=p.get("description", ""),
            endpoints=_load_endpoints(p.get("endpoints", [])),
            requests=_load_requests(p.get("requests", [])),
            planned_events=p.get("planned_events", {}) or {},
        )
        for idx, p in enumerate(raw.get("phases", []) or [])
    ]

    return Scenario(
        name=raw.get("name", "unnamed"),
        description=raw.get("description", ""),
        models=models,
        endpoints=endpoints,
        requests=requests,
        planned_events=raw.get("planned_events", {}) or {},
        canary=raw.get("canary"),
        phases=phases,
        traffic_windows=raw.get("traffic_windows", []) or [],
        key_events=raw.get("key_events", []) or [],
    )
