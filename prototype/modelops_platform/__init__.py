"""ModelOps Platform prototype.

A small, illustrative cost-aware inference router that demonstrates the
target Quince ML serving platform's core decisions: SLA-aware routing,
autoscaling signals, training/batch isolation, instance unavailability,
and canary auto-promotion vs human review.

This prototype is intentionally lightweight. It does not run real models or
talk to Kubernetes. It simulates the decision logic so the routing and
deployment policies can be inspected end-to-end.
"""

__version__ = "0.1.0"
