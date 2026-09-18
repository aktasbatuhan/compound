# Experiment entry point

Use [the Claude execution handoff](claude-experiment-handoff.md) as the starting
point. It identifies the current specifications, launch gates, spending bounds,
and evidence required for independent review.

## 1. Budget-first Flex pilot

Canonical protocol: [budget-curve-v2](flex-agentic/budget-curve-v2.md).
Canonical specification: [budget-curve-v2.json](flex-agentic/budget-curve-v2.json).

The primary question is delivered task success at a fixed agent-dollar allowance,
comparing realtime and Flex within the same model and workload. Time and delivery
failures remain secondary outcomes. The first pilot is descriptive and limited to
DeepSeek V4.1 Flash on Doubleword; it is not an all-provider ranking.

The old breadth/depth budgets, uncached pilot extrapolations and equivalence
recommendation formerly on this page are superseded. Historical specs and raw
results remain available; do not overwrite them or use them as the new launch
configuration. The equivalence analyzer is not the budget-curve reporter.

## 2. Fireworks comparison

This is a separate design, not an executable experiment yet. See the handoff's
Fireworks section. Exact replication needs the original harness and evaluation
configuration. A different harness must be labelled an independent comparison.

Do not treat a rate-card calculation on another provider's token mix as measured
cost, quality, cache performance, or a reliable funding estimate. The prior
cost-ranking and 30-to-50-task recommendation on this page are withdrawn as
execution guidance. Do not launch this experiment under the Flex pilot's $10 cap.
