/**
 * Tool-call linkage and focal-step selection.
 *
 * Both implement fixed rules from docs/langfuse-import-mapping.md. Where the
 * rule does not determine an answer the importer records `null` plus a
 * diagnostic — it never guesses.
 */
import type { ModelCallStep, Step, ToolExecutionStep } from "@compound/contract";
import { type Collector, DIAGNOSTICS } from "./diagnostics";
import type { ObservationView } from "./observations";
import { toolCallIdOf } from "./observations";

interface ToolCallIndexEntry {
  stepId: string;
  toolCallId: string;
  name: string;
}

function indexToolCalls(steps: Step[]): ToolCallIndexEntry[] {
  const entries: ToolCallIndexEntry[] = [];
  for (const step of steps) {
    if (step.type !== "model_call" || step.output == null) continue;
    for (const call of step.output.tool_calls ?? []) {
      entries.push({ stepId: step.step_id, toolCallId: call.id, name: call.name });
    }
  }
  return entries;
}

/**
 * Resolve `call_ref` for every tool_execution step, in place.
 *
 * Primary rule (mapping doc): match the observation's `tool_call_id` against the
 * generations' output tool_calls. Secondary rule: when the observation carries
 * no tool_call_id, a tool_execution whose name matches **exactly one** tool_call
 * in the whole trace is linked to it — a unique match, not a guess. Anything
 * ambiguous stays `null` with `unresolved_tool_call_ref`.
 */
export function resolveToolCallRefs(
  steps: Step[],
  views: Map<string, ObservationView>,
  collector: Collector,
): void {
  const index = indexToolCalls(steps);
  if (index.length === 0) {
    for (const step of steps) {
      if (step.type === "tool_execution") collector.diagnostic(DIAGNOSTICS.unresolvedToolCallRef);
    }
    return;
  }

  const claimed = new Set<string>();
  const toolSteps = steps.filter(
    (step): step is ToolExecutionStep => step.type === "tool_execution",
  );

  // Pass 1: explicit tool_call_id.
  const unmatched: ToolExecutionStep[] = [];
  for (const step of toolSteps) {
    const view = views.get(step.step_id);
    const toolCallId = view === undefined ? null : toolCallIdOf(view.raw);
    if (toolCallId === null) {
      unmatched.push(step);
      continue;
    }
    const matches = index.filter((entry) => entry.toolCallId === toolCallId);
    const match = matches.length === 1 ? matches[0] : undefined;
    if (match === undefined) {
      collector.diagnostic(DIAGNOSTICS.unresolvedToolCallRef);
      continue;
    }
    step.call_ref = { step_id: match.stepId, tool_call_id: match.toolCallId };
    claimed.add(match.toolCallId);
  }

  // Pass 2: unique name match against still-unclaimed tool_calls.
  for (const step of unmatched) {
    const matches = index.filter(
      (entry) => entry.name === step.name && !claimed.has(entry.toolCallId),
    );
    const match = matches.length === 1 ? matches[0] : undefined;
    if (match === undefined) {
      collector.diagnostic(DIAGNOSTICS.unresolvedToolCallRef);
      continue;
    }
    step.call_ref = { step_id: match.stepId, tool_call_id: match.toolCallId };
    claimed.add(match.toolCallId);
  }
}

function hasRecordedOutput(step: ToolExecutionStep): boolean {
  return step.output !== undefined && step.output !== null;
}

/**
 * Focal step per the v1 ordered rule:
 *
 * 1. Sole `model_call` in the trace → focal.
 * 2. Multiple generations → the last generation in the trace's main chain whose
 *    output carries no tool_calls (the "final answer" call), **if** every
 *    tool_execution has a resolvable `call_ref` and a recorded output.
 * 3. Otherwise `null` + `no_replayable_focal_call`.
 */
export function selectFocalStepId(steps: Step[], collector: Collector): string | null {
  const generations = steps.filter((step): step is ModelCallStep => step.type === "model_call");

  if (generations.length === 0) {
    collector.diagnostic(DIAGNOSTICS.noReplayableFocalCall);
    return null;
  }
  const sole = generations[0];
  if (generations.length === 1 && sole !== undefined) return sole.step_id;

  const last = generations[generations.length - 1];
  if (last === undefined || last.output == null) {
    collector.diagnostic(DIAGNOSTICS.noReplayableFocalCall);
    return null;
  }
  if ((last.output.tool_calls ?? []).length > 0) {
    collector.diagnostic(DIAGNOSTICS.noReplayableFocalCall);
    return null;
  }

  const toolSteps = steps.filter(
    (step): step is ToolExecutionStep => step.type === "tool_execution",
  );
  const replayable = toolSteps.every((step) => step.call_ref != null && hasRecordedOutput(step));
  if (!replayable) {
    collector.diagnostic(DIAGNOSTICS.noReplayableFocalCall);
    return null;
  }

  return last.step_id;
}
