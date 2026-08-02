/**
 * Content hashing for exact-duplicate detection.
 *
 * `content_hash` identifies *the same replayable work*, so two distinct traces
 * representing the same request collapse during curation. (Re-importing the
 * same export is handled far more cheaply by the unique `trace_id`.)
 *
 * Spec: docs/ingest-pipeline-v1.md, "Content hash".
 */
import { createHash } from "node:crypto";
import type { Trace } from "@compound/contract";

/**
 * Deterministic JSON: object keys sorted recursively, no insignificant
 * whitespace. Array order is significant and preserved — message order changes
 * meaning.
 */
export function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value) ?? "null";
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(",")}]`;
  }
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([, child]) => child !== undefined)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([key, child]) => `${JSON.stringify(key)}:${canonicalJson(child)}`);
  return `{${entries.join(",")}}`;
}

/**
 * Hash the replayable request of a trace.
 *
 * Computed on the focal model call's `{model, input, tools_available}` — the
 * request a candidate model would actually be given. When the trace also ran
 * tools, the recorded replay script (each `tool_execution`'s name/input/output,
 * in order) is folded in too (#6): a different script IS a different agentic
 * case, so two traces with the same focal request but different tool outcomes
 * must not collapse into one. Traces with no focal step are not replayable, so
 * they fall back to hashing the whole `steps` array, which still collapses
 * exact duplicates.
 *
 * Must be called AFTER redaction: two traces differing only in a secret that
 * was redacted away are the same work and should dedupe as such.
 */
export function computeContentHash(trace: Trace): string {
  const focal =
    trace.focal_step_id === null
      ? undefined
      : trace.steps.find(
          (step) => step.step_id === trace.focal_step_id && step.type === "model_call",
        );

  const replayScript = trace.steps
    .filter((step) => step.type === "tool_execution")
    .map((step) => ({ name: step.name, input: step.input ?? null, output: step.output ?? null }));

  const subject =
    focal !== undefined && focal.type === "model_call"
      ? {
          kind: "focal_request",
          model: focal.model ?? null,
          input: focal.input,
          tools_available: focal.tools_available ?? null,
          // Present only for agentic traces, so non-agentic hashes are unchanged.
          ...(replayScript.length > 0 ? { recorded_tool_results: replayScript } : {}),
        }
      : { kind: "steps", steps: trace.steps };

  return createHash("sha256").update(canonicalJson(subject)).digest("hex");
}
