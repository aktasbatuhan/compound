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
 * request a candidate model would actually be given. Traces with no focal step
 * are not replayable, so they fall back to hashing the whole `steps` array,
 * which still collapses exact duplicates.
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

  const subject =
    focal !== undefined && focal.type === "model_call"
      ? {
          kind: "focal_request",
          model: focal.model ?? null,
          input: focal.input,
          tools_available: focal.tools_available ?? null,
        }
      : { kind: "steps", steps: trace.steps };

  return createHash("sha256").update(canonicalJson(subject)).digest("hex");
}
