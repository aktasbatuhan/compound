/**
 * Case extraction: an eval-ready trace becomes at most one eval case.
 *
 * Provenance is assigned from what the trace carries, and is deliberately
 * conservative: `human_golden` and `synthetic_label` are NEVER assigned here.
 * They arise only through review or judging, because promoting observed or
 * synthetic output to golden without a human is exactly the mistake that makes
 * a candidate imitate the incumbent's errors (docs/curation-v1.md, SPEC.md).
 */
import { createHash } from "node:crypto";
import type { ModelCallStep, Outcome, Trace } from "@compound/contract";
import type { CaseProvenance } from "@compound/storage";

export class NotExtractableError extends Error {
  constructor(
    readonly traceId: string,
    readonly reason: string,
  ) {
    super(`trace ${traceId} cannot become a case: ${reason}`);
    this.name = "NotExtractableError";
  }
}

/** The replayable request a candidate model would be given. */
export interface CaseInput {
  model: string | null;
  input: ModelCallStep["input"];
  tools_available: ModelCallStep["tools_available"] | null;
}

/** A typed expected output plus the provenance that governs its use. */
export interface CaseExpected {
  provenance: CaseProvenance;
  /** The expected value; shape depends on provenance. May be null. */
  value: unknown;
}

export interface ExtractedCase {
  caseId: string;
  taskKey: string;
  sourceTraceId: string;
  contentHash: string;
  input: CaseInput;
  provenance: CaseProvenance;
  expected: CaseExpected["value"] | null;
}

function focalCall(trace: Trace): ModelCallStep {
  if (trace.focal_step_id === null) {
    throw new NotExtractableError(trace.trace_id, "no focal step");
  }
  const step = trace.steps.find(
    (candidate) => candidate.step_id === trace.focal_step_id && candidate.type === "model_call",
  );
  if (step === undefined || step.type !== "model_call") {
    throw new NotExtractableError(trace.trace_id, "focal step is not a model call");
  }
  return step;
}

/**
 * Decide provenance and expected value from the trace.
 *
 * Order matters: a deterministic outcome is the strongest evidence, then real
 * user feedback, then the incumbent's own output as a last resort. The output
 * is only ever `observed_output` here — never a golden.
 */
function typeExpected(trace: Trace, focal: ModelCallStep): CaseExpected {
  const outcome: Outcome | null | undefined = trace.outcome;

  if (outcome?.deterministic != null) {
    return { provenance: "deterministic_outcome", value: outcome.deterministic };
  }

  const humanFeedback = outcome?.feedback?.filter(
    (entry) =>
      entry.kind === "thumbs" ||
      entry.kind === "rating" ||
      entry.kind === "correction" ||
      entry.kind === "comment",
  );
  if (humanFeedback != null && humanFeedback.length > 0) {
    return { provenance: "user_feedback", value: humanFeedback };
  }

  // The incumbent's recorded answer: evidence, not ground truth. May be null
  // (a replayable case with no expected output, gradeable by assertion).
  return { provenance: "observed_output", value: focal.output ?? null };
}

/** Stable case id: `case:<sha256(task_key + content_hash)>` truncated. */
export function caseIdFor(taskKey: string, contentHash: string): string {
  const digest = createHash("sha256").update(`${taskKey}\n${contentHash}`).digest("hex");
  return `case:${digest.slice(0, 32)}`;
}

/**
 * Extract a case from an eval-ready trace.
 *
 * Throws `NotExtractableError` if the trace is not a replayable single call —
 * the caller (curation) treats that as "this trace does not become a case",
 * not as an error to surface to the user.
 */
export function extractCase(trace: Trace, options: { contentHash: string }): ExtractedCase {
  if (trace.task_key === null) {
    throw new NotExtractableError(trace.trace_id, "trace has no task_key");
  }
  const focal = focalCall(trace);
  const expected = typeExpected(trace, focal);

  return {
    caseId: caseIdFor(trace.task_key, options.contentHash),
    taskKey: trace.task_key,
    sourceTraceId: trace.trace_id,
    contentHash: options.contentHash,
    input: {
      model: focal.model ?? null,
      input: focal.input,
      tools_available: focal.tools_available ?? null,
    },
    provenance: expected.provenance,
    expected: expected.value,
  };
}
