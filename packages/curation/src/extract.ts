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

/**
 * Convert a trace's `tool_execution` steps into ordered replay results (#6). A
 * production agentic trace already records what each tool returned; without this
 * an imported agentic case reaches `--agentic` with an empty script and skips on
 * its first tool call. `arguments` are carried whenever the recorded execution
 * HAS an input (#8): a result is bound to the exact call that produced it, so a
 * candidate that calls the tool with different arguments does not receive a
 * result recorded for other arguments. The runner consumes these in order, so
 * repeated identical calls draw successive results. Trace order is preserved.
 */
function recordedToolResults(trace: Trace): RecordedToolResult[] {
  const executions = trace.steps.filter((step) => step.type === "tool_execution");
  const results: RecordedToolResult[] = [];
  for (const step of executions) {
    const output = step.output;
    const result = typeof output === "string" ? output : JSON.stringify(output ?? null);
    results.push({
      tool: step.name,
      // Bind to the recorded arguments whenever they exist, so a wrong-arg call
      // is not answered by an unrelated recorded result (#8). Only a genuinely
      // argument-less recorded call (no input) stays a wildcard.
      ...(step.input != null ? { arguments: step.input } : {}),
      result,
    });
  }
  return results;
}

/** The first model call in the trace — the request a candidate is given (#7). */
function firstModelCall(trace: Trace): ModelCallStep | undefined {
  return trace.steps.find((step): step is ModelCallStep => step.type === "model_call");
}

export class NotExtractableError extends Error {
  constructor(
    readonly traceId: string,
    readonly reason: string,
  ) {
    super(`trace ${traceId} cannot become a case: ${reason}`);
    this.name = "NotExtractableError";
  }
}

/**
 * A scripted tool result for an agentic replay (#6, #23): during a `recorded`
 * trajectory the runner answers the candidate's tool calls from these instead
 * of executing a live tool. Structurally matches @compound/execution's
 * `RecordedToolResult` (persisted as JSON, so no package dependency is needed).
 */
export interface RecordedToolResult {
  tool: string;
  /** When present, only a call whose arguments deep-equal this is answered. */
  arguments?: unknown;
  result: string;
}

/** The replayable request a candidate model would be given. */
export interface CaseInput {
  model: string | null;
  input: ModelCallStep["input"];
  tools_available: ModelCallStep["tools_available"] | null;
  /** Scripted tool results for agentic replay; present only when the trace ran tools (#6). */
  recorded_tool_results?: RecordedToolResult[];
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
  // The focal call is the trace's FINAL model call — the terminal answer, which
  // defines the expected output. But an agentic candidate must drive the whole
  // trajectory itself, so the request it is GIVEN is the FIRST model call (before
  // any tool ran), not the focal call's already-expanded transcript (#7).
  // Grading it against the focal answer starts replay at turn one, as production
  // did; using the focal request as input would hand the candidate the finished
  // conversation and let it answer without ever selecting a tool.
  const focal = focalCall(trace);
  const expected = typeExpected(trace, focal);
  // The whole trace (its tool_execution steps included) is covered by
  // `contentHash`, so scripting the replay changes the case identity too (#6).
  const recorded = recordedToolResults(trace);
  const requestRoot = recorded.length > 0 ? (firstModelCall(trace) ?? focal) : focal;

  return {
    caseId: caseIdFor(trace.task_key, options.contentHash),
    taskKey: trace.task_key,
    sourceTraceId: trace.trace_id,
    contentHash: options.contentHash,
    input: {
      model: requestRoot.model ?? null,
      input: requestRoot.input,
      tools_available: requestRoot.tools_available ?? null,
      ...(recorded.length > 0 ? { recorded_tool_results: recorded } : {}),
    },
    provenance: expected.provenance,
    expected: expected.value,
  };
}
