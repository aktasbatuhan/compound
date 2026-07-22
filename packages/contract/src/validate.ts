/**
 * Trace validation per docs/trace-contract-v1.md "Validation classes".
 *
 * - `eval_ready`: schema-valid, >=1 model_call, focal_step_id set and pointing
 *   at an existing model_call, focal call has input and output (or an explicit
 *   error being studied), usage or timestamps present.
 * - `diagnostic`: schema-valid but unreplayable; carries machine-readable
 *   `diagnostic_reasons` (stable snake_case strings).
 * - `rejected`: schema-invalid; never persisted, reported with zod issues.
 */
import type { z } from "zod";
import type { Message, ModelCallStep, Trace } from "./schemas";
import { TraceSchema } from "./schemas";

export const DIAGNOSTIC_REASONS = [
  "no_model_call_steps",
  "missing_focal_step_id",
  "focal_step_not_found",
  "focal_step_not_model_call",
  "focal_step_missing_input",
  "focal_step_missing_output_and_error",
  "focal_step_missing_usage_and_timestamps",
  "unsupported_content_part",
  "duplicate_step_ids",
] as const;

export type DiagnosticReason = (typeof DIAGNOSTIC_REASONS)[number];

export type ValidationResult =
  | { class: "eval_ready"; trace: Trace }
  | { class: "diagnostic"; trace: Trace; diagnostic_reasons: DiagnosticReason[] }
  | { class: "rejected"; issues: z.core.$ZodIssue[] };

export type ValidationClass = ValidationResult["class"];

function messageHasUnsupportedContent(message: Message): boolean {
  return (
    Array.isArray(message.content) && message.content.some((part) => part.type === "unsupported")
  );
}

function traceHasUnsupportedContent(trace: Trace): boolean {
  for (const step of trace.steps) {
    if (step.type !== "model_call") continue;
    if (step.input.some(messageHasUnsupportedContent)) return true;
    if (step.output != null && messageHasUnsupportedContent(step.output)) return true;
  }
  return false;
}

function collectDiagnosticReasons(trace: Trace): DiagnosticReason[] {
  const reasons: DiagnosticReason[] = [];

  const stepIds = trace.steps.map((step) => step.step_id);
  if (new Set(stepIds).size !== stepIds.length) {
    reasons.push("duplicate_step_ids");
  }

  const hasModelCall = trace.steps.some((step) => step.type === "model_call");
  if (!hasModelCall) {
    reasons.push("no_model_call_steps");
  }

  if (trace.focal_step_id === null) {
    reasons.push("missing_focal_step_id");
  } else {
    const focal = trace.steps.find((step) => step.step_id === trace.focal_step_id);
    if (focal === undefined) {
      reasons.push("focal_step_not_found");
    } else if (focal.type !== "model_call") {
      reasons.push("focal_step_not_model_call");
    } else {
      reasons.push(...collectFocalReasons(focal));
    }
  }

  if (traceHasUnsupportedContent(trace)) {
    reasons.push("unsupported_content_part");
  }

  return reasons;
}

function collectFocalReasons(focal: ModelCallStep): DiagnosticReason[] {
  const reasons: DiagnosticReason[] = [];
  if (focal.input.length === 0) {
    reasons.push("focal_step_missing_input");
  }
  if (focal.output == null && focal.error == null) {
    reasons.push("focal_step_missing_output_and_error");
  }
  const hasUsage = focal.usage != null;
  const hasTimestamps = focal.started_at != null && focal.ended_at != null;
  if (!hasUsage && !hasTimestamps) {
    reasons.push("focal_step_missing_usage_and_timestamps");
  }
  return reasons;
}

/**
 * Validate one candidate trace record and classify it.
 *
 * Never throws on bad input: schema failures come back as
 * `{ class: "rejected", issues }`.
 */
export function validate(input: unknown): ValidationResult {
  const parsed = TraceSchema.safeParse(input);
  if (!parsed.success) {
    return { class: "rejected", issues: parsed.error.issues };
  }
  const trace = parsed.data;
  const diagnosticReasons = collectDiagnosticReasons(trace);
  if (diagnosticReasons.length > 0) {
    return { class: "diagnostic", trace, diagnostic_reasons: diagnosticReasons };
  }
  return { class: "eval_ready", trace };
}
