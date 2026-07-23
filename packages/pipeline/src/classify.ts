/**
 * Merging normalization diagnostics with contract validation.
 *
 * Two independent sources produce reasons and both must be honoured:
 *
 * - **Normalization** reports what the *source data* could not express
 *   (`unparseable_generation_input`, `no_replayable_focal_call`, …).
 * - **Contract validation** reports what the *contract* requires
 *   (`missing_focal_step_id`, `unsupported_content_part`, …).
 *
 * A trace is `eval_ready` only when nothing anywhere had a complaint.
 *
 * Spec: docs/ingest-pipeline-v1.md, "Classification merge".
 */
import type { Trace, ValidationResult } from "@compound/contract";
import type { z } from "zod";

export type ClassifiedOutcome =
  | {
      outcome: "persist";
      trace: Trace;
      validationClass: "eval_ready" | "diagnostic";
      diagnosticReasons: string[];
    }
  | {
      /**
       * Our own code produced a trace that fails the contract. Ingest
       * guarantees schema-valid output and redaction preserves validity, so
       * this is a bug in Compound — categorically different from a malformed
       * input line, and never reported as the user's fault.
       */
      outcome: "internal_error";
      issues: z.core.$ZodIssue[];
    };

export function classify(
  validation: ValidationResult,
  normalizationDiagnostics: readonly string[],
): ClassifiedOutcome {
  if (validation.class === "rejected") {
    return { outcome: "internal_error", issues: validation.issues };
  }

  const reasons = new Set<string>(normalizationDiagnostics);
  if (validation.class === "diagnostic") {
    for (const reason of validation.diagnostic_reasons) reasons.add(reason);
  }

  const diagnosticReasons = [...reasons].sort();
  return {
    outcome: "persist",
    trace: validation.trace,
    // Any complaint from either source downgrades the trace.
    validationClass: diagnosticReasons.length > 0 ? "diagnostic" : "eval_ready",
    diagnosticReasons,
  };
}
