/**
 * Grade a stored case's OWN observed output against its task's assertions.
 *
 * This is free — it grades output that already exists, with no model call. It
 * answers "does the incumbent's output pass the checks I care about?", which is
 * a real sanity check on both the traffic and the assertion config, and it is
 * the same engine that will later grade candidate outputs.
 *
 * Grading a candidate model's output is a separate, paid step; this module does
 * not make any provider calls.
 */
import type { Message } from "@compound/contract";
import { evaluateAssertions } from "./evaluate";
import type { Assertion, AssertionReport } from "./types";

/** The stored case shape this needs — a subset of the storage CaseRow. */
export interface GradableCase {
  caseId: string;
  taskKey: string;
  /** The typed expected output; only `observed_output` carries a model message. */
  expected: unknown;
  provenance: string;
}

/**
 * A case's expected value is a model output message only when its provenance is
 * `observed_output` (the incumbent's recorded answer). Other provenance types
 * hold feedback or deterministic results, which are not a message to grade.
 */
function observedOutputMessage(gradable: GradableCase): Message | null {
  if (gradable.provenance !== "observed_output") return null;
  const expected = gradable.expected;
  if (expected === null || typeof expected !== "object") return null;
  // Structural check: it looks like a contract Message.
  if (!("role" in expected)) return null;
  return expected as Message;
}

export interface CaseAssertionReport extends AssertionReport {
  caseId: string;
  /** False when the case has no observed output message to grade. */
  graded: boolean;
}

/**
 * Grade one case's observed output against a task's assertions.
 *
 * A case with no observed output message (non-`observed_output` provenance, or
 * an absent output) is reported as `graded: false` rather than failed — there
 * is simply nothing deterministic to check yet.
 */
export function gradeCaseObservedOutput(
  gradable: GradableCase,
  assertions: readonly Assertion[],
): CaseAssertionReport {
  const output = observedOutputMessage(gradable);
  if (output === null) {
    return {
      caseId: gradable.caseId,
      graded: false,
      passed: false,
      score: 0,
      results: [],
    };
  }
  const report = evaluateAssertions(assertions, { output });
  return { caseId: gradable.caseId, graded: true, ...report };
}
