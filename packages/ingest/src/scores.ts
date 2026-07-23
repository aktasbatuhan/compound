/**
 * Scores → `Outcome`.
 *
 * Source mapping is fixed by docs/langfuse-import-mapping.md:
 * ANNOTATION→human, EVAL→judge, API→deterministic, and `dataType: CORRECTION`
 * becomes a `correction` feedback entry instead of a score.
 */
import type { Feedback, Outcome, Score } from "@compound/contract";
import { type Collector, DIAGNOSTICS } from "./diagnostics";
import { asFiniteNumber, asString, field, toIsoUtc } from "./values";

const SOURCE_MAP: Record<string, Score["source"]> = {
  ANNOTATION: "human",
  EVAL: "judge",
  API: "deterministic",
};

/** Why a score was not carried into the contract. */
export const SKIPPED_SCORE_REASONS = {
  nullTraceId: "null_trace_id",
  unmatchedTrace: "unmatched_trace",
  unknownSource: "unknown_source",
  nonNumericValue: "non_numeric_value",
  missingName: "missing_name",
} as const;

export interface OutcomeAccumulator {
  scores: Score[];
  feedback: Feedback[];
}

export function newOutcomeAccumulator(): OutcomeAccumulator {
  return { scores: [], feedback: [] };
}

/**
 * Fold one trace-scoped score into the accumulator.
 *
 * Returns a skip reason when the score cannot be represented: the contract's
 * `Score.value` is a number, so CATEGORICAL/TEXT scores (which carry only
 * `stringValue`) are counted rather than coerced.
 */
export function applyScore(
  raw: Record<string, unknown>,
  accumulator: OutcomeAccumulator,
  collector: Collector,
): string | null {
  const name = asString(field(raw, "name"));
  if (name === null) return SKIPPED_SCORE_REASONS.missingName;

  const at = toIsoUtc(field(raw, "timestamp", "createdAt", "created_at"));
  const dataType = asString(field(raw, "dataType", "data_type"))?.toUpperCase() ?? null;
  const value = field(raw, "value");
  const stringValue = asString(field(raw, "stringValue", "string_value"));

  if (dataType === "CORRECTION") {
    accumulator.feedback.push({
      kind: "correction",
      value: (stringValue ?? value ?? null) as Feedback["value"],
      at,
    });
    return null;
  }

  const sourceRaw = asString(field(raw, "source"))?.toUpperCase() ?? "";
  const source = SOURCE_MAP[sourceRaw];
  if (source === undefined) {
    collector.diagnostic(DIAGNOSTICS.unknownScoreSource);
    return SKIPPED_SCORE_REASONS.unknownSource;
  }

  const numeric = asFiniteNumber(value);
  if (numeric === null) {
    collector.diagnostic(DIAGNOSTICS.nonNumericScoreSkipped);
    return SKIPPED_SCORE_REASONS.nonNumericValue;
  }

  accumulator.scores.push({ name, value: numeric, source, at });
  return null;
}

/** `null` when the trace carried no representable outcome data. */
export function toOutcome(accumulator: OutcomeAccumulator): Outcome | null {
  if (accumulator.scores.length === 0 && accumulator.feedback.length === 0) return null;
  return {
    feedback: accumulator.feedback.length > 0 ? accumulator.feedback : null,
    scores: accumulator.scores.length > 0 ? accumulator.scores : null,
    deterministic: null,
  };
}
