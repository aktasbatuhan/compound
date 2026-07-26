/**
 * Optimization eligibility (docs/optimization-v1.md): optimize a candidate's
 * prompt only when the gap is worth closing. Pure — it reads a gate's stored
 * summary (outcome, delta, margin), never the sealed set.
 *
 * The band: a candidate already MEETING the gate has nothing to close; one that
 * is hopelessly far behind won't be rescued by a prompt (switch the model); the
 * useful middle — below the reference but within a closable ceiling, or
 * `insufficient_data` with a negative point estimate — is where GEPA earns its
 * keep. GLM at 92% vs Opus 100% (Δ −8pp) is the canonical eligible case.
 */
import type { GateOutcome } from "@compound/storage";

export interface GateSummary {
  outcome: GateOutcome;
  /** Candidate minus reference on the metric; negative means candidate is behind. */
  delta: number;
}

export interface EligibilityOptions {
  /** Max gap (reference − candidate) a prompt could plausibly close. Default 0.25. */
  ceiling?: number;
  /** Optimization budget in USD; ≤ 0 makes it ineligible. */
  budgetUsd?: number;
}

export type EligibilityReason = "eligible" | "already_meets" | "hopeless" | "no_gap" | "no_budget";

export interface Eligibility {
  eligible: boolean;
  reason: EligibilityReason;
  /** How far the candidate is behind the reference (reference − candidate), ≥ 0. */
  gap: number;
  detail: string;
}

export function assessEligibility(
  gate: GateSummary,
  options: EligibilityOptions = {},
): Eligibility {
  const ceiling = options.ceiling ?? 0.25;
  const gap = Math.max(0, -gate.delta);

  if (options.budgetUsd !== undefined && options.budgetUsd <= 0) {
    return { eligible: false, reason: "no_budget", gap, detail: "no optimization budget" };
  }
  if (gate.outcome === "meets_gate") {
    return {
      eligible: false,
      reason: "already_meets",
      gap,
      detail: "candidate already meets the gate — nothing to close",
    };
  }
  if (gap <= 0) {
    return {
      eligible: false,
      reason: "no_gap",
      gap,
      detail: "candidate is not behind the reference — a prompt pass would not help",
    };
  }
  if (gap > ceiling) {
    return {
      eligible: false,
      reason: "hopeless",
      gap,
      detail: `gap ${(gap * 100).toFixed(1)}pp exceeds the closable ceiling ${(ceiling * 100).toFixed(0)}pp — switch the model instead`,
    };
  }
  return {
    eligible: true,
    reason: "eligible",
    gap,
    detail: `candidate is ${(gap * 100).toFixed(1)}pp behind (within the ${(ceiling * 100).toFixed(0)}pp closable band) — worth a prompt pass`,
  };
}
