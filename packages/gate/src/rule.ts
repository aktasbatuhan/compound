/**
 * The pure decision rule: given the confidence interval and the counts, which
 * of the five outcomes applies (docs/gate-decision-v1.md, "The statistic").
 *
 * Kept side-effect-free and separate from I/O so the rule itself is exhaustively
 * testable.
 */
import type { GateMode, GateOutcome } from "@compound/storage";

export interface DecideInput {
  mode: GateMode;
  margin: number;
  ciLo: number;
  ciHi: number;
  /** Cases contributing to the diff (paired, graded, not abstained). */
  n: number;
  minCases: number;
  judgeAbstainedFraction: number;
  judgeAbstainMax: number;
}

export function decideOutcome(input: DecideInput): GateOutcome {
  // A judge that won't stand behind too many cases voids the decision, whatever
  // the numbers say. Assertion-only tasks never abstain, so this never fires
  // for them (fraction is 0).
  if (input.judgeAbstainedFraction > input.judgeAbstainMax) return "judge_abstained";

  // Too little evidence to conclude either way.
  if (input.n < input.minCases) return "insufficient_data";

  if (input.mode === "superiority") {
    if (input.ciLo > 0) return "meets_gate";
    if (input.ciHi < 0) return "fails_gate";
    return "no_reliable_improvement";
  }

  // Non-inferiority against margin m > 0: the candidate may be up to m worse.
  const m = input.margin;
  if (input.ciLo >= -m) return "meets_gate";
  if (input.ciHi < -m) return "fails_gate";
  return "insufficient_data";
}
