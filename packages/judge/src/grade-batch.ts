/**
 * Grade a batch of response texts with the judge, on earned trust — the bridge
 * that lets GEPA optimize FUZZY tasks (docs/judges-v1.md + docs/optimization-v1.md).
 *
 * The single hard rule: an UNCALIBRATED judge grades nothing. Optimization must
 * never chase an untrusted opinion, so this refuses (returns `calibrated:false`
 * with no verdicts) rather than emit weak scores. When calibrated, each text is
 * graded through the same money-safe, cache-first path as everything else, so a
 * re-grade of an identical (judge, rubric, prompt, output) pin is $0.
 */
import { judgeGradeOutput } from "./execute";
import { judgeTrust } from "./trust";
import type { JudgeConfig, JudgeExecutionContext, JudgeTrust } from "./types";

export interface JudgeBatchItem {
  caseId: string;
  responseText: string;
}

export interface JudgeBatchVerdict {
  caseId: string;
  status: "graded" | "cache_miss_dry_run" | "unparseable";
  /** Present only when status is `graded`. */
  score?: number;
  reasoning?: string;
  costUsd?: number;
}

export interface JudgeBatchResult {
  trust: JudgeTrust;
  /** Empty when the judge is uncalibrated — the caller must refuse, not guess. */
  verdicts: JudgeBatchVerdict[];
  totalCostUsd: number;
}

/**
 * Grade `items` with the judge if — and only if — it is calibrated for its pin.
 * The caller inspects `trust.calibrated`: false means refuse the whole batch.
 */
export async function judgeGradeBatch(
  ctx: JudgeExecutionContext,
  judge: JudgeConfig,
  items: readonly JudgeBatchItem[],
  minCalibrationCases?: number,
): Promise<JudgeBatchResult> {
  const trust = judgeTrust(ctx.db, judge, minCalibrationCases);
  if (!trust.calibrated) {
    return { trust, verdicts: [], totalCostUsd: 0 };
  }

  const verdicts: JudgeBatchVerdict[] = [];
  let totalCostUsd = 0;
  for (const item of items) {
    const outcome = await judgeGradeOutput(ctx, judge.rubric, item.responseText);
    if (outcome.status === "graded") {
      // Only a fresh call is new spend; a cache hit is $0 (the ledger records
      // no second charge for the same pin).
      const newCost = outcome.cached ? 0 : outcome.costUsd;
      totalCostUsd += newCost;
      verdicts.push({
        caseId: item.caseId,
        status: "graded",
        score: outcome.verdict.score,
        reasoning: outcome.verdict.reasoning,
        costUsd: newCost,
      });
    } else {
      verdicts.push({ caseId: item.caseId, status: outcome.status });
    }
  }
  return { trust, verdicts, totalCostUsd };
}
