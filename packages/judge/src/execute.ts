/**
 * Run the judge model on one response, under the same money-safety rules as the
 * experiment runner: cache-first (a re-grade is $0), the hard cap enforced before
 * every call, and zero provider calls unless paid runs are enabled. A judge is
 * just another completion, so it reuses the ledger and the content-addressed
 * cache wholesale (docs/judges-v1.md, "Judge execution").
 */
import {
  type CompletionRequest,
  type CompletionResponse,
  chargeableCost,
  completionFingerprint,
  estimateCost,
} from "@compound/execution";
import {
  cacheCompletion,
  getCachedCompletion,
  releaseSpend,
  reserveSpend,
  settleSpend,
} from "@compound/storage";
import {
  buildPointwiseMessages,
  type JudgeVerdict,
  messageText,
  parseJudgeVerdict,
} from "./prompt";
import type { JudgeExecutionContext } from "./types";

export type JudgeGradeOutcome =
  | {
      status: "graded";
      verdict: JudgeVerdict;
      fingerprint: string;
      costUsd: number;
      cached: boolean;
      /**
       * True when a paid judge call returned no usage, so `costUsd` is the
       * conservative estimate rather than a measured figure (#3). The caller can
       * surface this so a run's judge cost isn't quietly part-guessed.
       */
      costEstimated: boolean;
    }
  | { status: "cache_miss_dry_run"; fingerprint: string }
  | { status: "unparseable"; fingerprint: string };

/**
 * Grade one response text with the judge. The rubric and prompt live inside the
 * request messages, so the completion fingerprint already covers the judge
 * model, the rubric, and the prompt version — a different pin is a different
 * cache entry, and re-grading the same pin is free.
 */
export async function judgeGradeOutput(
  ctx: JudgeExecutionContext,
  rubric: string,
  responseText: string,
): Promise<JudgeGradeOutcome> {
  const request: CompletionRequest = {
    model: ctx.judgeModel,
    messages: buildPointwiseMessages(rubric, responseText),
  };
  const fingerprint = completionFingerprint({
    provider: ctx.providerName,
    request,
    providerRevision: ctx.providerRevision,
  });

  const hit = getCachedCompletion(ctx.db, fingerprint);
  let response: CompletionResponse;
  let costUsd = 0;
  let cached = false;
  // A paid call whose provider reported no usage is billed at the estimate, so
  // its cost is guessed rather than measured (#3). A cache hit keeps the cost
  // recorded on the row (already measured-or-estimated when first paid).
  let costEstimated = false;

  if (hit !== null) {
    cached = true;
    response = {
      output: hit.outputJson as CompletionResponse["output"],
      usage: (hit.usageJson as CompletionResponse["usage"]) ?? null,
      finishReason: hit.finishReason,
      resolvedModel: hit.resolvedModel,
      latencyMs: hit.latencyMs ?? 0,
    };
    costUsd = hit.costUsd;
  } else if (!ctx.paid) {
    return { status: "cache_miss_dry_run", fingerprint };
  } else {
    const estimate = estimateCost(request, ctx.price);
    const judgeExperimentId = `judge:${ctx.judgeModel}`;
    const reservationId = reserveSpend(ctx.db, {
      fingerprint,
      estimatedCost: estimate,
      experimentId: judgeExperimentId,
      experimentCapUsd: ctx.experimentCapUsd,
      globalHardLimitUsd: ctx.globalHardLimitUsd,
    });
    try {
      response = await ctx.provider.complete(request);
    } catch (error) {
      releaseSpend(ctx.db, reservationId);
      throw error;
    }
    // A real judge call whose usage is null must ledger the estimate, never $0
    // (#3): the judge shares the same Flex provider and ledger as the runner, so
    // a $0 judge charge silently leaks the experiment cap and the global limit.
    const charge = chargeableCost(response.usage, request, ctx.price);
    costUsd = charge.costUsd;
    costEstimated = !charge.usageKnown;
    // Persist before parsing, so a paid call is never lost to a parse error.
    settleSpend(ctx.db, reservationId, { fingerprint, costUsd, experimentId: judgeExperimentId });
    cacheCompletion(ctx.db, {
      fingerprint,
      provider: ctx.providerName,
      model: ctx.judgeModel,
      resolvedModel: response.resolvedModel,
      params: null,
      output: response.output,
      usage: response.usage,
      finishReason: response.finishReason,
      latencyMs: response.latencyMs,
      costUsd,
    });
  }

  const verdict = parseJudgeVerdict(response.output);
  if (verdict === null) {
    // Loud, not silent: a judge that won't answer in the contract yields no
    // verdict. The caller treats this as an abstain on the case, never a score.
    void messageText(response.output);
    return { status: "unparseable", fingerprint };
  }
  return { status: "graded", verdict, fingerprint, costUsd, cached, costEstimated };
}
