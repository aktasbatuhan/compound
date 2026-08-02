import { describe, expect, test } from "bun:test";
import type { Message } from "@compound/contract";
import type { CompletionRequest, CompletionResponse, Provider } from "@compound/execution";
import { totalSpendUsd } from "@compound/storage";
import { judgeGradeOutput } from "../src/execute";
import { freshDb, judgeCtx } from "./helpers";

/**
 * A judge provider that returns a valid verdict but NO usage — the shape a Flex
 * host gives when it omits token counts. The judge must still ledger the paid
 * call at the estimate, never $0 (#3): the judge shares the runner's ledger, so a
 * $0 charge silently leaks the experiment cap and the global limit.
 */
function noUsageJudge(score: number): Provider {
  return {
    name: "fake-judge",
    async complete(request: CompletionRequest): Promise<CompletionResponse> {
      return {
        output: {
          role: "assistant",
          content: JSON.stringify({ score, reasoning: "scripted" }),
        } as Message,
        usage: null,
        finishReason: "stop",
        resolvedModel: request.model,
        latencyMs: 1,
      };
    },
  };
}

describe("judge money-safety on null usage (#3)", () => {
  test("a paid judge call with no usage ledgers the estimate, not $0", async () => {
    const db = freshDb();
    const ctx = judgeCtx(db, noUsageJudge(1));
    const outcome = await judgeGradeOutput(ctx, "resolve the request", "a fine answer");
    expect(outcome.status).toBe("graded");
    if (outcome.status !== "graded") return;
    // The measured path is unavailable, so cost is the conservative estimate and
    // flagged as such — but it is strictly positive, and it is what got ledgered.
    expect(outcome.costEstimated).toBe(true);
    expect(outcome.costUsd).toBeGreaterThan(0);
    expect(totalSpendUsd(db)).toBeCloseTo(outcome.costUsd, 10);
  });

  test("re-grading the same pin is a $0 cache hit and adds no new spend", async () => {
    const db = freshDb();
    const ctx = judgeCtx(db, noUsageJudge(1));
    const first = await judgeGradeOutput(ctx, "resolve the request", "a fine answer");
    const spentAfterFirst = totalSpendUsd(db);
    const second = await judgeGradeOutput(ctx, "resolve the request", "a fine answer");
    expect(second.status).toBe("graded");
    if (second.status !== "graded") return;
    expect(second.cached).toBe(true);
    // A cache hit carries the cost recorded on the row but records no fresh spend.
    expect(second.costUsd).toBeCloseTo(first.status === "graded" ? first.costUsd : 0, 10);
    expect(totalSpendUsd(db)).toBeCloseTo(spentAfterFirst, 10);
  });
});
