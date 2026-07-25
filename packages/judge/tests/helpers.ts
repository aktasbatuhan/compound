import type { Message } from "@compound/contract";
import type {
  CompletionRequest,
  CompletionResponse,
  Provider,
  TokenPrice,
} from "@compound/execution";
import { type CompoundDatabase, createDatabase } from "@compound/storage";
import { messageText } from "../src/prompt";
import type { JudgeConfig, JudgeExecutionContext } from "../src/types";

export function freshDb(): CompoundDatabase {
  return createDatabase({ path: ":memory:", migrate: true });
}

export const PRICE: TokenPrice = { input: 1, output: 1 };

/**
 * A scripted judge provider: `scorer` maps the response text embedded in the
 * prompt to a score, so tests control the judge's opinion without a network.
 */
export function scriptedJudge(scorer: (responseText: string) => number): Provider {
  return {
    name: "fake-judge",
    async complete(request: CompletionRequest): Promise<CompletionResponse> {
      const userMsg = request.messages.find((m) => (m as { role?: string }).role === "user");
      const text = messageText((userMsg ?? null) as Message | null).replace(
        "RESPONSE TO SCORE:\n",
        "",
      );
      const score = scorer(text);
      return {
        output: {
          role: "assistant",
          content: JSON.stringify({ score, reasoning: "scripted" }),
        } as Message,
        usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
        finishReason: "stop",
        resolvedModel: request.model,
        latencyMs: 1,
      };
    },
  };
}

export function judgeCtx(
  db: CompoundDatabase,
  provider: Provider,
  overrides: Partial<JudgeExecutionContext> = {},
): JudgeExecutionContext {
  return {
    db,
    provider,
    providerName: "fake",
    judgeModel: "judge-model",
    price: PRICE,
    paid: true,
    experimentCapUsd: 10,
    globalHardLimitUsd: 100,
    ...overrides,
  };
}

export function judgeConfig(overrides: Partial<JudgeConfig> = {}): JudgeConfig {
  return {
    taskKey: "support",
    model: "judge-model",
    promptVersion: "v1",
    rubric: "The response resolves the user's request correctly and politely.",
    mode: "pointwise",
    calibrationThreshold: 0.6,
    decisionPoint: 0.5,
    ...overrides,
  };
}
