import { describe, expect, test } from "bun:test";
import { normalizeChatCompletion } from "../src/index";

const CHOICE = { message: { role: "assistant", content: "hi" }, finish_reason: "stop" };

describe("normalizeChatCompletion cached tokens (#34)", () => {
  test("captures cached tokens from an OpenAI-shaped usage payload", () => {
    const response = normalizeChatCompletion(
      "openrouter",
      {
        choices: [CHOICE],
        usage: {
          prompt_tokens: 1000,
          completion_tokens: 50,
          prompt_tokens_details: { cached_tokens: 800 },
        },
      },
      100,
    );
    expect(response.usage?.cached_input_tokens).toBe(800);
  });

  test("a reported 0 stays 0 — a working report of 'no cache hit'", () => {
    const response = normalizeChatCompletion(
      "openai",
      {
        choices: [CHOICE],
        usage: {
          prompt_tokens: 1000,
          completion_tokens: 50,
          prompt_tokens_details: { cached_tokens: 0 },
        },
      },
      100,
    );
    expect(response.usage?.cached_input_tokens).toBe(0);
  });

  test("yields NULL when the provider reports no cached-token field", () => {
    const response = normalizeChatCompletion(
      "doubleword",
      { choices: [CHOICE], usage: { prompt_tokens: 1000, completion_tokens: 50 } },
      100,
    );
    // null, not 0: "provider did not report" must stay distinguishable.
    expect(response.usage?.cached_input_tokens).toBeNull();
  });

  test("falls back to Anthropic-style cache_read_input_tokens (parse hook)", () => {
    const response = normalizeChatCompletion(
      "gateway",
      {
        choices: [CHOICE],
        usage: { prompt_tokens: 1000, completion_tokens: 50, cache_read_input_tokens: 640 },
      },
      100,
    );
    expect(response.usage?.cached_input_tokens).toBe(640);
  });
});
