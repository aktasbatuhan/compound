import { describe, expect, test } from "bun:test";
import { FlexProvider, responsesOutputText } from "../src/index";

/** A scripted fetch: submit returns an id, then GETs return the queued states. */
function scriptedFetch(states: unknown[]) {
  const calls: Array<{ method: string; path: string; body?: unknown }> = [];
  let getIndex = 0;
  const impl = (async (url: string, init?: RequestInit) => {
    const path = url.replace(/^https?:\/\/[^/]+/, "");
    const method = init?.method ?? "GET";
    calls.push({ method, path, body: init?.body ? JSON.parse(init.body as string) : undefined });
    if (method === "POST") {
      return new Response(JSON.stringify({ id: "resp-1", status: "queued" }), { status: 200 });
    }
    const state = states[Math.min(getIndex, states.length - 1)];
    getIndex += 1;
    return new Response(JSON.stringify(state), { status: 200 });
  }) as unknown as typeof fetch;
  return { impl, calls };
}

function makeProvider(states: unknown[]) {
  const { impl, calls } = scriptedFetch(states);
  const provider = new FlexProvider({
    name: "doubleword",
    baseUrl: "https://api.doubleword.ai/v1",
    apiKey: "test-key",
    pollIntervalMs: 1,
    sleep: async () => {},
    fetchImpl: impl,
  });
  return { provider, calls };
}

describe("responsesOutputText", () => {
  test("prefers the direct output_text field", () => {
    expect(responsesOutputText({ output_text: "hello" })).toBe("hello");
  });

  test("falls back to concatenating output item text parts", () => {
    const payload = {
      output: [
        {
          content: [
            { type: "output_text", text: "a" },
            { type: "text", text: "b" },
          ],
        },
      ],
    };
    expect(responsesOutputText(payload)).toBe("ab");
  });

  test("ignores non-text content", () => {
    const payload = { output: [{ content: [{ type: "reasoning", text: "hidden" }] }] };
    expect(responsesOutputText(payload)).toBe("");
  });
});

describe("FlexProvider.complete", () => {
  test("submits a background flex request then polls to completion", async () => {
    const { provider, calls } = makeProvider([
      { id: "resp-1", status: "in_progress" },
      { id: "resp-1", status: "in_progress" },
      {
        id: "resp-1",
        status: "completed",
        output_text: '{"ok":true}',
        model: "zai-org/GLM-5.2-FP8",
        usage: {
          input_tokens: 12,
          output_tokens: 5,
          output_tokens_details: { reasoning_tokens: 2 },
        },
      },
    ]);

    const result = await provider.complete({
      model: "zai-org/GLM-5.2-FP8",
      messages: [{ role: "user", content: "hi" }],
      params: { max_tokens: 4096, reasoning_effort: "minimal" },
    });

    // The POST used the flex background route with the right knobs.
    const post = calls.find((c) => c.method === "POST");
    if (post === undefined) throw new Error("expected a POST submission");
    const postBody = post.body as Record<string, unknown>;
    expect(post.path).toBe("/v1/responses");
    expect(postBody.background).toBe(true);
    expect(postBody.service_tier).toBe("flex");
    expect(postBody.max_output_tokens).toBe(4096);
    expect(postBody.reasoning).toEqual({ effort: "minimal" });

    // It polled until terminal, then returned the normalized completion.
    expect(calls.filter((c) => c.method === "GET").length).toBeGreaterThanOrEqual(2);
    expect(result.output.content).toBe('{"ok":true}');
    expect(result.usage).toEqual({ input_tokens: 12, output_tokens: 5, reasoning_tokens: 2 });
    expect(result.resolvedModel).toBe("zai-org/GLM-5.2-FP8");
  });

  test("returns immediately when the first poll is already complete", async () => {
    const { provider, calls } = makeProvider([
      {
        id: "resp-1",
        status: "completed",
        output_text: "done",
        usage: { input_tokens: 1, output_tokens: 1 },
      },
    ]);
    const result = await provider.complete({
      model: "m",
      messages: [{ role: "user", content: "x" }],
    });
    expect(result.output.content).toBe("done");
    expect(calls.filter((c) => c.method === "GET")).toHaveLength(1);
  });

  test("throws when the background response ends in a non-completed status", async () => {
    const { provider } = makeProvider([
      { id: "resp-1", status: "failed", error: { message: "capacity" } },
    ]);
    await expect(
      provider.complete({ model: "m", messages: [{ role: "user", content: "x" }] }),
    ).rejects.toThrow(/failed/);
  });
});
