import { describe, expect, test } from "bun:test";
import {
  FlexProvider,
  responsesOutputText,
  responsesToolCalls,
  toResponsesTools,
} from "../src/index";

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

describe("toResponsesTools", () => {
  test("flattens the chat-completions function envelope to Responses shape", () => {
    const chat = [
      {
        type: "function",
        function: { name: "dispute_charge", description: "d", parameters: { type: "object" } },
      },
    ];
    expect(toResponsesTools(chat)).toEqual([
      {
        type: "function",
        name: "dispute_charge",
        description: "d",
        parameters: { type: "object" },
      },
    ]);
  });

  test("passes through an already-flat tool", () => {
    const flat = [{ type: "function", name: "freeze_card", description: "f", parameters: {} }];
    expect(toResponsesTools(flat)).toEqual(flat);
  });
});

describe("responsesToolCalls", () => {
  test("parses a structured function_call output item", () => {
    const payload = {
      output: [
        {
          type: "function_call",
          call_id: "c1",
          name: "dispute_charge",
          arguments: '{"merchant":"ACME","amount":23}',
        },
      ],
    };
    expect(responsesToolCalls(payload)).toEqual([
      { id: "c1", name: "dispute_charge", arguments: { merchant: "ACME", amount: 23 } },
    ]);
  });

  test("falls back to parsing a <tool_call> text block", () => {
    const payload = {
      output_text:
        'Sure.\n<tool_call>\n{"name": "freeze_card", "arguments": {"card": "debit"}}\n</tool_call>',
    };
    expect(responsesToolCalls(payload)).toEqual([
      { id: "", name: "freeze_card", arguments: { card: "debit" } },
    ]);
  });

  test("returns no calls when there are none", () => {
    expect(responsesToolCalls({ output_text: "just a plain answer" })).toEqual([]);
  });

  test("malformed arguments never become a guess", () => {
    const payload = {
      output: [{ type: "function_call", call_id: "c", name: "t", arguments: "{not json" }],
    };
    expect(responsesToolCalls(payload)[0]?.arguments).toEqual({ _raw: "{not json" });
  });
});

describe("FlexProvider tools", () => {
  test("submits Responses-format tools and returns structured tool_calls", async () => {
    const { provider, calls } = makeProvider([
      {
        status: "completed",
        model: "zai-org/GLM-5.2-FP8",
        output: [
          {
            type: "function_call",
            call_id: "c1",
            name: "dispute_charge",
            arguments: '{"merchant":"ACME","amount":450}',
          },
        ],
        usage: { input_tokens: 10, output_tokens: 4 },
      },
    ]);
    const res = await provider.complete({
      model: "zai-org/GLM-5.2-FP8",
      messages: [{ role: "user", content: "dispute the ACME charge" }],
      tools: [
        {
          type: "function",
          function: { name: "dispute_charge", description: "d", parameters: { type: "object" } },
        },
      ],
    });
    const submit = calls.find((c) => c.method === "POST")?.body as {
      tools?: unknown[];
      tool_choice?: string;
    };
    expect(submit.tools).toEqual([
      {
        type: "function",
        name: "dispute_charge",
        description: "d",
        parameters: { type: "object" },
      },
    ]);
    expect(submit.tool_choice).toBe("auto");
    expect(res.output.tool_calls?.[0]?.name).toBe("dispute_charge");
    expect(res.output.tool_calls?.[0]?.arguments).toEqual({ merchant: "ACME", amount: 450 });
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
