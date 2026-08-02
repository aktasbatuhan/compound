import { describe, expect, test } from "bun:test";
import type { Message } from "@compound/contract";
import type { CompletionRequest, CompletionResponse, Provider } from "../src/provider";
import {
  MissingRecordedResultError,
  type RecordedToolResult,
  runTrajectory,
  ToolBlockedError,
  UnsupportedReplayPolicyError,
} from "../src/trajectory";

/** A provider that returns a scripted sequence of outputs (clamped to the last). */
class ScriptedProvider implements Provider {
  readonly name = "scripted";
  calls: CompletionRequest[] = [];
  private i = 0;
  constructor(private readonly outputs: Message[]) {}
  async complete(request: CompletionRequest): Promise<CompletionResponse> {
    this.calls.push(request);
    const output = this.outputs[Math.min(this.i, this.outputs.length - 1)] as Message;
    this.i += 1;
    return {
      output,
      usage: { input_tokens: 10, output_tokens: 5 },
      finishReason: "stop",
      resolvedModel: request.model,
      latencyMs: 2,
    };
  }
}

const toolCall = (id: string, name: string, args: Record<string, string | number>): Message => ({
  role: "assistant",
  content: null,
  tool_calls: [{ id, name, arguments: args }],
});
const answer = (text: string): Message => ({ role: "assistant", content: text });

const baseRequest: CompletionRequest = {
  model: "cand",
  messages: [{ role: "user", content: "dispute my $23 charge" }],
};

describe("runTrajectory", () => {
  test("drives a recorded multi-turn run and aggregates the trajectory", async () => {
    const provider = new ScriptedProvider([
      toolCall("c1", "get_charge", { id: "ch_1" }),
      toolCall("c2", "dispute_charge", { amount: 23 }),
      answer("Done — I disputed the $23 charge."),
    ]);
    const recorded: RecordedToolResult[] = [
      { tool: "get_charge", result: '{"amount":23}' },
      { tool: "dispute_charge", result: '{"ok":true}' },
    ];
    const result = await runTrajectory(provider, {
      request: baseRequest,
      recordedToolResults: recorded,
      policy: { default: "recorded" },
    });

    expect(result.turns).toBe(3);
    expect(result.truncated).toBe(false);
    // Every tool call across turns is collected, in order.
    expect(result.toolCalls.map((c) => c.name)).toEqual(["get_charge", "dispute_charge"]);
    // The aggregate graded output carries the final text AND all tool calls, so
    // the ordinary tool assertions grade the whole trajectory.
    expect(result.gradedOutput.content).toBe("Done — I disputed the $23 charge.");
    expect(result.gradedOutput.tool_calls?.map((c) => c.name)).toEqual([
      "get_charge",
      "dispute_charge",
    ]);
    // Usage and latency are summed across the three model calls.
    expect(result.usage.output_tokens).toBe(15);
    expect(result.latencyMs).toBe(6);
    // The transcript replays tool results back to the model as tool messages.
    const toolMsgs = result.transcript.filter((m) => m.role === "tool");
    expect(toolMsgs.map((m) => m.content)).toEqual(['{"amount":23}', '{"ok":true}']);
  });

  test("a blocked tool refuses to run during replay (no side effect)", async () => {
    const provider = new ScriptedProvider([toolCall("c1", "issue_refund", { amount: 23 })]);
    await expect(
      runTrajectory(provider, {
        request: baseRequest,
        policy: { default: "recorded", perTool: { issue_refund: "blocked" } },
      }),
    ).rejects.toBeInstanceOf(ToolBlockedError);
  });

  test("a mocked tool is answered with the stub and the run continues", async () => {
    const provider = new ScriptedProvider([toolCall("c1", "search", { q: "x" }), answer("ok")]);
    const result = await runTrajectory(provider, {
      request: baseRequest,
      policy: { default: "mocked" },
      mockResult: '{"stub":true}',
    });
    expect(result.turns).toBe(2);
    expect(result.transcript.find((m) => m.role === "tool")?.content).toBe('{"stub":true}');
  });

  test("a recorded run without a matching result fails clearly", async () => {
    const provider = new ScriptedProvider([toolCall("c1", "unknown_tool", {})]);
    await expect(
      runTrajectory(provider, { request: baseRequest, policy: { default: "recorded" } }),
    ).rejects.toBeInstanceOf(MissingRecordedResultError);
  });

  test("live_read_only is refused in v1 (documented follow-up)", async () => {
    const provider = new ScriptedProvider([toolCall("c1", "read_orders", {})]);
    await expect(
      runTrajectory(provider, { request: baseRequest, policy: { default: "live_read_only" } }),
    ).rejects.toBeInstanceOf(UnsupportedReplayPolicyError);
  });

  test("argument-specific recorded results answer the matching call", async () => {
    const provider = new ScriptedProvider([toolCall("c1", "get_order", { id: "B" }), answer("ok")]);
    const recorded: RecordedToolResult[] = [
      { tool: "get_order", arguments: { id: "A" }, result: '{"who":"A"}' },
      { tool: "get_order", arguments: { id: "B" }, result: '{"who":"B"}' },
    ];
    const result = await runTrajectory(provider, {
      request: baseRequest,
      recordedToolResults: recorded,
      policy: { default: "recorded" },
    });
    expect(result.transcript.find((m) => m.role === "tool")?.content).toBe('{"who":"B"}');
  });

  test("hitting the turn budget stops the run and marks it truncated", async () => {
    // A provider that always calls a tool never answers → cut off at maxTurns.
    const provider = new ScriptedProvider([toolCall("c1", "loop", {})]);
    const result = await runTrajectory(provider, {
      request: baseRequest,
      recordedToolResults: [{ tool: "loop", result: "{}" }],
      policy: { default: "recorded" },
      maxTurns: 3,
    });
    expect(result.turns).toBe(3);
    expect(result.truncated).toBe(true);
    expect(result.toolCalls).toHaveLength(3);
  });
});
