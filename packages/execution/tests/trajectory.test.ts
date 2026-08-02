import { describe, expect, test } from "bun:test";
import type { Message } from "@compound/contract";
import type { CompletionRequest, CompletionResponse, Provider } from "../src/provider";
import { type RecordedToolResult, runTrajectory } from "../src/trajectory";

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

  test("a blocked tool stops the trajectory (no side effect) but keeps the call it made", async () => {
    const provider = new ScriptedProvider([toolCall("c1", "issue_refund", { amount: 23 })]);
    const result = await runTrajectory(provider, {
      request: baseRequest,
      policy: { default: "recorded", perTool: { issue_refund: "blocked" } },
    });
    // The stop is reported, not thrown — so the caller can ledger the real call
    // that the model already made before we saw the blocked tool.
    expect(result.stop).toEqual({ reason: "tool_blocked", tool: "issue_refund" });
    expect(result.turns).toBe(1);
    expect(provider.calls).toHaveLength(1);
    // No tool result was ever replayed for the blocked call.
    expect(result.transcript.some((m) => m.role === "tool")).toBe(false);
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

  test("a recorded run without a matching result stops with a clear reason", async () => {
    const provider = new ScriptedProvider([toolCall("c1", "unknown_tool", {})]);
    const result = await runTrajectory(provider, {
      request: baseRequest,
      policy: { default: "recorded" },
    });
    expect(result.stop).toEqual({ reason: "missing_recorded_result", tool: "unknown_tool" });
    expect(result.truncated).toBe(false);
  });

  test("live_read_only is refused in v1 (documented follow-up)", async () => {
    const provider = new ScriptedProvider([toolCall("c1", "read_orders", {})]);
    const result = await runTrajectory(provider, {
      request: baseRequest,
      policy: { default: "live_read_only" },
    });
    expect(result.stop).toEqual({
      reason: "unsupported_policy",
      tool: "read_orders",
      policy: "live_read_only",
    });
  });

  test("argument-specific matching ignores key order (#10)", async () => {
    // The recorded arguments and the call's arguments carry the same keys in a
    // different order — a structural match, not a brittle JSON.stringify match.
    const provider = new ScriptedProvider([
      toolCall("c1", "book", { row: 1, col: 2 }),
      answer("ok"),
    ]);
    const recorded: RecordedToolResult[] = [
      { tool: "book", arguments: { col: 2, row: 1 }, result: '{"booked":true}' },
    ];
    const result = await runTrajectory(provider, {
      request: baseRequest,
      recordedToolResults: recorded,
      policy: { default: "recorded" },
    });
    expect(result.stop.reason).toBe("answered");
    expect(result.transcript.find((m) => m.role === "tool")?.content).toBe('{"booked":true}');
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

  test("onTurnComplete fires for every real call — including the blocked stop turn", async () => {
    // The model calls a blocked tool on turn 1. The call is real and must be
    // reported so the caller can ledger it (#23, money-safety) even though the
    // trajectory then stops with no gradeable outcome.
    const provider = new ScriptedProvider([toolCall("c1", "issue_refund", { amount: 23 })]);
    const completedTurns: number[] = [];
    const checkedTurns: number[] = [];
    const result = await runTrajectory(provider, {
      request: baseRequest,
      policy: { default: "recorded", perTool: { issue_refund: "blocked" } },
      beforeTurn: (turn) => checkedTurns.push(turn),
      onTurnComplete: (turn) => completedTurns.push(turn),
    });
    expect(result.stop.reason).toBe("tool_blocked");
    expect(checkedTurns).toEqual([1]); // budget checked before the one call
    expect(completedTurns).toEqual([1]); // and the one real call was reported
  });

  test("a beforeTurn that throws aborts the run after ledgering prior turns", async () => {
    // Simulates a budget error on turn 2: turn 1's call was made and reported,
    // then the check before turn 2 throws — no call is lost, none fabricated.
    const provider = new ScriptedProvider([toolCall("c1", "loop", {}), toolCall("c2", "loop", {})]);
    const completedTurns: number[] = [];
    await expect(
      runTrajectory(provider, {
        request: baseRequest,
        recordedToolResults: [{ tool: "loop", result: "{}" }],
        policy: { default: "recorded" },
        onTurnComplete: (turn) => completedTurns.push(turn),
        beforeTurn: (turn) => {
          if (turn === 2) throw new Error("budget exceeded");
        },
      }),
    ).rejects.toThrow("budget exceeded");
    expect(provider.calls).toHaveLength(1); // turn 2 never called the provider
    expect(completedTurns).toEqual([1]); // turn 1's real call was ledgered
  });

  test("hitting the turn budget stops the run and marks it truncated", async () => {
    // A provider that always calls a tool never answers → cut off at maxTurns.
    const provider = new ScriptedProvider([toolCall("c1", "loop", {})]);
    const result = await runTrajectory(provider, {
      request: baseRequest,
      // `mocked` answers every turn's tool call, so nothing runs out and the run
      // truncates on the turn budget — a `recorded` result is consumed once (#8),
      // which would instead stop the loop at turn 2 with missing_recorded_result.
      policy: { default: "mocked" },
      maxTurns: 3,
    });
    expect(result.turns).toBe(3);
    expect(result.truncated).toBe(true);
    expect(result.toolCalls).toHaveLength(3);
  });

  test("a recorded result with pinned arguments does not answer a different-arg call (#8)", async () => {
    // The trace recorded dispute_charge({amount:23})→ok. A candidate that calls
    // it with the WRONG amount must NOT receive the success result — replay stops
    // with missing_recorded_result rather than fabricating a matching outcome.
    const provider = new ScriptedProvider([toolCall("c1", "dispute_charge", { amount: 999 })]);
    const result = await runTrajectory(provider, {
      request: baseRequest,
      recordedToolResults: [{ tool: "dispute_charge", arguments: { amount: 23 }, result: "ok" }],
      policy: { default: "recorded" },
      maxTurns: 3,
    });
    expect(result.stop.reason).toBe("missing_recorded_result");
    expect(result.turns).toBe(1);
  });

  test("identical repeated calls draw recorded results in sequence, not the first twice (#8)", async () => {
    // A poll tool called twice with identical args must replay pending THEN done —
    // the recorded queue is consumed once per result, so the second call does not
    // re-serve the first.
    const provider = new ScriptedProvider([
      toolCall("c1", "poll", {}),
      toolCall("c2", "poll", {}),
      { role: "assistant", content: "finished" },
    ]);
    const result = await runTrajectory(provider, {
      request: baseRequest,
      recordedToolResults: [
        { tool: "poll", result: "pending" },
        { tool: "poll", result: "done" },
      ],
      policy: { default: "recorded" },
      maxTurns: 5,
    });
    expect(result.stop.reason).toBe("answered");
    const toolResults = result.transcript.filter((m) => m.role === "tool").map((m) => m.content);
    expect(toolResults).toEqual(["pending", "done"]);
  });
});
