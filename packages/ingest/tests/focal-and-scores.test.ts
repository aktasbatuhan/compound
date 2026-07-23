import { describe, expect, test } from "bun:test";
import type { ToolExecutionStep } from "@compound/contract";
import { DIAGNOSTICS } from "../src/index";
import { normalizeRecords } from "./helpers";

interface ObservationSpec {
  id: string;
  type: string;
  name?: string;
  startTime: string;
  input?: unknown;
  output?: unknown;
  metadata?: unknown;
}

function traceWith(observations: ObservationSpec[]): unknown[] {
  return [
    {
      id: "tr-focal",
      timestamp: "2026-07-22T09:00:00.000Z",
      metadata: {},
      tags: [],
      observations: observations.map((observation) => ({
        traceId: "tr-focal",
        parentObservationId: null,
        endTime: observation.startTime,
        model: "gpt-4.1-mini",
        modelParameters: {},
        level: "DEFAULT",
        environment: "production",
        ...observation,
      })),
      scores: [],
    },
  ];
}

function openaiToolCall(id: string, name: string): unknown {
  return {
    role: "assistant",
    content: null,
    tool_calls: [{ id, type: "function", function: { name, arguments: '{"q": "x"}' } }],
  };
}

describe("focal step selection", () => {
  test("rule 1: a sole model_call is focal", () => {
    const { traces } = normalizeRecords(
      traceWith([
        { id: "span", type: "SPAN", startTime: "2026-07-22T09:00:00.000Z" },
        {
          id: "gen",
          type: "GENERATION",
          startTime: "2026-07-22T09:00:01.000Z",
          input: [{ role: "user", content: "hi" }],
          output: "hello",
        },
      ]),
    );
    expect(traces[0]?.trace.focal_step_id).toBe("gen");
  });

  test("rule 2: the last generation with no tool_calls wins when the loop is replayable", () => {
    const { traces } = normalizeRecords(
      traceWith([
        {
          id: "gen-1",
          type: "GENERATION",
          startTime: "2026-07-22T09:00:01.000Z",
          input: [{ role: "user", content: "hi" }],
          output: openaiToolCall("call_1", "search"),
        },
        {
          id: "tool-1",
          type: "TOOL",
          name: "search",
          startTime: "2026-07-22T09:00:02.000Z",
          input: { q: "x" },
          output: { hits: 2 },
          metadata: { tool_call_id: "call_1" },
        },
        {
          id: "gen-2",
          type: "GENERATION",
          startTime: "2026-07-22T09:00:03.000Z",
          input: [{ role: "user", content: "hi" }],
          output: "there were 2 hits",
        },
      ]),
    );
    expect(traces[0]?.trace.focal_step_id).toBe("gen-2");
    expect(traces[0]?.diagnostics).toEqual([]);
  });

  test("rule 3: a tool_execution with no recorded output blocks focal selection", () => {
    const { traces } = normalizeRecords(
      traceWith([
        {
          id: "gen-1",
          type: "GENERATION",
          startTime: "2026-07-22T09:00:01.000Z",
          input: [{ role: "user", content: "hi" }],
          output: openaiToolCall("call_1", "search"),
        },
        {
          id: "tool-1",
          type: "TOOL",
          name: "search",
          startTime: "2026-07-22T09:00:02.000Z",
          input: { q: "x" },
          metadata: { tool_call_id: "call_1" },
        },
        {
          id: "gen-2",
          type: "GENERATION",
          startTime: "2026-07-22T09:00:03.000Z",
          input: [{ role: "user", content: "hi" }],
          output: "answer",
        },
      ]),
    );
    expect(traces[0]?.trace.focal_step_id).toBeNull();
    expect(traces[0]?.diagnostics).toContain(DIAGNOSTICS.noReplayableFocalCall);
  });

  test("rule 3: a last generation that still asks for tools is not focal", () => {
    const { traces } = normalizeRecords(
      traceWith([
        {
          id: "gen-1",
          type: "GENERATION",
          startTime: "2026-07-22T09:00:01.000Z",
          input: [{ role: "user", content: "hi" }],
          output: openaiToolCall("call_1", "search"),
        },
        {
          id: "gen-2",
          type: "GENERATION",
          startTime: "2026-07-22T09:00:03.000Z",
          input: [{ role: "user", content: "hi" }],
          output: openaiToolCall("call_2", "search"),
        },
      ]),
    );
    expect(traces[0]?.trace.focal_step_id).toBeNull();
  });

  test("a trace with no model_call has no focal step", () => {
    const { traces } = normalizeRecords(
      traceWith([{ id: "span", type: "SPAN", startTime: "2026-07-22T09:00:00.000Z" }]),
    );
    expect(traces[0]?.trace.focal_step_id).toBeNull();
    expect(traces[0]?.diagnostics).toContain(DIAGNOSTICS.noReplayableFocalCall);
  });

  test("steps are ordered by start time regardless of file order", () => {
    const { traces } = normalizeRecords(
      traceWith([
        {
          id: "later",
          type: "GENERATION",
          startTime: "2026-07-22T09:00:05.000Z",
          input: [{ role: "user", content: "hi" }],
          output: "b",
        },
        { id: "earlier", type: "SPAN", startTime: "2026-07-22T09:00:01.000Z" },
      ]),
    );
    expect(traces[0]?.trace.steps.map((step) => step.step_id)).toEqual(["earlier", "later"]);
  });
});

describe("tool call linkage", () => {
  test("a unique tool name links a tool_execution that carries no tool_call_id", () => {
    const { traces } = normalizeRecords(
      traceWith([
        {
          id: "gen-1",
          type: "GENERATION",
          startTime: "2026-07-22T09:00:01.000Z",
          input: [{ role: "user", content: "hi" }],
          output: openaiToolCall("call_1", "search"),
        },
        {
          id: "tool-1",
          type: "TOOL",
          name: "search",
          startTime: "2026-07-22T09:00:02.000Z",
          output: { hits: 1 },
        },
      ]),
    );
    const tool = traces[0]?.trace.steps[1] as ToolExecutionStep;
    expect(tool.call_ref).toEqual({ step_id: "gen-1", tool_call_id: "call_1" });
  });

  test("an ambiguous name is left unresolved rather than guessed", () => {
    const { traces } = normalizeRecords(
      traceWith([
        {
          id: "gen-1",
          type: "GENERATION",
          startTime: "2026-07-22T09:00:01.000Z",
          input: [{ role: "user", content: "hi" }],
          output: {
            role: "assistant",
            content: null,
            tool_calls: [
              { id: "call_1", type: "function", function: { name: "search", arguments: "{}" } },
              { id: "call_2", type: "function", function: { name: "search", arguments: "{}" } },
            ],
          },
        },
        {
          id: "tool-1",
          type: "TOOL",
          name: "search",
          startTime: "2026-07-22T09:00:02.000Z",
          output: { hits: 1 },
        },
        {
          id: "tool-2",
          type: "TOOL",
          name: "search",
          startTime: "2026-07-22T09:00:03.000Z",
          output: { hits: 2 },
        },
      ]),
    );
    const tools = traces[0]?.trace.steps.filter(
      (step): step is ToolExecutionStep => step.type === "tool_execution",
    );
    expect(tools?.map((step) => step.call_ref)).toEqual([null, null]);
    expect(traces[0]?.diagnostics).toContain(DIAGNOSTICS.unresolvedToolCallRef);
  });
});

describe("scores", () => {
  const trace = {
    id: "tr-scores",
    timestamp: "2026-07-22T09:00:00.000Z",
    metadata: {},
    tags: [],
    observations: [],
  };

  test("an unknown score source is skipped and counted", () => {
    const { traces, report } = normalizeRecords([
      {
        ...trace,
        scores: [
          {
            id: "s1",
            traceId: "tr-scores",
            name: "mystery",
            source: "PIPELINE",
            dataType: "NUMERIC",
            value: 1,
          },
        ],
      },
    ]);
    expect(traces[0]?.trace.outcome).toBeNull();
    expect(report.skippedScores).toEqual({ total: 1, byReason: { unknown_source: 1 } });
    expect(traces[0]?.diagnostics).toContain(DIAGNOSTICS.unknownScoreSource);
  });

  test("a trace with no scores has a null outcome", () => {
    const { traces } = normalizeRecords([{ ...trace, scores: [] }]);
    expect(traces[0]?.trace.outcome).toBeNull();
  });
});
