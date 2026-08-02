import { describe, expect, test } from "bun:test";
import type { Trace } from "@compound/contract";
import { validate } from "@compound/contract";
import { caseIdFor, extractCase, NotExtractableError } from "../src/extract";

function baseTrace(overrides: Partial<Trace> = {}): Trace {
  const raw = {
    schema: "compound.trace",
    schema_version: 1,
    trace_id: "langfuse:tr-1",
    task_key: "support",
    started_at: "2026-07-23T10:00:00Z",
    source: { importer: "test", importer_version: "1", source_ids: {} },
    steps: [
      {
        type: "model_call",
        step_id: "s1",
        model: "gpt-4o",
        input: [{ role: "user", content: "refund window?" }],
        output: { role: "assistant", content: "Thirty days." },
        usage: { input_tokens: 5, output_tokens: 2 },
        started_at: "2026-07-23T10:00:00Z",
        ended_at: "2026-07-23T10:00:01Z",
      },
    ],
    focal_step_id: "s1",
    permissions: { judging: true, optimization: true, fine_tuning: false },
    redactions: [],
    ...overrides,
  };
  const result = validate(raw);
  if (result.class === "rejected") throw new Error("bad base trace");
  return result.trace;
}

describe("extractCase", () => {
  test("extracts a replayable case with the focal request as input", () => {
    const c = extractCase(baseTrace(), { contentHash: "h1" });
    expect(c.taskKey).toBe("support");
    expect(c.sourceTraceId).toBe("langfuse:tr-1");
    expect(c.contentHash).toBe("h1");
    expect(c.input.model).toBe("gpt-4o");
    expect(c.input.input).toEqual([{ role: "user", content: "refund window?" }]);
    expect(c.caseId).toBe(caseIdFor("support", "h1"));
  });

  test("defaults provenance to observed_output — the incumbent's answer is evidence, not truth", () => {
    const c = extractCase(baseTrace(), { contentHash: "h1" });
    expect(c.provenance).toBe("observed_output");
    expect(c.expected).toEqual({ role: "assistant", content: "Thirty days." });
  });

  test("types a deterministic outcome as deterministic_outcome", () => {
    const c = extractCase(
      baseTrace({ outcome: { deterministic: { status: "success", detail: "exit 0" } } }),
      { contentHash: "h1" },
    );
    expect(c.provenance).toBe("deterministic_outcome");
  });

  test("types real user feedback as user_feedback", () => {
    const c = extractCase(baseTrace({ outcome: { feedback: [{ kind: "thumbs", value: "up" }] } }), {
      contentHash: "h1",
    });
    expect(c.provenance).toBe("user_feedback");
  });

  test("prefers a deterministic outcome over feedback", () => {
    const c = extractCase(
      baseTrace({
        outcome: {
          deterministic: { status: "failure" },
          feedback: [{ kind: "thumbs", value: "up" }],
        },
      }),
      { contentHash: "h1" },
    );
    expect(c.provenance).toBe("deterministic_outcome");
  });

  test("NEVER assigns human_golden or synthetic_label at extraction", () => {
    // Whatever the trace carries, extraction cannot mint a golden.
    for (const outcome of [
      undefined,
      { deterministic: { status: "success" as const } },
      { feedback: [{ kind: "correction" as const, value: "fixed" }] },
    ]) {
      const c = extractCase(baseTrace({ outcome }), { contentHash: crypto.randomUUID() });
      expect(c.provenance).not.toBe("human_golden");
      expect(c.provenance).not.toBe("synthetic_label");
    }
  });

  // An agentic trace: a focal model call plus recorded tool executions the
  // application ran. Extraction must script those as recorded_tool_results (#6),
  // or an imported agentic case reaches --agentic with an empty script.
  function agenticTrace(): Trace {
    const raw = {
      schema: "compound.trace",
      schema_version: 1,
      trace_id: "langfuse:agentic-1",
      task_key: "support",
      started_at: "2026-07-23T10:00:00Z",
      source: { importer: "test", importer_version: "1", source_ids: {} },
      steps: [
        {
          type: "model_call",
          step_id: "s1",
          model: "gpt-4o",
          input: [{ role: "user", content: "dispute my $23 charge" }],
          output: { role: "assistant", content: "Done." },
        },
        {
          type: "tool_execution",
          step_id: "t1",
          name: "get_charge",
          call_ref: { step_id: "s1", tool_call_id: "c1" },
          input: { id: "ch_1" },
          output: { amount: 23 },
        },
        {
          type: "tool_execution",
          step_id: "t2",
          name: "dispute_charge",
          call_ref: { step_id: "s1", tool_call_id: "c2" },
          input: { amount: 23 },
          output: "ok",
        },
      ],
      focal_step_id: "s1",
      permissions: { judging: true, optimization: true, fine_tuning: false },
      redactions: [],
    };
    const result = validate(raw);
    if (result.class === "rejected") throw new Error("bad agentic trace");
    return result.trace;
  }

  test("scripts recorded tool executions as replay results, in order (#6)", () => {
    const c = extractCase(agenticTrace(), { contentHash: "h-agentic" });
    expect(c.input.recorded_tool_results).toEqual([
      { tool: "get_charge", result: '{"amount":23}' },
      { tool: "dispute_charge", result: "ok" },
    ]);
  });

  test("a non-agentic trace carries no recorded_tool_results (unchanged shape)", () => {
    const c = extractCase(baseTrace(), { contentHash: "h1" });
    expect(c.input.recorded_tool_results).toBeUndefined();
  });

  test("disambiguates by arguments when a tool runs more than once (#6)", () => {
    const base = agenticTrace();
    const raw = {
      ...base,
      steps: [
        base.steps[0],
        {
          type: "tool_execution",
          step_id: "t1",
          name: "get_order",
          input: { id: "A" },
          output: "orderA",
        },
        {
          type: "tool_execution",
          step_id: "t2",
          name: "get_order",
          input: { id: "B" },
          output: "orderB",
        },
      ],
    };
    const result = validate(raw);
    if (result.class === "rejected") throw new Error("unexpected reject");
    const c = extractCase(result.trace, { contentHash: "h-dup" });
    expect(c.input.recorded_tool_results).toEqual([
      { tool: "get_order", arguments: { id: "A" }, result: "orderA" },
      { tool: "get_order", arguments: { id: "B" }, result: "orderB" },
    ]);
  });

  test("allows a null expected output (assertion-gradeable case)", () => {
    const noOutput = baseTrace();
    // The contract permits a focal call with an error and no output.
    const raw = {
      ...noOutput,
      steps: [{ ...noOutput.steps[0], output: null, error: "timeout" }],
    };
    const result = validate(raw);
    if (result.class === "rejected") throw new Error("unexpected reject");
    const c = extractCase(result.trace, { contentHash: "h1" });
    expect(c.expected).toBeNull();
    expect(c.provenance).toBe("observed_output");
  });

  test("refuses a trace with no task_key", () => {
    expect(() => extractCase(baseTrace({ task_key: null }), { contentHash: "h1" })).toThrow(
      NotExtractableError,
    );
  });

  test("refuses a trace with no focal step", () => {
    const raw = { ...baseTrace(), focal_step_id: null };
    const result = validate(raw);
    // This is a diagnostic trace; curation must not turn it into a case.
    const trace = result.class === "rejected" ? null : result.trace;
    expect(trace).not.toBeNull();
    expect(() => extractCase(trace as Trace, { contentHash: "h1" })).toThrow(NotExtractableError);
  });

  test("caseId is stable for the same task and content hash, distinct otherwise", () => {
    expect(caseIdFor("support", "h1")).toBe(caseIdFor("support", "h1"));
    expect(caseIdFor("support", "h1")).not.toBe(caseIdFor("billing", "h1"));
    expect(caseIdFor("support", "h1")).not.toBe(caseIdFor("support", "h2"));
  });
});
