import { describe, expect, test } from "bun:test";
import type { ModelCallStep, ToolExecutionStep } from "@compound/contract";
import { DIAGNOSTICS, DIALECTS } from "../src/index";
import { expectContractValid, normalizeFixture } from "./helpers";

describe("API-shaped export: OpenAI wrapper with tools", () => {
  const { traces, report } = normalizeFixture("api-openai-tools.json");
  const normalized = traces[0];
  if (normalized === undefined) throw new Error("expected one trace");
  const trace = normalized.trace;

  test("emits one contract-valid trace", () => {
    expect(traces).toHaveLength(1);
    expectContractValid(traces);
    expect(trace.schema).toBe("compound.trace");
    expect(trace.schema_version).toBe(1);
  });

  test("trace_id is prefixed with importer and project", () => {
    expect(trace.trace_id).toBe("langfuse:proj-main:tr-9c1f");
    expect(trace.source).toEqual({
      importer: "langfuse",
      importer_version: "0.1.0",
      source_ids: { trace_id: "tr-9c1f", project_id: "proj-main" },
      exported_at: "2026-07-22T12:00:00.000Z",
    });
  });

  test("task_key comes from metadata and is consumed out of it", () => {
    expect(trace.task_key).toBe("support.invoice_triage");
    expect(trace.metadata).toEqual({ channel: "email" });
  });

  test("trace attributes map straight across", () => {
    expect(trace.started_at).toBe("2026-07-20T14:03:11.000Z");
    expect(trace.ended_at).toBe("2026-07-20T14:03:19.412Z");
    expect(trace.session_id).toBe("sess-991");
    expect(trace.user_ref).toBe("user-a41f9c");
    expect(trace.environment).toBe("production");
    expect(trace.release).toBe("app-v2.14.0");
    expect(trace.tags).toEqual(["support", "beta"]);
  });

  test("permissions come from options and redactions are always empty", () => {
    expect(trace.permissions).toEqual({ judging: true, optimization: true, fine_tuning: false });
    expect(trace.redactions).toEqual([]);
  });

  test("observations become ordered steps of the mapped types", () => {
    expect(trace.steps.map((step) => [step.step_id, step.type])).toEqual([
      ["gen-1", "model_call"],
      ["tool-1", "tool_execution"],
      ["gen-2", "model_call"],
    ]);
  });

  test("OpenAI wrapper input becomes messages plus tools_available", () => {
    const gen = trace.steps[0] as ModelCallStep;
    expect(gen.input).toEqual([
      {
        role: "system",
        content: "You triage customer invoice questions.",
        tool_calls: null,
        tool_call_id: null,
      },
      {
        role: "user",
        content: "Where is invoice INV-4471?",
        tool_calls: null,
        tool_call_id: null,
      },
    ]);
    expect(gen.tools_available).toEqual([
      {
        name: "lookup_invoice",
        description: "Look up an invoice by id.",
        parameters: {
          type: "object",
          properties: { invoice_id: { type: "string" } },
          required: ["invoice_id"],
        },
      },
    ]);
    expect(gen.model).toBe("gpt-4.1-mini");
    expect(gen.resolved_model).toBe("clx-model-gpt41mini");
    expect(gen.params).toEqual({ temperature: 0, max_tokens: 1024 });
    expect(gen.provider).toBeNull();
  });

  test("OpenAI tool_call arguments are JSON-parsed into an object", () => {
    const gen = trace.steps[0] as ModelCallStep;
    expect(gen.output?.tool_calls).toEqual([
      { id: "call_ab12", name: "lookup_invoice", arguments: { invoice_id: "INV-4471" } },
    ]);
    expect(report.dialects).toContain(DIALECTS.openaiToolCalls);
  });

  test("usageDetails buckets are summed into inclusive contract usage", () => {
    const gen = trace.steps[0] as ModelCallStep;
    expect(gen.usage).toEqual({
      input_tokens: 412,
      output_tokens: 28,
      reasoning_tokens: 8,
      cached_input_tokens: 32,
      total_tokens: 440,
    });
    expect(gen.cost_usd).toBe(0.00058);
  });

  test("usageDetails wins over the deprecated usage object", () => {
    const gen = trace.steps[0] as ModelCallStep;
    expect(gen.usage?.input_tokens).not.toBe(999);
    expect(report.dialects).toContain(DIALECTS.usageDetails);
  });

  test("tool_execution resolves call_ref against the generation output", () => {
    const tool = trace.steps[1] as ToolExecutionStep;
    expect(tool.name).toBe("lookup_invoice");
    expect(tool.call_ref).toEqual({ step_id: "gen-1", tool_call_id: "call_ab12" });
    expect(tool.output).toEqual({ status: "paid", paid_at: "2026-07-02", amount_usd: 1240 });
    expect(tool.replay_policy).toBeNull();
  });

  test("focal step is the final answer generation", () => {
    expect(trace.focal_step_id).toBe("gen-2");
    // The only diagnostic is the CATEGORICAL score the contract cannot hold;
    // nothing about the replayable call itself is uncertain.
    expect(normalized.diagnostics).toEqual([DIAGNOSTICS.nonNumericScoreSkipped]);
  });

  test("tool-role messages survive into the focal call's input", () => {
    const focal = trace.steps[2] as ModelCallStep;
    expect(focal.input[3]).toEqual({
      role: "tool",
      content: '{"status": "paid", "paid_at": "2026-07-02"}',
      tool_calls: null,
      tool_call_id: "call_ab12",
    });
  });

  test("scores map to outcome by source, corrections to feedback", () => {
    expect(trace.outcome?.scores).toEqual([
      { name: "helpfulness", value: 0.82, source: "judge", at: "2026-07-20T14:05:00.000Z" },
      { name: "thumbs", value: 1, source: "human", at: "2026-07-20T14:06:12.000Z" },
    ]);
    expect(trace.outcome?.feedback).toEqual([
      {
        kind: "correction",
        value: "Invoice INV-4471 was paid on 2026-07-02.",
        at: "2026-07-20T14:07:30.000Z",
      },
    ]);
  });

  test("a non-numeric score is skipped and counted", () => {
    expect(report.skippedScores.total).toBe(1);
    expect(report.skippedScores.byReason).toEqual({ non_numeric_value: 1 });
    expect(report.diagnosticReasons).toEqual({
      [DIAGNOSTICS.nonNumericScoreSkipped]: 1,
    });
  });

  test("report describes the detected surface and casing", () => {
    expect(report.surface).toBe("api");
    expect(report.casing).toEqual(["camelCase"]);
    expect(report.format).toBe("json_array");
    expect(report.counts).toEqual({
      recordsSeen: 8,
      recordsRejected: 0,
      traceRecords: 1,
      observationRecords: 3,
      scoreRecords: 4,
      tracesNormalized: 1,
    });
    expect(report.rejected).toEqual([]);
    expect(report.unknownObservationTypes).toEqual([]);
  });
});

describe("API-shaped export: LangChain callback trace", () => {
  const { traces, report } = normalizeFixture("api-langchain.json");
  const normalized = traces[0];
  if (normalized === undefined) throw new Error("expected one trace");
  const trace = normalized.trace;

  test("emits one contract-valid trace", () => {
    expectContractValid(traces);
  });

  test("task_key falls back to the task: tag", () => {
    expect(trace.task_key).toBe("agents.research");
    expect(trace.metadata).toEqual({ framework: "langchain", graph: "research" });
  });

  test("SPAN becomes an opaque other step keeping type, level and metadata", () => {
    const span = trace.steps[0];
    expect(span?.type).toBe("other");
    expect(span).toEqual({
      type: "other",
      step_id: "span-root",
      parent_step_id: null,
      name: "AgentExecutor",
      data: {
        type: "SPAN",
        level: "DEFAULT",
        status_message: null,
        metadata: { run_id: "b7c1" },
      },
    });
  });

  test("nesting is preserved through parent_step_id", () => {
    expect(trace.steps[1]?.parent_step_id).toBe("span-root");
  });

  test("the typed LangChain tool_calls field wins over additional_kwargs", () => {
    const gen = trace.steps[1] as ModelCallStep;
    expect(gen.output?.tool_calls).toEqual([
      {
        id: "call_lc_1",
        name: "web_search",
        arguments: { query: "compound eval tooling", top_k: 5 },
      },
    ]);
    expect(report.dialects).toContain(DIALECTS.langchainToolCalls);
    expect(report.dialects).not.toContain(DIALECTS.additionalKwargsToolCalls);
  });

  test("Anthropic cache buckets sum into input_tokens with cached as a subset", () => {
    const gen = trace.steps[1] as ModelCallStep;
    expect(gen.usage).toEqual({
      // 128 input + 1024 cache_read + 256 cache_creation; cached is the read bucket.
      input_tokens: 1408,
      output_tokens: 41,
      reasoning_tokens: null,
      cached_input_tokens: 1024,
      total_tokens: 1449,
    });
  });

  test("content parts normalize to contract text parts", () => {
    const gen = trace.steps[3] as ModelCallStep;
    expect(gen.output?.content).toEqual([
      { type: "text", text: "Three things changed in eval tooling this quarter." },
    ]);
  });

  test("focal is the final generation once the tool loop is resolvable", () => {
    const tool = trace.steps[2] as ToolExecutionStep;
    expect(tool.call_ref).toEqual({ step_id: "gen-a", tool_call_id: "call_lc_1" });
    expect(trace.focal_step_id).toBe("gen-b");
    expect(normalized.diagnostics).toEqual([]);
  });
});

describe("API-shaped export: v4 trace with no trace-level I/O", () => {
  const { traces, report } = normalizeFixture("api-v4-no-trace-io.jsonl");

  test("root-observation I/O carries the trace", () => {
    expectContractValid(traces);
    expect(traces).toHaveLength(2);
    const gen = traces[0]?.trace.steps[0] as ModelCallStep;
    expect(gen.input).toHaveLength(1);
    expect(gen.output?.content).toContain("Thanks so much");
  });

  test("task_key is null when neither metadata nor tags carry one", () => {
    expect(traces[0]?.trace.task_key).toBeNull();
    expect(traces[1]?.trace.task_key).toBeNull();
  });

  test("sole generation is the focal step", () => {
    expect(traces[0]?.trace.focal_step_id).toBe("gen-v4");
    expect(traces[1]?.trace.focal_step_id).toBe("gen-v4-2");
  });

  test("deprecated usage is used when usageDetails is absent", () => {
    const first = traces[0]?.trace.steps[0] as ModelCallStep;
    expect(first.usage).toEqual({
      input_tokens: 120,
      output_tokens: 45,
      reasoning_tokens: null,
      cached_input_tokens: null,
      total_tokens: 165,
    });
    const second = traces[1]?.trace.steps[0] as ModelCallStep;
    expect(second.usage).toEqual({
      input_tokens: 80,
      output_tokens: 22,
      reasoning_tokens: null,
      cached_input_tokens: null,
      total_tokens: 102,
    });
    expect(report.dialects).toContain(DIALECTS.deprecatedUsage);
  });

  test("JSONL of API records is read line by line", () => {
    expect(report.format).toBe("jsonl");
    expect(report.surface).toBe("api");
  });
});
