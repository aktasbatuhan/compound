import { describe, expect, test } from "bun:test";
import type { ModelCallStep, OtherStep } from "@compound/contract";
import { DIAGNOSTICS, DIALECTS } from "../src/index";
import { expectContractValid, normalizeFixture } from "./helpers";

describe("blob export: traces + observations joined on trace_id", () => {
  const { traces, report } = normalizeFixture("blob-join.jsonl", { projectId: undefined });
  const normalized = traces[0];
  if (normalized === undefined) throw new Error("expected one trace");
  const trace = normalized.trace;

  test("snake_case records join into one contract-valid trace", () => {
    expectContractValid(traces);
    expect(traces).toHaveLength(1);
    expect(report.surface).toBe("blob_join");
    expect(report.casing).toEqual(["snake_case"]);
  });

  test("project id falls back to the record's project_id for the trace_id prefix", () => {
    expect(trace.trace_id).toBe("langfuse:proj-blob:tr-blob-1");
    expect(trace.source.source_ids).toEqual({ trace_id: "tr-blob-1", project_id: "proj-blob" });
  });

  test("blob text timestamps are read as UTC", () => {
    expect(trace.started_at).toBe("2026-07-19T09:12:03.481Z");
    expect(trace.ended_at).toBe("2026-07-19T09:12:11.480Z");
  });

  test("snake_case trace attributes map across", () => {
    expect(trace.task_key).toBe("reports.nightly");
    expect(trace.metadata).toEqual({ cron: "0 3 * * *" });
    expect(trace.user_ref).toBe("user-77");
    expect(trace.session_id).toBeNull();
    expect(trace.release).toBe("app-v2.13.1");
    expect(trace.environment).toBe("staging");
  });

  test("legacy functions become tool defs with a params note", () => {
    const gen = trace.steps[0] as ModelCallStep;
    expect(gen.model).toBe("gpt-4o-mini");
    expect(gen.tools_available).toEqual([
      {
        name: "fetch_incidents",
        description: "Fetch incidents in a window.",
        parameters: { type: "object", properties: { since: { type: "string" } } },
      },
    ]);
    expect(gen.params).toEqual({
      temperature: 0.1,
      _legacy_functions: true,
      _legacy_function_call: "auto",
    });
    expect(report.dialects).toContain(DIALECTS.legacyFunctions);
  });

  test("unparseable tool arguments are kept raw and flagged", () => {
    const gen = trace.steps[0] as ModelCallStep;
    expect(gen.output?.tool_calls).toEqual([
      { id: "call_bad", name: "fetch_incidents", arguments: { _raw: '{"since": ' } },
    ]);
    expect(normalized.diagnostics).toContain(DIAGNOSTICS.toolArgumentsUnparseable);
  });

  test("deprecated usage and cost_details map across", () => {
    const gen = trace.steps[0] as ModelCallStep;
    expect(gen.usage).toEqual({
      input_tokens: 210,
      output_tokens: 15,
      reasoning_tokens: null,
      cached_input_tokens: null,
      total_tokens: 225,
    });
    expect(gen.cost_usd).toBe(0.0009);
  });

  test("an unknown observation type is preserved as an other step and reported", () => {
    const mystery = trace.steps[1] as OtherStep;
    expect(mystery.type).toBe("other");
    expect(mystery.data).toEqual({
      type: "MYSTERY",
      level: "WARNING",
      status_message: "emitted by a newer SDK",
      metadata: { sdk: "langfuse-python 4.9.0" },
    });
    expect(report.unknownObservationTypes).toEqual(["MYSTERY"]);
    expect(normalized.diagnostics).toContain(DIAGNOSTICS.unknownObservationType);
  });

  test("the sole generation is focal even though its output holds tool_calls", () => {
    expect(trace.focal_step_id).toBe("obs-blob-gen");
  });

  test("malformed lines are rejected by line number without failing the file", () => {
    expect(report.rejected).toEqual([
      { line: 4, reason: "malformed_json_line" },
      { line: 5, reason: "record_not_an_object" },
    ]);
    expect(report.counts.tracesNormalized).toBe(1);
  });

  test("scores with a null trace_id and scores for absent traces are skipped and counted", () => {
    expect(trace.outcome?.scores).toEqual([
      { name: "accuracy", value: 0.7, source: "judge", at: "2026-07-19T09:21:00.000Z" },
    ]);
    expect(report.skippedScores).toEqual({
      total: 2,
      byReason: { null_trace_id: 1, unmatched_trace: 1 },
    });
  });
});

describe("blob export: enriched observations_v2 alone", () => {
  const { traces, report } = normalizeFixture("blob-observations-v2.jsonl", {
    projectId: undefined,
  });
  const normalized = traces[0];
  if (normalized === undefined) throw new Error("expected one trace");
  const trace = normalized.trace;

  test("a trace is reconstructed from denormalized observation columns", () => {
    expectContractValid(traces);
    expect(report.surface).toBe("enriched_observations");
    expect(trace.trace_id).toBe("langfuse:proj-v2:tr-v2-1");
    expect(trace.started_at).toBe("2026-07-18T22:05:00.000Z");
    expect(trace.ended_at).toBe("2026-07-18T22:05:31.750Z");
    expect(trace.user_ref).toBe("user-v2-9");
    expect(trace.session_id).toBe("sess-v2-3");
    expect(trace.release).toBe("agent-v3.0.0");
    expect(trace.environment).toBe("production");
    expect(trace.tags).toEqual(["prod", "research"]);
  });

  test("with no trace record, task_key comes from the root observation metadata", () => {
    expect(trace.task_key).toBe("research.deep_dive");
    expect(trace.metadata).toBeNull();
  });

  test("AGENT maps to an other step and GENERATION to a model_call", () => {
    expect(trace.steps.map((step) => step.type)).toEqual(["other", "model_call"]);
    const gen = trace.steps[1] as ModelCallStep;
    expect(gen.parent_step_id).toBe("obs-v2-root");
    expect(gen.model).toBe("claude-opus-4-1");
    expect(gen.resolved_model).toBe("clx-model-opus41");
  });

  test("snake_case usage_details sum with cache buckets on the prompt side", () => {
    const gen = trace.steps[1] as ModelCallStep;
    expect(gen.usage).toEqual({
      input_tokens: 1200,
      output_tokens: 60,
      reasoning_tokens: 10,
      cached_input_tokens: 900,
      total_tokens: 1260,
    });
    expect(gen.cost_usd).toBe(0.004);
  });

  test("the sole generation is focal and nothing is uncertain", () => {
    expect(trace.focal_step_id).toBe("obs-v2-gen");
    expect(normalized.diagnostics).toEqual([]);
  });
});
