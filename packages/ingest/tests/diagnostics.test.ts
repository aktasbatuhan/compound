import { describe, expect, test } from "bun:test";
import type { ModelCallStep, ToolExecutionStep } from "@compound/contract";
import { DIAGNOSTICS } from "../src/index";
import {
  expectContractValid,
  normalizeFixture,
  normalizeRecords,
  traceWithGeneration,
} from "./helpers";

describe("diagnostic cases fixture", () => {
  const { traces, report } = normalizeFixture("api-diagnostic-cases.json");
  const normalized = traces[0];
  if (normalized === undefined) throw new Error("expected one trace");
  const trace = normalized.trace;

  test("still emits a contract-valid trace", () => {
    expectContractValid(traces);
  });

  test("an unrecognized generation input yields an empty input plus a reason", () => {
    const gen = trace.steps[0] as ModelCallStep;
    expect(gen.input).toEqual([]);
    expect(normalized.diagnostics).toContain(DIAGNOSTICS.unparseableGenerationInput);
  });

  test("a non-JSON arguments string is kept raw under _raw", () => {
    const gen = trace.steps[0] as ModelCallStep;
    expect(gen.output?.tool_calls?.[0]?.arguments).toEqual({ _raw: "SELECT * FROM invoices" });
    expect(normalized.diagnostics).toContain(DIAGNOSTICS.toolArgumentsUnparseable);
  });

  test("an ERROR-level observation carries its statusMessage as the step error", () => {
    const tool = trace.steps[1] as ToolExecutionStep;
    expect(tool.error).toBe("tool timed out");
    expect(tool.output).toBeUndefined();
  });

  test("an unmatchable tool_execution keeps a null call_ref", () => {
    const tool = trace.steps[1] as ToolExecutionStep;
    expect(tool.call_ref).toBeNull();
    expect(normalized.diagnostics).toContain(DIAGNOSTICS.unresolvedToolCallRef);
  });

  test("focal is null when the tool loop is not replayable", () => {
    expect(trace.focal_step_id).toBeNull();
    expect(normalized.diagnostics).toContain(DIAGNOSTICS.noReplayableFocalCall);
  });

  test("an unknown role is dropped and a legacy function role becomes tool", () => {
    const gen = trace.steps[2] as ModelCallStep;
    expect(gen.input.map((message) => message.role)).toEqual(["user", "tool"]);
    expect(gen.input[1]?.tool_call_id).toBeNull();
    expect(normalized.diagnostics).toContain(DIAGNOSTICS.unknownMessageRole);
    expect(normalized.diagnostics).toContain(DIAGNOSTICS.legacyFunctionRole);
  });

  test("non-text content becomes an unsupported part", () => {
    const gen = trace.steps[2] as ModelCallStep;
    expect(gen.input[0]?.content).toEqual([
      { type: "text", text: "What does this screenshot show?" },
      { type: "unsupported", media_type: "image_url" },
    ]);
    expect(normalized.diagnostics).toContain(DIAGNOSTICS.unsupportedContentPart);
  });

  test("diagnostics are sorted, deduplicated, and counted in the report", () => {
    expect(normalized.diagnostics).toEqual([...normalized.diagnostics].sort());
    expect(new Set(normalized.diagnostics).size).toBe(normalized.diagnostics.length);
    for (const reason of normalized.diagnostics) {
      expect(report.diagnosticReasons[reason]).toBe(1);
    }
  });
});

describe("generation input shapes", () => {
  test("a bare message array is accepted", () => {
    const { traces } = normalizeRecords(
      traceWithGeneration({ input: [{ role: "user", content: "hi" }] }),
    );
    const gen = traces[0]?.trace.steps[0] as ModelCallStep;
    expect(gen.input).toHaveLength(1);
    expect(traces[0]?.diagnostics).toEqual([]);
  });

  test("an absent input is empty but not a diagnostic on its own", () => {
    const { traces } = normalizeRecords(traceWithGeneration({ input: null }));
    const gen = traces[0]?.trace.steps[0] as ModelCallStep;
    expect(gen.input).toEqual([]);
    expect(traces[0]?.diagnostics).not.toContain(DIAGNOSTICS.unparseableGenerationInput);
  });

  test("an arbitrary object input is unparseable", () => {
    const { traces } = normalizeRecords(traceWithGeneration({ input: { prompt: "hi" } }));
    expect(traces[0]?.diagnostics).toContain(DIAGNOSTICS.unparseableGenerationInput);
  });

  test("an arbitrary object output yields a null output rather than a guess", () => {
    const { traces } = normalizeRecords(traceWithGeneration({ output: { rows: [1, 2] } }));
    const gen = traces[0]?.trace.steps[0] as ModelCallStep;
    expect(gen.output).toBeNull();
    expect(traces[0]?.diagnostics).toContain(DIAGNOSTICS.unparseableGenerationOutput);
  });

  test("a string output becomes the assistant message", () => {
    const { traces } = normalizeRecords(traceWithGeneration({ output: "done" }));
    const gen = traces[0]?.trace.steps[0] as ModelCallStep;
    expect(gen.output).toEqual({
      role: "assistant",
      content: "done",
      tool_calls: null,
      tool_call_id: null,
    });
  });

  test("a tool_call with no id gets a synthetic one and a reason", () => {
    const { traces } = normalizeRecords(
      traceWithGeneration({
        output: {
          role: "assistant",
          content: null,
          tool_calls: [{ name: "search", args: { q: "x" } }],
        },
      }),
    );
    const gen = traces[0]?.trace.steps[0] as ModelCallStep;
    expect(gen.output?.tool_calls?.[0]?.id).toBe("gen-1:m0:tc0");
    expect(traces[0]?.diagnostics).toContain(DIAGNOSTICS.toolCallMissingId);
  });
});

describe("usage fallbacks", () => {
  test("an empty usageDetails falls back to the deprecated usage object", () => {
    const { traces } = normalizeRecords(
      traceWithGeneration({ usageDetails: {}, usage: { input: 10, output: 4, total: 14 } }),
    );
    const gen = traces[0]?.trace.steps[0] as ModelCallStep;
    expect(gen.usage?.input_tokens).toBe(10);
  });

  test("no usage anywhere leaves usage null and cost null", () => {
    const { traces } = normalizeRecords(traceWithGeneration());
    const gen = traces[0]?.trace.steps[0] as ModelCallStep;
    expect(gen.usage).toBeNull();
    expect(gen.cost_usd).toBeNull();
  });

  test("total_tokens is summed when the source omits it", () => {
    const { traces } = normalizeRecords(
      traceWithGeneration({ usageDetails: { input: 7, output: 3 } }),
    );
    const gen = traces[0]?.trace.steps[0] as ModelCallStep;
    expect(gen.usage?.total_tokens).toBe(10);
  });

  test("cost comes from costDetails.total and is never recomputed", () => {
    const { traces } = normalizeRecords(
      traceWithGeneration({
        usageDetails: { input: 7, output: 3 },
        costDetails: { input: 1, output: 2, total: 9.5 },
      }),
    );
    const gen = traces[0]?.trace.steps[0] as ModelCallStep;
    expect(gen.cost_usd).toBe(9.5);
  });
});
