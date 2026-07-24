import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { TraceSchema } from "@compound/contract";
import type { NormalizedTrace, NormalizeOptions } from "../src/index";
import { DIAGNOSTICS, JSON_REJECTION_REASONS, normalizeJsonExport } from "../src/index";

const FIXTURES = join(import.meta.dir, "..", "fixtures");

function loadFixture(name: string): string {
  return readFileSync(join(FIXTURES, name), "utf8");
}

const OPTIONS: NormalizeOptions = {
  defaultPermissions: { judging: true, optimization: true, fine_tuning: false },
  importerVersion: "0.1.0",
  exportedAt: "2026-07-24T09:00:00.000Z",
};

function normalizeFixture(
  name: string,
  overrides: Partial<NormalizeOptions> = {},
): ReturnType<typeof normalizeJsonExport> {
  return normalizeJsonExport(loadFixture(name), { ...OPTIONS, ...overrides });
}

function normalizeRecords(
  records: unknown[],
  overrides: Partial<NormalizeOptions> = {},
): ReturnType<typeof normalizeJsonExport> {
  return normalizeJsonExport(records, { ...OPTIONS, ...overrides });
}

/** Every trace this importer emits must satisfy the contract schema. */
function expectContractValid(traces: NormalizedTrace[]): void {
  for (const normalized of traces) {
    const parsed = TraceSchema.safeParse(normalized.trace);
    if (!parsed.success) {
      throw new Error(
        `trace ${normalized.trace.trace_id} is not contract-valid: ${JSON.stringify(parsed.error.issues, null, 2)}`,
      );
    }
  }
}

describe("normalizeJsonExport — envelope", () => {
  test("a fully-formed contract trace round-trips unchanged except the id prefix", () => {
    const original = JSON.parse(loadFixture("json-full-contract.json"))[0];
    const { traces, report } = normalizeFixture("json-full-contract.json");

    expect(traces).toHaveLength(1);
    expectContractValid(traces);

    const trace = traces[0]?.trace;
    expect(trace?.trace_id).toBe("json:trace-abc");
    // Everything else is preserved byte-for-byte: swap the id back and compare.
    expect({ ...trace, trace_id: original.trace_id }).toEqual(original);
    expect(traces[0]?.diagnostics).toEqual([]);
    expect(report.surface).toBe("json");
    expect(report.counts.tracesNormalized).toBe(1);
  });

  test("a lenient trace missing the envelope gets it filled", () => {
    const { traces } = normalizeFixture("json-lenient.json");
    expect(traces).toHaveLength(1);
    expectContractValid(traces);

    const trace = traces[0]?.trace;
    expect(trace?.schema).toBe("compound.trace");
    expect(trace?.schema_version).toBe(1);
    expect(trace?.trace_id).toBe("json:lenient-1");
    expect(trace?.source).toEqual({
      importer: "json",
      importer_version: "0.1.0",
      source_ids: { trace_id: "lenient-1" },
      exported_at: "2026-07-24T09:00:00.000Z",
    });
    expect(trace?.permissions).toEqual(OPTIONS.defaultPermissions);
    expect(trace?.redactions).toEqual([]);
  });

  test("does not invent trace content, only the wrapper fields", () => {
    const { traces } = normalizeFixture("json-lenient.json");
    const trace = traces[0]?.trace;
    // Content came straight from the record.
    expect(trace?.task_key).toBeNull();
    expect(trace?.started_at).toBe("2026-07-24T11:00:00.000Z");
    expect(trace?.steps).toHaveLength(1);
    expect(trace?.focal_step_id).toBe("m1");
  });

  test("an already-prefixed trace_id is not prefixed again", () => {
    const { traces } = normalizeRecords([
      {
        trace_id: "json:already",
        task_key: null,
        started_at: "2026-07-24T10:00:00.000Z",
        steps: [
          {
            type: "model_call",
            step_id: "s",
            model: "gpt-4o",
            input: [{ role: "user", content: "hi" }],
            output: { role: "assistant", content: "yo" },
            usage: { input_tokens: 1, output_tokens: 1 },
          },
        ],
        focal_step_id: "s",
      },
    ]);
    expect(traces[0]?.trace.trace_id).toBe("json:already");
  });
});

describe("normalizeJsonExport — focal", () => {
  test("a single model_call with no focal is auto-set", () => {
    const { traces } = normalizeFixture("json-auto-focal.json");
    expect(traces).toHaveLength(1);
    expectContractValid(traces);
    expect(traces[0]?.trace.focal_step_id).toBe("only-call");
    expect(traces[0]?.diagnostics).toEqual([]);
  });

  test("an explicit focal_step_id is honored over auto-selection", () => {
    const { traces } = normalizeRecords([
      {
        trace_id: "multi",
        task_key: null,
        started_at: "2026-07-24T10:00:00.000Z",
        steps: [
          {
            type: "model_call",
            step_id: "one",
            model: "gpt-4o",
            input: [{ role: "user", content: "a" }],
            output: { role: "assistant", content: "b" },
            usage: { input_tokens: 1, output_tokens: 1 },
          },
          {
            type: "model_call",
            step_id: "two",
            model: "gpt-4o",
            input: [{ role: "user", content: "c" }],
            output: { role: "assistant", content: "d" },
            usage: { input_tokens: 1, output_tokens: 1 },
          },
        ],
        focal_step_id: "two",
      },
    ]);
    expect(traces[0]?.trace.focal_step_id).toBe("two");
  });

  test("multiple model_calls with no focal are left null and flagged diagnostic", () => {
    const { traces } = normalizeRecords([
      {
        trace_id: "multi-nofocal",
        task_key: null,
        started_at: "2026-07-24T10:00:00.000Z",
        steps: [
          {
            type: "model_call",
            step_id: "one",
            model: "gpt-4o",
            input: [{ role: "user", content: "a" }],
            output: { role: "assistant", content: "b" },
            usage: { input_tokens: 1, output_tokens: 1 },
          },
          {
            type: "model_call",
            step_id: "two",
            model: "gpt-4o",
            input: [{ role: "user", content: "c" }],
            output: { role: "assistant", content: "d" },
            usage: { input_tokens: 1, output_tokens: 1 },
          },
        ],
      },
    ]);
    expect(traces[0]?.trace.focal_step_id).toBeNull();
    expect(traces[0]?.diagnostics).toContain(DIAGNOSTICS.noReplayableFocalCall);
    expectContractValid(traces);
  });
});

describe("normalizeJsonExport — rejections", () => {
  test("a trace missing steps is rejected with a counted reason", () => {
    const { traces, report } = normalizeFixture("json-missing-steps.json");
    expect(traces).toHaveLength(0);
    expect(report.rejected).toEqual([
      { line: 1, reason: JSON_REJECTION_REASONS.traceMissingSteps },
    ]);
    expect(report.counts.recordsRejected).toBe(1);
  });

  test("a record with steps but no model_call is rejected", () => {
    const { traces, report } = normalizeRecords([
      {
        trace_id: "only-tool",
        task_key: null,
        started_at: "2026-07-24T10:00:00.000Z",
        steps: [{ type: "other", step_id: "o", name: "retrieval" }],
      },
    ]);
    expect(traces).toHaveLength(0);
    expect(report.rejected[0]?.reason).toBe(JSON_REJECTION_REASONS.traceNoModelCall);
  });

  test("a record with no trace_id is rejected", () => {
    const { traces, report } = normalizeFixture("json-no-id.json");
    expect(traces).toHaveLength(0);
    expect(report.rejected).toEqual([
      { line: 1, reason: JSON_REJECTION_REASONS.recordMissingTraceId },
    ]);
  });

  test("a structurally invalid trace is rejected rather than emitted", () => {
    const { traces, report } = normalizeRecords([
      {
        trace_id: "bad-step",
        task_key: null,
        started_at: "2026-07-24T10:00:00.000Z",
        // model_call missing the required `input` array.
        steps: [{ type: "model_call", step_id: "s", model: "gpt-4o" }],
        focal_step_id: "s",
      },
    ]);
    expect(traces).toHaveLength(0);
    expect(report.rejected[0]?.reason).toBe(JSON_REJECTION_REASONS.traceSchemaInvalid);
  });
});

describe("normalizeJsonExport — input surfaces", () => {
  test("JSONL with a malformed line rejects only that line and keeps the good ones", () => {
    const { traces, report } = normalizeFixture("json-mixed.jsonl");
    expect(report.format).toBe("jsonl");
    expect(traces.map((normalized) => normalized.trace.trace_id)).toEqual([
      "json:jsonl-1",
      "json:jsonl-2",
    ]);
    expect(report.rejected).toEqual([
      { line: 2, reason: JSON_REJECTION_REASONS.malformedJsonLine },
    ]);
    expect(report.counts.tracesNormalized).toBe(2);
    expectContractValid(traces);
  });

  test("an empty array normalizes to nothing without failing", () => {
    const { traces, report } = normalizeJsonExport("[]", OPTIONS);
    expect(traces).toHaveLength(0);
    expect(report.counts.recordsSeen).toBe(0);
    expect(report.surface).toBe("json");
  });

  test("a non-array JSON document is rejected as a whole", () => {
    const { traces, report } = normalizeJsonExport('{"trace_id":"x"}', OPTIONS);
    // A leading `{` is read as JSONL: one object line that is missing steps.
    expect(traces).toHaveLength(0);
    expect(report.counts.recordsRejected).toBe(1);
  });
});
