import { describe, expect, test } from "bun:test";
import { classifyRecord, detectCasing, normalizeLangfuseExport } from "../src/index";
import {
  expectContractValid,
  loadFixture,
  normalizeFixture,
  normalizeRecords,
  OPTIONS,
} from "./helpers";

const FIXTURES = [
  "api-openai-tools.json",
  "api-langchain.json",
  "api-v4-no-trace-io.jsonl",
  "api-diagnostic-cases.json",
  "blob-join.jsonl",
  "blob-observations-v2.jsonl",
];

describe("every fixture produces contract-valid traces", () => {
  for (const name of FIXTURES) {
    test(name, () => {
      const { traces } = normalizeFixture(name);
      expect(traces.length).toBeGreaterThan(0);
      expectContractValid(traces);
      for (const normalized of traces) {
        expect(normalized.trace.redactions).toEqual([]);
        expect(normalized.trace.permissions).toEqual(OPTIONS.defaultPermissions);
        expect(normalized.trace.source.importer).toBe("langfuse");
      }
    });
  }
});

describe("input surfaces", () => {
  test("an already-parsed array is accepted", () => {
    const records: unknown[] = JSON.parse(loadFixture("api-openai-tools.json"));
    const { traces, report } = normalizeLangfuseExport(records, OPTIONS);
    expect(report.format).toBe("records");
    expect(traces).toHaveLength(1);
  });

  test("an empty input yields an empty report, not an error", () => {
    const { traces, report } = normalizeLangfuseExport([], OPTIONS);
    expect(traces).toEqual([]);
    expect(report.surface).toBe("unknown");
    expect(report.casing).toEqual(["unknown"]);
    expect(report.counts.recordsSeen).toBe(0);
  });

  test("a JSON array of records is read as one file", () => {
    const { report } = normalizeFixture("api-openai-tools.json");
    expect(report.format).toBe("json_array");
  });

  test("a syntactically broken JSON array is one rejection, not a crash", () => {
    const { traces, report } = normalizeLangfuseExport('[{"id": "x",', OPTIONS);
    expect(traces).toEqual([]);
    expect(report.rejected).toEqual([{ line: 1, reason: "file_not_valid_json" }]);
  });

  test("blank lines in JSONL are ignored", () => {
    const line =
      '{"id":"tr-a","timestamp":"2026-07-22T10:00:00.000Z","tags":[],"metadata":{},"observations":[]}';
    const { report } = normalizeLangfuseExport(`\n${line}\n\n`, OPTIONS);
    expect(report.rejected).toEqual([]);
    expect(report.counts.recordsSeen).toBe(1);
  });

  test("a record that is not an object or has no id is rejected on its line", () => {
    const { report } = normalizeLangfuseExport('"nope"\n{"timestamp":"2026-07-22T10:00:00.000Z"}', {
      ...OPTIONS,
    });
    expect(report.rejected).toEqual([
      { line: 1, reason: "record_not_an_object" },
      { line: 2, reason: "record_missing_id" },
    ]);
  });

  test("an unrecognized record shape is rejected without failing the file", () => {
    const good =
      '{"id":"tr-a","timestamp":"2026-07-22T10:00:00.000Z","tags":[],"metadata":{},"observations":[{"id":"g","traceId":"tr-a","type":"GENERATION","startTime":"2026-07-22T10:00:00.000Z","input":[{"role":"user","content":"x"}],"output":"y"}]}';
    const { traces, report } = normalizeLangfuseExport(`{"id":"weird","foo":1}\n${good}`, OPTIONS);
    expect(report.rejected).toEqual([{ line: 1, reason: "unrecognized_record_shape" }]);
    expect(traces).toHaveLength(1);
  });

  test("an observation with no trace_id is rejected", () => {
    const { report } = normalizeLangfuseExport(
      '{"id":"o1","type":"GENERATION","start_time":"2026-07-22 10:00:00.000000"}',
      OPTIONS,
    );
    expect(report.rejected).toEqual([{ line: 1, reason: "observation_missing_trace_id" }]);
  });

  test("casing is detected per record, so mixed files are normal input", () => {
    expect(detectCasing({ traceId: "a", startTime: "b" })).toBe("camelCase");
    expect(detectCasing({ trace_id: "a", start_time: "b" })).toBe("snake_case");
    expect(detectCasing({ id: "a" })).toBe("unknown");

    const camel =
      '{"id":"o1","traceId":"tr-mix","type":"GENERATION","startTime":"2026-07-22T10:00:00.000Z","input":[{"role":"user","content":"x"}],"output":"y"}';
    const snake =
      '{"id":"o2","trace_id":"tr-mix","type":"SPAN","start_time":"2026-07-22 10:00:01.000000","parent_observation_id":"o1"}';
    const { traces, report } = normalizeLangfuseExport(`${camel}\n${snake}`, OPTIONS);
    expect(report.casing).toEqual(["camelCase", "snake_case"]);
    expect(traces).toHaveLength(1);
    expect(traces[0]?.trace.steps.map((step) => step.step_id)).toEqual(["o1", "o2"]);
  });

  test("records are classified by shape", () => {
    expect(classifyRecord({ id: "x", timestamp: "t" })).toBe("trace");
    expect(classifyRecord({ id: "x", type: "GENERATION", startTime: "t" })).toBe("observation");
    expect(classifyRecord({ id: "x", name: "n", source: "EVAL" })).toBe("score");
    expect(classifyRecord({ id: "x" })).toBeNull();
  });
});

describe("trace_id prefixing", () => {
  test("the project segment is omitted when no project is known", () => {
    const { traces } = normalizeLangfuseExport(
      '{"id":"tr-np","timestamp":"2026-07-22T10:00:00.000Z","tags":[],"metadata":{},"observations":[]}',
      { ...OPTIONS, projectId: undefined },
    );
    expect(traces[0]?.trace.trace_id).toBe("langfuse:tr-np");
    expect(traces[0]?.trace.source.source_ids).toEqual({ trace_id: "tr-np" });
  });

  test("the option wins over a record's project_id", () => {
    const { traces } = normalizeFixture("blob-join.jsonl", { projectId: "override" });
    expect(traces[0]?.trace.trace_id).toBe("langfuse:override:tr-blob-1");
  });
});

describe("task_key resolution", () => {
  const base = {
    id: "tr-tk",
    timestamp: "2026-07-22T10:00:00.000Z",
    observations: [],
    scores: [],
  };

  test("metadata wins over a task tag", () => {
    const { traces } = normalizeRecords([
      { ...base, metadata: { task_key: "from.metadata" }, tags: ["task:from.tag"] },
    ]);
    expect(traces[0]?.trace.task_key).toBe("from.metadata");
    expect(traces[0]?.trace.metadata).toEqual({});
  });

  test("a task tag is the fallback", () => {
    const { traces } = normalizeRecords([{ ...base, metadata: {}, tags: ["x", "task:from.tag"] }]);
    expect(traces[0]?.trace.task_key).toBe("from.tag");
  });

  test("neither present leaves the unassigned bucket", () => {
    const { traces } = normalizeRecords([{ ...base, metadata: {}, tags: ["task:"] }]);
    expect(traces[0]?.trace.task_key).toBeNull();
  });
});

describe("report totals", () => {
  test("counts, histogram and dialects describe the whole run", () => {
    const { report } = normalizeFixture("api-diagnostic-cases.json");
    expect(report.counts.recordsSeen).toBe(4);
    expect(report.counts.tracesNormalized).toBe(1);
    expect(Object.values(report.diagnosticReasons).every((count) => count === 1)).toBe(true);
    expect(report.dialects.length).toBeGreaterThan(0);
    expect([...report.dialects].sort()).toEqual(report.dialects);
  });
});
