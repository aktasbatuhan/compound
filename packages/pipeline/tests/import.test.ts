import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import type { CompoundDatabase } from "@compound/storage";
import {
  createDatabase,
  getImportBatch,
  getTraceByTraceId,
  listTraces,
  migrate,
} from "@compound/storage";
import { runImport, UnsupportedImporterError } from "../src/index";

let db: CompoundDatabase;

beforeEach(() => {
  db = createDatabase();
  migrate(db);
});

afterEach(() => {
  db.close();
});

/** A minimal Langfuse API-shaped export with one replayable generation. */
function langfuseExport(overrides: Record<string, unknown> = {}): string {
  return JSON.stringify([
    {
      id: "tr-1",
      timestamp: "2026-07-23T10:00:00Z",
      tags: [],
      public: false,
      environment: "production",
      metadata: { task_key: "support" },
      observations: [
        {
          id: "gen-1",
          traceId: "tr-1",
          type: "GENERATION",
          startTime: "2026-07-23T10:00:00Z",
          endTime: "2026-07-23T10:00:02Z",
          level: "DEFAULT",
          environment: "production",
          model: "gpt-4o",
          input: [{ role: "user", content: "what is our refund window?" }],
          output: { role: "assistant", content: "Thirty days." },
          usageDetails: { input: 12, output: 4 },
          costDetails: { total: 0.002 },
        },
      ],
      scores: [],
      ...overrides,
    },
  ]);
}

describe("runImport", () => {
  test("imports a Langfuse export end to end", () => {
    const { batch, report } = runImport(db, { importer: "langfuse", content: langfuseExport() });

    expect(batch.status).toBe("completed");
    expect(report.counts).toMatchObject({ eval_ready: 1, diagnostic: 0, rejected: 0 });

    const stored = getTraceByTraceId(db, "langfuse:tr-1");
    expect(stored).not.toBeNull();
    expect(stored?.validationClass).toBe("eval_ready");
    expect(stored?.taskKey).toBe("support");
    expect(stored?.focalModel).toBe("gpt-4o");
    expect(stored?.contentHash).toMatch(/^[0-9a-f]{64}$/);
  });

  test("stamps the config's ingest permissions onto imported traces", () => {
    const config = {
      ingest: { default_permissions: { judging: true, optimization: false, fine_tuning: false } },
    } as never;
    runImport(db, { importer: "langfuse", content: langfuseExport(), config });

    const stored = getTraceByTraceId(db, "langfuse:tr-1");
    expect(stored?.permissions).toEqual({
      judging: true,
      optimization: false,
      fineTuning: false,
    });
  });

  test("redacts before persisting, so no secret reaches storage", () => {
    const withSecret = JSON.parse(langfuseExport());
    withSecret[0].observations[0].input = [
      { role: "user", content: "my key is sk-proj-AbCdEf0123456789XyZwVuTsRqPoNmLkJiHgFe ok?" },
    ];
    const config = {
      redaction: { rules: [{ name: "s", applies_to: ["**"], detector: "secret" }] },
    } as never;

    const { report } = runImport(db, {
      importer: "langfuse",
      content: JSON.stringify(withSecret),
      config,
    });

    const stored = getTraceByTraceId(db, "langfuse:tr-1");
    const persisted = JSON.stringify(stored);
    expect(persisted).not.toContain("sk-proj-AbCdEf0123456789XyZwVuTsRqPoNmLkJiHgFe");
    expect(persisted).toContain("⟦redacted:secret⟧");
    // Surrounding context survives, which is what keeps the case usable.
    expect(persisted).toContain("my key is");
    expect(report.redactions_by_rule).toEqual({ secret: 1 });
  });

  test("a trace the source could not express is diagnostic, not eval_ready", () => {
    const unparseable = JSON.parse(langfuseExport());
    unparseable[0].observations[0].input = 42; // not a recognizable input shape

    runImport(db, { importer: "langfuse", content: JSON.stringify(unparseable) });

    const stored = getTraceByTraceId(db, "langfuse:tr-1");
    expect(stored?.validationClass).toBe("diagnostic");
    expect(stored?.diagnosticReasons.length).toBeGreaterThan(0);
  });

  test("merges normalization and contract reasons into one sorted set", () => {
    const noOutput = JSON.parse(langfuseExport());
    noOutput[0].observations[0].input = 42;
    delete noOutput[0].observations[0].output;

    const { report } = runImport(db, {
      importer: "langfuse",
      content: JSON.stringify(noOutput),
    });

    const stored = getTraceByTraceId(db, "langfuse:tr-1");
    const reasons = stored?.diagnosticReasons ?? [];
    expect(reasons).toEqual([...reasons].sort());
    expect(new Set(reasons).size).toBe(reasons.length);
    // Both sources contributed: the source shape and the contract requirement.
    expect(reasons).toContain("unparseable_generation_input");
    expect(Object.keys(report.diagnostic_reasons ?? {}).length).toBeGreaterThan(0);
  });

  test("counts malformed input lines without persisting them", () => {
    const jsonl = ["not json at all", JSON.parse(langfuseExport())[0]]
      .map((entry) => (typeof entry === "string" ? entry : JSON.stringify(entry)))
      .join("\n");

    const { report } = runImport(db, { importer: "langfuse", content: jsonl });

    expect(report.counts?.rejected).toBe(1);
    expect(report.rejected_lines).toEqual([1]);
    expect(report.counts?.eval_ready).toBe(1);
    expect(listTraces(db, {})).toHaveLength(1);
  });

  test("rejected line content is never stored, only line numbers and reasons", () => {
    const jsonl = [
      "sk-proj-SecretInAMalformedLine0123456789abcdef",
      JSON.parse(langfuseExport())[0],
    ]
      .map((entry) => (typeof entry === "string" ? entry : JSON.stringify(entry)))
      .join("\n");

    const { batch } = runImport(db, { importer: "langfuse", content: jsonl });

    const stored = getImportBatch(db, batch.id);
    expect(JSON.stringify(stored?.report)).not.toContain("SecretInAMalformedLine");
    expect(stored?.report?.rejected_lines).toEqual([1]);
    expect(Object.keys(stored?.report?.rejected_reasons ?? {}).length).toBeGreaterThan(0);
  });

  test("re-importing the same export skips duplicates instead of failing", () => {
    runImport(db, { importer: "langfuse", content: langfuseExport() });
    const second = runImport(db, { importer: "langfuse", content: langfuseExport() });

    expect(second.batch.status).toBe("completed");
    expect(second.report.counts?.duplicate).toBe(1);
    expect(second.report.duplicate_trace_ids).toEqual(["langfuse:tr-1"]);
    expect(listTraces(db, {})).toHaveLength(1);
  });

  test("identical work imported under different trace ids shares a content hash", () => {
    runImport(db, { importer: "langfuse", content: langfuseExport() });
    const other = JSON.parse(langfuseExport());
    other[0].id = "tr-2";
    other[0].observations[0].traceId = "tr-2";
    runImport(db, { importer: "langfuse", content: JSON.stringify(other) });

    const first = getTraceByTraceId(db, "langfuse:tr-1");
    const second = getTraceByTraceId(db, "langfuse:tr-2");
    expect(second?.contentHash).toBe(first?.contentHash as string);
  });

  test("records the batch report on the batch itself", () => {
    const { batch } = runImport(db, { importer: "langfuse", content: langfuseExport() });
    const stored = getImportBatch(db, batch.id);

    expect(stored?.report?.surface).toBe("api");
    // The trace record and its embedded observation are each a source record.
    expect(stored?.report?.source_counts).toMatchObject({
      traceRecords: 1,
      observationRecords: 1,
      tracesNormalized: 1,
      recordsRejected: 0,
    });
    expect(stored?.importer).toBe("langfuse");
    expect(stored?.sourceFingerprint).toMatch(/^[0-9a-f]{64}$/);
  });

  test("rejects an unsupported importer before creating a batch", () => {
    expect(() => runImport(db, { importer: "braintrust", content: "[]" })).toThrow(
      UnsupportedImporterError,
    );
    expect(listTraces(db, {})).toHaveLength(0);
  });

  test("an empty export completes with zero counts rather than failing", () => {
    const { batch, report } = runImport(db, { importer: "langfuse", content: "[]" });
    expect(batch.status).toBe("completed");
    expect(report.counts).toMatchObject({ eval_ready: 0, diagnostic: 0 });
  });
});
