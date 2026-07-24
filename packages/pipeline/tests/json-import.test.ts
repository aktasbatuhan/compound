import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import type { CompoundDatabase } from "@compound/storage";
import {
  createDatabase,
  getImportBatch,
  getTraceByTraceId,
  listTraces,
  migrate,
} from "@compound/storage";
import { runImport } from "../src/index";

let db: CompoundDatabase;

beforeEach(() => {
  db = createDatabase();
  migrate(db);
});

afterEach(() => {
  db.close();
});

/** A lenient, near-contract trace with one replayable model_call and no envelope. */
function jsonExport(overrides: Record<string, unknown> = {}): string {
  return JSON.stringify([
    {
      trace_id: "t-json-1",
      task_key: "support",
      started_at: "2026-07-24T10:00:00.000Z",
      steps: [
        {
          type: "model_call",
          step_id: "call-1",
          model: "gpt-4o",
          input: [{ role: "user", content: "what is our refund window?" }],
          output: { role: "assistant", content: "Thirty days." },
          usage: { input_tokens: 12, output_tokens: 4 },
          started_at: "2026-07-24T10:00:00.000Z",
          ended_at: "2026-07-24T10:00:02.000Z",
        },
      ],
      ...overrides,
    },
  ]);
}

describe("runImport — json importer", () => {
  test("imports plain trace JSON end to end into a stored, eval-ready trace", () => {
    const { batch, report } = runImport(db, { importer: "json", content: jsonExport() });

    expect(batch.status).toBe("completed");
    expect(batch.importer).toBe("json");
    expect(report.counts).toMatchObject({ eval_ready: 1, diagnostic: 0, rejected: 0 });
    expect(report.surface).toBe("json");

    const stored = getTraceByTraceId(db, "json:t-json-1");
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
    runImport(db, { importer: "json", content: jsonExport(), config });

    const stored = getTraceByTraceId(db, "json:t-json-1");
    expect(stored?.permissions).toEqual({ judging: true, optimization: false, fineTuning: false });
  });

  test("a record missing steps is rejected, counted, and never persisted", () => {
    const good = JSON.stringify(JSON.parse(jsonExport())[0]);
    const noSteps = JSON.stringify({
      trace_id: "no-steps",
      started_at: "2026-07-24T10:00:00.000Z",
    });
    const content = [good, noSteps].join("\n");

    const { report } = runImport(db, { importer: "json", content });

    expect(report.counts?.eval_ready).toBe(1);
    expect(report.counts?.rejected).toBe(1);
    expect(report.rejected_reasons).toMatchObject({ trace_missing_steps: 1 });
    expect(listTraces(db, {})).toHaveLength(1);
  });

  test("a trace with no focal is stored as diagnostic, not eval-ready", () => {
    const twoCalls = jsonExport({
      steps: [
        {
          type: "model_call",
          step_id: "a",
          model: "gpt-4o",
          input: [{ role: "user", content: "hi" }],
          output: { role: "assistant", content: "one" },
          usage: { input_tokens: 1, output_tokens: 1 },
        },
        {
          type: "model_call",
          step_id: "b",
          model: "gpt-4o",
          input: [{ role: "user", content: "hi again" }],
          output: { role: "assistant", content: "two" },
          usage: { input_tokens: 1, output_tokens: 1 },
        },
      ],
    });

    runImport(db, { importer: "json", content: twoCalls });

    const stored = getTraceByTraceId(db, "json:t-json-1");
    expect(stored?.validationClass).toBe("diagnostic");
    expect(stored?.diagnosticReasons).toContain("missing_focal_step_id");
  });

  test("re-importing the same JSON skips duplicates instead of failing", () => {
    runImport(db, { importer: "json", content: jsonExport() });
    const second = runImport(db, { importer: "json", content: jsonExport() });

    expect(second.batch.status).toBe("completed");
    expect(second.report.counts?.duplicate).toBe(1);
    expect(listTraces(db, {})).toHaveLength(1);
  });

  test("records the json surface on the batch report", () => {
    const { batch } = runImport(db, { importer: "json", content: jsonExport() });
    const stored = getImportBatch(db, batch.id);
    expect(stored?.report?.surface).toBe("json");
    expect(stored?.importer).toBe("json");
  });
});
