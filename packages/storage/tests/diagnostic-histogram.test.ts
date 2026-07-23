import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import type { CompoundDatabase } from "../src/index";
import { countTracesByDiagnosticReason, insertTraces } from "../src/index";
import { freshDatabase, newBatch, recordFromFixture } from "./helpers";

let db: CompoundDatabase;

beforeEach(() => {
  db = freshDatabase();
});

afterEach(() => {
  db.close();
});

describe("countTracesByDiagnosticReason", () => {
  test("is empty on an empty store", () => {
    expect(countTracesByDiagnosticReason(db)).toEqual([]);
  });

  test("is empty when every trace is eval_ready", () => {
    const batch = newBatch(db);
    insertTraces(db, batch.id, [
      recordFromFixture("eval-ready-full.json", (t) => {
        t.trace_id = "ready-1";
      }),
    ]);
    expect(countTracesByDiagnosticReason(db)).toEqual([]);
  });

  /**
   * A trace is counted once per reason it carries, so these counts sum to at
   * least the number of diagnostic traces rather than exactly to it. The
   * no-model-calls fixture also lacks a focal step, which is why
   * `missing_focal_step_id` legitimately outnumbers the traces that are only
   * missing a focal step.
   */
  test("counts each reason a trace carries, largest bucket first", () => {
    const batch = newBatch(db);
    insertTraces(db, batch.id, [
      recordFromFixture("diagnostic-missing-focal.json", (t) => {
        t.trace_id = "missing-1";
      }),
      recordFromFixture("diagnostic-missing-focal.json", (t) => {
        t.trace_id = "missing-2";
      }),
      recordFromFixture("diagnostic-no-model-calls.json", (t) => {
        t.trace_id = "no-calls-1";
      }),
      recordFromFixture("eval-ready-full.json", (t) => {
        t.trace_id = "ready-1";
      }),
    ]);

    const histogram = countTracesByDiagnosticReason(db);
    expect(histogram).toEqual([
      { reason: "missing_focal_step_id", count: 3 },
      { reason: "no_model_call_steps", count: 1 },
    ]);
  });

  test("honours a task_key filter, including the unassigned bucket", () => {
    const batch = newBatch(db);
    insertTraces(db, batch.id, [
      recordFromFixture("diagnostic-missing-focal.json", (t) => {
        t.trace_id = "keyed";
        t.task_key = "support";
      }),
      recordFromFixture("diagnostic-missing-focal.json", (t) => {
        t.trace_id = "unassigned";
        t.task_key = null;
      }),
    ]);

    expect(countTracesByDiagnosticReason(db, { taskKey: "support" })).toEqual([
      { reason: "missing_focal_step_id", count: 1 },
    ]);
    expect(countTracesByDiagnosticReason(db, { taskKey: null })).toEqual([
      { reason: "missing_focal_step_id", count: 1 },
    ]);
  });

  test("honours an import_batch_id filter", () => {
    const first = newBatch(db);
    const second = newBatch(db);
    insertTraces(db, first.id, [
      recordFromFixture("diagnostic-missing-focal.json", (t) => {
        t.trace_id = "first-1";
      }),
    ]);
    insertTraces(db, second.id, [
      recordFromFixture("diagnostic-no-model-calls.json", (t) => {
        t.trace_id = "second-1";
      }),
    ]);

    expect(countTracesByDiagnosticReason(db, { importBatchId: first.id })).toEqual([
      { reason: "missing_focal_step_id", count: 1 },
    ]);
    const secondHistogram = countTracesByDiagnosticReason(db, { importBatchId: second.id });
    expect(secondHistogram.map((row) => row.reason)).toContain("no_model_call_steps");
  });
});
