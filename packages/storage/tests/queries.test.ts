import { beforeEach, describe, expect, test } from "bun:test";
import type { CompoundDatabase } from "../src/index";
import {
  countTraces,
  countTracesByTaskKey,
  countTracesByValidationClass,
  insertTraces,
  listTraces,
} from "../src/index";
import { freshDatabase, newBatch, recordFromFixture } from "./helpers";

/**
 * Seed built from the contract fixtures, patched only in the fields the query
 * under test discriminates on (trace_id, task_key, started_at).
 *
 *  batch A: t1 eval_ready  triage    2026-07-01
 *           t2 eval_ready  null      2026-07-02
 *           t3 diagnostic  triage    2026-07-03
 *  batch B: t4 diagnostic  null      2026-07-04
 *           t5 eval_ready  summarize 2026-07-05
 */
interface Seed {
  handle: CompoundDatabase;
  batchA: string;
  batchB: string;
}

function seeded(): Seed {
  const handle = freshDatabase();
  const batchA = newBatch(handle).id;
  const batchB = newBatch(handle).id;

  insertTraces(handle, batchA, [
    recordFromFixture("eval-ready-full.json", (trace) => {
      trace.trace_id = "t1";
      trace.task_key = "support.invoice_triage";
      trace.started_at = "2026-07-01T00:00:00Z";
    }),
    recordFromFixture("eval-ready-full.json", (trace) => {
      trace.trace_id = "t2";
      trace.task_key = null;
      trace.started_at = "2026-07-02T00:00:00Z";
    }),
    recordFromFixture("diagnostic-missing-focal.json", (trace) => {
      trace.trace_id = "t3";
      trace.task_key = "support.invoice_triage";
      trace.started_at = "2026-07-03T00:00:00Z";
    }),
  ]);
  insertTraces(handle, batchB, [
    recordFromFixture("diagnostic-missing-focal.json", (trace) => {
      trace.trace_id = "t4";
      trace.task_key = null;
      trace.started_at = "2026-07-04T00:00:00Z";
    }),
    recordFromFixture("eval-ready-full.json", (trace) => {
      trace.trace_id = "t5";
      trace.task_key = "docs.summarize";
      trace.started_at = "2026-07-05T00:00:00Z";
    }),
  ]);
  return { handle, batchA, batchB };
}

let seed: Seed;

beforeEach(() => {
  seed = seeded();
});

function ids(rows: ReturnType<typeof listTraces>): string[] {
  return rows.map((row) => row.traceId);
}

describe("listTraces", () => {
  test("defaults to newest first and returns parsed contract traces", () => {
    const rows = listTraces(seed.handle);
    expect(ids(rows)).toEqual(["t5", "t4", "t3", "t2", "t1"]);
    const first = rows[0];
    if (first === undefined) throw new Error("expected rows");
    expect(first.trace.schema).toBe("compound.trace");
    expect(first.trace.trace_id).toBe("t5");
    expect(first.trace.steps.length).toBeGreaterThan(0);
  });

  test("ascending order", () => {
    expect(ids(listTraces(seed.handle, { order: "started_at_asc" }))).toEqual([
      "t1",
      "t2",
      "t3",
      "t4",
      "t5",
    ]);
  });

  test("filters by task_key", () => {
    expect(ids(listTraces(seed.handle, { taskKey: "support.invoice_triage" }))).toEqual([
      "t3",
      "t1",
    ]);
    expect(ids(listTraces(seed.handle, { taskKey: "docs.summarize" }))).toEqual(["t5"]);
    expect(ids(listTraces(seed.handle, { taskKey: "nope" }))).toEqual([]);
  });

  test("explicit null task_key selects the unassigned bucket", () => {
    expect(ids(listTraces(seed.handle, { taskKey: null }))).toEqual(["t4", "t2"]);
  });

  test("omitting task_key does not filter (undefined is not unassigned)", () => {
    expect(ids(listTraces(seed.handle, { taskKey: undefined }))).toHaveLength(5);
  });

  test("filters by validation_class, singly and as a set", () => {
    expect(ids(listTraces(seed.handle, { validationClass: "eval_ready" }))).toEqual([
      "t5",
      "t2",
      "t1",
    ]);
    expect(ids(listTraces(seed.handle, { validationClass: "diagnostic" }))).toEqual(["t4", "t3"]);
    expect(
      ids(listTraces(seed.handle, { validationClass: ["eval_ready", "diagnostic"] })),
    ).toHaveLength(5);
  });

  test("filters by import_batch_id", () => {
    expect(ids(listTraces(seed.handle, { importBatchId: seed.batchA }))).toEqual([
      "t3",
      "t2",
      "t1",
    ]);
    expect(ids(listTraces(seed.handle, { importBatchId: seed.batchB }))).toEqual(["t5", "t4"]);
  });

  test("filters by started_at range (inclusive, ISO string or Date)", () => {
    expect(
      ids(
        listTraces(seed.handle, {
          startedAtFrom: "2026-07-02T00:00:00Z",
          startedAtTo: "2026-07-04T00:00:00Z",
        }),
      ),
    ).toEqual(["t4", "t3", "t2"]);
    expect(
      ids(listTraces(seed.handle, { startedAtFrom: new Date("2026-07-05T00:00:00Z") })),
    ).toEqual(["t5"]);
    expect(ids(listTraces(seed.handle, { startedAtTo: new Date("2026-07-01T00:00:00Z") }))).toEqual(
      ["t1"],
    );
  });

  test("filters by content_hash", () => {
    expect(
      ids(listTraces(seed.handle, { contentHash: "hash:diagnostic-missing-focal.json" })),
    ).toEqual(["t4", "t3"]);
  });

  test("combines filters", () => {
    expect(
      ids(
        listTraces(seed.handle, {
          taskKey: null,
          validationClass: "eval_ready",
        }),
      ),
    ).toEqual(["t2"]);
    expect(
      ids(
        listTraces(seed.handle, {
          importBatchId: seed.batchA,
          validationClass: "diagnostic",
        }),
      ),
    ).toEqual(["t3"]);
  });

  test("limit and offset", () => {
    expect(ids(listTraces(seed.handle, { limit: 2 }))).toEqual(["t5", "t4"]);
    expect(ids(listTraces(seed.handle, { limit: 2, offset: 2 }))).toEqual(["t3", "t2"]);
    expect(ids(listTraces(seed.handle, { offset: 4 }))).toEqual(["t1"]);
    expect(ids(listTraces(seed.handle, { validationClass: "eval_ready", limit: 1 }))).toEqual([
      "t5",
    ]);
  });
});

describe("counts", () => {
  test("countTraces honours filters", () => {
    expect(countTraces(seed.handle)).toBe(5);
    expect(countTraces(seed.handle, { taskKey: null })).toBe(2);
    expect(countTraces(seed.handle, { importBatchId: seed.batchB })).toBe(2);
  });

  test("countTracesByValidationClass always reports both classes", () => {
    expect(countTracesByValidationClass(seed.handle)).toEqual({
      eval_ready: 3,
      diagnostic: 2,
    });
    expect(countTracesByValidationClass(seed.handle, { importBatchId: seed.batchB })).toEqual({
      eval_ready: 1,
      diagnostic: 1,
    });
    expect(countTracesByValidationClass(seed.handle, { taskKey: "docs.summarize" })).toEqual({
      eval_ready: 1,
      diagnostic: 0,
    });
  });

  test("countTracesByTaskKey buckets unassigned under null, largest first", () => {
    expect(countTracesByTaskKey(seed.handle)).toEqual([
      { taskKey: "support.invoice_triage", count: 2 },
      { taskKey: null, count: 2 },
      { taskKey: "docs.summarize", count: 1 },
    ]);
  });

  test("countTracesByTaskKey honours a validation_class filter", () => {
    // Equal counts: named keys sort before the unassigned bucket.
    expect(countTracesByTaskKey(seed.handle, { validationClass: "diagnostic" })).toEqual([
      { taskKey: "support.invoice_triage", count: 1 },
      { taskKey: null, count: 1 },
    ]);
  });

  test("counts on an empty store", () => {
    const empty = freshDatabase();
    expect(countTraces(empty)).toBe(0);
    expect(countTracesByValidationClass(empty)).toEqual({ eval_ready: 0, diagnostic: 0 });
    expect(countTracesByTaskKey(empty)).toEqual([]);
    empty.close();
  });
});
