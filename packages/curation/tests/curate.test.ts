import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { type ValidationResult, validate } from "@compound/contract";
import {
  type CompoundDatabase,
  countCases,
  countCasesByPartition,
  createDatabase,
  createImportBatch,
  getCase,
  insertTraces,
  listCases,
  migrate,
  openDecisionFirewall,
  traceRecordFromValidation,
} from "@compound/storage";
import { curateTask } from "../src/curate";

let db: CompoundDatabase;

beforeEach(() => {
  db = createDatabase();
  migrate(db);
});

afterEach(() => {
  db.close();
});

function evalReadyTrace(id: string, taskKey: string | null, content: string) {
  return {
    schema: "compound.trace",
    schema_version: 1,
    trace_id: id,
    task_key: taskKey,
    started_at: "2026-07-23T10:00:00Z",
    source: { importer: "test", importer_version: "1", source_ids: {} },
    steps: [
      {
        type: "model_call",
        step_id: "s1",
        model: "gpt-4o",
        input: [{ role: "user", content }],
        output: { role: "assistant", content: "answer" },
        usage: { input_tokens: 5, output_tokens: 2 },
        started_at: "2026-07-23T10:00:00Z",
        ended_at: "2026-07-23T10:00:01Z",
      },
    ],
    focal_step_id: "s1",
    permissions: { judging: true, optimization: true, fine_tuning: false },
    redactions: [],
  };
}

function seedTrace(id: string, taskKey: string | null, content: string, contentHash: string): void {
  const result: ValidationResult = validate(evalReadyTrace(id, taskKey, content));
  const record = traceRecordFromValidation(result, contentHash);
  if (record === null) throw new Error("trace rejected");
  const batch = createImportBatch(db, {
    importer: "test",
    importerVersion: "1",
    sourceFingerprint: id,
  });
  insertTraces(db, batch.id, [record]);
}

describe("curateTask", () => {
  test("turns eval-ready traces into partitioned, provenance-typed cases", () => {
    for (let i = 0; i < 40; i += 1) {
      seedTrace(`langfuse:tr-${i}`, "support", `question ${i}`, `hash-${i}`);
    }

    const report = curateTask(db, { taskKey: "support" });
    expect(report.tracesScanned).toBe(40);
    expect(report.casesCreated).toBe(40);
    expect(report.duplicates).toBe(0);
    expect(countCases(db, { taskKey: "support" }) + sealedCount()).toBe(40);
  });

  function sealedCount(): number {
    const token = openDecisionFirewall("test count");
    return listCases(db, { partition: "decision_test", openDecisionFirewall: token }).length;
  }

  test("is idempotent: re-running creates no new cases and moves no partitions", () => {
    for (let i = 0; i < 20; i += 1) {
      seedTrace(`langfuse:tr-${i}`, "support", `q${i}`, `hash-${i}`);
    }
    const first = curateTask(db, { taskKey: "support" });
    const partitionsBefore = countCasesByPartition(db, "support");

    const second = curateTask(db, { taskKey: "support" });
    const partitionsAfter = countCasesByPartition(db, "support");

    expect(first.casesCreated).toBe(20);
    expect(second.casesCreated).toBe(0);
    expect(second.duplicates).toBe(20);
    expect(partitionsAfter).toEqual(partitionsBefore);
  });

  test("byPartition counts only created cases, never inflated by content-duplicates", () => {
    // Six traces, three distinct content hashes → three cases, three duplicates.
    for (let i = 0; i < 3; i += 1) {
      seedTrace(`langfuse:a-${i}`, "support", `q${i}`, `dup-hash-${i}`);
      seedTrace(`langfuse:b-${i}`, "support", `q${i}`, `dup-hash-${i}`);
    }
    const report = curateTask(db, { taskKey: "support" });
    expect(report.casesCreated).toBe(3);
    expect(report.duplicates).toBe(3);
    // The bug: byPartition summed to 6 (each case counted with its duplicate).
    const partitionTotal = Object.values(report.byPartition).reduce((a, b) => a + b, 0);
    expect(partitionTotal).toBe(report.casesCreated);
    const provenanceTotal = Object.values(report.byProvenance).reduce((a, b) => a + b, 0);
    expect(provenanceTotal).toBe(report.casesCreated);
  });

  test("only curates the requested task", () => {
    seedTrace("langfuse:a", "support", "q", "h-a");
    seedTrace("langfuse:b", "billing", "q", "h-b");

    const report = curateTask(db, { taskKey: "support" });
    expect(report.tracesScanned).toBe(1);
    expect(report.casesCreated).toBe(1);
  });

  test("assigns a deterministic partition from the content hash", () => {
    seedTrace("langfuse:x", "support", "q", "fixed-hash");
    curateTask(db, { taskKey: "support" });

    // Re-derive the same case id to look it up regardless of partition.
    const token = openDecisionFirewall("test lookup");
    const all = listCases(db, { openDecisionFirewall: token });
    expect(all).toHaveLength(1);
    const partition = all[0]?.partition;

    // A fresh DB with the same hash lands in the same partition.
    const db2 = createDatabase();
    migrate(db2);
    const result = validate(evalReadyTrace("langfuse:x", "support", "q"));
    const record = traceRecordFromValidation(result, "fixed-hash");
    const batch = createImportBatch(db2, {
      importer: "t",
      importerVersion: "1",
      sourceFingerprint: "x",
    });
    if (record === null) throw new Error("rejected");
    insertTraces(db2, batch.id, [record]);
    curateTask(db2, { taskKey: "support" });
    const all2 = listCases(db2, { openDecisionFirewall: openDecisionFirewall("t") });
    expect(all2[0]?.partition).toBe(partition as never);
    db2.close();
  });

  test("skips diagnostic traces rather than forcing them into cases", () => {
    // A trace with no focal step is diagnostic; it must not become a case.
    const noFocal = { ...evalReadyTrace("langfuse:d", "support", "q"), focal_step_id: null };
    const result = validate(noFocal);
    const record = traceRecordFromValidation(result, "hash-d");
    const batch = createImportBatch(db, {
      importer: "t",
      importerVersion: "1",
      sourceFingerprint: "d",
    });
    if (record !== null) insertTraces(db, batch.id, [record]);

    const report = curateTask(db, { taskKey: "support" });
    // The diagnostic trace is not eval_ready, so it is never scanned.
    expect(report.casesCreated).toBe(0);
  });

  test("preserves provenance from the trace outcome", () => {
    const withDeterministic = {
      ...evalReadyTrace("langfuse:det", "support", "q"),
      outcome: { deterministic: { status: "success" } },
    };
    const result = validate(withDeterministic);
    const record = traceRecordFromValidation(result, "hash-det");
    const batch = createImportBatch(db, {
      importer: "t",
      importerVersion: "1",
      sourceFingerprint: "det",
    });
    if (record === null) throw new Error("rejected");
    insertTraces(db, batch.id, [record]);

    curateTask(db, { taskKey: "support" });
    const token = openDecisionFirewall("test");
    const cases = listCases(db, { openDecisionFirewall: token });
    expect(cases[0]?.provenance).toBe("deterministic_outcome");
  });

  test("newly created cases keep the source trace lineage", () => {
    seedTrace("langfuse:lineage", "support", "q", "hash-lineage");
    curateTask(db, { taskKey: "support" });
    const token = openDecisionFirewall("test");
    const [c] = listCases(db, { openDecisionFirewall: token });
    expect(c?.sourceTraceId).toBe("langfuse:lineage");
    if (c) expect(getCase(db, c.caseId)?.contentHash).toBe("hash-lineage");
  });
});
