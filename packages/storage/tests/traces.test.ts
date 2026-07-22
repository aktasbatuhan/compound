import { describe, expect, test } from "bun:test";
import { validate } from "@compound/contract";
import {
  countTraces,
  getTraceByTraceId,
  InvalidTraceRecordError,
  insertTraces,
  traceRecordFromValidation,
  UnknownImportBatchError,
} from "../src/index";
import { freshDatabase, loadContractFixture, newBatch, recordFromFixture } from "./helpers";

describe("insertTraces round-trip", () => {
  test("eval_ready fixture survives storage byte-for-byte", () => {
    const handle = freshDatabase();
    const batch = newBatch(handle);
    const original = loadContractFixture("eval-ready-full.json");

    const result = insertTraces(handle, batch.id, [
      recordFromFixture("eval-ready-full.json", () => {}, "sha256:content-1"),
    ]);
    expect(result).toEqual({
      inserted: 1,
      insertedTraceIds: ["langfuse:proj-main:tr-8f3a2c91"],
      duplicates: 0,
      duplicateTraceIds: [],
    });

    const stored = getTraceByTraceId(handle, "langfuse:proj-main:tr-8f3a2c91");
    if (stored === null) throw new Error("expected the trace to be stored");
    expect(stored.trace).toEqual(original as never);
    handle.close();
  });

  test("extracted columns are derived from the payload", () => {
    const handle = freshDatabase();
    const batch = newBatch(handle);
    insertTraces(handle, batch.id, [
      recordFromFixture("eval-ready-full.json", () => {}, "sha256:content-1"),
    ]);
    const stored = getTraceByTraceId(handle, "langfuse:proj-main:tr-8f3a2c91");
    if (stored === null) throw new Error("expected the trace to be stored");

    expect(stored.taskKey).toBe("support.invoice_triage");
    expect(stored.validationClass).toBe("eval_ready");
    expect(stored.diagnosticReasons).toEqual([]);
    expect(stored.importBatchId).toBe(batch.id);
    expect(stored.startedAt.toISOString()).toBe("2026-07-20T14:03:11.000Z");
    expect(stored.endedAt?.toISOString()).toBe("2026-07-20T14:03:19.412Z");
    expect(stored.environment).toBe("production");
    expect(stored.release).toBe("app-v2.14.0");
    expect(stored.sessionId).toBe("sess-991");
    expect(stored.userRef).toBe("user-a41f9c");
    // Focal model/provider come from the focal model_call (gen-2), not gen-1.
    expect(stored.focalStepId).toBe("gen-2");
    expect(stored.focalModel).toBe("gpt-4.1-mini");
    expect(stored.focalProvider).toBe("openai");
    expect(stored.permissions).toEqual({
      judging: true,
      optimization: true,
      fineTuning: false,
    });
    expect(stored.contentHash).toBe("sha256:content-1");
    expect(stored.id).toMatch(/^[0-9a-f-]{36}$/);
    handle.close();
  });

  test("nullable columns stay null on a minimal trace", () => {
    const handle = freshDatabase();
    const batch = newBatch(handle);
    insertTraces(handle, batch.id, [recordFromFixture("eval-ready-minimal.json")]);
    const stored = getTraceByTraceId(handle, "jsonl:local:tr-0001");
    if (stored === null) throw new Error("expected the trace to be stored");

    expect(stored.taskKey).toBeNull();
    expect(stored.endedAt).toBeNull();
    expect(stored.environment).toBeNull();
    expect(stored.release).toBeNull();
    expect(stored.sessionId).toBeNull();
    expect(stored.userRef).toBeNull();
    // The minimal fixture's model_call declares no provider/model.
    expect(stored.focalStepId).toBe("gen-1");
    expect(stored.focalModel).toBeNull();
    expect(stored.focalProvider).toBeNull();
    handle.close();
  });

  test("diagnostic traces persist with their reasons", () => {
    const handle = freshDatabase();
    const batch = newBatch(handle);
    insertTraces(handle, batch.id, [
      recordFromFixture("diagnostic-missing-focal.json"),
      recordFromFixture("diagnostic-unsupported-content.json"),
      recordFromFixture("diagnostic-no-model-calls.json"),
    ]);

    const missingFocal = getTraceByTraceId(handle, "langfuse:proj-main:tr-nofocal01");
    expect(missingFocal?.validationClass).toBe("diagnostic");
    expect(missingFocal?.diagnosticReasons).toEqual(["missing_focal_step_id"]);
    expect(missingFocal?.focalStepId).toBeNull();
    expect(missingFocal?.focalModel).toBeNull();

    const unsupported = getTraceByTraceId(handle, "langfuse:proj-main:tr-imginput1");
    expect(unsupported?.diagnosticReasons).toEqual(["unsupported_content_part"]);
    // The focal step exists here, so model/provider extraction still applies.
    expect(unsupported?.focalStepId).toBe("gen-1");

    const noModelCalls = getTraceByTraceId(handle, "langfuse:proj-main:tr-spansonly1");
    expect(noModelCalls?.diagnosticReasons).toContain("no_model_call_steps");
    handle.close();
  });

  test("rejected fixtures cannot be turned into records", () => {
    const handle = freshDatabase();
    for (const name of ["rejected-bad-envelope.json", "rejected-wrong-types.json"]) {
      expect(traceRecordFromValidation(validate(loadContractFixture(name)), "h")).toBeNull();
    }
    expect(countTraces(handle)).toBe(0);
    handle.close();
  });

  test("class and reasons must agree", () => {
    const handle = freshDatabase();
    const batch = newBatch(handle);
    const evalReady = recordFromFixture("eval-ready-full.json");
    expect(() =>
      insertTraces(handle, batch.id, [{ ...evalReady, diagnosticReasons: ["bogus"] }]),
    ).toThrow(InvalidTraceRecordError);

    const diagnostic = recordFromFixture("diagnostic-missing-focal.json");
    expect(() =>
      insertTraces(handle, batch.id, [{ ...diagnostic, diagnosticReasons: [] }]),
    ).toThrow(InvalidTraceRecordError);
    // Nothing was written by the rejected calls.
    expect(countTraces(handle)).toBe(0);
    handle.close();
  });

  test("inserting into an unknown batch throws before writing", () => {
    const handle = freshDatabase();
    expect(() =>
      insertTraces(handle, "no-such-batch", [recordFromFixture("eval-ready-full.json")]),
    ).toThrow(UnknownImportBatchError);
    expect(countTraces(handle)).toBe(0);
    handle.close();
  });

  test("getTraceByTraceId returns null for an unknown trace_id", () => {
    const handle = freshDatabase();
    expect(getTraceByTraceId(handle, "langfuse:nope")).toBeNull();
    handle.close();
  });
});

describe("duplicate trace_id handling", () => {
  test("re-importing an overlapping export skips and counts duplicates", () => {
    const handle = freshDatabase();
    const first = newBatch(handle);
    const second = newBatch(handle);

    insertTraces(handle, first.id, [
      recordFromFixture("eval-ready-full.json"),
      recordFromFixture("eval-ready-minimal.json"),
    ]);

    const result = insertTraces(handle, second.id, [
      recordFromFixture("eval-ready-full.json", () => {}, "sha256:different"),
      recordFromFixture("diagnostic-missing-focal.json"),
    ]);
    expect(result.inserted).toBe(1);
    expect(result.insertedTraceIds).toEqual(["langfuse:proj-main:tr-nofocal01"]);
    expect(result.duplicates).toBe(1);
    expect(result.duplicateTraceIds).toEqual(["langfuse:proj-main:tr-8f3a2c91"]);

    // The duplicate did not overwrite the original row.
    const stored = getTraceByTraceId(handle, "langfuse:proj-main:tr-8f3a2c91");
    expect(stored?.importBatchId).toBe(first.id);
    expect(stored?.contentHash).toBe("hash:eval-ready-full.json");
    expect(countTraces(handle)).toBe(3);
    handle.close();
  });

  test("duplicates inside one batch are skipped, not crashed on", () => {
    const handle = freshDatabase();
    const batch = newBatch(handle);
    const result = insertTraces(handle, batch.id, [
      recordFromFixture("eval-ready-full.json"),
      recordFromFixture("eval-ready-full.json"),
      recordFromFixture("eval-ready-minimal.json"),
    ]);
    expect(result.inserted).toBe(2);
    expect(result.duplicates).toBe(1);
    expect(countTraces(handle)).toBe(2);
    handle.close();
  });
});
