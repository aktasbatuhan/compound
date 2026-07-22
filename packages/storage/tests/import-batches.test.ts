import { describe, expect, test } from "bun:test";
import {
  completeImportBatch,
  createImportBatch,
  failImportBatch,
  getImportBatch,
  ImportBatchNotFoundError,
  type ImportReport,
  listImportBatches,
} from "../src/index";
import { freshDatabase } from "./helpers";

const REPORT: ImportReport = {
  counts: { eval_ready: 12, diagnostic: 3, rejected: 2 },
  diagnostic_reasons: { missing_focal_step_id: 2, unsupported_content_part: 1 },
  dialects: ["openai_tool_calls", "langchain_tool_calls"],
  skipped_scores: 4,
  rejected_lines: [17, 41],
};

describe("import batch lifecycle", () => {
  test("created batches start running with no report", () => {
    const handle = freshDatabase();
    const batch = createImportBatch(handle, {
      importer: "langfuse",
      importerVersion: "0.1.0",
      sourceFingerprint: "sha256:abc",
    });
    expect(batch.status).toBe("running");
    expect(batch.report).toBeNull();
    expect(batch.completedAt).toBeNull();
    expect(batch.id).toMatch(/^[0-9a-f-]{36}$/);
    expect(batch.startedAt).toBeInstanceOf(Date);
    handle.close();
  });

  test("completing stores the report and closes the batch", () => {
    const handle = freshDatabase();
    const batch = createImportBatch(handle, {
      importer: "langfuse",
      importerVersion: "0.1.0",
      sourceFingerprint: "sha256:abc",
    });
    const completed = completeImportBatch(handle, batch.id, REPORT);
    expect(completed.status).toBe("completed");
    expect(completed.completedAt).toBeInstanceOf(Date);
    expect(completed.report).toEqual(REPORT);

    const reloaded = getImportBatch(handle, batch.id);
    expect(reloaded?.report?.rejected_lines).toEqual([17, 41]);
    expect(reloaded?.report?.diagnostic_reasons).toEqual({
      missing_focal_step_id: 2,
      unsupported_content_part: 1,
    });
    handle.close();
  });

  test("failing with a report stores it", () => {
    const handle = freshDatabase();
    const batch = createImportBatch(handle, {
      importer: "langfuse",
      importerVersion: "0.1.0",
      sourceFingerprint: "sha256:abc",
    });
    const failed = failImportBatch(handle, batch.id, { ...REPORT, error: "unreadable file" });
    expect(failed.status).toBe("failed");
    expect(failed.report?.error).toBe("unreadable file");
    expect(failed.completedAt).toBeInstanceOf(Date);
    handle.close();
  });

  test("failing without a report keeps the batch closable before one exists", () => {
    const handle = freshDatabase();
    const batch = createImportBatch(handle, {
      importer: "langfuse",
      importerVersion: "0.1.0",
      sourceFingerprint: "sha256:abc",
    });
    const failed = failImportBatch(handle, batch.id);
    expect(failed.status).toBe("failed");
    expect(failed.report).toBeNull();
    handle.close();
  });

  test("completing or failing an unknown batch throws", () => {
    const handle = freshDatabase();
    expect(() => completeImportBatch(handle, "nope", REPORT)).toThrow(ImportBatchNotFoundError);
    expect(() => failImportBatch(handle, "nope")).toThrow(ImportBatchNotFoundError);
    handle.close();
  });

  test("getImportBatch returns null for an unknown id", () => {
    const handle = freshDatabase();
    expect(getImportBatch(handle, "nope")).toBeNull();
    handle.close();
  });
});

describe("listImportBatches", () => {
  function seed(handle: ReturnType<typeof freshDatabase>) {
    const a = createImportBatch(handle, {
      importer: "langfuse",
      importerVersion: "0.1.0",
      sourceFingerprint: "sha256:a",
      startedAt: new Date("2026-07-01T00:00:00Z"),
    });
    const b = createImportBatch(handle, {
      importer: "plain_json",
      importerVersion: "0.1.0",
      sourceFingerprint: "sha256:b",
      startedAt: new Date("2026-07-02T00:00:00Z"),
    });
    const c = createImportBatch(handle, {
      importer: "langfuse",
      importerVersion: "0.1.0",
      sourceFingerprint: "sha256:c",
      startedAt: new Date("2026-07-03T00:00:00Z"),
    });
    return { a, b, c };
  }

  test("newest first", () => {
    const handle = freshDatabase();
    const { a, b, c } = seed(handle);
    expect(listImportBatches(handle).map((row) => row.id)).toEqual([c.id, b.id, a.id]);
    handle.close();
  });

  test("filters by status and importer", () => {
    const handle = freshDatabase();
    const { a, b, c } = seed(handle);
    completeImportBatch(handle, c.id, REPORT);
    failImportBatch(handle, a.id);

    expect(listImportBatches(handle, { status: "running" }).map((row) => row.id)).toEqual([b.id]);
    expect(listImportBatches(handle, { importer: "langfuse" }).map((row) => row.id)).toEqual([
      c.id,
      a.id,
    ]);
    expect(
      listImportBatches(handle, { importer: "langfuse", status: "failed" }).map((row) => row.id),
    ).toEqual([a.id]);
    handle.close();
  });

  test("limit and offset", () => {
    const handle = freshDatabase();
    const { a, b, c } = seed(handle);
    expect(listImportBatches(handle, { limit: 2 }).map((row) => row.id)).toEqual([c.id, b.id]);
    expect(listImportBatches(handle, { offset: 1 }).map((row) => row.id)).toEqual([b.id, a.id]);
    expect(listImportBatches(handle, { limit: 1, offset: 2 }).map((row) => row.id)).toEqual([a.id]);
    handle.close();
  });
});
