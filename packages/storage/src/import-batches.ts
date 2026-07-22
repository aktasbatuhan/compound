/**
 * Import batch repository.
 *
 * One batch per import run. The batch owns the import report
 * (docs/langfuse-import-mapping.md, "Import report") — including the counted
 * line numbers of `rejected` records, which are never persisted as traces.
 */
import { and, count, desc, eq } from "drizzle-orm";
import type { CompoundDatabase } from "./db";
import { paginate } from "./pagination";
import type { ImportBatchRow, ImportBatchStatus, ImportReport } from "./schema";
import { importBatches } from "./schema";

/** Thrown when an operation targets an import batch id that does not exist. */
export class ImportBatchNotFoundError extends Error {
  constructor(readonly batchId: string) {
    super(`import batch not found: ${batchId}`);
    this.name = "ImportBatchNotFoundError";
  }
}

export interface CreateImportBatchInput {
  importer: string;
  importerVersion: string;
  /** Caller-supplied hash of the input files. */
  sourceFingerprint: string;
  /** Defaults to a fresh uuid. */
  id?: string;
  /** Defaults to now. */
  startedAt?: Date;
}

export interface ListImportBatchesFilter {
  status?: ImportBatchStatus;
  importer?: string;
  limit?: number;
  offset?: number;
}

/** Open a new batch in `running` state, with no report yet. */
export function createImportBatch(
  handle: CompoundDatabase,
  input: CreateImportBatchInput,
): ImportBatchRow {
  const [row] = handle.db
    .insert(importBatches)
    .values({
      id: input.id ?? crypto.randomUUID(),
      importer: input.importer,
      importerVersion: input.importerVersion,
      sourceFingerprint: input.sourceFingerprint,
      status: "running",
      startedAt: input.startedAt ?? new Date(),
      completedAt: null,
      report: null,
      createdAt: new Date(),
    })
    .returning()
    .all();
  if (row === undefined) throw new Error("failed to create import batch");
  return row;
}

/** Close a batch successfully, storing its report. */
export function completeImportBatch(
  handle: CompoundDatabase,
  id: string,
  report: ImportReport,
  completedAt: Date = new Date(),
): ImportBatchRow {
  return finishImportBatch(handle, id, "completed", report, completedAt);
}

/**
 * Close a batch as failed. The report is optional: a batch can fail before it
 * has produced one, in which case the existing (possibly null) report is kept.
 */
export function failImportBatch(
  handle: CompoundDatabase,
  id: string,
  report?: ImportReport,
  completedAt: Date = new Date(),
): ImportBatchRow {
  return finishImportBatch(handle, id, "failed", report, completedAt);
}

function finishImportBatch(
  handle: CompoundDatabase,
  id: string,
  status: Extract<ImportBatchStatus, "completed" | "failed">,
  report: ImportReport | undefined,
  completedAt: Date,
): ImportBatchRow {
  const values = report === undefined ? { status, completedAt } : { status, completedAt, report };
  const [row] = handle.db
    .update(importBatches)
    .set(values)
    .where(eq(importBatches.id, id))
    .returning()
    .all();
  if (row === undefined) throw new ImportBatchNotFoundError(id);
  return row;
}

export function getImportBatch(handle: CompoundDatabase, id: string): ImportBatchRow | null {
  const [row] = handle.db.select().from(importBatches).where(eq(importBatches.id, id)).all();
  return row ?? null;
}

function batchConditions(filter: ListImportBatchesFilter) {
  const conditions = [];
  if (filter.status !== undefined) conditions.push(eq(importBatches.status, filter.status));
  if (filter.importer !== undefined) conditions.push(eq(importBatches.importer, filter.importer));
  return conditions.length > 0 ? and(...conditions) : undefined;
}

/** Newest batches first (by `started_at`, tie-broken by `id` for determinism). */
export function listImportBatches(
  handle: CompoundDatabase,
  filter: ListImportBatchesFilter = {},
): ImportBatchRow[] {
  const query = handle.db
    .select()
    .from(importBatches)
    .where(batchConditions(filter))
    .orderBy(desc(importBatches.startedAt), importBatches.id)
    .$dynamic();
  return paginate(query, filter).all();
}

/**
 * Total batches matching `filter`, ignoring `limit`/`offset` — the denominator
 * for a paginated listing.
 */
export function countImportBatches(
  handle: CompoundDatabase,
  filter: ListImportBatchesFilter = {},
): number {
  const [row] = handle.db
    .select({ value: count() })
    .from(importBatches)
    .where(batchConditions(filter))
    .all();
  return row?.value ?? 0;
}
