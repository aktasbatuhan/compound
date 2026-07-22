/**
 * Trace repository.
 *
 * Only `eval_ready` and `diagnostic` traces are persisted; `rejected` records
 * are never stored and exist only as counted line numbers in the batch report
 * (docs/trace-contract-v1.md, "Validation classes").
 *
 * Every extracted column is derived HERE from the payload, so a caller cannot
 * desync the queryable columns from the stored contract trace.
 */
import { type Trace, TraceSchema, type ValidationResult } from "@compound/contract";
import { and, asc, count, desc, eq, gte, inArray, isNull, lte, sql } from "drizzle-orm";
import type { CompoundDatabase } from "./db";
import { type Pagination, paginate } from "./pagination";
import type { PersistedValidationClass, TraceRow } from "./schema";
import { importBatches, traces } from "./schema";

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/**
 * One trace to persist: a validated contract trace paired with the outcome of
 * validation and a caller-supplied content hash of its replayable content.
 */
export interface TraceRecordInput {
  trace: Trace;
  validationClass: PersistedValidationClass;
  /** Required (non-empty) for `diagnostic`, must be empty for `eval_ready`. */
  diagnosticReasons?: readonly string[];
  /** Caller-supplied hash of the replayable content, for later dedupe. */
  contentHash: string;
  /** Internal surrogate id; defaults to a fresh uuid. */
  id?: string;
}

/**
 * Adapt a `validate()` result into a persistable record. Returns `null` for
 * `rejected` results, which are never persisted — the caller reports them by
 * line number in the batch report instead.
 */
export function traceRecordFromValidation(
  result: ValidationResult,
  contentHash: string,
  id?: string,
): TraceRecordInput | null {
  if (result.class === "rejected") return null;
  const record: TraceRecordInput = {
    trace: result.trace,
    validationClass: result.class,
    contentHash,
    diagnosticReasons: result.class === "diagnostic" ? result.diagnostic_reasons : [],
  };
  return id === undefined ? record : { ...record, id };
}

/** Thrown when a record contradicts the contract's validation classes. */
export class InvalidTraceRecordError extends Error {
  constructor(
    readonly traceId: string,
    message: string,
  ) {
    super(`invalid trace record ${traceId}: ${message}`);
    this.name = "InvalidTraceRecordError";
  }
}

/** Thrown when an insert targets an import batch that does not exist. */
export class UnknownImportBatchError extends Error {
  constructor(readonly batchId: string) {
    super(`unknown import batch: ${batchId}`);
    this.name = "UnknownImportBatchError";
  }
}

/**
 * Outcome of a bulk insert. Duplicate `trace_id`s are skipped and reported —
 * re-importing an overlapping export is a normal user action, not a batch
 * failure.
 */
export interface InsertTracesResult {
  inserted: number;
  insertedTraceIds: string[];
  duplicates: number;
  duplicateTraceIds: string[];
}

/** A persisted trace: the stored columns plus the parsed contract trace. */
export interface StoredTrace {
  id: string;
  traceId: string;
  importBatchId: string;
  taskKey: string | null;
  validationClass: PersistedValidationClass;
  diagnosticReasons: string[];
  startedAt: Date;
  endedAt: Date | null;
  environment: string | null;
  release: string | null;
  sessionId: string | null;
  userRef: string | null;
  focalStepId: string | null;
  focalModel: string | null;
  focalProvider: string | null;
  permissions: { judging: boolean; optimization: boolean; fineTuning: boolean };
  contentHash: string;
  createdAt: Date;
  /** The full contract trace, re-parsed through the contract schema. */
  trace: Trace;
}

/**
 * Filters for trace queries.
 *
 * `taskKey`: omit for no filter; pass `null` for the explicit "unassigned"
 * bucket (traces whose `task_key` is null); pass a string to match one key.
 */
export interface TraceFilter {
  taskKey?: string | null;
  validationClass?: PersistedValidationClass | readonly PersistedValidationClass[];
  importBatchId?: string;
  contentHash?: string;
  /** Inclusive lower bound on the trace's `started_at`. */
  startedAtFrom?: Date | string;
  /** Inclusive upper bound on the trace's `started_at`. */
  startedAtTo?: Date | string;
}

export interface ListTracesOptions extends TraceFilter, Pagination {
  /** Defaults to `started_at_desc`. Ties break on `trace_id` ascending. */
  order?: "started_at_asc" | "started_at_desc";
}

// ---------------------------------------------------------------------------
// Derivation from the payload
// ---------------------------------------------------------------------------

function toDate(iso: string): Date {
  return new Date(iso);
}

function focalModelCall(trace: Trace) {
  if (trace.focal_step_id == null) return null;
  const step = trace.steps.find((candidate) => candidate.step_id === trace.focal_step_id);
  return step !== undefined && step.type === "model_call" ? step : null;
}

/**
 * Derive the queryable columns from the contract trace. The requested `model`
 * and `provider` identifiers are used (not `resolved_model`), because the
 * task_key x model matrix is keyed on what was asked for.
 */
function toInsertValues(record: TraceRecordInput, importBatchId: string) {
  const { trace } = record;
  const reasons = record.diagnosticReasons ?? [];
  if (record.validationClass === "diagnostic" && reasons.length === 0) {
    throw new InvalidTraceRecordError(trace.trace_id, "diagnostic traces require reasons");
  }
  if (record.validationClass === "eval_ready" && reasons.length > 0) {
    throw new InvalidTraceRecordError(trace.trace_id, "eval_ready traces must have no reasons");
  }
  const focal = focalModelCall(trace);
  return {
    id: record.id ?? crypto.randomUUID(),
    traceId: trace.trace_id,
    importBatchId,
    taskKey: trace.task_key,
    validationClass: record.validationClass,
    diagnosticReasons: [...reasons],
    startedAt: toDate(trace.started_at),
    endedAt: trace.ended_at == null ? null : toDate(trace.ended_at),
    environment: trace.environment ?? null,
    release: trace.release ?? null,
    sessionId: trace.session_id ?? null,
    userRef: trace.user_ref ?? null,
    focalStepId: trace.focal_step_id,
    focalModel: focal?.model ?? null,
    focalProvider: focal?.provider ?? null,
    permissionJudging: trace.permissions.judging,
    permissionOptimization: trace.permissions.optimization,
    permissionFineTuning: trace.permissions.fine_tuning,
    contentHash: record.contentHash,
    payload: trace,
    createdAt: new Date(),
  };
}

function toStoredTrace(row: TraceRow): StoredTrace {
  return {
    id: row.id,
    traceId: row.traceId,
    importBatchId: row.importBatchId,
    taskKey: row.taskKey,
    validationClass: row.validationClass,
    diagnosticReasons: row.diagnosticReasons,
    startedAt: row.startedAt,
    endedAt: row.endedAt,
    environment: row.environment,
    release: row.release,
    sessionId: row.sessionId,
    userRef: row.userRef,
    focalStepId: row.focalStepId,
    focalModel: row.focalModel,
    focalProvider: row.focalProvider,
    permissions: {
      judging: row.permissionJudging,
      optimization: row.permissionOptimization,
      fineTuning: row.permissionFineTuning,
    },
    contentHash: row.contentHash,
    createdAt: row.createdAt,
    trace: TraceSchema.parse(row.payload),
  };
}

// ---------------------------------------------------------------------------
// Writes
// ---------------------------------------------------------------------------

/**
 * Bulk-insert traces for one batch inside a single transaction.
 *
 * A `trace_id` that already exists (in the store or earlier in `records`) is
 * skipped and counted; it never aborts the batch.
 */
export function insertTraces(
  handle: CompoundDatabase,
  batchId: string,
  records: readonly TraceRecordInput[],
): InsertTracesResult {
  const batchExists = handle.db
    .select({ id: importBatches.id })
    .from(importBatches)
    .where(eq(importBatches.id, batchId))
    .all();
  if (batchExists.length === 0) throw new UnknownImportBatchError(batchId);

  // Derive (and validate) everything before opening the transaction so a bad
  // record fails loudly without leaving a half-written batch.
  const values = records.map((record) => toInsertValues(record, batchId));

  return handle.db.transaction((tx) => {
    const insertedTraceIds: string[] = [];
    const duplicateTraceIds: string[] = [];
    for (const value of values) {
      const returned = tx
        .insert(traces)
        .values(value)
        .onConflictDoNothing({ target: traces.traceId })
        .returning({ traceId: traces.traceId })
        .all();
      if (returned.length > 0) insertedTraceIds.push(value.traceId);
      else duplicateTraceIds.push(value.traceId);
    }
    return {
      inserted: insertedTraceIds.length,
      insertedTraceIds,
      duplicates: duplicateTraceIds.length,
      duplicateTraceIds,
    };
  });
}

// ---------------------------------------------------------------------------
// Reads
// ---------------------------------------------------------------------------

function buildConditions(filter: TraceFilter) {
  const conditions = [];
  if ("taskKey" in filter) {
    if (filter.taskKey === null) conditions.push(isNull(traces.taskKey));
    else if (filter.taskKey !== undefined) conditions.push(eq(traces.taskKey, filter.taskKey));
  }
  if (filter.validationClass !== undefined) {
    conditions.push(
      Array.isArray(filter.validationClass)
        ? inArray(traces.validationClass, [...filter.validationClass])
        : eq(traces.validationClass, filter.validationClass as PersistedValidationClass),
    );
  }
  if (filter.importBatchId !== undefined) {
    conditions.push(eq(traces.importBatchId, filter.importBatchId));
  }
  if (filter.contentHash !== undefined) {
    conditions.push(eq(traces.contentHash, filter.contentHash));
  }
  if (filter.startedAtFrom !== undefined) {
    conditions.push(gte(traces.startedAt, new Date(filter.startedAtFrom)));
  }
  if (filter.startedAtTo !== undefined) {
    conditions.push(lte(traces.startedAt, new Date(filter.startedAtTo)));
  }
  return conditions.length > 0 ? and(...conditions) : undefined;
}

/** Look a trace up by its contract `trace_id` (globally unique). */
export function getTraceByTraceId(handle: CompoundDatabase, traceId: string): StoredTrace | null {
  const [row] = handle.db.select().from(traces).where(eq(traces.traceId, traceId)).all();
  return row === undefined ? null : toStoredTrace(row);
}

export function listTraces(
  handle: CompoundDatabase,
  options: ListTracesOptions = {},
): StoredTrace[] {
  const orderColumn =
    options.order === "started_at_asc" ? asc(traces.startedAt) : desc(traces.startedAt);
  const query = handle.db
    .select()
    .from(traces)
    .where(buildConditions(options))
    .orderBy(orderColumn, asc(traces.traceId))
    .$dynamic();
  return paginate(query, options).all().map(toStoredTrace);
}

/** Total number of traces matching a filter (ignores limit/offset). */
export function countTraces(handle: CompoundDatabase, filter: TraceFilter = {}): number {
  const [row] = handle.db
    .select({ value: count() })
    .from(traces)
    .where(buildConditions(filter))
    .all();
  return row?.value ?? 0;
}

/**
 * Counts per validation class. Both classes are always present (zero when
 * absent), so callers can render the diagnostic queue without a null check.
 */
export function countTracesByValidationClass(
  handle: CompoundDatabase,
  filter: TraceFilter = {},
): Record<PersistedValidationClass, number> {
  const rows = handle.db
    .select({ validationClass: traces.validationClass, value: count() })
    .from(traces)
    .where(buildConditions(filter))
    .groupBy(traces.validationClass)
    .all();
  const result: Record<PersistedValidationClass, number> = { eval_ready: 0, diagnostic: 0 };
  for (const row of rows) result[row.validationClass] = row.value;
  return result;
}

export interface TaskKeyCount {
  /** `null` is the "unassigned" bucket. */
  taskKey: string | null;
  count: number;
}

/**
 * Counts per `task_key`, largest bucket first. The unassigned bucket appears as
 * `taskKey: null` when any unassigned trace matches the filter.
 */
export function countTracesByTaskKey(
  handle: CompoundDatabase,
  filter: TraceFilter = {},
): TaskKeyCount[] {
  return handle.db
    .select({ taskKey: traces.taskKey, count: count() })
    .from(traces)
    .where(buildConditions(filter))
    .groupBy(traces.taskKey)
    .orderBy(desc(count()), sql`${traces.taskKey} is null`, asc(traces.taskKey))
    .all();
}
