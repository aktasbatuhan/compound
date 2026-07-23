/**
 * Drizzle schema for the Compound local store (SQLite).
 *
 * Two tables:
 *
 * - `import_batches` — one row per import run, carrying the import report
 *   described in docs/langfuse-import-mapping.md ("Import report").
 * - `traces` — one row per PERSISTED contract trace. Only `eval_ready` and
 *   `diagnostic` records are stored; `rejected` records are never persisted and
 *   exist only as counted line numbers in the batch report
 *   (docs/trace-contract-v1.md, "Validation classes").
 *
 * Timestamps are stored as epoch milliseconds (`integer` + `timestamp_ms`) so
 * range filters and ordering are correct; the contract's ISO-8601 strings are
 * preserved verbatim inside `payload`, which is the authoritative copy.
 */
import { relations, sql } from "drizzle-orm";
import { index, integer, sqliteTable, text, uniqueIndex } from "drizzle-orm/sqlite-core";

/** Lifecycle of an import batch. */
export const IMPORT_BATCH_STATUSES = ["running", "completed", "failed"] as const;
export type ImportBatchStatus = (typeof IMPORT_BATCH_STATUSES)[number];

/** The only two validation classes that reach persistence. */
export const PERSISTED_VALIDATION_CLASSES = ["eval_ready", "diagnostic"] as const;
export type PersistedValidationClass = (typeof PERSISTED_VALIDATION_CLASSES)[number];

export const importBatches = sqliteTable("import_batches", {
  /** Internal uuid. */
  id: text("id").primaryKey(),
  /** Importer name, e.g. `langfuse`. */
  importer: text("importer").notNull(),
  importerVersion: text("importer_version").notNull(),
  /** Caller-supplied hash of the input files (files, export surface, casing). */
  sourceFingerprint: text("source_fingerprint").notNull(),
  status: text("status", { enum: IMPORT_BATCH_STATUSES }).notNull(),
  startedAt: integer("started_at", { mode: "timestamp_ms" }).notNull(),
  completedAt: integer("completed_at", { mode: "timestamp_ms" }),
  /**
   * JSON import report: counts by validation class, diagnostic-reason
   * histogram, dialects seen, skipped scores, rejected line numbers.
   * Null while the batch is still running.
   */
  report: text("report", { mode: "json" }).$type<ImportReport>(),
  createdAt: integer("created_at", { mode: "timestamp_ms" })
    .notNull()
    .default(sql`(unixepoch() * 1000)`),
});

export const traces = sqliteTable(
  "traces",
  {
    /** Internal surrogate uuid. */
    id: text("id").primaryKey(),
    /** The contract `trace_id`; importers prefix the source, so it is globally unique. */
    traceId: text("trace_id").notNull(),
    importBatchId: text("import_batch_id")
      .notNull()
      .references(() => importBatches.id),
    /** `null` routes the trace to the "unassigned" bucket. */
    taskKey: text("task_key"),
    validationClass: text("validation_class", { enum: PERSISTED_VALIDATION_CLASSES }).notNull(),
    /** JSON array of stable snake_case reasons; empty for `eval_ready`. */
    diagnosticReasons: text("diagnostic_reasons", { mode: "json" })
      .notNull()
      .$type<string[]>()
      .default(sql`'[]'`),
    startedAt: integer("started_at", { mode: "timestamp_ms" }).notNull(),
    endedAt: integer("ended_at", { mode: "timestamp_ms" }),
    environment: text("environment"),
    release: text("release"),
    sessionId: text("session_id"),
    userRef: text("user_ref"),
    focalStepId: text("focal_step_id"),
    /** Extracted from the focal model_call for task_key x model matrix queries. */
    focalModel: text("focal_model"),
    focalProvider: text("focal_provider"),
    permissionJudging: integer("permission_judging", { mode: "boolean" }).notNull(),
    permissionOptimization: integer("permission_optimization", { mode: "boolean" }).notNull(),
    permissionFineTuning: integer("permission_fine_tuning", { mode: "boolean" }).notNull(),
    /** Caller-supplied hash of the replayable content, for later dedupe. */
    contentHash: text("content_hash").notNull(),
    /** The FULL contract trace as imported; authoritative. */
    payload: text("payload", { mode: "json" }).notNull(),
    createdAt: integer("created_at", { mode: "timestamp_ms" })
      .notNull()
      .default(sql`(unixepoch() * 1000)`),
  },
  (table) => [
    uniqueIndex("traces_trace_id_unique").on(table.traceId),
    index("traces_task_key_idx").on(table.taskKey),
    index("traces_validation_class_idx").on(table.validationClass),
    index("traces_started_at_idx").on(table.startedAt),
    index("traces_content_hash_idx").on(table.contentHash),
    index("traces_import_batch_id_idx").on(table.importBatchId),
  ],
);

/** Typed provenance of a case's expected output (docs/curation-v1.md). */
export const CASE_PROVENANCES = [
  "observed_output",
  "human_golden",
  "deterministic_outcome",
  "user_feedback",
  "synthetic_label",
] as const;
export type CaseProvenance = (typeof CASE_PROVENANCES)[number];

/** Immutable data partitions; `decision_test` is sealed (docs/curation-v1.md). */
export const CASE_PARTITIONS = [
  "optimization_train",
  "optimizer_validation",
  "judge_calibration",
  "decision_test",
] as const;
export type CasePartition = (typeof CASE_PARTITIONS)[number];

/** Human review state of a case. */
export const CASE_REVIEW_STATES = ["unreviewed", "approved", "rejected", "needs_edit"] as const;
export type CaseReviewState = (typeof CASE_REVIEW_STATES)[number];

/**
 * Eval cases extracted from eval-ready traces (docs/curation-v1.md).
 *
 * `input` and `expected` hold the replayable request and the typed expected
 * output. `partition` is assigned once from `content_hash` and is immutable —
 * reshuffling it would silently move sealed decision data.
 */
export const cases = sqliteTable(
  "cases",
  {
    id: text("id").primaryKey(),
    /** Stable case identifier derived from the source trace. */
    caseId: text("case_id").notNull(),
    /** A case always belongs to a task; unassigned traces do not become cases. */
    taskKey: text("task_key").notNull(),
    /** Lineage back to the evidence. */
    sourceTraceId: text("source_trace_id").notNull(),
    /** Dedupe key, carried from the trace; drives partition assignment. */
    contentHash: text("content_hash").notNull(),
    provenance: text("provenance", { enum: CASE_PROVENANCES }).notNull(),
    partition: text("partition", { enum: CASE_PARTITIONS }).notNull(),
    reviewState: text("review_state", { enum: CASE_REVIEW_STATES }).notNull().default("unreviewed"),
    /** The focal call's replayable request: `{model?, input, tools_available?}`. */
    input: text("input", { mode: "json" }).notNull(),
    /** Typed expected output; may be null (assertion-gradeable without one). */
    expected: text("expected", { mode: "json" }),
    /** How many duplicate traces collapsed into this case (>= 0). */
    duplicateCount: integer("duplicate_count").notNull().default(0),
    createdAt: integer("created_at", { mode: "timestamp_ms" })
      .notNull()
      .default(sql`(unixepoch() * 1000)`),
    updatedAt: integer("updated_at", { mode: "timestamp_ms" })
      .notNull()
      .default(sql`(unixepoch() * 1000)`),
  },
  (table) => [
    // Dedupe is within a task_key: the same request under two tasks is two cases.
    uniqueIndex("cases_task_content_unique").on(table.taskKey, table.contentHash),
    uniqueIndex("cases_case_id_unique").on(table.caseId),
    index("cases_task_key_idx").on(table.taskKey),
    index("cases_partition_idx").on(table.partition),
    index("cases_provenance_idx").on(table.provenance),
    index("cases_review_state_idx").on(table.reviewState),
    index("cases_source_trace_id_idx").on(table.sourceTraceId),
  ],
);

export type CaseRow = typeof cases.$inferSelect;

export const importBatchesRelations = relations(importBatches, ({ many }) => ({
  traces: many(traces),
}));

export const tracesRelations = relations(traces, ({ one }) => ({
  importBatch: one(importBatches, {
    fields: [traces.importBatchId],
    references: [importBatches.id],
  }),
}));

/**
 * Import report stored with each batch for lineage
 * (docs/langfuse-import-mapping.md, "Import report").
 *
 * The shape is open: importers may add their own keys. The named fields are the
 * ones this layer and the diagnostic-queue views rely on.
 */
export interface ImportReport {
  /** Counts by validation class, including `rejected` (which is never persisted). */
  counts?: {
    eval_ready?: number;
    diagnostic?: number;
    rejected?: number;
    duplicate?: number;
    /**
     * Traces that normalized cleanly and then failed contract validation.
     * That means Compound produced an invalid trace, so this counts OUR bugs —
     * deliberately separate from `rejected`, which counts malformed source
     * records (docs/ingest-pipeline-v1.md).
     */
    internal_normalization_errors?: number;
  };
  /** Histogram of diagnostic reasons across the batch. */
  diagnostic_reasons?: Record<string, number>;
  /** Source dialects seen (e.g. `openai_tool_calls`, `langchain_tool_calls`). */
  dialects?: string[];
  /** Scores dropped because they were not trace-scoped. */
  skipped_scores?: number;
  /** Line numbers of rejected records; the records themselves are never stored. */
  rejected_lines?: number[];
  /** `trace_id`s that were already present and therefore skipped. */
  duplicate_trace_ids?: string[];
  /** Failure detail when the batch ended in `failed`. */
  error?: string;
  [key: string]: unknown;
}

export type ImportBatchRow = typeof importBatches.$inferSelect;
export type TraceRow = typeof traces.$inferSelect;
