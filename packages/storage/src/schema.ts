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
import {
  index,
  integer,
  primaryKey,
  real,
  sqliteTable,
  text,
  uniqueIndex,
} from "drizzle-orm/sqlite-core";

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

// ---------------------------------------------------------------------------
// Execution: experiments, completion cache, spend ledger (docs/execution-v1.md)
// ---------------------------------------------------------------------------

export const EXPERIMENT_STATUSES = ["running", "completed", "failed"] as const;
export type ExperimentStatus = (typeof EXPERIMENT_STATUSES)[number];

/** One run of a candidate model against a task's cases in a partition. */
export const experiments = sqliteTable(
  "experiments",
  {
    id: text("id").primaryKey(),
    taskKey: text("task_key").notNull(),
    candidateModel: text("candidate_model").notNull(),
    provider: text("provider").notNull(),
    partition: text("partition", { enum: CASE_PARTITIONS }).notNull(),
    status: text("status", { enum: EXPERIMENT_STATUSES }).notNull(),
    /** true if real provider calls were permitted; false for a dry run. */
    paid: integer("paid", { mode: "boolean" }).notNull().default(false),
    /** JSON run report: counts, pass rate, cost, cache hits, run fingerprint. */
    report: text("report", { mode: "json" }).$type<ExperimentReport>(),
    startedAt: integer("started_at", { mode: "timestamp_ms" }).notNull(),
    completedAt: integer("completed_at", { mode: "timestamp_ms" }),
    createdAt: integer("created_at", { mode: "timestamp_ms" })
      .notNull()
      .default(sql`(unixepoch() * 1000)`),
  },
  (table) => [
    index("experiments_task_key_idx").on(table.taskKey),
    index("experiments_candidate_model_idx").on(table.candidateModel),
  ],
);

export type ExperimentRow = typeof experiments.$inferSelect;

/**
 * Content-addressed completion cache. A hit costs $0 and makes no provider
 * call, so a re-run of an identical experiment is free (docs/execution-v1.md).
 */
export const completions = sqliteTable("completions", {
  /** Fingerprint over {provider, model, params, input, provider_revision}. */
  fingerprint: text("fingerprint").primaryKey(),
  provider: text("provider").notNull(),
  model: text("model").notNull(),
  /** The resolved model id the provider actually served, when reported. */
  resolvedModel: text("resolved_model"),
  paramsJson: text("params_json", { mode: "json" }),
  outputJson: text("output_json", { mode: "json" }).notNull(),
  usageJson: text("usage_json", { mode: "json" }),
  finishReason: text("finish_reason"),
  latencyMs: integer("latency_ms"),
  costUsd: real("cost_usd").notNull().default(0),
  createdAt: integer("created_at", { mode: "timestamp_ms" })
    .notNull()
    .default(sql`(unixepoch() * 1000)`),
});

export type CompletionRow = typeof completions.$inferSelect;

/**
 * Durable, append-only spend ledger. Every paid call appends a record with its
 * fingerprint; the running total is the sum. A fingerprint present here was
 * already charged and is never double-charged.
 */
export const spendRecords = sqliteTable(
  "spend_records",
  {
    id: text("id").primaryKey(),
    experimentId: text("experiment_id"),
    fingerprint: text("fingerprint").notNull(),
    costUsd: real("cost_usd").notNull(),
    createdAt: integer("created_at", { mode: "timestamp_ms" })
      .notNull()
      .default(sql`(unixepoch() * 1000)`),
  },
  (table) => [
    uniqueIndex("spend_records_fingerprint_unique").on(table.fingerprint),
    index("spend_records_experiment_id_idx").on(table.experimentId),
  ],
);

export type SpendRecordRow = typeof spendRecords.$inferSelect;

/** Stored on each experiment (docs/execution-v1.md, "Experiment and run"). */
export interface ExperimentReport {
  cases_total?: number;
  cases_graded?: number;
  cases_skipped?: number;
  passed?: number;
  pass_rate?: number;
  mean_score?: number;
  cache_hits?: number;
  provider_calls?: number;
  estimated_cost_usd?: number;
  actual_cost_usd?: number;
  run_fingerprint?: string;
  skip_reasons?: Record<string, number>;
  error?: string;
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Gate decision (docs/gate-decision-v1.md, Step 6)
// ---------------------------------------------------------------------------

/** Per-case outcome of one experiment, persisted so a gate can pair by case. */
export const experimentResults = sqliteTable(
  "experiment_results",
  {
    experimentId: text("experiment_id")
      .notNull()
      .references(() => experiments.id),
    caseId: text("case_id").notNull(),
    status: text("status", { enum: ["graded", "skipped", "cache_miss_dry_run"] }).notNull(),
    /** Null unless status is `graded`. */
    passed: integer("passed", { mode: "boolean" }),
    score: real("score"),
    /** True if a judge abstained on this case (judge-graded tasks only). */
    judgeAbstained: integer("judge_abstained", { mode: "boolean" }).notNull().default(false),
    /**
     * Fingerprint of the completion this case produced, so a judge can fetch the
     * exact cached output it graded (docs/judges-v1.md). Null for skipped cases.
     */
    completionFingerprint: text("completion_fingerprint"),
    createdAt: integer("created_at", { mode: "timestamp_ms" })
      .notNull()
      .default(sql`(unixepoch() * 1000)`),
  },
  (table) => [
    primaryKey({ columns: [table.experimentId, table.caseId] }),
    index("experiment_results_experiment_id_idx").on(table.experimentId),
  ],
);

export type ExperimentResultRow = typeof experimentResults.$inferSelect;

export const GATE_METRICS = ["pass_rate", "mean_score"] as const;
export type GateMetric = (typeof GATE_METRICS)[number];

export const GATE_MODES = ["non_inferiority", "superiority"] as const;
export type GateMode = (typeof GATE_MODES)[number];

export const GATE_OUTCOMES = [
  "meets_gate",
  "fails_gate",
  "insufficient_data",
  "judge_abstained",
  "no_reliable_improvement",
] as const;
export type GateOutcome = (typeof GATE_OUTCOMES)[number];

/**
 * A pre-declared decision rule, content-hashed BEFORE any decision-partition
 * result exists (docs/gate-decision-v1.md, "Pre-declared rule"). `specHash` is
 * unique: re-declaring the identical rule reuses the row.
 */
export const gateSpecs = sqliteTable(
  "gate_specs",
  {
    id: text("id").primaryKey(),
    specHash: text("spec_hash").notNull(),
    taskKey: text("task_key").notNull(),
    candidateModel: text("candidate_model").notNull(),
    referenceModel: text("reference_model").notNull(),
    metric: text("metric", { enum: GATE_METRICS }).notNull(),
    mode: text("mode", { enum: GATE_MODES }).notNull(),
    margin: real("margin").notNull(),
    confidence: real("confidence").notNull(),
    minCases: integer("min_cases").notNull(),
    judgeAbstainMax: real("judge_abstain_max").notNull().default(0),
    /**
     * When the candidate ran with an optimized prompt, its content hash — part
     * of the pre-declared rule, so "model M with prompt P" is a different rule
     * than plain "model M". Null for a baseline gate.
     */
    candidatePromptHash: text("candidate_prompt_hash"),
    /** Provenance: the optimization artifact whose prompt was under test. */
    optimizationRunId: text("optimization_run_id").references(() => optimizationRuns.id),
    /**
     * The providers each side ran on, when a provider was named (the provider
     * axis). Part of the rule when set, so "model M on A" vs "model M on B" is a
     * distinct declaration. Null means the model's default provider.
     */
    candidateProvider: text("candidate_provider"),
    referenceProvider: text("reference_provider"),
    /** The stated reason for opening the sealed partition — required. */
    firewallReason: text("firewall_reason").notNull(),
    createdAt: integer("created_at", { mode: "timestamp_ms" })
      .notNull()
      .default(sql`(unixepoch() * 1000)`),
  },
  (table) => [
    uniqueIndex("gate_specs_spec_hash_unique").on(table.specHash),
    index("gate_specs_task_key_idx").on(table.taskKey),
  ],
);

export type GateSpecRow = typeof gateSpecs.$inferSelect;

/** The decided outcome of a gate: the verdict plus the number and its CI. */
export const gateResults = sqliteTable(
  "gate_results",
  {
    id: text("id").primaryKey(),
    gateSpecId: text("gate_spec_id")
      .notNull()
      .references(() => gateSpecs.id),
    candidateExperimentId: text("candidate_experiment_id")
      .notNull()
      .references(() => experiments.id),
    referenceExperimentId: text("reference_experiment_id")
      .notNull()
      .references(() => experiments.id),
    outcome: text("outcome", { enum: GATE_OUTCOMES }).notNull(),
    /** Candidate-minus-reference point estimate on the chosen metric. */
    delta: real("delta").notNull(),
    ciLo: real("ci_lo").notNull(),
    ciHi: real("ci_hi").notNull(),
    n: integer("n").notNull(),
    candidateRate: real("candidate_rate").notNull(),
    referenceRate: real("reference_rate").notNull(),
    judgeAbstainedFraction: real("judge_abstained_fraction").notNull().default(0),
    decidedAt: integer("decided_at", { mode: "timestamp_ms" })
      .notNull()
      .default(sql`(unixepoch() * 1000)`),
  },
  (table) => [
    index("gate_results_gate_spec_id_idx").on(table.gateSpecId),
    index("gate_results_task_lookup_idx").on(table.candidateExperimentId),
  ],
);

export type GateResultRow = typeof gateResults.$inferSelect;

// ---------------------------------------------------------------------------
// Judge calibration (docs/judges-v1.md)
// ---------------------------------------------------------------------------

export const JUDGE_MODES = ["pointwise", "pairwise"] as const;
export type JudgeMode = (typeof JUDGE_MODES)[number];

/**
 * A measured calibration of a judge against human labels, PINNED to the exact
 * (task, judge_model, prompt_version, rubric_hash) it was measured on. Changing
 * any pin field invalidates the calibration — the judge returns to uncalibrated
 * until re-measured (docs/judges-v1.md, "Pin judge model + prompt versions").
 * The latest row per pin is authoritative.
 */
export const judgeCalibrations = sqliteTable(
  "judge_calibrations",
  {
    id: text("id").primaryKey(),
    taskKey: text("task_key").notNull(),
    judgeModel: text("judge_model").notNull(),
    promptVersion: text("prompt_version").notNull(),
    rubricHash: text("rubric_hash").notNull(),
    mode: text("mode", { enum: JUDGE_MODES }).notNull(),
    /** Cohen's kappa agreement with human labels, with its bootstrap CI. */
    agreementKappa: real("agreement_kappa").notNull(),
    kappaCiLo: real("kappa_ci_lo").notNull(),
    kappaCiHi: real("kappa_ci_hi").notNull(),
    /** Number of human-labelled calibration cases used. */
    n: integer("n").notNull(),
    /** Fraction of pairwise judgments that flipped on order swap (pairwise only). */
    positionBiasRate: real("position_bias_rate").notNull().default(0),
    /** The threshold this measurement was compared against. */
    threshold: real("threshold").notNull(),
    /** True iff agreement met the threshold: the judge may feed a gate. */
    calibrated: integer("calibrated", { mode: "boolean" }).notNull(),
    measuredAt: integer("measured_at", { mode: "timestamp_ms" })
      .notNull()
      .default(sql`(unixepoch() * 1000)`),
  },
  (table) => [
    index("judge_calibrations_task_key_idx").on(table.taskKey),
    index("judge_calibrations_pin_idx").on(
      table.taskKey,
      table.judgeModel,
      table.promptVersion,
      table.rubricHash,
    ),
  ],
);

export type JudgeCalibrationRow = typeof judgeCalibrations.$inferSelect;

// ---------------------------------------------------------------------------
// Optimization artifacts (docs/optimization-v1.md, Step 7)
// ---------------------------------------------------------------------------

/**
 * One GEPA optimization run: the seed prompt improved into an optimized prompt,
 * with before/after validation scores and full provenance. The optimized prompt
 * is a PROPOSAL — adopting it is a separate, gated step (re-gate on the sealed
 * set + human approval). Optimization never self-certifies.
 */
export const optimizationRuns = sqliteTable(
  "optimization_runs",
  {
    id: text("id").primaryKey(),
    taskKey: text("task_key").notNull(),
    candidateModel: text("candidate_model").notNull(),
    seedPrompt: text("seed_prompt").notNull(),
    optimizedPrompt: text("optimized_prompt").notNull(),
    beforeValScore: real("before_val_score").notNull(),
    afterValScore: real("after_val_score").notNull(),
    valCases: integer("val_cases").notNull(),
    reflectionCalls: integer("reflection_calls").notNull().default(0),
    /** Why this run was run: the eligibility reason (or "forced"). */
    eligibilityReason: text("eligibility_reason"),
    costUsd: real("cost_usd").notNull().default(0),
    createdAt: integer("created_at", { mode: "timestamp_ms" })
      .notNull()
      .default(sql`(unixepoch() * 1000)`),
  },
  (table) => [index("optimization_runs_task_key_idx").on(table.taskKey)],
);

export type OptimizationRunRow = typeof optimizationRuns.$inferSelect;

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
