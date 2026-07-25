/**
 * Execution repository: budget ledger, completion cache, experiments.
 *
 * The money-safety invariants live here (docs/execution-v1.md):
 * - the ledger is durable and append-only, and never double-charges a
 *   fingerprint;
 * - a cached completion costs $0 and is served without a provider call.
 */
import { and, desc, eq, sql } from "drizzle-orm";
import type { CompoundDatabase } from "./db";
import {
  type CasePartition,
  type CompletionRow,
  completions,
  type ExperimentReport,
  type ExperimentResultRow,
  type ExperimentRow,
  type ExperimentStatus,
  experimentResults,
  experiments,
  spendRecords,
} from "./schema";

// --- budget ledger ---------------------------------------------------------

export class BudgetExceededError extends Error {
  constructor(
    readonly attempted: number,
    readonly limit: number,
    readonly scope: "experiment" | "global",
  ) {
    super(
      `estimated call would exceed the ${scope} budget: ` +
        `$${attempted.toFixed(6)} > $${limit.toFixed(6)}`,
    );
    this.name = "BudgetExceededError";
  }
}

/** Total spend recorded in the global ledger, in USD. */
export function totalSpendUsd(handle: CompoundDatabase): number {
  const [row] = handle.db
    .select({ total: sql<number>`coalesce(sum(${spendRecords.costUsd}), 0)` })
    .from(spendRecords)
    .all();
  return row?.total ?? 0;
}

/** Spend recorded against one experiment, in USD. */
export function experimentSpendUsd(handle: CompoundDatabase, experimentId: string): number {
  const [row] = handle.db
    .select({ total: sql<number>`coalesce(sum(${spendRecords.costUsd}), 0)` })
    .from(spendRecords)
    .where(eq(spendRecords.experimentId, experimentId))
    .all();
  return row?.total ?? 0;
}

/** Whether a fingerprint has already been charged (so it must not be again). */
export function isFingerprintCharged(handle: CompoundDatabase, fingerprint: string): boolean {
  const rows = handle.db
    .select({ id: spendRecords.id })
    .from(spendRecords)
    .where(eq(spendRecords.fingerprint, fingerprint))
    .all();
  return rows.length > 0;
}

export interface RecordSpendInput {
  fingerprint: string;
  costUsd: number;
  experimentId?: string;
}

/**
 * Append a spend record. Idempotent on fingerprint: a fingerprint already in
 * the ledger is a no-op that returns the existing total, so a retried call is
 * never charged twice.
 */
export function recordSpend(handle: CompoundDatabase, input: RecordSpendInput): void {
  if (isFingerprintCharged(handle, input.fingerprint)) return;
  handle.db
    .insert(spendRecords)
    .values({
      id: crypto.randomUUID(),
      experimentId: input.experimentId ?? null,
      fingerprint: input.fingerprint,
      costUsd: input.costUsd,
    })
    .run();
}

/**
 * Enforce both caps before a call. Throws `BudgetExceededError` if adding
 * `estimatedCost` would exceed the per-experiment cap or the global hard limit.
 * A fingerprint already charged is free to proceed (it will hit the cache).
 */
export function requireBudgetHeadroom(
  handle: CompoundDatabase,
  params: {
    fingerprint: string;
    estimatedCost: number;
    experimentId: string;
    experimentCapUsd: number;
    globalHardLimitUsd: number;
  },
): void {
  if (isFingerprintCharged(handle, params.fingerprint)) return;

  const globalAfter = totalSpendUsd(handle) + params.estimatedCost;
  if (globalAfter > params.globalHardLimitUsd) {
    throw new BudgetExceededError(globalAfter, params.globalHardLimitUsd, "global");
  }
  const experimentAfter = experimentSpendUsd(handle, params.experimentId) + params.estimatedCost;
  if (experimentAfter > params.experimentCapUsd) {
    throw new BudgetExceededError(experimentAfter, params.experimentCapUsd, "experiment");
  }
}

// --- completion cache ------------------------------------------------------

export function getCachedCompletion(
  handle: CompoundDatabase,
  fingerprint: string,
): CompletionRow | null {
  const [row] = handle.db
    .select()
    .from(completions)
    .where(eq(completions.fingerprint, fingerprint))
    .all();
  return row ?? null;
}

export interface CacheCompletionInput {
  fingerprint: string;
  provider: string;
  model: string;
  resolvedModel?: string | null;
  params: unknown;
  output: unknown;
  usage?: unknown;
  finishReason?: string | null;
  latencyMs?: number | null;
  costUsd: number;
}

/** Store a completion. Idempotent: an existing fingerprint is left untouched. */
export function cacheCompletion(handle: CompoundDatabase, input: CacheCompletionInput): void {
  if (getCachedCompletion(handle, input.fingerprint) !== null) return;
  handle.db
    .insert(completions)
    .values({
      fingerprint: input.fingerprint,
      provider: input.provider,
      model: input.model,
      resolvedModel: input.resolvedModel ?? null,
      paramsJson: input.params,
      outputJson: input.output,
      usageJson: input.usage ?? null,
      finishReason: input.finishReason ?? null,
      latencyMs: input.latencyMs ?? null,
      costUsd: input.costUsd,
    })
    .run();
}

// --- experiments -----------------------------------------------------------

export interface CreateExperimentInput {
  taskKey: string;
  candidateModel: string;
  provider: string;
  partition: CasePartition;
  paid: boolean;
  id?: string;
}

export function createExperiment(
  handle: CompoundDatabase,
  input: CreateExperimentInput,
): ExperimentRow {
  const id = input.id ?? crypto.randomUUID();
  handle.db
    .insert(experiments)
    .values({
      id,
      taskKey: input.taskKey,
      candidateModel: input.candidateModel,
      provider: input.provider,
      partition: input.partition,
      status: "running",
      paid: input.paid,
      startedAt: new Date(),
    })
    .run();
  return getExperiment(handle, id) as ExperimentRow;
}

export function finishExperiment(
  handle: CompoundDatabase,
  id: string,
  status: Extract<ExperimentStatus, "completed" | "failed">,
  report: ExperimentReport,
): ExperimentRow {
  handle.db
    .update(experiments)
    .set({ status, report, completedAt: new Date() })
    .where(eq(experiments.id, id))
    .run();
  return getExperiment(handle, id) as ExperimentRow;
}

export function getExperiment(handle: CompoundDatabase, id: string): ExperimentRow | null {
  const [row] = handle.db.select().from(experiments).where(eq(experiments.id, id)).all();
  return row ?? null;
}

// --- per-case results ------------------------------------------------------

export interface CaseResultInput {
  caseId: string;
  status: "graded" | "skipped" | "cache_miss_dry_run";
  passed?: boolean;
  score?: number;
  judgeAbstained?: boolean;
}

/**
 * Persist the per-case outcomes of an experiment so a gate can pair candidate
 * and reference by `case_id`. Idempotent per (experiment, case): re-running an
 * identical experiment overwrites its own rows rather than duplicating them.
 */
export function recordCaseResults(
  handle: CompoundDatabase,
  experimentId: string,
  results: readonly CaseResultInput[],
): void {
  if (results.length === 0) return;
  for (const r of results) {
    handle.db
      .insert(experimentResults)
      .values({
        experimentId,
        caseId: r.caseId,
        status: r.status,
        passed: r.passed ?? null,
        score: r.score ?? null,
        judgeAbstained: r.judgeAbstained ?? false,
      })
      .onConflictDoUpdate({
        target: [experimentResults.experimentId, experimentResults.caseId],
        set: {
          status: r.status,
          passed: r.passed ?? null,
          score: r.score ?? null,
          judgeAbstained: r.judgeAbstained ?? false,
        },
      })
      .run();
  }
}

export function getExperimentResults(
  handle: CompoundDatabase,
  experimentId: string,
): ExperimentResultRow[] {
  return handle.db
    .select()
    .from(experimentResults)
    .where(eq(experimentResults.experimentId, experimentId))
    .orderBy(experimentResults.caseId)
    .all();
}

export interface ListExperimentsFilter {
  taskKey?: string;
  candidateModel?: string;
  limit?: number;
  offset?: number;
}

export function listExperiments(
  handle: CompoundDatabase,
  filter: ListExperimentsFilter = {},
): ExperimentRow[] {
  const conditions = [];
  if (filter.taskKey !== undefined) conditions.push(eq(experiments.taskKey, filter.taskKey));
  if (filter.candidateModel !== undefined) {
    conditions.push(eq(experiments.candidateModel, filter.candidateModel));
  }
  let query = handle.db
    .select()
    .from(experiments)
    .where(conditions.length > 0 ? and(...conditions) : undefined)
    .orderBy(desc(experiments.startedAt), experiments.id)
    .$dynamic();
  if (filter.limit !== undefined) query = query.limit(filter.limit);
  if (filter.offset !== undefined) query = query.offset(filter.offset);
  return query.all();
}
