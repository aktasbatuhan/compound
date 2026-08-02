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
import { serviceTierOf, usageTokens } from "./telemetry";

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
  /** Upstream host a broker (e.g. OpenRouter) dispatched to; null for direct providers (#9). */
  upstreamProvider?: string | null;
  params: unknown;
  output: unknown;
  usage?: unknown;
  finishReason?: string | null;
  latencyMs?: number | null;
  /** Async-queue portion of latency for a flex route; null for sync providers. */
  queueMs?: number | null;
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
      upstreamProvider: input.upstreamProvider ?? null,
      paramsJson: input.params,
      outputJson: input.output,
      usageJson: input.usage ?? null,
      finishReason: input.finishReason ?? null,
      latencyMs: input.latencyMs ?? null,
      queueMs: input.queueMs ?? null,
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
  /** Fingerprint of the completion this case produced (for later judge grading). */
  completionFingerprint?: string;
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
        completionFingerprint: r.completionFingerprint ?? null,
      })
      .onConflictDoUpdate({
        target: [experimentResults.experimentId, experimentResults.caseId],
        set: {
          status: r.status,
          passed: r.passed ?? null,
          score: r.score ?? null,
          judgeAbstained: r.judgeAbstained ?? false,
          completionFingerprint: r.completionFingerprint ?? null,
        },
      })
      .run();
  }
}

/**
 * Overwrite a case's grade with a judge's verdict: its score, whether it passed
 * the judge's decision point, and whether the judge abstained (an uncalibrated
 * judge always abstains, and the gate then excludes the case). Only the judgment
 * fields change; status and the completion fingerprint are left intact.
 */
export function setCaseJudgment(
  handle: CompoundDatabase,
  experimentId: string,
  caseId: string,
  judgment: { score: number; passed: boolean; judgeAbstained: boolean },
): void {
  handle.db
    .update(experimentResults)
    .set({
      score: judgment.score,
      passed: judgment.passed,
      judgeAbstained: judgment.judgeAbstained,
    })
    .where(
      and(eq(experimentResults.experimentId, experimentId), eq(experimentResults.caseId, caseId)),
    )
    .run();
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

/** One experiment's outcome for a single case, joined to the output it produced. */
export interface CaseOutcomeRow {
  experimentId: string;
  candidateModel: string;
  partition: CasePartition;
  status: string;
  passed: boolean | null;
  score: number | null;
  /** The model's completion (parsed message), or null if it was never executed. */
  output: unknown;
  costUsd: number | null;
  startedAt: Date;
}

/**
 * Every experiment's outcome for one case, newest first: how each candidate model
 * scored on it and what it actually output. Joins the per-case grade to the
 * experiment that produced it and the cached completion that holds the output — so
 * a reviewer can see, side by side, why two models disagreed on the same case.
 */
export function listCaseOutcomes(handle: CompoundDatabase, caseId: string): CaseOutcomeRow[] {
  const rows = handle.db
    .select({
      experimentId: experimentResults.experimentId,
      candidateModel: experiments.candidateModel,
      partition: experiments.partition,
      status: experimentResults.status,
      passed: experimentResults.passed,
      score: experimentResults.score,
      output: completions.outputJson,
      costUsd: completions.costUsd,
      startedAt: experiments.startedAt,
    })
    .from(experimentResults)
    .innerJoin(experiments, eq(experimentResults.experimentId, experiments.id))
    .leftJoin(completions, eq(experimentResults.completionFingerprint, completions.fingerprint))
    .where(eq(experimentResults.caseId, caseId))
    .orderBy(desc(experiments.startedAt))
    .all();
  return rows.map((r) => ({
    experimentId: r.experimentId,
    candidateModel: r.candidateModel,
    partition: r.partition,
    status: r.status,
    passed: r.passed ?? null,
    score: r.score ?? null,
    output: r.output ?? null,
    costUsd: r.costUsd ?? null,
    startedAt: r.startedAt,
  }));
}

/** An experiment's score-and-cost summary — the two axes of a model comparison. */
export interface ExperimentScoreCost {
  /** Total per-case rows. */
  cases: number;
  /** Rows with a pass/fail grade (a judge may abstain, leaving some ungraded). */
  graded: number;
  passed: number;
  meanScore: number | null;
  /** Sum of the cost of every case's completion (cached completions count at $0). */
  totalCostUsd: number;
  /** Cost averaged over the cases that have a completion. */
  meanCostUsd: number | null;
  /** Mean full-request latency (ms) over completions that recorded one; flex queue included. */
  meanLatencyMs: number | null;
  /** Completions that contributed a latency sample (for aggregating the mean). */
  latencyCount: number;
  /** Mean output tokens/sec (output tokens over full latency), matching telemetry's TPS. */
  meanTps: number | null;
  /** Completions that contributed a TPS sample. */
  tpsCount: number;
  /** The provider these completions ran on (one per experiment). */
  provider: string | null;
  /** The upstream host a broker (OpenRouter) dispatched to, when echoed; else null (#9). */
  upstreamProvider: string | null;
  /** The service tier the run used (Doubleword flex/default/scale), else null (#9 speed axis). */
  serviceTier: string | null;
}

/**
 * Roll one experiment up to its pass rate, mean score, and cost — joining each
 * case's grade to the completion that produced it. This is the score half that
 * `telemetry` (cost/latency only) does not carry, so a caller can compare models
 * on quality and price together.
 */
export function experimentScoreCost(
  handle: CompoundDatabase,
  experimentId: string,
): ExperimentScoreCost {
  const rows = handle.db
    .select({
      passed: experimentResults.passed,
      score: experimentResults.score,
      costUsd: completions.costUsd,
      latencyMs: completions.latencyMs,
      usageJson: completions.usageJson,
      provider: completions.provider,
      upstreamProvider: completions.upstreamProvider,
      paramsJson: completions.paramsJson,
    })
    .from(experimentResults)
    .leftJoin(completions, eq(experimentResults.completionFingerprint, completions.fingerprint))
    .where(eq(experimentResults.experimentId, experimentId))
    .all();

  let graded = 0;
  let passed = 0;
  let scoreSum = 0;
  let scoreN = 0;
  let totalCostUsd = 0;
  let costN = 0;
  let latencySum = 0;
  let latencyCount = 0;
  let tpsSum = 0;
  let tpsCount = 0;
  let provider: string | null = null;
  let upstreamProvider: string | null = null;
  let serviceTier: string | null = null;
  for (const r of rows) {
    if (r.passed !== null) {
      graded += 1;
      if (r.passed) passed += 1;
    }
    if (r.score !== null) {
      scoreSum += r.score;
      scoreN += 1;
    }
    if (r.costUsd !== null) {
      totalCostUsd += r.costUsd;
      costN += 1;
    }
    if (r.latencyMs !== null && r.latencyMs > 0) {
      latencySum += r.latencyMs;
      latencyCount += 1;
      // TPS matches telemetry: output tokens over full request latency.
      const output = usageTokens(r.usageJson).output;
      if (output > 0) {
        tpsSum += output / (r.latencyMs / 1000);
        tpsCount += 1;
      }
    }
    if (provider === null && r.provider !== null) provider = r.provider;
    if (upstreamProvider === null && r.upstreamProvider != null)
      upstreamProvider = r.upstreamProvider;
    if (serviceTier === null) serviceTier = serviceTierOf(r.paramsJson);
  }
  return {
    cases: rows.length,
    graded,
    passed,
    meanScore: scoreN > 0 ? scoreSum / scoreN : null,
    totalCostUsd,
    meanCostUsd: costN > 0 ? totalCostUsd / costN : null,
    meanLatencyMs: latencyCount > 0 ? latencySum / latencyCount : null,
    latencyCount,
    meanTps: tpsCount > 0 ? tpsSum / tpsCount : null,
    tpsCount,
    provider,
    upstreamProvider,
    serviceTier,
  };
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
