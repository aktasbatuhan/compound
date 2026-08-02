/**
 * Gate-decision repository: the pre-declared rule and its decided outcome
 * (docs/gate-decision-v1.md, Step 6).
 *
 * The invariant enforced here: a `GateSpec` is stored (and content-hashed)
 * before results exist, and a spec hash is unique, so re-declaring the identical
 * rule reuses the same row rather than minting a second one.
 */

import { createHash } from "node:crypto";
import { and, desc, eq, inArray } from "drizzle-orm";
import type { CompoundDatabase } from "./db";
import {
  cases,
  experimentResults,
  type GateMetric,
  type GateMode,
  type GateOutcome,
  type GateResultRow,
  type GateSpecRow,
  gateDecisionCases,
  gateResults,
  gateSpecs,
} from "./schema";

export interface CreateGateSpecInput {
  specHash: string;
  taskKey: string;
  candidateModel: string;
  referenceModel: string;
  metric: GateMetric;
  mode: GateMode;
  margin: number;
  confidence: number;
  minCases: number;
  judgeAbstainMax: number;
  /** The coverage threshold the rule was declared with (#6); part of its identity. */
  maxSkipFraction?: number;
  /** Content hash of an optimized candidate prompt under test; absent for a baseline gate. */
  candidatePromptHash?: string;
  /** The optimization artifact that supplied the prompt (provenance). */
  optimizationRunId?: string;
  /** The provider each side ran on, when named (the provider axis). */
  candidateProvider?: string;
  referenceProvider?: string;
  firewallReason: string;
}

/**
 * Insert the pre-declared spec, or return the existing row if this exact spec
 * (same `specHash`) was already declared. Storing-before-results is the caller's
 * responsibility (the CLI/API declare the spec before running the decision-
 * partition experiments); the unique hash guarantees the rule can't be quietly
 * edited into a different one after the fact.
 */
export function createGateSpec(handle: CompoundDatabase, input: CreateGateSpecInput): GateSpecRow {
  const existing = getGateSpecByHash(handle, input.specHash);
  if (existing !== null) return existing;
  const id = crypto.randomUUID();
  handle.db
    .insert(gateSpecs)
    .values({
      id,
      specHash: input.specHash,
      taskKey: input.taskKey,
      candidateModel: input.candidateModel,
      referenceModel: input.referenceModel,
      metric: input.metric,
      mode: input.mode,
      margin: input.margin,
      confidence: input.confidence,
      minCases: input.minCases,
      judgeAbstainMax: input.judgeAbstainMax,
      maxSkipFraction: input.maxSkipFraction ?? null,
      candidatePromptHash: input.candidatePromptHash ?? null,
      optimizationRunId: input.optimizationRunId ?? null,
      candidateProvider: input.candidateProvider ?? null,
      referenceProvider: input.referenceProvider ?? null,
      firewallReason: input.firewallReason,
    })
    .run();
  return getGateSpec(handle, id) as GateSpecRow;
}

export function getGateSpec(handle: CompoundDatabase, id: string): GateSpecRow | null {
  const [row] = handle.db.select().from(gateSpecs).where(eq(gateSpecs.id, id)).all();
  return row ?? null;
}

export function getGateSpecByHash(handle: CompoundDatabase, specHash: string): GateSpecRow | null {
  const [row] = handle.db.select().from(gateSpecs).where(eq(gateSpecs.specHash, specHash)).all();
  return row ?? null;
}

export interface RecordGateResultInput {
  gateSpecId: string;
  candidateExperimentId: string;
  referenceExperimentId: string;
  outcome: GateOutcome;
  delta: number;
  ciLo: number;
  ciHi: number;
  n: number;
  candidateRate: number;
  referenceRate: number;
  judgeAbstainedFraction: number;
  /** Fingerprint of the sealed set this verdict was decided on (the peeking guard, #22). */
  decisionPartitionVersion?: string | null;
}

export function recordGateResult(
  handle: CompoundDatabase,
  input: RecordGateResultInput,
): GateResultRow {
  const id = crypto.randomUUID();
  handle.db
    .insert(gateResults)
    .values({ id, ...input })
    .run();
  return getGateResult(handle, id) as GateResultRow;
}

export function getGateResult(handle: CompoundDatabase, id: string): GateResultRow | null {
  const [row] = handle.db.select().from(gateResults).where(eq(gateResults.id, id)).all();
  return row ?? null;
}

export interface GateResultWithSpec {
  result: GateResultRow;
  spec: GateSpecRow;
}

/**
 * The exact held-out cohort two experiments decided on: the cases present in
 * BOTH result sets, identified by content hash. This is what a gate actually
 * examined, so the peeking guard is keyed to reality, not to the current shape
 * of the partition (#22, #3). Returns a stable version hash of the sorted
 * content hashes and the membership list.
 */
export function decidedCohort(
  handle: CompoundDatabase,
  candidateExperimentId: string,
  referenceExperimentId: string,
): { contentHashes: string[]; version: string | null } {
  const candRows = handle.db
    .select({ caseId: experimentResults.caseId, contentHash: experimentResults.contentHash })
    .from(experimentResults)
    .where(eq(experimentResults.experimentId, candidateExperimentId))
    .all();
  const refIds = new Set(
    handle.db
      .select({ caseId: experimentResults.caseId })
      .from(experimentResults)
      .where(eq(experimentResults.experimentId, referenceExperimentId))
      .all()
      .map((r) => r.caseId),
  );
  const shared = candRows.filter((r) => refIds.has(r.caseId));
  if (shared.length === 0) return { contentHashes: [], version: null };

  // Prefer the content hash recorded ON the result row (survives case pruning,
  // #5); fall back to the live `cases` table for pre-migration rows that never
  // stored one.
  const needsFallback = shared.filter((r) => r.contentHash == null).map((r) => r.caseId);
  const fallback = new Map<string, string>();
  if (needsFallback.length > 0) {
    for (const row of handle.db
      .select({ caseId: cases.caseId, contentHash: cases.contentHash })
      .from(cases)
      .where(inArray(cases.caseId, needsFallback))
      .all()) {
      fallback.set(row.caseId, row.contentHash);
    }
  }
  const contentHashes: string[] = [];
  for (const r of shared) {
    const hash = r.contentHash ?? fallback.get(r.caseId);
    if (hash != null) contentHashes.push(hash);
  }
  // Fail closed: if the cohort can't be FULLY reconstructed (cases pruned and no
  // recorded hash), report no version so the caller does not silently treat an
  // unverifiable cohort as "never decided" (#5).
  if (contentHashes.length < shared.length) return { contentHashes: [], version: null };
  contentHashes.sort();
  const digest = createHash("sha256");
  for (const h of contentHashes) digest.update(h).update("\n");
  return { contentHashes, version: `sha256:${digest.digest("hex")}` };
}

/**
 * Content hashes of a task's SEALED decision partition — the full cohort a gate
 * would decide on, resolved WITHOUT opening the firewall (a set of hashes is not
 * the sealed labels). Used by the CLI to preflight the peeking guard BEFORE
 * paying to re-run the decision experiments (#2), and to attribute coverage
 * omissions against the true sealed set rather than only the cases that happened
 * to be graded (#11).
 */
export function decisionTestCohortHashes(handle: CompoundDatabase, taskKey: string): string[] {
  return handle.db
    .select({ contentHash: cases.contentHash, caseId: cases.caseId })
    .from(cases)
    .where(and(eq(cases.taskKey, taskKey), eq(cases.partition, "decision_test")))
    .all()
    .map((r) => r.contentHash);
}

/** Case ids of a task's sealed decision partition (for coverage attribution, #11). */
export function decisionTestCaseIds(handle: CompoundDatabase, taskKey: string): string[] {
  return handle.db
    .select({ caseId: cases.caseId })
    .from(cases)
    .where(and(eq(cases.taskKey, taskKey), eq(cases.partition, "decision_test")))
    .all()
    .map((r) => r.caseId);
}

/** Record the exact cases a persisted verdict decided (the peeking-guard ground truth). */
export function recordDecisionCohort(
  handle: CompoundDatabase,
  gateResultId: string,
  contentHashes: readonly string[],
): void {
  if (contentHashes.length === 0) return;
  for (const contentHash of contentHashes) {
    handle.db
      .insert(gateDecisionCases)
      .values({ gateResultId, contentHash })
      .onConflictDoNothing()
      .run();
  }
}

export interface PriorDecisions {
  /** Prior verdicts whose decided cohort OVERLAPS the one being decided. */
  count: number;
  /** When such an overlapping set was first decided, or null if never. */
  firstDecidedAt: Date | null;
  /** How many overlapping priors put an optimized prompt under test (informational). */
  adoptionCount: number;
  /**
   * Prior verdicts for this task with NO recorded cohort membership (decided
   * before the guard, or their experiments were pruned). Their overlap can't be
   * proven, so an enabled guard treats their presence as a reason to require an
   * explicit --force (#9).
   */
  legacyCount: number;
}

/**
 * How the held-out labels being decided have ALREADY been examined — the
 * multiple-comparisons budget for #22/#3. A prior verdict counts when its
 * recorded cohort shares ANY case with `cohortHashes`, so reusing even part of
 * the sealed set (e.g. 100 of 101 cases) is caught; extending the cohort by one
 * case no longer resets the budget. Priors with no recorded membership are
 * surfaced separately as `legacyCount`.
 */
export function priorDecisions(
  handle: CompoundDatabase,
  taskKey: string,
  cohortHashes: readonly string[],
): PriorDecisions {
  const empty: PriorDecisions = {
    count: 0,
    firstDecidedAt: null,
    adoptionCount: 0,
    legacyCount: 0,
  };
  if (cohortHashes.length === 0) return empty;
  const cohortSet = new Set(cohortHashes);
  const rows = handle.db
    .select({
      id: gateResults.id,
      decidedAt: gateResults.decidedAt,
      candidatePromptHash: gateSpecs.candidatePromptHash,
      optimizationRunId: gateSpecs.optimizationRunId,
    })
    .from(gateResults)
    .innerJoin(gateSpecs, eq(gateResults.gateSpecId, gateSpecs.id))
    .where(eq(gateSpecs.taskKey, taskKey))
    .all();
  if (rows.length === 0) return empty;

  let firstDecidedAt: Date | null = null;
  let adoptionCount = 0;
  let overlapCount = 0;
  let legacyCount = 0;
  for (const row of rows) {
    const members = handle.db
      .select({ contentHash: gateDecisionCases.contentHash })
      .from(gateDecisionCases)
      .where(eq(gateDecisionCases.gateResultId, row.id))
      .all();
    if (members.length === 0) {
      legacyCount += 1;
      continue;
    }
    if (!members.some((m) => cohortSet.has(m.contentHash))) continue;
    overlapCount += 1;
    if (firstDecidedAt === null || row.decidedAt < firstDecidedAt) firstDecidedAt = row.decidedAt;
    if (row.candidatePromptHash != null || row.optimizationRunId != null) adoptionCount += 1;
  }
  return { count: overlapCount, firstDecidedAt, adoptionCount, legacyCount };
}

/** Every decided gate, newest first, each joined to the rule it was decided under. */
export function listGateResults(handle: CompoundDatabase, limit = 100): GateResultWithSpec[] {
  const rows = handle.db
    .select()
    .from(gateResults)
    .innerJoin(gateSpecs, eq(gateResults.gateSpecId, gateSpecs.id))
    .orderBy(desc(gateResults.decidedAt))
    .limit(limit)
    .all();
  return rows.map((r) => ({ result: r.gate_results, spec: r.gate_specs }));
}
