/**
 * Gate-decision repository: the pre-declared rule and its decided outcome
 * (docs/gate-decision-v1.md, Step 6).
 *
 * The invariant enforced here: a `GateSpec` is stored (and content-hashed)
 * before results exist, and a spec hash is unique, so re-declaring the identical
 * rule reuses the same row rather than minting a second one.
 */
import { desc, eq } from "drizzle-orm";
import type { CompoundDatabase } from "./db";
import {
  type GateMetric,
  type GateMode,
  type GateOutcome,
  type GateResultRow,
  type GateSpecRow,
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
