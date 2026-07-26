/**
 * Optimization-run repository (docs/optimization-v1.md, Step 7): the stored
 * artifact of a GEPA run — a proposed prompt improvement with provenance.
 */
import { desc, eq } from "drizzle-orm";
import type { CompoundDatabase } from "./db";
import { type OptimizationRunRow, optimizationRuns } from "./schema";

export interface RecordOptimizationInput {
  taskKey: string;
  candidateModel: string;
  seedPrompt: string;
  optimizedPrompt: string;
  beforeValScore: number;
  afterValScore: number;
  valCases: number;
  reflectionCalls: number;
  eligibilityReason?: string;
  costUsd?: number;
}

export function recordOptimizationRun(
  handle: CompoundDatabase,
  input: RecordOptimizationInput,
): OptimizationRunRow {
  const id = crypto.randomUUID();
  handle.db
    .insert(optimizationRuns)
    .values({
      id,
      taskKey: input.taskKey,
      candidateModel: input.candidateModel,
      seedPrompt: input.seedPrompt,
      optimizedPrompt: input.optimizedPrompt,
      beforeValScore: input.beforeValScore,
      afterValScore: input.afterValScore,
      valCases: input.valCases,
      reflectionCalls: input.reflectionCalls,
      eligibilityReason: input.eligibilityReason ?? null,
      costUsd: input.costUsd ?? 0,
    })
    .run();
  return getOptimizationRun(handle, id) as OptimizationRunRow;
}

export function getOptimizationRun(
  handle: CompoundDatabase,
  id: string,
): OptimizationRunRow | null {
  const [row] = handle.db.select().from(optimizationRuns).where(eq(optimizationRuns.id, id)).all();
  return row ?? null;
}

export function listOptimizationRuns(
  handle: CompoundDatabase,
  taskKey?: string,
  limit = 100,
): OptimizationRunRow[] {
  const base = handle.db.select().from(optimizationRuns).$dynamic();
  const filtered = taskKey !== undefined ? base.where(eq(optimizationRuns.taskKey, taskKey)) : base;
  return filtered.orderBy(desc(optimizationRuns.createdAt)).limit(limit).all();
}
