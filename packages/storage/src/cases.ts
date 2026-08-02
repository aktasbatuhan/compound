/**
 * Case repository.
 *
 * The decision partition is sealed HERE: `listCases` refuses to return
 * `decision_test` cases unless the caller passes an explicit, auditable
 * `openDecisionFirewall` token. Optimization, selection, and calibration
 * callers never hold one, so they cannot see sealed data by construction
 * (docs/curation-v1.md).
 */
import { createHash } from "node:crypto";
import { and, asc, count, desc, eq, inArray, ne, sql } from "drizzle-orm";
import type { CompoundDatabase } from "./db";
import { type Pagination, paginate } from "./pagination";
import {
  type CasePartition,
  type CaseProvenance,
  type CaseReviewState,
  type CaseRow,
  cases,
} from "./schema";

/** A new case to persist. Partition and provenance are decided by curation. */
export interface CaseInsert {
  caseId: string;
  taskKey: string;
  sourceTraceId: string;
  contentHash: string;
  provenance: CaseProvenance;
  partition: CasePartition;
  input: unknown;
  expected: unknown;
  id?: string;
}

export interface InsertCasesResult {
  inserted: number;
  insertedCaseIds: string[];
  /** Cases whose (task_key, content_hash) already existed: deduped, not errors. */
  duplicates: number;
  duplicateContentHashes: string[];
}

export class UnknownCaseError extends Error {
  constructor(readonly caseId: string) {
    super(`unknown case: ${caseId}`);
    this.name = "UnknownCaseError";
  }
}

/**
 * A capability token that opens the decision-test firewall for one read.
 *
 * It is intentionally awkward to obtain: a caller must state a reason, which is
 * the audit record of why sealed data was read. Holding one of these is the
 * only way `listCases` will return `decision_test` cases.
 */
export interface DecisionFirewallToken {
  readonly reason: string;
  readonly openedAt: Date;
}

export function openDecisionFirewall(reason: string): DecisionFirewallToken {
  if (reason.trim().length === 0) {
    throw new Error("opening the decision firewall requires a non-empty reason");
  }
  return { reason, openedAt: new Date() };
}

/**
 * Insert cases, skipping any whose (task_key, content_hash) already exists.
 *
 * Dedupe is within a task: the first case for a piece of work wins, and later
 * duplicates are counted rather than inserted, so a re-run of curation never
 * creates a second case for the same request nor reshuffles its partition.
 */
export function insertCases(
  handle: CompoundDatabase,
  records: readonly CaseInsert[],
): InsertCasesResult {
  const insertedCaseIds: string[] = [];
  const duplicateContentHashes: string[] = [];

  handle.db.transaction((tx) => {
    for (const record of records) {
      const existing = tx
        .select({ id: cases.id })
        .from(cases)
        .where(and(eq(cases.taskKey, record.taskKey), eq(cases.contentHash, record.contentHash)))
        .all();

      if (existing.length > 0) {
        duplicateContentHashes.push(record.contentHash);
        tx.update(cases)
          .set({ duplicateCount: sql`${cases.duplicateCount} + 1` })
          .where(eq(cases.id, existing[0]?.id as string))
          .run();
        continue;
      }

      const id = record.id ?? crypto.randomUUID();
      tx.insert(cases)
        .values({
          id,
          caseId: record.caseId,
          taskKey: record.taskKey,
          sourceTraceId: record.sourceTraceId,
          contentHash: record.contentHash,
          provenance: record.provenance,
          partition: record.partition,
          input: record.input,
          expected: record.expected ?? null,
        })
        .run();
      insertedCaseIds.push(record.caseId);
    }
  });

  return {
    inserted: insertedCaseIds.length,
    insertedCaseIds,
    duplicates: duplicateContentHashes.length,
    duplicateContentHashes,
  };
}

export interface CaseFilter {
  taskKey?: string;
  partition?: CasePartition | readonly CasePartition[];
  provenance?: CaseProvenance | readonly CaseProvenance[];
  reviewState?: CaseReviewState | readonly CaseReviewState[];
  sourceTraceId?: string;
}

export interface ListCasesOptions extends CaseFilter, Pagination {
  order?: "created_at_asc" | "created_at_desc";
  /**
   * The decision-test firewall. Without it, `decision_test` cases are excluded
   * from every result — the seal. With it, they are included and the read is
   * on the caller's audited responsibility.
   */
  openDecisionFirewall?: DecisionFirewallToken;
}

function inClause<T>(column: Parameters<typeof inArray>[0], value: T | readonly T[]) {
  return Array.isArray(value) ? inArray(column, [...value]) : eq(column, value as never);
}

function buildCaseConditions(options: ListCasesOptions | CaseFilter, sealed: boolean) {
  const conditions = [];
  if (options.taskKey !== undefined) conditions.push(eq(cases.taskKey, options.taskKey));
  if (options.partition !== undefined) {
    conditions.push(inClause(cases.partition, options.partition));
  }
  if (options.provenance !== undefined) {
    conditions.push(inClause(cases.provenance, options.provenance));
  }
  if (options.reviewState !== undefined) {
    conditions.push(inClause(cases.reviewState, options.reviewState));
  }
  if (options.sourceTraceId !== undefined) {
    conditions.push(eq(cases.sourceTraceId, options.sourceTraceId));
  }
  // The seal: exclude decision_test unless the firewall was explicitly opened.
  if (sealed) conditions.push(ne(cases.partition, "decision_test"));
  return conditions.length > 0 ? and(...conditions) : undefined;
}

export function listCases(handle: CompoundDatabase, options: ListCasesOptions = {}): CaseRow[] {
  const sealed = options.openDecisionFirewall === undefined;
  const order = options.order === "created_at_asc" ? asc : desc;
  const query = handle.db
    .select()
    .from(cases)
    .where(buildCaseConditions(options, sealed))
    .orderBy(order(cases.createdAt), asc(cases.caseId))
    .$dynamic();
  return paginate(query, options).all();
}

export function countCases(handle: CompoundDatabase, filter: CaseFilter = {}): number {
  // Counting also respects the seal: a caller with no firewall cannot even
  // learn the decision set's size through a filtered count.
  const [row] = handle.db
    .select({ value: count() })
    .from(cases)
    .where(buildCaseConditions(filter, true))
    .all();
  return row?.value ?? 0;
}

export interface PartitionCount {
  partition: CasePartition;
  count: number;
}

/**
 * Counts per partition. This is the one place decision_test size is visible
 * without the firewall, because knowing "5% is sealed" is not the same as
 * reading the sealed cases, and the dashboard needs the split.
 */
export function countCasesByPartition(
  handle: CompoundDatabase,
  taskKey?: string,
): PartitionCount[] {
  return handle.db
    .select({ partition: cases.partition, count: count() })
    .from(cases)
    .where(taskKey !== undefined ? eq(cases.taskKey, taskKey) : undefined)
    .groupBy(cases.partition)
    .orderBy(desc(count()))
    .all();
}

/**
 * A fingerprint of a task's SEALED decision_test set: a hash of its case content
 * hashes. Two properties matter for the peeking guard (#22): it is stable while
 * the held-out set is unchanged (so repeated decisions on it can be counted),
 * and it changes the moment the set is re-curated (a genuinely new held-out set
 * is a fresh test whose decision budget resets). Returns null when the task has
 * no sealed cases. Like `countCasesByPartition`, this reveals a hash, never the
 * sealed cases themselves, so it needs no firewall token.
 */
export function sealedPartitionVersion(handle: CompoundDatabase, taskKey: string): string | null {
  const rows = handle.db
    .select({ contentHash: cases.contentHash })
    .from(cases)
    .where(and(eq(cases.taskKey, taskKey), eq(cases.partition, "decision_test")))
    .orderBy(asc(cases.contentHash))
    .all();
  if (rows.length === 0) return null;
  const digest = createHash("sha256");
  for (const row of rows) digest.update(row.contentHash).update("\n");
  return `sha256:${digest.digest("hex")}`;
}

export interface CaseTaskKeyCount {
  taskKey: string;
  count: number;
}

/** Case count per task key — the tasks that have curated cases, most-first. */
export function countCasesByTaskKey(handle: CompoundDatabase): CaseTaskKeyCount[] {
  return handle.db
    .select({ taskKey: cases.taskKey, count: count() })
    .from(cases)
    .groupBy(cases.taskKey)
    .orderBy(desc(count()))
    .all();
}

export interface ProvenanceCount {
  provenance: CaseProvenance;
  count: number;
}

export function countCasesByProvenance(
  handle: CompoundDatabase,
  taskKey?: string,
): ProvenanceCount[] {
  return handle.db
    .select({ provenance: cases.provenance, count: count() })
    .from(cases)
    .where(taskKey !== undefined ? eq(cases.taskKey, taskKey) : undefined)
    .groupBy(cases.provenance)
    .orderBy(desc(count()))
    .all();
}

export function getCase(handle: CompoundDatabase, caseId: string): CaseRow | null {
  const [row] = handle.db.select().from(cases).where(eq(cases.caseId, caseId)).all();
  return row ?? null;
}

/**
 * Review a case. Approving with an edited expected output is the ONLY path that
 * promotes provenance to `human_golden` — the human validation SPEC.md requires
 * before observed/synthetic output can be treated as ground truth.
 */
export interface ReviewCaseInput {
  reviewState: CaseReviewState;
  /** When present, replaces the expected output. */
  expected?: unknown;
  /** Set true to promote to human_golden; only valid with reviewState approved. */
  promoteToGolden?: boolean;
}

export class InvalidPromotionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InvalidPromotionError";
  }
}

export function reviewCase(
  handle: CompoundDatabase,
  caseId: string,
  input: ReviewCaseInput,
): CaseRow {
  const current = getCase(handle, caseId);
  if (current === null) throw new UnknownCaseError(caseId);

  if (input.promoteToGolden && input.reviewState !== "approved") {
    throw new InvalidPromotionError("promotion to human_golden requires reviewState 'approved'");
  }

  const update: Partial<typeof cases.$inferInsert> = {
    reviewState: input.reviewState,
    updatedAt: new Date(),
  };
  if (input.expected !== undefined) update.expected = input.expected;
  if (input.promoteToGolden) update.provenance = "human_golden";

  handle.db.update(cases).set(update).where(eq(cases.caseId, caseId)).run();
  return getCase(handle, caseId) as CaseRow;
}
