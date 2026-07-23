/**
 * Curation: turn eval-ready traces into partitioned, provenance-typed cases.
 *
 * Reads eval-ready traces for a task, extracts a case from each replayable one,
 * assigns a firewalled partition from its content hash, and persists — deduping
 * within the task. Diagnostic traces and non-replayable traces are counted and
 * skipped, never forced into cases.
 *
 * Spec: docs/curation-v1.md.
 */
import type { Trace } from "@compound/contract";
import {
  type CaseInsert,
  type CompoundDatabase,
  insertCases,
  listTraces,
  type StoredTrace,
} from "@compound/storage";
import { extractCase, NotExtractableError } from "./extract";
import { assignPartition, DEFAULT_PARTITION_RATIOS, type PartitionRatios } from "./partition";

export interface CurateOptions {
  /** Curate one task at a time; a case must belong to a task. */
  taskKey: string;
  ratios?: PartitionRatios;
  /** Partition salt version; bump only to deliberately re-partition. */
  partitionVersion?: string;
  /** How many eval-ready traces to scan per call. */
  batchSize?: number;
}

export interface CurateReport {
  taskKey: string;
  tracesScanned: number;
  casesCreated: number;
  duplicates: number;
  /** Traces that were eval-ready but not replayable as a single call. */
  skippedNotExtractable: number;
  skipReasons: Record<string, number>;
  byPartition: Record<string, number>;
  byProvenance: Record<string, number>;
}

/**
 * Curate the eval-ready traces of one task into cases.
 *
 * Idempotent by construction: dedupe is on (task_key, content_hash) and
 * partition assignment is a pure function of the content hash, so re-running
 * curation neither creates duplicate cases nor moves an existing case between
 * partitions.
 */
export function curateTask(db: CompoundDatabase, options: CurateOptions): CurateReport {
  const ratios = options.ratios ?? DEFAULT_PARTITION_RATIOS;
  const traces: StoredTrace[] = listTraces(db, {
    taskKey: options.taskKey,
    validationClass: "eval_ready",
    limit: options.batchSize ?? 1000,
  });

  const report: CurateReport = {
    taskKey: options.taskKey,
    tracesScanned: traces.length,
    casesCreated: 0,
    duplicates: 0,
    skippedNotExtractable: 0,
    skipReasons: {},
    byPartition: {},
    byProvenance: {},
  };

  const records: CaseInsert[] = [];
  for (const stored of traces) {
    const trace: Trace = stored.trace;
    let extracted: ReturnType<typeof extractCase>;
    try {
      extracted = extractCase(trace, { contentHash: stored.contentHash });
    } catch (error) {
      if (error instanceof NotExtractableError) {
        report.skippedNotExtractable += 1;
        report.skipReasons[error.reason] = (report.skipReasons[error.reason] ?? 0) + 1;
        continue;
      }
      throw error;
    }

    const partition = assignPartition(
      extracted.contentHash,
      { taskKey: options.taskKey, version: options.partitionVersion },
      ratios,
    );

    records.push({
      caseId: extracted.caseId,
      taskKey: extracted.taskKey,
      sourceTraceId: extracted.sourceTraceId,
      contentHash: extracted.contentHash,
      provenance: extracted.provenance,
      partition,
      input: extracted.input,
      expected: extracted.expected,
    });

    report.byPartition[partition] = (report.byPartition[partition] ?? 0) + 1;
    report.byProvenance[extracted.provenance] =
      (report.byProvenance[extracted.provenance] ?? 0) + 1;
  }

  const result = insertCases(db, records);
  report.casesCreated = result.inserted;
  report.duplicates = result.duplicates;

  // byPartition/byProvenance above count extraction attempts; correct them to
  // reflect only newly created cases so the report is not inflated by dupes.
  if (result.duplicates > 0) {
    const createdHashes = new Set(
      records
        .filter((record) => result.insertedCaseIds.includes(record.caseId))
        .map((record) => record.contentHash),
    );
    report.byPartition = {};
    report.byProvenance = {};
    for (const record of records) {
      if (!createdHashes.has(record.contentHash)) continue;
      report.byPartition[record.partition] = (report.byPartition[record.partition] ?? 0) + 1;
      report.byProvenance[record.provenance] = (report.byProvenance[record.provenance] ?? 0) + 1;
    }
  }

  return report;
}
