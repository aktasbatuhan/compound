/**
 * The ingest pipeline: normalize → redact → validate → classify → persist.
 *
 * Spec: docs/ingest-pipeline-v1.md.
 *
 * Redaction runs BEFORE validation so the object we validate is byte-for-byte
 * the object we persist, and before persistence so no unredacted value is ever
 * written to the database.
 */

import { createHash } from "node:crypto";
import type { CompoundConfig } from "@compound/config";
import { validate } from "@compound/contract";
import {
  normalizeJsonExport,
  normalizeLangfuseExport,
  normalizeOtelExport,
} from "@compound/ingest";
import { redactTrace } from "@compound/redaction";
import {
  type CompoundDatabase,
  completeImportBatch,
  createImportBatch,
  failImportBatch,
  type ImportBatchRow,
  type ImportReport,
  insertTraces,
  type TraceRecordInput,
} from "@compound/storage";
import { classify } from "./classify";
import { computeContentHash } from "./content-hash";

/** Permissions applied when the config declares none. */
export const DEFAULT_INGEST_PERMISSIONS = {
  judging: true,
  optimization: true,
  fine_tuning: false,
} as const;

export const SUPPORTED_IMPORTERS = ["langfuse", "json", "otel"] as const;
export type SupportedImporter = (typeof SUPPORTED_IMPORTERS)[number];

export class UnsupportedImporterError extends Error {
  constructor(readonly importer: string) {
    super(`unsupported importer '${importer}'; supported: ${SUPPORTED_IMPORTERS.join(", ")}`);
    this.name = "UnsupportedImporterError";
  }
}

export interface RunImportOptions {
  importer: string;
  /** Raw export bytes. */
  content: string;
  config?: CompoundConfig;
  importerVersion?: string;
  projectId?: string;
  exportedAt?: string;
  /** Recorded on the batch; defaults to a hash of the content. */
  sourceFingerprint?: string;
}

export interface RunImportResult {
  batch: ImportBatchRow;
  report: ImportReport;
}

function sourceFingerprintOf(content: string): string {
  return createHash("sha256").update(content).digest("hex");
}

/**
 * Run one import end to end.
 *
 * The batch lifecycle is always closed: on success it completes with the full
 * report, and on an unexpected throw it is marked `failed` with whatever was
 * accumulated, so a crashed import never leaves a batch stuck in `running`.
 */
export function runImport(db: CompoundDatabase, options: RunImportOptions): RunImportResult {
  if (!SUPPORTED_IMPORTERS.includes(options.importer as SupportedImporter)) {
    throw new UnsupportedImporterError(options.importer);
  }

  const importerVersion = options.importerVersion ?? "1";
  const permissions = options.config?.ingest?.default_permissions ?? DEFAULT_INGEST_PERMISSIONS;
  const redactionConfig = options.config?.redaction;

  const batch = createImportBatch(db, {
    importer: options.importer,
    importerVersion,
    sourceFingerprint: options.sourceFingerprint ?? sourceFingerprintOf(options.content),
  });

  try {
    const normalizeOptions = {
      defaultPermissions: permissions,
      importerVersion,
      projectId: options.projectId,
      exportedAt: options.exportedAt,
    };
    const { traces, report: source } =
      options.importer === "json"
        ? normalizeJsonExport(options.content, normalizeOptions)
        : options.importer === "otel"
          ? normalizeOtelExport(options.content, normalizeOptions)
          : normalizeLangfuseExport(options.content, normalizeOptions);

    const records: TraceRecordInput[] = [];
    const diagnosticReasons: Record<string, number> = {};
    const redactionsByRule: Record<string, number> = {};
    const internalErrors: Array<{ trace_id: string; issues: unknown }> = [];
    let evalReady = 0;
    let diagnostic = 0;

    for (const normalized of traces) {
      // Redact first: what we validate is exactly what we persist.
      const redacted = redactTrace(normalized.trace, redactionConfig);
      for (const record of redacted.redactions) {
        redactionsByRule[record.rule] = (redactionsByRule[record.rule] ?? 0) + 1;
      }

      const classified = classify(validate(redacted.trace), normalized.diagnostics);
      if (classified.outcome === "internal_error") {
        internalErrors.push({
          trace_id: normalized.trace.trace_id,
          issues: classified.issues,
        });
        continue;
      }

      for (const reason of classified.diagnosticReasons) {
        diagnosticReasons[reason] = (diagnosticReasons[reason] ?? 0) + 1;
      }
      if (classified.validationClass === "eval_ready") evalReady += 1;
      else diagnostic += 1;

      records.push({
        trace: classified.trace,
        validationClass: classified.validationClass,
        diagnosticReasons: classified.diagnosticReasons,
        contentHash: computeContentHash(classified.trace),
      });
    }

    const inserted = insertTraces(db, batch.id, records);

    const report: ImportReport = {
      counts: {
        eval_ready: evalReady,
        diagnostic,
        rejected: source.counts.recordsRejected,
        duplicate: inserted.duplicates,
        internal_normalization_errors: internalErrors.length,
      },
      diagnostic_reasons: diagnosticReasons,
      dialects: source.dialects,
      skipped_scores: source.skippedScores.total,
      skipped_scores_by_reason: source.skippedScores.byReason,
      rejected_lines: source.rejected.map((entry) => entry.line),
      // Reasons are stable snake_case strings; the rejected records themselves
      // are never stored, since a malformed line may still hold sensitive data.
      rejected_reasons: source.rejected.reduce<Record<string, number>>((acc, entry) => {
        acc[entry.reason] = (acc[entry.reason] ?? 0) + 1;
        return acc;
      }, {}),
      duplicate_trace_ids: inserted.duplicateTraceIds,
      redactions_by_rule: redactionsByRule,
      surface: source.surface,
      casing: source.casing,
      format: source.format,
      unknown_observation_types: source.unknownObservationTypes,
      source_counts: source.counts,
      ...(internalErrors.length > 0 ? { internal_errors: internalErrors } : {}),
    };

    return { batch: completeImportBatch(db, batch.id, report), report };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const report: ImportReport = { error: message };
    failImportBatch(db, batch.id, report);
    throw error;
  }
}
