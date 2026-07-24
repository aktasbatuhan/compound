/**
 * Public types for the Langfuse importer.
 *
 * Source of truth: docs/langfuse-import-mapping.md (the mapping) and
 * docs/trace-contract-v1.md (the output). This package only *normalizes*:
 * it does not redact, validate, or persist.
 */
import type { Permissions, Trace } from "@compound/contract";

/** One contract trace plus the reasons normalization could not be clean. */
export interface NormalizedTrace {
  /** Contract trace. `redactions` is always `[]` — redaction is a later pass. */
  trace: Trace;
  /** Stable snake_case reasons, deduplicated and sorted. */
  diagnostics: string[];
}

export interface NormalizeOptions {
  /** Stamped onto every trace; comes from `ingest.default_permissions`. */
  defaultPermissions: Permissions;
  /** Recorded in `trace.source.importer_version`. */
  importerVersion: string;
  /** Used for the `langfuse:<projectId>:<id>` trace_id prefix. */
  projectId?: string;
  /** Recorded in `trace.source.exported_at`. */
  exportedAt?: string;
}

/** How the input bytes were read. */
export type InputFormat = "json_array" | "jsonl" | "records";

/**
 * Which documented export surface the records came from.
 *
 * - `api` — `TraceWithFullDetails` with embedded observations/scores.
 * - `blob_join` — separate trace and observation records joined on trace_id.
 * - `enriched_observations` — `observations_v2` alone (trace context denormalized).
 * - `traces_only` — trace records with no observations at all.
 * - `json` — plain contract-shaped trace JSON (the `@compound/ingest` JSON importer).
 * - `unknown` — nothing recognizable was read.
 */
export type ExportSurface =
  | "api"
  | "blob_join"
  | "enriched_observations"
  | "traces_only"
  | "json"
  | "unknown";

export type Casing = "camelCase" | "snake_case" | "unknown";

/** A record that could not be turned into input, with the 1-based line it came from. */
export interface RejectedRecord {
  line: number;
  reason: string;
}

export interface ImportSourceReport {
  /** Detected export surface (see {@link ExportSurface}). */
  surface: ExportSurface;
  /** Every per-record casing observed, in first-seen order. */
  casing: Casing[];
  format: InputFormat;
  counts: {
    recordsSeen: number;
    recordsRejected: number;
    traceRecords: number;
    observationRecords: number;
    scoreRecords: number;
    tracesNormalized: number;
  };
  /** Rejected line numbers with a stable snake_case reason. */
  rejected: RejectedRecord[];
  /** Histogram of every diagnostic reason emitted across all traces. */
  diagnosticReasons: Record<string, number>;
  /** Message/tool-call dialects encountered (see DIALECTS in `dialects.ts`). */
  dialects: string[];
  /** Scores dropped rather than mapped, with the reason breakdown. */
  skippedScores: {
    total: number;
    byReason: Record<string, number>;
  };
  /** Observation `type` values not in the documented set, in first-seen order. */
  unknownObservationTypes: string[];
}

export interface NormalizeResult {
  traces: NormalizedTrace[];
  report: ImportSourceReport;
}
