/**
 * `normalizeLangfuseExport` — Langfuse export records into contract traces.
 *
 * Specs: docs/langfuse-import-mapping.md (mapping), docs/trace-contract-v1.md
 * (output), docs/reference/langfuse-export-reference-20260722.md (source
 * schema). This package normalizes only: redaction, validation and persistence
 * are separate passes, so every trace leaves here with `redactions: []` and the
 * caller's `defaultPermissions`.
 */
import { type Step, TRACE_SCHEMA_NAME, TRACE_SCHEMA_VERSION, type Trace } from "@compound/contract";
import { Collector } from "./diagnostics";
import { resolveToolCallRefs, selectFocalStepId } from "./linking";
import {
  OBSERVATION_TYPE_MAP,
  type ObservationView,
  toStep,
  viewObservation,
} from "./observations";
import { parseInput, REJECTION_REASONS } from "./parse";
import { applyScore, newOutcomeAccumulator, SKIPPED_SCORE_REASONS, toOutcome } from "./scores";
import type {
  Casing,
  ExportSurface,
  ImportSourceReport,
  NormalizedTrace,
  NormalizeOptions,
  NormalizeResult,
  RejectedRecord,
} from "./types";
import {
  asMetadataObject,
  asString,
  asStringArray,
  earliest,
  field,
  isRecord,
  latest,
  toIsoUtc,
} from "./values";

export const IMPORTER_NAME = "langfuse";

const TASK_TAG_PREFIX = "task:";

interface SourceRecord {
  raw: Record<string, unknown>;
  line: number;
}

interface TraceGroup {
  id: string;
  order: number;
  trace: SourceRecord | null;
  observations: SourceRecord[];
  scores: SourceRecord[];
}

/**
 * Normalize one Langfuse export into contract traces plus a source report.
 *
 * Accepts the two documented surfaces — API-shaped `TraceWithFullDetails`
 * (camelCase, observations and scores embedded) and blob-style snake_case
 * records (`traces` + `observations` joined on trace_id, or enriched
 * `observations_v2` alone) — as a JSON array, JSONL text, or an already-parsed
 * array. Never performs I/O.
 */
export function normalizeLangfuseExport(
  input: string | unknown[],
  options: NormalizeOptions,
): NormalizeResult {
  const parsed = parseInput(input);
  const rejected: RejectedRecord[] = [...parsed.rejected];
  const collector = new Collector();

  const casing: Casing[] = [];
  const groups = new Map<string, TraceGroup>();
  const counts = { trace: 0, observation: 0, score: 0 };
  const skippedScores = new Map<string, number>();
  const unknownObservationTypes: string[] = [];
  let sawEmbedded = false;

  const noteCasing = (value: Casing): void => {
    if (!casing.includes(value)) casing.push(value);
  };
  const skipScore = (reason: string): void => {
    skippedScores.set(reason, (skippedScores.get(reason) ?? 0) + 1);
  };
  const group = (traceId: string): TraceGroup => {
    const existing = groups.get(traceId);
    if (existing !== undefined) return existing;
    const created: TraceGroup = {
      id: traceId,
      order: groups.size,
      trace: null,
      observations: [],
      scores: [],
    };
    groups.set(traceId, created);
    return created;
  };

  for (const record of parsed.records) {
    noteCasing(record.casing);
    counts[record.kind] += 1;

    if (record.kind === "trace") {
      const target = group(record.value.id as string);
      target.trace = { raw: record.value, line: record.line };
      const embeddedObservations = field(record.value, "observations");
      if (Array.isArray(embeddedObservations)) {
        for (const entry of embeddedObservations) {
          if (!isRecord(entry) || typeof entry.id !== "string") continue;
          sawEmbedded = true;
          counts.observation += 1;
          noteCasing(record.casing);
          target.observations.push({ raw: entry, line: record.line });
        }
      }
      const embeddedScores = field(record.value, "scores");
      if (Array.isArray(embeddedScores)) {
        for (const entry of embeddedScores) {
          if (!isRecord(entry)) continue;
          sawEmbedded = true;
          counts.score += 1;
          target.scores.push({ raw: entry, line: record.line });
        }
      }
      continue;
    }

    const traceId = asString(field(record.value, "traceId", "trace_id"));

    if (record.kind === "observation") {
      if (traceId === null) {
        rejected.push({ line: record.line, reason: REJECTION_REASONS.observationMissingTraceId });
        counts.observation -= 1;
        continue;
      }
      group(traceId).observations.push({ raw: record.value, line: record.line });
      continue;
    }

    // Score. v3 scores can attach to a session or dataset run instead of a
    // trace, so a null trace_id is expected input, skipped and counted.
    if (traceId === null) {
      skipScore(SKIPPED_SCORE_REASONS.nullTraceId);
      continue;
    }
    group(traceId).scores.push({ raw: record.value, line: record.line });
  }

  const traces: NormalizedTrace[] = [];
  for (const target of [...groups.values()].sort((a, b) => a.order - b.order)) {
    if (target.trace === null && target.observations.length === 0) {
      // Scores referencing a trace that is not in this export.
      for (let index = 0; index < target.scores.length; index += 1) {
        skipScore(SKIPPED_SCORE_REASONS.unmatchedTrace);
      }
      continue;
    }
    const normalized = normalizeGroup(
      target,
      options,
      collector,
      rejected,
      unknownObservationTypes,
      skipScore,
    );
    if (normalized !== null) traces.push(normalized);
  }

  const report: ImportSourceReport = {
    surface: detectSurface(sawEmbedded, counts.trace, counts.observation),
    casing: casing.length > 0 ? casing : ["unknown"],
    format: parsed.format,
    counts: {
      recordsSeen: counts.trace + counts.observation + counts.score,
      recordsRejected: rejected.length,
      traceRecords: counts.trace,
      observationRecords: counts.observation,
      scoreRecords: counts.score,
      tracesNormalized: traces.length,
    },
    rejected: [...rejected].sort((a, b) => a.line - b.line),
    diagnosticReasons: Object.fromEntries([...collector.reasonHistogram.entries()].sort()),
    dialects: [...collector.dialects].sort(),
    skippedScores: {
      total: [...skippedScores.values()].reduce((sum, value) => sum + value, 0),
      byReason: Object.fromEntries([...skippedScores.entries()].sort()),
    },
    unknownObservationTypes,
  };

  return { traces, report };
}

function detectSurface(
  sawEmbedded: boolean,
  traceRecords: number,
  observationRecords: number,
): ExportSurface {
  if (sawEmbedded) return "api";
  if (traceRecords > 0 && observationRecords > 0) return "blob_join";
  if (observationRecords > 0) return "enriched_observations";
  if (traceRecords > 0) return "traces_only";
  return "unknown";
}

function normalizeGroup(
  target: TraceGroup,
  options: NormalizeOptions,
  collector: Collector,
  rejected: RejectedRecord[],
  unknownObservationTypes: string[],
  skipScore: (reason: string) => void,
): NormalizedTrace | null {
  const traceRaw = target.trace?.raw ?? null;

  const views: ObservationView[] = target.observations.map((entry, index) => {
    const view = viewObservation(entry.raw, entry.raw.id as string, index, collector);
    if (
      OBSERVATION_TYPE_MAP[view.type] === undefined &&
      !unknownObservationTypes.includes(view.type)
    ) {
      unknownObservationTypes.push(view.type);
    }
    return view;
  });
  views.sort(compareObservations);

  const steps: Step[] = views.map((view) => toStep(view, collector));
  const viewsById = new Map(views.map((view) => [view.id, view]));
  resolveToolCallRefs(steps, viewsById, collector);
  const focalStepId = selectFocalStepId(steps, collector);

  const startedAt =
    toIsoUtc(traceRaw === null ? undefined : field(traceRaw, "timestamp")) ??
    earliest(views.map((view) => view.startedAt));
  if (startedAt === null) {
    rejected.push({
      line: target.trace?.line ?? target.observations[0]?.line ?? 0,
      reason: REJECTION_REASONS.traceMissingStartedAt,
    });
    collector.takeTraceDiagnostics();
    return null;
  }

  // Trace context: the trace record when present, else the denormalized columns
  // on the enriched observations.
  const context = (...names: string[]): unknown => {
    if (traceRaw !== null) {
      const value = field(traceRaw, ...names);
      if (value !== undefined) return value;
    }
    for (const view of views) {
      const value = field(view.raw, ...names);
      if (value !== undefined) return value;
    }
    return undefined;
  };

  const tags = asStringArray(context("tags")) ?? [];
  const metadata = traceRaw === null ? null : asMetadataObject(field(traceRaw, "metadata"));
  const taskKey = resolveTaskKey(metadata, views, tags, traceRaw !== null);

  const outcome = newOutcomeAccumulator();
  for (const score of target.scores) {
    const reason = applyScore(score.raw, outcome, collector);
    if (reason !== null) skipScore(reason);
  }

  const projectId = options.projectId ?? asString(context("projectId", "project_id")) ?? undefined;
  const sourceIds: Record<string, string> = { trace_id: target.id };
  if (projectId !== undefined) sourceIds.project_id = projectId;

  const trace: Trace = {
    schema: TRACE_SCHEMA_NAME,
    schema_version: TRACE_SCHEMA_VERSION,
    trace_id: prefixTraceId(target.id, projectId),
    task_key: taskKey,
    started_at: startedAt,
    ended_at: latest(views.map((view) => view.endedAt)),
    source: {
      importer: IMPORTER_NAME,
      importer_version: options.importerVersion,
      source_ids: sourceIds,
      exported_at: options.exportedAt ?? null,
    },
    session_id: asString(context("sessionId", "session_id")),
    user_ref: asString(context("userId", "user_id")),
    environment: asString(context("environment")),
    release: asString(context("release")),
    metadata: metadata as Trace["metadata"],
    tags,
    steps,
    focal_step_id: focalStepId,
    outcome: toOutcome(outcome),
    permissions: { ...options.defaultPermissions },
    // Redaction is a separate pass; this importer never masks a value.
    redactions: [],
  };

  return { trace, diagnostics: collector.takeTraceDiagnostics() };
}

/** `langfuse:<projectId>:<id>`; the project segment is omitted when unknown. */
export function prefixTraceId(id: string, projectId: string | undefined): string {
  return projectId === undefined ? `${IMPORTER_NAME}:${id}` : `${IMPORTER_NAME}:${projectId}:${id}`;
}

/**
 * `metadata.task_key` first, then a `task:<key>` tag, else null (the
 * "unassigned" bucket). The key is consumed out of the copied trace metadata so
 * it is not duplicated in the payload.
 *
 * With no trace record (enriched `observations_v2` alone) there is no trace
 * metadata to read, so the root observation's metadata is used instead —
 * `propagate_attributes()` puts the same key there — and left in place, since
 * observation metadata is not copied onto the trace.
 */
function resolveTaskKey(
  metadata: Record<string, unknown> | null,
  views: ObservationView[],
  tags: string[],
  hasTraceRecord: boolean,
): string | null {
  if (metadata !== null) {
    const fromMetadata = asString(metadata.task_key);
    delete metadata.task_key;
    if (fromMetadata !== null) return fromMetadata;
  }

  if (!hasTraceRecord) {
    for (const view of views) {
      if (view.parentId !== null) continue;
      const observationMetadata = field(view.raw, "metadata");
      if (!isRecord(observationMetadata)) continue;
      const fromObservation = asString(observationMetadata.task_key);
      if (fromObservation !== null) return fromObservation;
    }
  }

  for (const tag of tags) {
    if (tag.startsWith(TASK_TAG_PREFIX) && tag.length > TASK_TAG_PREFIX.length) {
      return tag.slice(TASK_TAG_PREFIX.length);
    }
  }
  return null;
}

function compareObservations(a: ObservationView, b: ObservationView): number {
  const left = a.startedAt ?? "";
  const right = b.startedAt ?? "";
  if (left !== right) return left < right ? -1 : 1;
  return a.order - b.order;
}
