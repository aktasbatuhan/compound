/**
 * `normalizeJsonExport` — plain contract-shaped trace JSON into contract traces.
 *
 * The spec's second day-one ingest path (SPEC.md: "Import: Langfuse export, or
 * plain trace JSON"). Unlike the Langfuse importer, the input is already in — or
 * nearly in — the Compound contract shape (`docs/trace-contract-v1.md`): each
 * record is one `Trace`. This importer accepts fully-formed contract traces
 * as-is (prefixing the id), and fills in only the envelope/wrapper fields for a
 * lenient hand-written record. It never invents trace *content*, never performs
 * I/O, and — like every normalizer — leaves `redactions: []` and the caller's
 * `defaultPermissions`; redaction is a later pipeline pass.
 *
 * Mirrors the `normalizeLangfuseExport` API exactly so the pipeline can dispatch
 * to either interchangeably.
 */
import {
  TRACE_SCHEMA_NAME,
  TRACE_SCHEMA_VERSION,
  type Trace,
  TraceSchema,
} from "@compound/contract";
import { Collector, DIAGNOSTICS } from "./diagnostics";
import type {
  Casing,
  ImportSourceReport,
  InputFormat,
  NormalizedTrace,
  NormalizeOptions,
  NormalizeResult,
  RejectedRecord,
} from "./types";
import { asString, isRecord } from "./values";

export const IMPORTER_NAME = "json";

/**
 * Stable snake_case rejection reasons for the JSON importer. Malformed-input
 * reasons mirror the Langfuse importer; the content reasons are specific to
 * contract-shaped records.
 */
export const JSON_REJECTION_REASONS = {
  /** A JSONL line did not parse as JSON. */
  malformedJsonLine: "malformed_json_line",
  /** A string starting with `[` was not valid JSON. */
  fileNotValidJson: "file_not_valid_json",
  /** A `[`-led file parsed to something other than an array. */
  fileNotAJsonArray: "file_not_a_json_array",
  /** A record was not a JSON object. */
  recordNotAnObject: "record_not_an_object",
  /** A record carried no `trace_id`; there is no stable id to key on. */
  recordMissingTraceId: "record_missing_trace_id",
  /** A record had no `steps` (missing or empty); there is no replayable work. */
  traceMissingSteps: "trace_missing_steps",
  /** A record's steps held nothing that could be a `model_call`. */
  traceNoModelCall: "trace_no_model_call",
  /** The filled record did not satisfy the contract `TraceSchema`. */
  traceSchemaInvalid: "trace_schema_invalid",
} as const;

/**
 * A `trace_id` "already looks prefixed" when it starts with a lowercase source
 * token and a colon (`json:`, `langfuse:…`). Such ids are used verbatim so a
 * previously-exported contract trace does not get double-prefixed; anything else
 * gets the `json:` prefix, mirroring the Langfuse importer's `langfuse:` prefix.
 */
const SOURCE_PREFIX = /^[a-z][a-z0-9_-]*:/;

interface JsonRecord {
  line: number;
  value: Record<string, unknown>;
}

interface JsonParseResult {
  records: JsonRecord[];
  rejected: RejectedRecord[];
  format: InputFormat;
}

/**
 * Normalize plain trace JSON into contract traces plus a source report.
 *
 * Accepts a JSON array, JSONL text, or an already-parsed array. Each record is
 * one contract `Trace` (or a lenient near-contract record). Never performs I/O.
 */
export function normalizeJsonExport(
  input: string | unknown[],
  options: NormalizeOptions,
): NormalizeResult {
  const parsed = parseJsonInput(input);
  const rejected: RejectedRecord[] = [...parsed.rejected];
  const collector = new Collector();

  const traces: NormalizedTrace[] = [];
  let traceRecords = 0;

  for (const record of parsed.records) {
    traceRecords += 1;
    const normalized = normalizeRecord(record, options, collector, rejected);
    if (normalized !== null) traces.push(normalized);
  }

  const report: ImportSourceReport = {
    surface: "json",
    // Contract records are snake_case by definition; we do not detect a dialect
    // for the JSON surface, so casing is reported as not-applicable (empty).
    casing: [] as Casing[],
    format: parsed.format,
    counts: {
      recordsSeen: traceRecords,
      recordsRejected: rejected.length,
      traceRecords,
      observationRecords: 0,
      scoreRecords: 0,
      tracesNormalized: traces.length,
    },
    rejected: [...rejected].sort((a, b) => a.line - b.line),
    diagnosticReasons: Object.fromEntries([...collector.reasonHistogram.entries()].sort()),
    // Message/tool-call dialects and scores are Langfuse-surface concepts; the
    // JSON surface carries contract-normal messages, so both are empty here.
    dialects: [...collector.dialects].sort(),
    skippedScores: { total: 0, byReason: {} },
    unknownObservationTypes: [],
  };

  return { traces, report };
}

/**
 * Turn one contract-shaped record into a `NormalizedTrace`, or reject it.
 *
 * Rejections are per-record, counted with a line number, and never fail the
 * file — exactly like the Langfuse importer. The importer guarantees its output
 * satisfies `TraceSchema`, so a record that cannot be made valid is rejected as
 * user input rather than emitted as an internal error.
 */
function normalizeRecord(
  record: JsonRecord,
  options: NormalizeOptions,
  collector: Collector,
  rejected: RejectedRecord[],
): NormalizedTrace | null {
  const raw = record.value;

  const rawTraceId = asString(raw.trace_id);
  if (rawTraceId === null) {
    rejected.push({ line: record.line, reason: JSON_REJECTION_REASONS.recordMissingTraceId });
    return null;
  }

  const steps = raw.steps;
  if (!Array.isArray(steps) || steps.length === 0) {
    rejected.push({ line: record.line, reason: JSON_REJECTION_REASONS.traceMissingSteps });
    return null;
  }

  const modelCalls = steps.filter(
    (step): step is Record<string, unknown> => isRecord(step) && step.type === "model_call",
  );
  if (modelCalls.length === 0) {
    rejected.push({ line: record.line, reason: JSON_REJECTION_REASONS.traceNoModelCall });
    return null;
  }

  // Fill only the envelope/wrapper fields, and only when absent — a fully-formed
  // contract trace round-trips unchanged except for the id prefix. Trace content
  // (task_key, started_at, steps, outcome, …) is never invented.
  const candidate: Record<string, unknown> = { ...raw };
  candidate.schema = raw.schema ?? TRACE_SCHEMA_NAME;
  candidate.schema_version = raw.schema_version ?? TRACE_SCHEMA_VERSION;
  candidate.trace_id = prefixTraceId(rawTraceId);
  candidate.source = isRecord(raw.source)
    ? raw.source
    : {
        importer: IMPORTER_NAME,
        importer_version: options.importerVersion,
        source_ids: { trace_id: rawTraceId },
        exported_at: options.exportedAt ?? null,
      };
  candidate.permissions = isRecord(raw.permissions)
    ? raw.permissions
    : { ...options.defaultPermissions };
  // Redaction is a separate pass; this importer never masks a value.
  candidate.redactions = Array.isArray(raw.redactions) ? raw.redactions : [];
  candidate.focal_step_id = resolveFocalStepId(raw.focal_step_id, modelCalls);

  const validated = TraceSchema.safeParse(candidate);
  if (!validated.success) {
    rejected.push({ line: record.line, reason: JSON_REJECTION_REASONS.traceSchemaInvalid });
    return null;
  }

  const trace: Trace = validated.data;
  // A trace with no focal call is diagnostic, not eval-ready. The contract
  // validator flags it too; emitting the normalizer diagnostic keeps the report
  // histogram honest about why (mirrors the Langfuse importer).
  if (trace.focal_step_id === null) {
    collector.diagnostic(DIAGNOSTICS.noReplayableFocalCall);
  }

  return { trace, diagnostics: collector.takeTraceDiagnostics() };
}

/**
 * Honor an explicit `focal_step_id`; otherwise auto-set it only when there is
 * exactly one `model_call` (an unambiguous focal). With several model calls and
 * no explicit focal, leave `null` — the contract validator makes it diagnostic.
 */
function resolveFocalStepId(
  explicit: unknown,
  modelCalls: Record<string, unknown>[],
): string | null {
  const provided = asString(explicit);
  if (provided !== null) return provided;
  if (modelCalls.length === 1) {
    const sole = modelCalls[0];
    const stepId = sole === undefined ? null : asString(sole.step_id);
    if (stepId !== null) return stepId;
  }
  return null;
}

/** `json:<id>`, unless the id already looks prefixed, mirroring `langfuse:<id>`. */
export function prefixTraceId(id: string): string {
  return SOURCE_PREFIX.test(id) ? id : `${IMPORTER_NAME}:${id}`;
}

/**
 * Read the input surface.
 *
 * A string starting with `[` is read as a JSON array; anything else is read as
 * JSONL. An already-parsed array is used as-is (record index as line number).
 * Mirrors the Langfuse importer's input reading, but each record is a whole
 * contract trace rather than a Langfuse trace/observation/score.
 */
function parseJsonInput(input: string | unknown[]): JsonParseResult {
  if (Array.isArray(input)) {
    return classifyEntries(
      input.map((value, index) => ({ line: index + 1, value })),
      "records",
    );
  }

  const trimmed = input.trim();
  if (trimmed.startsWith("[")) {
    let value: unknown;
    try {
      value = JSON.parse(trimmed);
    } catch {
      return {
        records: [],
        rejected: [{ line: 1, reason: JSON_REJECTION_REASONS.fileNotValidJson }],
        format: "json_array",
      };
    }
    if (!Array.isArray(value)) {
      return {
        records: [],
        rejected: [{ line: 1, reason: JSON_REJECTION_REASONS.fileNotAJsonArray }],
        format: "json_array",
      };
    }
    return classifyEntries(
      value.map((entry, index) => ({ line: index + 1, value: entry })),
      "json_array",
    );
  }

  const entries: { line: number; value: unknown }[] = [];
  const rejected: RejectedRecord[] = [];
  input.split(/\r?\n/).forEach((rawLine, index) => {
    const line = index + 1;
    if (rawLine.trim() === "") return;
    try {
      entries.push({ line, value: JSON.parse(rawLine) });
    } catch {
      rejected.push({ line, reason: JSON_REJECTION_REASONS.malformedJsonLine });
    }
  });

  const result = classifyEntries(entries, "jsonl");
  return { ...result, rejected: [...rejected, ...result.rejected].sort((a, b) => a.line - b.line) };
}

function classifyEntries(
  entries: { line: number; value: unknown }[],
  format: InputFormat,
): JsonParseResult {
  const records: JsonRecord[] = [];
  const rejected: RejectedRecord[] = [];

  for (const entry of entries) {
    if (!isRecord(entry.value)) {
      rejected.push({ line: entry.line, reason: JSON_REJECTION_REASONS.recordNotAnObject });
      continue;
    }
    records.push({ line: entry.line, value: entry.value });
  }

  return { records, rejected, format };
}
