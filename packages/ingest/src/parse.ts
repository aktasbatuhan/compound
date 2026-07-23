/**
 * Input reading and per-record classification.
 *
 * A malformed line is rejected on its own line number and never fails the file,
 * and casing is detected per record — mixed files are normal input, not an
 * error (docs/langfuse-import-mapping.md, "Accepted input surfaces").
 */
import type { Casing, InputFormat, RejectedRecord } from "./types";
import { field, isRecord } from "./values";

export interface RawRecord {
  /** 1-based line number (JSONL) or record index (JSON array / in-memory array). */
  line: number;
  value: Record<string, unknown>;
  kind: RecordKind;
  casing: Casing;
}

export type RecordKind = "trace" | "observation" | "score";

export interface ParseResult {
  records: RawRecord[];
  rejected: RejectedRecord[];
  format: InputFormat;
}

export const REJECTION_REASONS = {
  malformedJsonLine: "malformed_json_line",
  fileNotValidJson: "file_not_valid_json",
  fileNotAJsonArray: "file_not_a_json_array",
  recordNotAnObject: "record_not_an_object",
  recordMissingId: "record_missing_id",
  observationMissingTraceId: "observation_missing_trace_id",
  unrecognizedRecordShape: "unrecognized_record_shape",
  traceMissingStartedAt: "trace_missing_started_at",
} as const;

const CAMEL_MARKERS = [
  "traceId",
  "startTime",
  "endTime",
  "parentObservationId",
  "usageDetails",
  "costDetails",
  "sessionId",
  "userId",
  "statusMessage",
  "modelParameters",
  "providedModelName",
  "completionStartTime",
  "dataType",
  "stringValue",
  "observationId",
  "projectId",
  "modelId",
  "createdAt",
];

const SNAKE_MARKERS = [
  "trace_id",
  "start_time",
  "end_time",
  "parent_observation_id",
  "usage_details",
  "cost_details",
  "session_id",
  "user_id",
  "status_message",
  "model_parameters",
  "provided_model_name",
  "completion_start_time",
  "data_type",
  "string_value",
  "observation_id",
  "project_id",
  "model_id",
  "created_at",
];

/** Per-record casing, by counting known field-name markers. Ties read as unknown. */
export function detectCasing(record: Record<string, unknown>): Casing {
  let camel = 0;
  let snake = 0;
  for (const key of Object.keys(record)) {
    if (CAMEL_MARKERS.includes(key)) camel += 1;
    if (SNAKE_MARKERS.includes(key)) snake += 1;
  }
  if (camel > snake) return "camelCase";
  if (snake > camel) return "snake_case";
  return "unknown";
}

const SCORE_SOURCES = ["ANNOTATION", "API", "EVAL"];

/**
 * Classify a record as trace / observation / score.
 *
 * Scores are recognized by their `source` enum or `dataType`; observations by a
 * `type` plus a start time; traces by an `id` plus a `timestamp`.
 */
export function classifyRecord(record: Record<string, unknown>): RecordKind | null {
  const source = field(record, "source");
  const dataType = field(record, "dataType", "data_type");
  if (
    (typeof source === "string" && SCORE_SOURCES.includes(source.toUpperCase())) ||
    typeof dataType === "string"
  ) {
    return "score";
  }

  const hasStart = field(record, "startTime", "start_time") !== undefined;
  if (typeof field(record, "type") === "string" && hasStart) return "observation";

  if (field(record, "timestamp") !== undefined) return "trace";
  if (Array.isArray(field(record, "observations"))) return "trace";
  return null;
}

/**
 * Read the input surface.
 *
 * A string starting with `[` is read as a JSON array; anything else is read as
 * JSONL. An already-parsed array is used as-is (record index as line number).
 */
export function parseInput(input: string | unknown[]): ParseResult {
  if (Array.isArray(input)) {
    return classifyAll(
      input.map((value, index) => ({ line: index + 1, value })),
      "records",
    );
  }

  const trimmed = input.trim();
  if (trimmed.startsWith("[")) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      return {
        records: [],
        rejected: [{ line: 1, reason: REJECTION_REASONS.fileNotValidJson }],
        format: "json_array",
      };
    }
    if (!Array.isArray(parsed)) {
      return {
        records: [],
        rejected: [{ line: 1, reason: REJECTION_REASONS.fileNotAJsonArray }],
        format: "json_array",
      };
    }
    return classifyAll(
      parsed.map((value, index) => ({ line: index + 1, value })),
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
      rejected.push({ line, reason: REJECTION_REASONS.malformedJsonLine });
    }
  });

  const result = classifyAll(entries, "jsonl");
  return { ...result, rejected: [...rejected, ...result.rejected].sort((a, b) => a.line - b.line) };
}

function classifyAll(
  entries: { line: number; value: unknown }[],
  format: InputFormat,
): ParseResult {
  const records: RawRecord[] = [];
  const rejected: RejectedRecord[] = [];

  for (const entry of entries) {
    if (!isRecord(entry.value)) {
      rejected.push({ line: entry.line, reason: REJECTION_REASONS.recordNotAnObject });
      continue;
    }
    const kind = classifyRecord(entry.value);
    if (kind === null) {
      rejected.push({ line: entry.line, reason: REJECTION_REASONS.unrecognizedRecordShape });
      continue;
    }
    if (typeof entry.value.id !== "string" || entry.value.id === "") {
      rejected.push({ line: entry.line, reason: REJECTION_REASONS.recordMissingId });
      continue;
    }
    records.push({
      line: entry.line,
      value: entry.value,
      kind,
      casing: detectCasing(entry.value),
    });
  }

  return { records, rejected, format };
}
