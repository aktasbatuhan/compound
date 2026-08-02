/**
 * `normalizeOtelExport` — OpenTelemetry GenAI spans (OTLP/JSON) into contract traces.
 *
 * The spec's telemetry ingest path (SPEC.md, "OTel GenAI ingest"): a file/collector
 * OTLP/JSON export whose LLM spans follow the GenAI semantic conventions. This
 * importer reads the widely-deployed FLAT-ATTRIBUTE convention emitted by
 * OpenLLMetry/Traceloop, Logfire, Langtrace and Phoenix — `gen_ai.*` (and the
 * legacy `llm.*` alias) attributes — plus the newer structured
 * `gen_ai.input.messages` / `gen_ai.output.messages` JSON payloads.
 *
 * Like every normalizer it only *normalizes*: no redaction, no persistence, no
 * I/O; `redactions` is always `[]` and permissions come from the caller. Message
 * and tool-call shaping is delegated to the shared `normalizeGeneration*` helpers
 * so dialects and diagnostics stay identical across importers. A span whose
 * prompt cannot be recovered yields an empty-input model_call — the contract
 * validator then classes the trace `diagnostic`, never a silent eval case.
 *
 * Mirrors the `normalizeJsonExport` / `normalizeLangfuseExport` API so the
 * pipeline can dispatch to any importer interchangeably.
 */
import {
  type Message,
  type ModelCallStep,
  type Step,
  TRACE_SCHEMA_NAME,
  TRACE_SCHEMA_VERSION,
  type Trace,
  TraceSchema,
  type Usage,
} from "@compound/contract";
import { Collector, DIAGNOSTICS } from "./diagnostics";
import { selectFocalStepId } from "./linking";
import { normalizeGenerationInput, normalizeGenerationOutput } from "./messages";
import type {
  Casing,
  ImportSourceReport,
  InputFormat,
  NormalizedTrace,
  NormalizeOptions,
  NormalizeResult,
  RejectedRecord,
} from "./types";
import { asFiniteNumber, asString, earliest, isRecord, latest } from "./values";

export const IMPORTER_NAME = "otel";

/** Stable snake_case rejection reasons for the OTel importer. */
export const OTEL_REJECTION_REASONS = {
  /** The input string did not parse as JSON (or JSONL). */
  fileNotValidJson: "file_not_valid_json",
  /** Parsed, but no `resourceSpans` were found anywhere. */
  noResourceSpans: "no_resource_spans",
  /** A span carried no `traceId`/`spanId`; it cannot be keyed or linked. */
  spanMissingIds: "span_missing_ids",
} as const;

/** A GenAI/LLM attribute namespace: `gen_ai.*` (current) or `llm.*` (legacy). */
const GENAI_KEY = /^(gen_ai|llm)\./;

/** `gen_ai.prompt.3.content` / `llm.completion.0.role` → { kind, index, field }. */
const INDEXED_MESSAGE_KEY =
  /^(?:gen_ai|llm)\.(prompt|completion)\.(\d+)\.(role|content|finish_reason|tool_calls)$/;

interface OtelSpan {
  /** 1-based ordinal across the whole file, used as the rejected-record line. */
  ordinal: number;
  traceId: string;
  spanId: string;
  parentSpanId: string | null;
  name: string;
  startedAt: string | null;
  endedAt: string | null;
  /** Flattened span attributes (OTLP AnyValue decoded to plain JSON). */
  attrs: Record<string, unknown>;
  /** Resource-level attributes in scope for this span (service, deployment, …). */
  resourceAttrs: Record<string, unknown>;
  error: string | null;
}

/**
 * Normalize an OTLP/JSON trace export into contract traces plus a source report.
 *
 * Accepts a single `ExportTraceServiceRequest` JSON object, JSONL where each line
 * is one such object (the OTLP file exporter's shape), or an already-parsed
 * object/array. Both camelCase (`resourceSpans`) and snake_case (`resource_spans`)
 * OTLP keys are read. Never performs I/O.
 */
export function normalizeOtelExport(
  input: string | unknown,
  options: NormalizeOptions,
): NormalizeResult {
  const rejected: RejectedRecord[] = [];
  const collector = new Collector();

  const parsed = parseOtlpInput(input);
  if (parsed.rejected !== null) rejected.push(parsed.rejected);

  const spans = collectSpans(parsed.documents, rejected);

  // Group spans into one contract trace per OTLP traceId, preserving first-seen order.
  const byTrace = new Map<string, OtelSpan[]>();
  for (const span of spans) {
    const group = byTrace.get(span.traceId);
    if (group === undefined) byTrace.set(span.traceId, [span]);
    else group.push(span);
  }

  const traces: NormalizedTrace[] = [];
  let genaiSpans = 0;
  for (const [traceId, group] of byTrace) {
    const normalized = normalizeTrace(traceId, group, options, collector);
    genaiSpans += normalized.genaiSpans;
    traces.push(normalized.result);
  }

  const report: ImportSourceReport = {
    surface: "unknown",
    casing: [] as Casing[],
    format: parsed.format,
    counts: {
      recordsSeen: spans.length,
      recordsRejected: rejected.length,
      // OTel has no separate trace/observation/score records; a "generation" is a
      // GenAI span, reported here so the count means the same thing across surfaces.
      traceRecords: byTrace.size,
      observationRecords: genaiSpans,
      scoreRecords: 0,
      tracesNormalized: traces.length,
    },
    rejected: [...rejected].sort((a, b) => a.line - b.line),
    diagnosticReasons: Object.fromEntries([...collector.reasonHistogram.entries()].sort()),
    dialects: [...collector.dialects].sort(),
    skippedScores: { total: 0, byReason: {} },
    unknownObservationTypes: [],
  };

  return { traces, report };
}

interface ParsedOtlp {
  /** Each element is one decoded `ExportTraceServiceRequest`-shaped object. */
  documents: Record<string, unknown>[];
  format: InputFormat;
  /** A whole-file rejection (unparseable, or nothing span-shaped), or null. */
  rejected: RejectedRecord | null;
}

const NOT_JSON: RejectedRecord = { line: 1, reason: OTEL_REJECTION_REASONS.fileNotValidJson };

function parseOtlpInput(input: string | unknown): ParsedOtlp {
  // Already-parsed value (array of documents, or one document).
  if (Array.isArray(input)) return finishDocuments(input, "records");
  if (isRecord(input)) return finishDocuments([input], "records");
  if (typeof input !== "string") return { documents: [], format: "records", rejected: NOT_JSON };

  const trimmed = input.trim();
  if (trimmed.length === 0) return { documents: [], format: "records", rejected: NOT_JSON };

  // A single JSON document (object or array) parses whole. On failure we do NOT
  // reject yet — the OTLP file exporter writes one JSON object PER LINE, whose
  // first character is also `{`, so we fall through to JSONL parsing.
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    try {
      const value = JSON.parse(trimmed);
      return finishDocuments(
        Array.isArray(value) ? value : [value],
        Array.isArray(value) ? "json_array" : "records",
      );
    } catch {
      // fall through to JSONL
    }
  }

  // JSONL (or a failed whole-parse): one JSON value per line. If nothing at all
  // parses, the input was not JSON; a parsed-but-span-less doc is a different
  // (no_resource_spans) failure handled by finishDocuments.
  const documents: Record<string, unknown>[] = [];
  let anyParsed = false;
  for (const line of trimmed.split("\n")) {
    const text = line.trim();
    if (text.length === 0) continue;
    try {
      const value = JSON.parse(text);
      anyParsed = true;
      if (isRecord(value)) documents.push(value);
    } catch {
      // skip a malformed line
    }
  }
  if (!anyParsed) return { documents: [], format: "jsonl", rejected: NOT_JSON };
  return finishDocuments(documents, "jsonl");
}

function finishDocuments(values: unknown[], format: InputFormat): ParsedOtlp {
  const documents = values.filter(isRecord);
  const hasSpans = documents.some((doc) => resourceSpansOf(doc).length > 0);
  return {
    documents,
    format,
    rejected: hasSpans ? null : { line: 1, reason: OTEL_REJECTION_REASONS.noResourceSpans },
  };
}

/** OTLP keys come in camelCase or snake_case; read either. */
function pick(record: Record<string, unknown>, camel: string, snake: string): unknown {
  return record[camel] ?? record[snake];
}

function resourceSpansOf(doc: Record<string, unknown>): Record<string, unknown>[] {
  const rs = pick(doc, "resourceSpans", "resource_spans");
  return Array.isArray(rs) ? rs.filter(isRecord) : [];
}

/** Walk resourceSpans → scopeSpans → spans, flattening each into an OtelSpan. */
function collectSpans(
  documents: Record<string, unknown>[],
  rejected: RejectedRecord[],
): OtelSpan[] {
  const spans: OtelSpan[] = [];
  let ordinal = 0;

  for (const doc of documents) {
    for (const resourceSpan of resourceSpansOf(doc)) {
      const resource = pick(resourceSpan, "resource", "resource");
      const resourceAttrs = isRecord(resource)
        ? attrsToRecord(pick(resource, "attributes", "attributes"))
        : {};
      const scopeSpans = pick(resourceSpan, "scopeSpans", "scope_spans");
      const scopes = Array.isArray(scopeSpans) ? scopeSpans.filter(isRecord) : [];

      for (const scope of scopes) {
        const rawSpans = pick(scope, "spans", "spans");
        const list = Array.isArray(rawSpans) ? rawSpans.filter(isRecord) : [];
        for (const raw of list) {
          ordinal += 1;
          const traceId = asString(pick(raw, "traceId", "trace_id"));
          const spanId = asString(pick(raw, "spanId", "span_id"));
          if (traceId === null || spanId === null) {
            rejected.push({ line: ordinal, reason: OTEL_REJECTION_REASONS.spanMissingIds });
            continue;
          }
          const parentSpanId = asString(pick(raw, "parentSpanId", "parent_span_id"));
          spans.push({
            ordinal,
            traceId,
            spanId,
            parentSpanId: parentSpanId !== null && parentSpanId.length > 0 ? parentSpanId : null,
            name: asString(raw.name) ?? "span",
            startedAt: nanoToIso(pick(raw, "startTimeUnixNano", "start_time_unix_nano")),
            endedAt: nanoToIso(pick(raw, "endTimeUnixNano", "end_time_unix_nano")),
            attrs: attrsToRecord(pick(raw, "attributes", "attributes")),
            resourceAttrs,
            error: statusError(pick(raw, "status", "status")),
          });
        }
      }
    }
  }
  return spans;
}

interface NormalizedOtelTrace {
  result: NormalizedTrace;
  genaiSpans: number;
}

function normalizeTrace(
  traceId: string,
  group: OtelSpan[],
  options: NormalizeOptions,
  collector: Collector,
): NormalizedOtelTrace {
  const steps: Step[] = [];
  let genaiSpans = 0;

  for (const span of group) {
    if (isGenAiSpan(span.attrs)) {
      genaiSpans += 1;
      steps.push(modelCallStep(span, collector));
    } else {
      // Non-GenAI spans are preserved for lineage but never evaluated.
      steps.push({
        type: "other",
        step_id: span.spanId,
        ...(span.parentSpanId !== null ? { parent_step_id: span.parentSpanId } : {}),
        name: span.name,
      });
    }
  }

  const startedAt =
    earliest(group.map((s) => s.startedAt)) ?? options.exportedAt ?? new Date(0).toISOString();
  const endedAt = latest(group.map((s) => s.endedAt));
  const attrsFor = (key: string): unknown =>
    group.map((s) => s.attrs[key] ?? s.resourceAttrs[key]).find((v) => v !== undefined);

  const taskKey =
    asString(attrsFor("gen_ai.compound.task_key")) ??
    asString(attrsFor("compound.task_key")) ??
    asString(attrsFor("task_key"));
  const environment =
    asString(attrsFor("deployment.environment.name")) ??
    asString(attrsFor("deployment.environment"));

  const focalStepId = selectFocalStepId(steps, collector);
  if (focalStepId === null) collector.diagnostic(DIAGNOSTICS.noReplayableFocalCall);

  const candidate = {
    schema: TRACE_SCHEMA_NAME,
    schema_version: TRACE_SCHEMA_VERSION,
    trace_id: prefixTraceId(traceId, options.projectId),
    task_key: taskKey,
    started_at: startedAt,
    ended_at: endedAt,
    source: {
      importer: IMPORTER_NAME,
      importer_version: options.importerVersion,
      source_ids: { trace_id: traceId },
      exported_at: options.exportedAt ?? null,
    },
    ...(environment !== null ? { environment } : {}),
    steps,
    focal_step_id: focalStepId,
    permissions: { ...options.defaultPermissions },
    redactions: [],
  };

  const validated = TraceSchema.safeParse(candidate);
  if (!validated.success) {
    // The importer guarantees schema-valid output; a fabricated invalid trace is
    // an internal bug, so we surface it as a minimal diagnostic trace rather than
    // throwing and failing the whole file. This should not happen in practice.
    collector.diagnostic("otel_internal_normalization_error");
    const fallback: Trace = {
      schema: TRACE_SCHEMA_NAME,
      schema_version: TRACE_SCHEMA_VERSION,
      trace_id: prefixTraceId(traceId, options.projectId),
      task_key: taskKey,
      started_at: startedAt,
      source: {
        importer: IMPORTER_NAME,
        importer_version: options.importerVersion,
        source_ids: { trace_id: traceId },
        exported_at: options.exportedAt ?? null,
      },
      steps: [],
      focal_step_id: null,
      permissions: { ...options.defaultPermissions },
      redactions: [],
    };
    return {
      result: { trace: fallback, diagnostics: collector.takeTraceDiagnostics() },
      genaiSpans,
    };
  }

  return {
    result: { trace: validated.data, diagnostics: collector.takeTraceDiagnostics() },
    genaiSpans,
  };
}

/** Build a model_call step from one GenAI span, delegating message shaping. */
function modelCallStep(span: OtelSpan, collector: Collector): ModelCallStep {
  const attrs = span.attrs;

  const input = reconstructInput(attrs);
  const normalizedInput = normalizeGenerationInput(input, collector, span.spanId);
  const output = reconstructOutput(attrs, collector, span.spanId);

  const provider = asString(attrs["gen_ai.system"]) ?? asString(attrs["llm.system"]);
  const model = asString(attrs["gen_ai.request.model"]) ?? asString(attrs["llm.request.model"]);
  const resolvedModel =
    asString(attrs["gen_ai.response.model"]) ?? asString(attrs["llm.response.model"]);

  const params = requestParams(attrs);
  const usage = reconstructUsage(attrs);

  const step: ModelCallStep = {
    type: "model_call",
    step_id: span.spanId,
    ...(span.parentSpanId !== null ? { parent_step_id: span.parentSpanId } : {}),
    ...(provider !== null ? { provider } : {}),
    ...(model !== null ? { model } : {}),
    ...(resolvedModel !== null ? { resolved_model: resolvedModel } : {}),
    ...(params !== null ? { params } : {}),
    input: normalizedInput.messages,
    ...(normalizedInput.tools !== null ? { tools_available: normalizedInput.tools } : {}),
    ...(output !== null ? { output } : {}),
    ...(finishReason(attrs) !== null ? { finish_reason: finishReason(attrs) as string } : {}),
    ...(usage !== null ? { usage } : {}),
    ...(span.startedAt !== null ? { started_at: span.startedAt } : {}),
    ...(span.endedAt !== null ? { ended_at: span.endedAt } : {}),
    ...(span.error !== null ? { error: span.error } : {}),
  };
  return step;
}

/**
 * Reconstruct the generation input as an OpenAI-shaped message array so the
 * shared `normalizeGenerationInput` can shape it. Prefer the structured
 * `gen_ai.input.messages` payload; otherwise assemble the flat
 * `gen_ai.prompt.{i}.{role,content}` attributes. Returns `undefined` when there
 * is nothing to read — an absent prompt is not "unparseable".
 */
function reconstructInput(attrs: Record<string, unknown>): unknown {
  const structured = parseMaybeJsonArray(attrs["gen_ai.input.messages"]);
  if (structured !== null) return structured;
  const flat = indexedMessages(attrs, "prompt");
  return flat.length > 0 ? flat : undefined;
}

/** Reconstruct the assistant output message from structured or flat attributes. */
function reconstructOutput(
  attrs: Record<string, unknown>,
  collector: Collector,
  stepId: string,
): Message | null {
  const structured = parseMaybeJsonArray(attrs["gen_ai.output.messages"]);
  if (structured !== null && structured.length > 0) {
    return normalizeGenerationOutput(structured[0], collector, stepId);
  }
  const flat = indexedMessages(attrs, "completion");
  if (flat.length > 0) return normalizeGenerationOutput(flat[0], collector, stepId);
  return null;
}

/**
 * Collect `<ns>.<kind>.<i>.<field>` attributes into an index-ordered array of
 * OpenAI-shaped message objects. `content` that is a JSON tool_calls string is
 * attached as `tool_calls` so the shared normalizer maps the OpenAI dialect.
 */
function indexedMessages(
  attrs: Record<string, unknown>,
  kind: "prompt" | "completion",
): Record<string, unknown>[] {
  const byIndex = new Map<number, Record<string, unknown>>();
  for (const [key, value] of Object.entries(attrs)) {
    const match = INDEXED_MESSAGE_KEY.exec(key);
    if (match === null || match[1] !== kind) continue;
    const index = Number.parseInt(match[2] as string, 10);
    const field = match[3] as string;
    const entry = byIndex.get(index) ?? {};
    if (field === "tool_calls") {
      const parsed = parseMaybeJsonArray(value);
      if (parsed !== null) entry.tool_calls = parsed;
    } else {
      entry[field] = value;
    }
    byIndex.set(index, entry);
  }
  return [...byIndex.entries()].sort((a, b) => a[0] - b[0]).map(([, entry]) => entry);
}

function requestParams(attrs: Record<string, unknown>): Record<string, number> | null {
  const params: Record<string, number> = {};
  const temperature =
    asFiniteNumber(attrs["gen_ai.request.temperature"]) ??
    asFiniteNumber(attrs["llm.request.temperature"]);
  const maxTokens =
    asFiniteNumber(attrs["gen_ai.request.max_tokens"]) ??
    asFiniteNumber(attrs["llm.request.max_tokens"]);
  const topP =
    asFiniteNumber(attrs["gen_ai.request.top_p"]) ?? asFiniteNumber(attrs["llm.request.top_p"]);
  if (temperature !== null) params.temperature = temperature;
  if (maxTokens !== null) params.max_tokens = maxTokens;
  if (topP !== null) params.top_p = topP;
  return Object.keys(params).length > 0 ? params : null;
}

/** Contract `Usage` from GenAI token attributes (current or legacy names). */
function reconstructUsage(attrs: Record<string, unknown>): Usage | null {
  const input =
    asFiniteNumber(attrs["gen_ai.usage.input_tokens"]) ??
    asFiniteNumber(attrs["gen_ai.usage.prompt_tokens"]) ??
    asFiniteNumber(attrs["llm.usage.prompt_tokens"]);
  const output =
    asFiniteNumber(attrs["gen_ai.usage.output_tokens"]) ??
    asFiniteNumber(attrs["gen_ai.usage.completion_tokens"]) ??
    asFiniteNumber(attrs["llm.usage.completion_tokens"]);
  if (input === null && output === null) return null;
  const total =
    asFiniteNumber(attrs["gen_ai.usage.total_tokens"]) ??
    asFiniteNumber(attrs["llm.usage.total_tokens"]);
  const reasoning = asFiniteNumber(attrs["gen_ai.usage.reasoning_tokens"]);
  return {
    input_tokens: input !== null ? Math.max(0, Math.trunc(input)) : 0,
    output_tokens: output !== null ? Math.max(0, Math.trunc(output)) : 0,
    ...(reasoning !== null ? { reasoning_tokens: Math.max(0, Math.trunc(reasoning)) } : {}),
    ...(total !== null ? { total_tokens: Math.max(0, Math.trunc(total)) } : {}),
  };
}

function finishReason(attrs: Record<string, unknown>): string | null {
  const reasons = attrs["gen_ai.response.finish_reasons"];
  if (Array.isArray(reasons) && reasons.length > 0) {
    const first = asString(reasons[0]);
    if (first !== null) return first;
  }
  return (
    asString(attrs["gen_ai.completion.0.finish_reason"]) ??
    asString(attrs["llm.completion.0.finish_reason"])
  );
}

function isGenAiSpan(attrs: Record<string, unknown>): boolean {
  for (const key of Object.keys(attrs)) {
    if (GENAI_KEY.test(key)) return true;
  }
  return false;
}

/** `otel:<traceId>` (or `otel:<projectId>:<traceId>`), mirroring `langfuse:` / `json:`. */
export function prefixTraceId(id: string, projectId?: string): string {
  if (/^[a-z][a-z0-9_-]*:/.test(id)) return id;
  return projectId !== undefined ? `${IMPORTER_NAME}:${projectId}:${id}` : `${IMPORTER_NAME}:${id}`;
}

// ---------------------------------------------------------------------------
// OTLP value decoding
// ---------------------------------------------------------------------------

/** Decode `[{key, value: AnyValue}, …]` into a flat `Record`. */
function attrsToRecord(raw: unknown): Record<string, unknown> {
  const record: Record<string, unknown> = {};
  if (!Array.isArray(raw)) return record;
  for (const attr of raw) {
    if (!isRecord(attr)) continue;
    const key = asString(attr.key);
    if (key === null) continue;
    record[key] = otlpValue(attr.value);
  }
  return record;
}

/** One OTLP AnyValue → plain JSON. int64 arrives as a string in OTLP/JSON. */
function otlpValue(value: unknown): unknown {
  if (!isRecord(value)) return undefined;
  if ("stringValue" in value) return value.stringValue;
  if ("intValue" in value) {
    const n = value.intValue;
    return typeof n === "string" ? Number(n) : n;
  }
  if ("doubleValue" in value) return value.doubleValue;
  if ("boolValue" in value) return value.boolValue;
  if (
    "arrayValue" in value &&
    isRecord(value.arrayValue) &&
    Array.isArray(value.arrayValue.values)
  ) {
    return value.arrayValue.values.map(otlpValue);
  }
  if ("kvlistValue" in value && isRecord(value.kvlistValue)) {
    return attrsToRecord(value.kvlistValue.values);
  }
  return undefined;
}

/** OTLP span status → an error string, only when the status code is ERROR (2). */
function statusError(status: unknown): string | null {
  if (!isRecord(status)) return null;
  const code = status.code;
  const isError = code === 2 || code === "STATUS_CODE_ERROR";
  if (!isError) return null;
  return asString(status.message) ?? "span status ERROR";
}

/** Nanoseconds-since-epoch (string or number) → ISO-8601 UTC, or null. */
function nanoToIso(value: unknown): string | null {
  const raw = asString(value) ?? (typeof value === "number" ? String(value) : null);
  if (raw === null) return null;
  const digits = raw.trim();
  if (!/^\d+$/.test(digits)) return null;
  const ms = Number(BigInt(digits) / 1_000_000n);
  const date = new Date(ms);
  return Number.isNaN(date.getTime()) ? null : date.toISOString();
}

/** Parse a value that may be a JSON array string (or already an array). */
function parseMaybeJsonArray(value: unknown): unknown[] | null {
  if (Array.isArray(value)) return value;
  if (typeof value !== "string") return null;
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}
