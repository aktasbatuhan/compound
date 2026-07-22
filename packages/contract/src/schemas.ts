/**
 * Zod schemas for the Compound portable trace contract v1.
 *
 * Source of truth: docs/trace-contract-v1.md (repo root). Field names, types,
 * and semantics follow that document exactly; anything an importer cannot
 * determine is `null`/absent, never guessed.
 */
import { z } from "zod";

/** ISO-8601 UTC timestamp (must be Z-terminated; offsets are not accepted). */
export const IsoUtcDateTimeSchema = z.iso.datetime();

// ---------------------------------------------------------------------------
// Content parts and messages
// ---------------------------------------------------------------------------

export const TextContentPartSchema = z.strictObject({
  type: z.literal("text"),
  text: z.string(),
});

export const RedactedContentPartSchema = z.strictObject({
  type: z.literal("redacted"),
  marker: z.string(),
});

/**
 * Images/audio are out of scope for v1 replay; importers record them as
 * `unsupported`, which forces the trace to `diagnostic`.
 */
export const UnsupportedContentPartSchema = z.strictObject({
  type: z.literal("unsupported"),
  media_type: z.string().nullish(),
});

export const ContentPartSchema = z.discriminatedUnion("type", [
  TextContentPartSchema,
  RedactedContentPartSchema,
  UnsupportedContentPartSchema,
]);

/**
 * A tool call emitted by an assistant message. `arguments` is always a parsed
 * JSON object in the contract; importers normalize provider dialects and an
 * unparseable arguments string forces `diagnostic`, never a stored guess.
 */
export const ToolCallSchema = z.strictObject({
  id: z.string(),
  name: z.string(),
  arguments: z.record(z.string(), z.json()),
});

/** OpenAI-compatible normalized message form. */
export const MessageSchema = z.strictObject({
  role: z.enum(["system", "user", "assistant", "tool"]),
  content: z.union([z.string(), z.array(ContentPartSchema), z.null()]),
  tool_calls: z.array(ToolCallSchema).nullish(),
  tool_call_id: z.string().nullish(),
});

/**
 * JSON-schema tool definition available to a model call (OpenAI function
 * schema passthrough).
 */
export const ToolDefSchema = z.strictObject({
  name: z.string(),
  description: z.string().nullish(),
  parameters: z.record(z.string(), z.json()).nullish(),
});

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

/**
 * Token usage for one model call.
 *
 * Semantics (per contract doc): `input_tokens` is the TOTAL prompt token count
 * including cached buckets; `cached_input_tokens` is an informational subset of
 * it; `reasoning_tokens` is a subset of `output_tokens`. Importers converting
 * from mutually-exclusive bucket schemes (e.g. Langfuse `usageDetails`) must
 * sum buckets accordingly.
 */
export const UsageSchema = z.strictObject({
  input_tokens: z.int().nonnegative(),
  output_tokens: z.int().nonnegative(),
  reasoning_tokens: z.int().nonnegative().nullish(),
  cached_input_tokens: z.int().nonnegative().nullish(),
  total_tokens: z.int().nonnegative().nullish(),
});

// ---------------------------------------------------------------------------
// Steps (discriminated union on `type`)
// ---------------------------------------------------------------------------

/** One LLM request/response. */
export const ModelCallStepSchema = z.strictObject({
  type: z.literal("model_call"),
  step_id: z.string(),
  parent_step_id: z.string().nullish(),
  provider: z.string().nullish(),
  model: z.string().nullish(),
  resolved_model: z.string().nullish(),
  params: z.record(z.string(), z.json()).nullish(),
  input: z.array(MessageSchema),
  tools_available: z.array(ToolDefSchema).nullish(),
  output: MessageSchema.nullish(),
  finish_reason: z.string().nullish(),
  usage: UsageSchema.nullish(),
  cost_usd: z.number().nullish(),
  started_at: IsoUtcDateTimeSchema.nullish(),
  ended_at: IsoUtcDateTimeSchema.nullish(),
  error: z.string().nullish(),
});

/** Which model tool_call a tool_execution step answers. */
export const CallRefSchema = z.strictObject({
  step_id: z.string(),
  tool_call_id: z.string(),
});

export const ReplayPolicySchema = z.enum(["recorded", "mocked", "live_read_only", "blocked"]);

/** The application running a tool. */
export const ToolExecutionStepSchema = z.strictObject({
  type: z.literal("tool_execution"),
  step_id: z.string(),
  parent_step_id: z.string().nullish(),
  name: z.string(),
  call_ref: CallRefSchema.nullish(),
  input: z.json().optional(),
  output: z.json().optional(),
  /** `null` inherits task config. */
  replay_policy: ReplayPolicySchema.nullish(),
  started_at: IsoUtcDateTimeSchema.nullish(),
  ended_at: IsoUtcDateTimeSchema.nullish(),
  error: z.string().nullish(),
});

/**
 * Preserved-but-opaque span (retrieval, custom event). Never used by
 * evaluation; kept for lineage.
 */
export const OtherStepSchema = z.strictObject({
  type: z.literal("other"),
  step_id: z.string(),
  parent_step_id: z.string().nullish(),
  name: z.string(),
  data: z.json().optional(),
});

export const StepSchema = z.discriminatedUnion("type", [
  ModelCallStepSchema,
  ToolExecutionStepSchema,
  OtherStepSchema,
]);

// ---------------------------------------------------------------------------
// Outcome
// ---------------------------------------------------------------------------

export const FeedbackSchema = z.strictObject({
  kind: z.enum(["thumbs", "rating", "correction", "comment"]),
  value: z.json(),
  at: IsoUtcDateTimeSchema.nullish(),
});

export const ScoreSchema = z.strictObject({
  name: z.string(),
  value: z.number(),
  source: z.enum(["human", "judge", "deterministic"]),
  at: IsoUtcDateTimeSchema.nullish(),
});

export const DeterministicOutcomeSchema = z.strictObject({
  status: z.enum(["success", "failure"]),
  detail: z.json().optional(),
});

/** Evidence, not ground truth: seeds case provenance. */
export const OutcomeSchema = z.strictObject({
  feedback: z.array(FeedbackSchema).nullish(),
  scores: z.array(ScoreSchema).nullish(),
  deterministic: DeterministicOutcomeSchema.nullish(),
});

// ---------------------------------------------------------------------------
// Permissions and redaction
// ---------------------------------------------------------------------------

/**
 * Usage permissions carried by every trace. The contract default is
 * `{ judging: true, optimization: true, fine_tuning: false }`; applying that
 * default is import-batch policy, so the schema requires explicit values.
 */
export const PermissionsSchema = z.strictObject({
  judging: z.boolean(),
  optimization: z.boolean(),
  fine_tuning: z.boolean(),
});

/** Marker for a field the importer masked. The original value is never persisted. */
export const RedactionSchema = z.strictObject({
  path: z.string(),
  rule: z.union([z.literal("secret"), z.literal("pii"), z.string().regex(/^custom:.+$/)]),
  marker: z.string(),
});

// ---------------------------------------------------------------------------
// Source and trace envelope
// ---------------------------------------------------------------------------

export const SourceSchema = z.strictObject({
  importer: z.string(),
  importer_version: z.string(),
  source_ids: z.record(z.string(), z.string()),
  exported_at: IsoUtcDateTimeSchema.nullish(),
});

/**
 * One trace record, including the envelope
 * (`schema: "compound.trace"`, `schema_version: 1`).
 */
export const TraceSchema = z.strictObject({
  schema: z.literal("compound.trace"),
  schema_version: z.literal(1),
  trace_id: z.string().min(1),
  /** Application-supplied; `null` routes to the "unassigned" bucket. */
  task_key: z.string().nullable(),
  started_at: IsoUtcDateTimeSchema,
  ended_at: IsoUtcDateTimeSchema.nullish(),
  source: SourceSchema,
  session_id: z.string().nullish(),
  /** Opaque handle; raw user IDs are redacted to it. */
  user_ref: z.string().nullish(),
  environment: z.string().nullish(),
  release: z.string().nullish(),
  /** Post-redaction application metadata. */
  metadata: z.record(z.string(), z.json()).nullish(),
  tags: z.array(z.string()).nullish(),
  /** Ordered; at least one `model_call` for `eval_ready`. */
  steps: z.array(StepSchema),
  /** The candidate-replaceable model call. `null` forces `diagnostic`. */
  focal_step_id: z.string().nullable(),
  outcome: OutcomeSchema.nullish(),
  permissions: PermissionsSchema,
  /** Markers for every field the importer masked (may be empty). */
  redactions: z.array(RedactionSchema),
});

// ---------------------------------------------------------------------------
// Inferred TypeScript types
// ---------------------------------------------------------------------------

export type IsoUtcDateTime = z.infer<typeof IsoUtcDateTimeSchema>;
export type TextContentPart = z.infer<typeof TextContentPartSchema>;
export type RedactedContentPart = z.infer<typeof RedactedContentPartSchema>;
export type UnsupportedContentPart = z.infer<typeof UnsupportedContentPartSchema>;
export type ContentPart = z.infer<typeof ContentPartSchema>;
export type ToolCall = z.infer<typeof ToolCallSchema>;
export type Message = z.infer<typeof MessageSchema>;
export type ToolDef = z.infer<typeof ToolDefSchema>;
export type Usage = z.infer<typeof UsageSchema>;
export type ModelCallStep = z.infer<typeof ModelCallStepSchema>;
export type CallRef = z.infer<typeof CallRefSchema>;
export type ReplayPolicy = z.infer<typeof ReplayPolicySchema>;
export type ToolExecutionStep = z.infer<typeof ToolExecutionStepSchema>;
export type OtherStep = z.infer<typeof OtherStepSchema>;
export type Step = z.infer<typeof StepSchema>;
export type Feedback = z.infer<typeof FeedbackSchema>;
export type Score = z.infer<typeof ScoreSchema>;
export type DeterministicOutcome = z.infer<typeof DeterministicOutcomeSchema>;
export type Outcome = z.infer<typeof OutcomeSchema>;
export type Permissions = z.infer<typeof PermissionsSchema>;
export type Redaction = z.infer<typeof RedactionSchema>;
export type Source = z.infer<typeof SourceSchema>;
export type Trace = z.infer<typeof TraceSchema>;
