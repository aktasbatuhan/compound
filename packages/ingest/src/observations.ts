/**
 * Observation → Step mapping.
 *
 * The importer is observations-first: observations are the primary records and
 * trace-level `input`/`output` is never read (Langfuse v4 removes it). See the
 * record mapping table in docs/langfuse-import-mapping.md.
 */
import type { ModelCallStep, OtherStep, Step, ToolExecutionStep } from "@compound/contract";
import { type Collector, DIAGNOSTICS } from "./diagnostics";
import { normalizeGenerationInput, normalizeGenerationOutput } from "./messages";
import { normalizeCost, normalizeUsage } from "./usage";
import { asString, field, isRecord, toIsoUtc } from "./values";

/** Documented observation types and the contract step they become. */
export const OBSERVATION_TYPE_MAP: Record<string, Step["type"]> = {
  GENERATION: "model_call",
  EMBEDDING: "model_call",
  TOOL: "tool_execution",
  SPAN: "other",
  EVENT: "other",
  AGENT: "other",
  CHAIN: "other",
  RETRIEVER: "other",
  EVALUATOR: "other",
  GUARDRAIL: "other",
};

/** A parsed observation record with the fields the mapping needs. */
export interface ObservationView {
  id: string;
  type: string;
  stepType: Step["type"];
  raw: Record<string, unknown>;
  parentId: string | null;
  startedAt: string | null;
  endedAt: string | null;
  /** Position in the source file; the tiebreaker when start times are equal. */
  order: number;
}

export function observationType(raw: Record<string, unknown>): string {
  return asString(raw.type)?.toUpperCase() ?? "";
}

export function viewObservation(
  raw: Record<string, unknown>,
  id: string,
  order: number,
  collector: Collector,
): ObservationView {
  const type = observationType(raw);
  const mapped = OBSERVATION_TYPE_MAP[type];
  if (mapped === undefined) collector.diagnostic(DIAGNOSTICS.unknownObservationType);

  const startRaw = field(raw, "startTime", "start_time");
  const endRaw = field(raw, "endTime", "end_time");
  const startedAt = toIsoUtc(startRaw);
  const endedAt = toIsoUtc(endRaw);
  if (
    (startRaw !== undefined && startedAt === null) ||
    (endRaw !== undefined && endedAt === null)
  ) {
    collector.diagnostic(DIAGNOSTICS.unparseableTimestamp);
  }

  return {
    id,
    type,
    /** Unknown types are preserved as opaque `other` steps, never dropped. */
    stepType: mapped ?? "other",
    raw,
    parentId: asString(field(raw, "parentObservationId", "parent_observation_id")),
    startedAt,
    endedAt,
    order,
  };
}

/** `level: ERROR` (+ `statusMessage`) becomes the step's error text. */
function errorText(raw: Record<string, unknown>): string | null {
  const level = asString(field(raw, "level"))?.toUpperCase();
  if (level !== "ERROR") return null;
  return asString(field(raw, "statusMessage", "status_message")) ?? "ERROR";
}

export function toModelCallStep(view: ObservationView, collector: Collector): ModelCallStep {
  const raw = view.raw;
  const { messages, tools, paramNotes } = normalizeGenerationInput(
    field(raw, "input"),
    collector,
    view.id,
  );
  const rawOutput = field(raw, "output");
  const output = normalizeGenerationOutput(rawOutput, collector, view.id);

  const modelParameters = field(raw, "modelParameters", "model_parameters");
  const baseParams = isRecord(modelParameters) ? { ...modelParameters } : {};
  const params = { ...baseParams, ...paramNotes };

  return {
    type: "model_call",
    step_id: view.id,
    parent_step_id: view.parentId,
    // Langfuse has no provider field; the contract keeps it null rather than
    // inferring one from the model name.
    provider: null,
    model: asString(field(raw, "model", "providedModelName", "provided_model_name")),
    resolved_model: asString(field(raw, "modelId", "model_id")),
    params: Object.keys(params).length > 0 ? (params as ModelCallStep["params"]) : null,
    input: messages,
    tools_available: tools,
    output,
    finish_reason: isRecord(rawOutput)
      ? asString(field(rawOutput, "finish_reason", "finishReason"))
      : null,
    usage: normalizeUsage(
      field(raw, "usageDetails", "usage_details"),
      field(raw, "usage"),
      collector,
    ),
    cost_usd: normalizeCost(field(raw, "costDetails", "cost_details")),
    started_at: view.startedAt,
    ended_at: view.endedAt,
    error: errorText(raw),
  };
}

export function toToolExecutionStep(
  view: ObservationView,
  collector: Collector,
): ToolExecutionStep {
  const raw = view.raw;
  const name = asString(field(raw, "name"));
  if (name === null) collector.diagnostic(DIAGNOSTICS.toolExecutionMissingName);

  return {
    type: "tool_execution",
    step_id: view.id,
    parent_step_id: view.parentId,
    name: name ?? "unknown",
    // Resolved in a second pass, once every model_call step exists.
    call_ref: null,
    input: field(raw, "input") as ToolExecutionStep["input"],
    output: field(raw, "output") as ToolExecutionStep["output"],
    // `null` inherits the task's replay policy; Langfuse carries no such field.
    replay_policy: null,
    started_at: view.startedAt,
    ended_at: view.endedAt,
    error: errorText(raw),
  };
}

/**
 * Opaque span. Per the mapping doc I/O is dropped and metadata kept; the
 * contract's `other` step has no metadata field of its own, so it rides inside
 * `data` alongside the source type, level and status message.
 */
export function toOtherStep(view: ObservationView): OtherStep {
  const raw = view.raw;
  const data: Record<string, unknown> = {
    type: view.type,
    level: asString(field(raw, "level")),
    status_message: asString(field(raw, "statusMessage", "status_message")),
  };
  const metadata = field(raw, "metadata");
  if (metadata !== undefined) data.metadata = metadata;

  return {
    type: "other",
    step_id: view.id,
    parent_step_id: view.parentId,
    name: asString(field(raw, "name")) ?? view.type.toLowerCase() ?? "observation",
    data: data as OtherStep["data"],
  };
}

export function toStep(view: ObservationView, collector: Collector): Step {
  switch (view.stepType) {
    case "model_call":
      return toModelCallStep(view, collector);
    case "tool_execution":
      return toToolExecutionStep(view, collector);
    default:
      return toOtherStep(view);
  }
}

/**
 * The tool_call id a TOOL observation answers.
 *
 * Langfuse has no dedicated column, so the documented carriers are checked in
 * order: observation metadata, then the observation's own input/output objects.
 */
export function toolCallIdOf(raw: Record<string, unknown>): string | null {
  const metadata = field(raw, "metadata");
  if (isRecord(metadata)) {
    const fromMetadata = asString(field(metadata, "tool_call_id", "toolCallId"));
    if (fromMetadata !== null) return fromMetadata;
  }
  for (const key of ["input", "output"]) {
    const value = field(raw, key);
    if (isRecord(value)) {
      const found = asString(field(value, "tool_call_id", "toolCallId"));
      if (found !== null) return found;
    }
  }
  return asString(field(raw, "tool_call_id", "toolCallId"));
}
