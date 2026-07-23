/**
 * Generation I/O normalization: Langfuse's schemaless `input`/`output` into the
 * contract's `Message[]` / `Message`.
 *
 * Implements "Generation input normalization" and "Tool-call dialect
 * normalization" in docs/langfuse-import-mapping.md.
 */
import type { ContentPart, Message, ToolCall, ToolDef } from "@compound/contract";
import { type Collector, DIAGNOSTICS, DIALECTS } from "./diagnostics";
import { asString, isRecord } from "./values";

type Role = Message["role"];

/**
 * Role aliases. LangChain's callback handler already normalizes Human/AI/System,
 * but raw dumps carry the original names, so both are accepted. `developer` is
 * OpenAI's own system alias. `function` is legacy and handled separately.
 */
const ROLE_ALIASES: Record<string, Role> = {
  system: "system",
  developer: "system",
  user: "user",
  human: "user",
  assistant: "assistant",
  ai: "assistant",
  tool: "tool",
  toolmessage: "tool",
  humanmessage: "user",
  aimessage: "assistant",
  systemmessage: "system",
};

export interface NormalizedGenerationInput {
  messages: Message[];
  tools: ToolDef[] | null;
  /** Extra `params` entries the wrapper implied (legacy function calling). */
  paramNotes: Record<string, unknown>;
}

/**
 * Normalize a generation's `input`.
 *
 * Accepted shapes, per the mapping doc: a message array; `{messages, tools}`
 * (OpenAI wrapper); `{messages, functions, function_call}` (legacy). Anything
 * else yields `messages: []` and the `unparseable_generation_input` diagnostic —
 * the importer never guesses a prompt.
 *
 * An absent input is *not* unparseable: it yields `[]` with no diagnostic, and
 * the contract validator flags the resulting focal step on its own.
 */
export function normalizeGenerationInput(
  raw: unknown,
  collector: Collector,
  stepId: string,
): NormalizedGenerationInput {
  const empty: NormalizedGenerationInput = { messages: [], tools: null, paramNotes: {} };
  if (raw === undefined || raw === null) return empty;

  if (Array.isArray(raw)) {
    collector.dialect(DIALECTS.messagesArray);
    return { messages: normalizeMessages(raw, collector, stepId), tools: null, paramNotes: {} };
  }

  if (isRecord(raw) && Array.isArray(raw.messages)) {
    const messages = normalizeMessages(raw.messages, collector, stepId);
    const paramNotes: Record<string, unknown> = {};
    let tools: ToolDef[] | null = null;

    if (Array.isArray(raw.tools)) {
      collector.dialect(DIALECTS.openaiToolsWrapper);
      tools = normalizeToolDefs(raw.tools);
    } else if (Array.isArray(raw.functions)) {
      collector.dialect(DIALECTS.legacyFunctions);
      collector.diagnostic(DIAGNOSTICS.legacyFunctions);
      tools = normalizeToolDefs(raw.functions);
      paramNotes._legacy_functions = true;
      if (raw.function_call !== undefined) {
        paramNotes._legacy_function_call = raw.function_call;
      }
    } else {
      collector.dialect(DIALECTS.messagesArray);
    }
    return { messages, tools, paramNotes };
  }

  collector.diagnostic(DIAGNOSTICS.unparseableGenerationInput);
  return empty;
}

/**
 * Normalize a generation's `output` into the assistant message.
 *
 * Accepts a bare string, a message-shaped object, or a single-element array of
 * one (some SDKs wrap the completion). Anything else returns `null` with the
 * `unparseable_generation_output` diagnostic rather than a stringified guess.
 */
export function normalizeGenerationOutput(
  raw: unknown,
  collector: Collector,
  stepId: string,
): Message | null {
  if (raw === undefined || raw === null) return null;

  if (typeof raw === "string") {
    return { role: "assistant", content: raw, tool_calls: null, tool_call_id: null };
  }

  if (Array.isArray(raw)) {
    if (raw.length === 1) return normalizeGenerationOutput(raw[0], collector, stepId);
    collector.diagnostic(DIAGNOSTICS.unparseableGenerationOutput);
    return null;
  }

  if (isRecord(raw) && isMessageShaped(raw)) {
    return normalizeMessage(raw, collector, stepId, 0, "assistant");
  }

  collector.diagnostic(DIAGNOSTICS.unparseableGenerationOutput);
  return null;
}

function isMessageShaped(record: Record<string, unknown>): boolean {
  return (
    "role" in record ||
    "content" in record ||
    "tool_calls" in record ||
    "additional_kwargs" in record
  );
}

export function normalizeMessages(raw: unknown[], collector: Collector, stepId: string): Message[] {
  const messages: Message[] = [];
  raw.forEach((entry, index) => {
    const message = normalizeMessage(entry, collector, stepId, index);
    if (message !== null) messages.push(message);
  });
  return messages;
}

function normalizeMessage(
  raw: unknown,
  collector: Collector,
  stepId: string,
  index: number,
  defaultRole?: Role,
): Message | null {
  if (typeof raw === "string") {
    if (defaultRole === undefined) {
      collector.diagnostic(DIAGNOSTICS.unknownMessageRole);
      return null;
    }
    return { role: defaultRole, content: raw, tool_calls: null, tool_call_id: null };
  }
  if (!isRecord(raw)) {
    collector.diagnostic(DIAGNOSTICS.unknownMessageRole);
    return null;
  }

  const rawRole = asString(raw.role) ?? asString(raw.type);
  let role: Role | undefined = defaultRole;
  let legacyFunctionRole = false;

  if (rawRole !== null) {
    const key = rawRole.toLowerCase();
    if (key === "function") {
      role = "tool";
      legacyFunctionRole = true;
      collector.diagnostic(DIAGNOSTICS.legacyFunctionRole);
    } else {
      role = ROLE_ALIASES[key];
    }
  }

  if (role === undefined) {
    collector.diagnostic(DIAGNOSTICS.unknownMessageRole);
    return null;
  }

  const toolCalls = extractToolCalls(raw, collector, stepId, index);
  const toolCallId = legacyFunctionRole ? null : asString(raw.tool_call_id);

  return {
    role,
    content: normalizeContent(raw.content, collector),
    tool_calls: toolCalls,
    tool_call_id: toolCallId,
  };
}

/**
 * Content into `string | ContentPart[] | null`.
 *
 * Non-text parts (images, audio) become `unsupported` parts, which the contract
 * validator turns into a diagnostic — v1 replay does not handle them.
 */
function normalizeContent(raw: unknown, collector: Collector): string | ContentPart[] | null {
  if (raw === undefined || raw === null) return null;
  if (typeof raw === "string") return raw;
  if (typeof raw === "number" || typeof raw === "boolean") return String(raw);

  if (Array.isArray(raw)) {
    const parts: ContentPart[] = [];
    for (const entry of raw) {
      if (typeof entry === "string") {
        parts.push({ type: "text", text: entry });
        continue;
      }
      if (isRecord(entry) && entry.type === "text" && typeof entry.text === "string") {
        parts.push({ type: "text", text: entry.text });
        continue;
      }
      collector.diagnostic(DIAGNOSTICS.unsupportedContentPart);
      parts.push({
        type: "unsupported",
        media_type: isRecord(entry) ? asString(entry.type) : null,
      });
    }
    return parts;
  }

  collector.diagnostic(DIAGNOSTICS.unsupportedContentPart);
  return [{ type: "unsupported", media_type: "application/json" }];
}

/**
 * Tool calls in either dialect.
 *
 * The typed `tool_calls` field wins over the verbatim
 * `additional_kwargs.tool_calls` blob whenever both are present, per the
 * mapping doc — LangChain emits both and only the typed one is authoritative.
 */
function extractToolCalls(
  message: Record<string, unknown>,
  collector: Collector,
  stepId: string,
  messageIndex: number,
): ToolCall[] | null {
  const typed = message.tool_calls;
  if (Array.isArray(typed) && typed.length > 0) {
    return normalizeToolCallArray(typed, collector, stepId, messageIndex);
  }

  const extra = message.additional_kwargs;
  if (isRecord(extra) && Array.isArray(extra.tool_calls) && extra.tool_calls.length > 0) {
    collector.dialect(DIALECTS.additionalKwargsToolCalls);
    return normalizeToolCallArray(extra.tool_calls, collector, stepId, messageIndex);
  }

  return null;
}

function normalizeToolCallArray(
  raw: unknown[],
  collector: Collector,
  stepId: string,
  messageIndex: number,
): ToolCall[] {
  const calls: ToolCall[] = [];
  raw.forEach((entry, index) => {
    const call = normalizeToolCall(entry, collector, `${stepId}:m${messageIndex}:tc${index}`);
    if (call !== null) calls.push(call);
  });
  return calls;
}

function normalizeToolCall(
  raw: unknown,
  collector: Collector,
  syntheticId: string,
): ToolCall | null {
  if (!isRecord(raw)) return null;

  const fn = isRecord(raw.function) ? raw.function : null;
  if (fn !== null) collector.dialect(DIALECTS.openaiToolCalls);
  else if (isRecord(raw.args)) collector.dialect(DIALECTS.langchainToolCalls);

  let id = asString(raw.id);
  if (id === null) {
    collector.diagnostic(DIAGNOSTICS.toolCallMissingId);
    id = syntheticId;
  }

  let name = asString(fn?.name) ?? asString(raw.name);
  if (name === null) {
    collector.diagnostic(DIAGNOSTICS.toolCallMissingName);
    name = "unknown";
  }

  const rawArguments = fn !== null ? fn.arguments : raw.args;
  return {
    id,
    name,
    arguments: normalizeToolArguments(rawArguments, collector) as ToolCall["arguments"],
  };
}

/**
 * OpenAI serializes arguments as a JSON string; LangChain emits a dict. A string
 * that does not parse to an object is kept verbatim under `_raw` and flagged,
 * so the trace becomes diagnostic instead of carrying a guessed argument set.
 */
function normalizeToolArguments(raw: unknown, collector: Collector): Record<string, unknown> {
  if (isRecord(raw)) return { ...raw };
  if (raw === undefined || raw === null) return {};
  if (typeof raw !== "string") {
    collector.diagnostic(DIAGNOSTICS.toolArgumentsUnparseable);
    return { _raw: raw };
  }
  try {
    const parsed: unknown = JSON.parse(raw);
    if (isRecord(parsed)) return parsed;
  } catch {
    // fall through to the raw form below
  }
  collector.diagnostic(DIAGNOSTICS.toolArgumentsUnparseable);
  return { _raw: raw };
}

/**
 * OpenAI function-schema passthrough. Accepts both the wrapped
 * `{type: "function", function: {…}}` tools form and the legacy bare
 * `{name, description, parameters}` functions form.
 */
export function normalizeToolDefs(raw: unknown[]): ToolDef[] | null {
  const defs: ToolDef[] = [];
  for (const entry of raw) {
    if (!isRecord(entry)) continue;
    const spec = isRecord(entry.function) ? entry.function : entry;
    const name = asString(spec.name);
    if (name === null) continue;
    defs.push({
      name,
      description: asString(spec.description),
      parameters: isRecord(spec.parameters) ? (spec.parameters as ToolDef["parameters"]) : null,
    });
  }
  return defs.length > 0 ? defs : null;
}
