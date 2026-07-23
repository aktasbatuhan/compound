/**
 * `redactTrace` — remove sensitive values from a contract trace before it is
 * persisted, and describe what was removed.
 *
 * The walk visits the trace's redactable nodes, addressing each with a concrete
 * path (`steps[2].input[0].content`). At every node the active rule set is the
 * rules whose `applies_to` matches that path, plus the rules that matched an
 * ancestor — matching `steps[*].input` therefore applies the rule to every
 * string beneath it. `field_allowlist` is checked first and wins.
 *
 * Three replacement shapes, per docs/trace-contract-v1.md:
 *
 * - string value: only the matched substrings become the marker; surrounding
 *   text survives, because context is what keeps a redacted trace usable as an
 *   eval case;
 * - free-form JSON value targeted whole (a `tool_execution` input, an `other`
 *   step's `data`, a metadata value): the entire value becomes the marker
 *   string;
 * - a `{type:"text"}` content part whose whole text is redacted becomes a
 *   `{type:"redacted", marker}` part; a partial match stays a text part with
 *   markers inline.
 *
 * The original value is never returned, recorded, thrown or logged.
 */
import type {
  ContentPart,
  Message,
  Outcome,
  Redaction,
  Step,
  ToolCall,
  ToolDef,
  Trace,
} from "@compound/contract";
import type { CompiledRedaction, CompiledRule, RedactionConfig } from "./compile";
import { compileRedactionCached } from "./compile";
import { applyMatches, findMatches } from "./detectors";
import type { PathSegment } from "./paths";
import { formatPath, index, key, matchesPath, matchesPathOrAncestor } from "./paths";

export interface RedactionResult {
  /** The redacted trace. Its `redactions` array carries the markers. */
  trace: Trace;
  /**
   * The records produced by *this* call, in traversal order. `result.trace`
   * carries these appended to any redactions the trace already had (an
   * importer may have masked fields upstream), so the two differ when the
   * input was already partly redacted.
   */
  redactions: Redaction[];
}

/**
 * Structurally identical to zod's `JSONType`, which is what `z.json()` fields
 * (`tool_execution` input/output, `other.data`, metadata values) infer to.
 */
type Json = string | number | boolean | null | Json[] | { [key: string]: Json };

interface Hit {
  readonly rule: string;
  readonly marker: string;
}

interface WalkContext {
  readonly compiled: CompiledRedaction;
  readonly records: Redaction[];
}

const NO_RULES: readonly CompiledRule[] = [];

function isPlainObject(value: unknown): value is Record<string, Json> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isAllowlisted(ctx: WalkContext, path: readonly PathSegment[]): boolean {
  return ctx.compiled.allowlist.some((pattern) => matchesPath(pattern, path));
}

/** Rules active at `path`: config order, inherited rules included. */
function activeAt(
  ctx: WalkContext,
  path: readonly PathSegment[],
  inherited: readonly CompiledRule[],
): readonly CompiledRule[] {
  const active = ctx.compiled.rules.filter(
    (rule) =>
      inherited.includes(rule) || rule.appliesTo.some((pattern) => matchesPath(pattern, path)),
  );
  return active.length === 0 ? NO_RULES : active;
}

/**
 * Whether any allowlist entry points inside this subtree. If one does, a
 * whole-value replacement would swallow it, so the walker descends instead.
 */
function subtreeHasAllowlisted(
  ctx: WalkContext,
  value: Json,
  path: readonly PathSegment[],
): boolean {
  if (ctx.compiled.allowlist.some((pattern) => matchesPathOrAncestor(pattern, path))) return true;
  if (Array.isArray(value)) {
    return value.some((item, i) => subtreeHasAllowlisted(ctx, item, [...path, index(i)]));
  }
  if (isPlainObject(value)) {
    return Object.entries(value).some(([k, item]) =>
      subtreeHasAllowlisted(ctx, item, [...path, key(k)]),
    );
  }
  return false;
}

/**
 * Apply every active rule to one string, in config order. One hit per rule that
 * changed something, however many substrings it matched.
 */
function redactStringValue(
  active: readonly CompiledRule[],
  value: string,
): { value: string; hits: Hit[] } {
  let out = value;
  const hits: Hit[] = [];
  for (const rule of active) {
    const matches = findMatches(rule.patterns, out);
    if (matches.length === 0) continue;
    out = applyMatches(out, matches, rule.marker);
    hits.push({ rule: rule.rule, marker: rule.marker });
  }
  return { value: out, hits };
}

function record(ctx: WalkContext, path: readonly PathSegment[], hit: Hit): void {
  ctx.records.push({ path: formatPath(path), rule: hit.rule, marker: hit.marker });
}

/** A string leaf: substring replacement, one record per rule that fired. */
function redactString(
  ctx: WalkContext,
  value: string,
  path: readonly PathSegment[],
  inherited: readonly CompiledRule[],
): string {
  if (isAllowlisted(ctx, path)) return value;
  const active = activeAt(ctx, path, inherited);
  if (active.length === 0) return value;
  const result = redactStringValue(active, value);
  for (const hit of result.hits) record(ctx, path, hit);
  return result.value;
}

/**
 * A free-form JSON value (`tool_execution` input/output, `other.data`, a
 * metadata value, tool-call arguments, model params). Targeted whole, a
 * non-string value is replaced by the marker; otherwise the walk descends.
 */
function redactFreeJson(
  ctx: WalkContext,
  value: Json,
  path: readonly PathSegment[],
  inherited: readonly CompiledRule[],
): Json {
  if (isAllowlisted(ctx, path)) return value;
  if (typeof value === "string") return redactString(ctx, value, path, inherited);

  const active = activeAt(ctx, path, inherited);
  if (!Array.isArray(value) && !isPlainObject(value)) return value;

  if (active.length > 0 && !subtreeHasAllowlisted(ctx, value, path)) {
    const serialized = JSON.stringify(value) ?? "";
    for (const rule of active) {
      if (findMatches(rule.patterns, serialized).length === 0) continue;
      record(ctx, path, { rule: rule.rule, marker: rule.marker });
      return rule.marker;
    }
  }

  if (Array.isArray(value)) {
    return value.map((item, i) => redactFreeJson(ctx, item, [...path, index(i)], active));
  }
  return redactJsonRecord(ctx, value, path, inherited);
}

/**
 * A contract-typed `Record<string, json>` container (`metadata`, `params`,
 * `tools_available[].parameters`, tool-call `arguments`). The container itself
 * is never replaced — the contract requires an object there — so a rule
 * matching it applies to its values instead.
 */
function redactJsonRecord(
  ctx: WalkContext,
  value: Record<string, Json>,
  path: readonly PathSegment[],
  inherited: readonly CompiledRule[],
): Record<string, Json> {
  if (isAllowlisted(ctx, path)) return value;
  const active = activeAt(ctx, path, inherited);
  const out: Record<string, Json> = {};
  for (const [name, item] of Object.entries(value)) {
    out[name] = redactFreeJson(ctx, item, [...path, key(name)], active);
  }
  return out;
}

function redactContentPart(
  ctx: WalkContext,
  part: ContentPart,
  path: readonly PathSegment[],
  inherited: readonly CompiledRule[],
): ContentPart {
  // `redacted` parts are already markers; `unsupported` carries only a media
  // type. Neither holds trace content.
  if (part.type !== "text") return part;
  if (isAllowlisted(ctx, path)) return part;

  const partActive = activeAt(ctx, path, inherited);
  const textPath = [...path, key("text")];
  if (isAllowlisted(ctx, textPath)) return part;
  const active = activeAt(ctx, textPath, partActive);
  if (active.length === 0) return part;

  const result = redactStringValue(active, part.text);
  if (result.hits.length === 0) return part;

  const whole = result.hits.find((hit) => hit.marker === result.value);
  if (whole !== undefined) {
    record(ctx, path, whole);
    return { type: "redacted", marker: whole.marker };
  }
  for (const hit of result.hits) record(ctx, textPath, hit);
  return { type: "text", text: result.value };
}

function redactToolCall(
  ctx: WalkContext,
  call: ToolCall,
  path: readonly PathSegment[],
  inherited: readonly CompiledRule[],
): ToolCall {
  if (isAllowlisted(ctx, path)) return call;
  const active = activeAt(ctx, path, inherited);
  return {
    ...call,
    // `id` is linkage (`tool_call_id`, `call_ref`) and is never redacted.
    name: redactString(ctx, call.name, [...path, key("name")], active),
    arguments: redactJsonRecord(ctx, call.arguments, [...path, key("arguments")], active),
  };
}

function redactMessage(
  ctx: WalkContext,
  message: Message,
  path: readonly PathSegment[],
  inherited: readonly CompiledRule[],
): Message {
  if (isAllowlisted(ctx, path)) return message;
  const active = activeAt(ctx, path, inherited);
  const out: Message = { ...message };

  const contentPath = [...path, key("content")];
  if (typeof message.content === "string") {
    out.content = redactString(ctx, message.content, contentPath, active);
  } else if (Array.isArray(message.content) && !isAllowlisted(ctx, contentPath)) {
    const contentActive = activeAt(ctx, contentPath, active);
    out.content = message.content.map((part, i) =>
      redactContentPart(ctx, part, [...contentPath, index(i)], contentActive),
    );
  }

  if (message.tool_calls != null) {
    const callsPath = [...path, key("tool_calls")];
    if (!isAllowlisted(ctx, callsPath)) {
      const callsActive = activeAt(ctx, callsPath, active);
      out.tool_calls = message.tool_calls.map((call, i) =>
        redactToolCall(ctx, call, [...callsPath, index(i)], callsActive),
      );
    }
  }

  return out;
}

function redactToolDef(
  ctx: WalkContext,
  def: ToolDef,
  path: readonly PathSegment[],
  inherited: readonly CompiledRule[],
): ToolDef {
  if (isAllowlisted(ctx, path)) return def;
  const active = activeAt(ctx, path, inherited);
  const out: ToolDef = { ...def };
  out.name = redactString(ctx, def.name, [...path, key("name")], active);
  if (typeof def.description === "string") {
    out.description = redactString(ctx, def.description, [...path, key("description")], active);
  }
  if (def.parameters != null) {
    out.parameters = redactJsonRecord(ctx, def.parameters, [...path, key("parameters")], active);
  }
  return out;
}

function redactStep(
  ctx: WalkContext,
  step: Step,
  path: readonly PathSegment[],
  inherited: readonly CompiledRule[],
): Step {
  if (isAllowlisted(ctx, path)) return step;
  const active = activeAt(ctx, path, inherited);
  const string = (value: string, name: string): string =>
    redactString(ctx, value, [...path, key(name)], active);

  if (step.type === "model_call") {
    const out = { ...step };
    if (typeof step.provider === "string") out.provider = string(step.provider, "provider");
    if (typeof step.model === "string") out.model = string(step.model, "model");
    if (typeof step.resolved_model === "string") {
      out.resolved_model = string(step.resolved_model, "resolved_model");
    }
    if (typeof step.finish_reason === "string") {
      out.finish_reason = string(step.finish_reason, "finish_reason");
    }
    if (typeof step.error === "string") out.error = string(step.error, "error");
    if (step.params != null) {
      out.params = redactJsonRecord(ctx, step.params, [...path, key("params")], active);
    }

    const inputPath = [...path, key("input")];
    if (!isAllowlisted(ctx, inputPath)) {
      const inputActive = activeAt(ctx, inputPath, active);
      out.input = step.input.map((message, i) =>
        redactMessage(ctx, message, [...inputPath, index(i)], inputActive),
      );
    }

    if (step.output != null) {
      out.output = redactMessage(ctx, step.output, [...path, key("output")], active);
    }

    if (step.tools_available != null) {
      const toolsPath = [...path, key("tools_available")];
      if (!isAllowlisted(ctx, toolsPath)) {
        const toolsActive = activeAt(ctx, toolsPath, active);
        out.tools_available = step.tools_available.map((def, i) =>
          redactToolDef(ctx, def, [...toolsPath, index(i)], toolsActive),
        );
      }
    }
    return out;
  }

  if (step.type === "tool_execution") {
    const out = { ...step };
    out.name = string(step.name, "name");
    if (typeof step.error === "string") out.error = string(step.error, "error");
    if (step.input !== undefined) {
      out.input = redactFreeJson(ctx, step.input, [...path, key("input")], active);
    }
    if (step.output !== undefined) {
      out.output = redactFreeJson(ctx, step.output, [...path, key("output")], active);
    }
    return out;
  }

  const out = { ...step };
  out.name = string(step.name, "name");
  if (step.data !== undefined) {
    out.data = redactFreeJson(ctx, step.data, [...path, key("data")], active);
  }
  return out;
}

function redactOutcome(
  ctx: WalkContext,
  outcome: Outcome,
  path: readonly PathSegment[],
  inherited: readonly CompiledRule[],
): Outcome {
  if (isAllowlisted(ctx, path)) return outcome;
  const active = activeAt(ctx, path, inherited);
  const out: Outcome = { ...outcome };

  if (outcome.feedback != null) {
    const feedbackPath = [...path, key("feedback")];
    const feedbackActive = activeAt(ctx, feedbackPath, active);
    out.feedback = outcome.feedback.map((entry, i) => {
      const entryPath = [...feedbackPath, index(i)];
      if (isAllowlisted(ctx, entryPath)) return entry;
      const entryActive = activeAt(ctx, entryPath, feedbackActive);
      return {
        ...entry,
        value: redactFreeJson(ctx, entry.value, [...entryPath, key("value")], entryActive),
      };
    });
  }

  if (outcome.scores != null) {
    const scoresPath = [...path, key("scores")];
    const scoresActive = activeAt(ctx, scoresPath, active);
    out.scores = outcome.scores.map((score, i) => {
      const scorePath = [...scoresPath, index(i)];
      if (isAllowlisted(ctx, scorePath)) return score;
      const scoreActive = activeAt(ctx, scorePath, scoresActive);
      return {
        ...score,
        name: redactString(ctx, score.name, [...scorePath, key("name")], scoreActive),
      };
    });
  }

  if (outcome.deterministic != null && outcome.deterministic.detail !== undefined) {
    const detPath = [...path, key("deterministic")];
    if (!isAllowlisted(ctx, detPath)) {
      const detActive = activeAt(ctx, detPath, active);
      out.deterministic = {
        ...outcome.deterministic,
        detail: redactFreeJson(
          ctx,
          outcome.deterministic.detail,
          [...detPath, key("detail")],
          detActive,
        ),
      };
    }
  }

  return out;
}

/**
 * Redact `trace` according to `config`.
 *
 * With `config` `undefined` — or a config with no rules — the trace comes back
 * unchanged and `redactions` is empty: redaction is opt-in, and doing nothing
 * is an explicit outcome rather than an accident. The input is never mutated.
 *
 * Throws `RedactionConfigError` if the config cannot be compiled (an invalid
 * `pattern` or `applies_to` path). Compile once with `compileRedaction` if you
 * want that failure before the first trace.
 */
export function redactTrace(trace: Trace, config: RedactionConfig | undefined): RedactionResult {
  const compiled = compileRedactionCached(config);
  const clone = structuredClone(trace);
  if (compiled === undefined) return { trace: clone, redactions: [] };

  const ctx: WalkContext = { compiled, records: [] };
  const root: PathSegment[] = [];

  const out: Trace = clone;
  const active = activeAt(ctx, root, NO_RULES);

  const string = (value: string, name: string): string =>
    redactString(ctx, value, [key(name)], active);

  if (typeof clone.session_id === "string") out.session_id = string(clone.session_id, "session_id");
  if (typeof clone.user_ref === "string") out.user_ref = string(clone.user_ref, "user_ref");
  if (typeof clone.environment === "string") {
    out.environment = string(clone.environment, "environment");
  }
  if (typeof clone.release === "string") out.release = string(clone.release, "release");

  if (clone.metadata != null) {
    out.metadata = redactJsonRecord(ctx, clone.metadata, [key("metadata")], active);
  }

  if (clone.tags != null) {
    const tagsPath = [key("tags")];
    if (!isAllowlisted(ctx, tagsPath)) {
      const tagsActive = activeAt(ctx, tagsPath, active);
      out.tags = clone.tags.map((tag, i) =>
        redactString(ctx, tag, [...tagsPath, index(i)], tagsActive),
      );
    }
  }

  const stepsPath = [key("steps")];
  if (!isAllowlisted(ctx, stepsPath)) {
    const stepsActive = activeAt(ctx, stepsPath, active);
    out.steps = clone.steps.map((step, i) =>
      redactStep(ctx, step, [...stepsPath, index(i)], stepsActive),
    );
  }

  if (clone.outcome != null) {
    out.outcome = redactOutcome(ctx, clone.outcome, [key("outcome")], active);
  }

  out.redactions = [...clone.redactions, ...ctx.records];
  return { trace: out, redactions: ctx.records };
}
