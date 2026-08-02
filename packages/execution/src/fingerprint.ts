/**
 * Completion fingerprinting and cost estimation.
 *
 * The fingerprint is what makes the cache correct and the ledger honest: two
 * requests that would produce the same paid call share a fingerprint, so the
 * second is served from cache at $0 (docs/execution-v1.md).
 */
import { createHash } from "node:crypto";
import type { CompletionRequest, CompletionUsage } from "./provider";

/** Deterministic JSON (keys sorted recursively); array order preserved. */
export function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value) ?? "null";
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(",")}]`;
  }
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([, child]) => child !== undefined)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([key, child]) => `${JSON.stringify(key)}:${canonicalJson(child)}`);
  return `{${entries.join(",")}}`;
}

export interface FingerprintInput {
  provider: string;
  request: CompletionRequest;
  /** Provider revision/version, so a provider change invalidates the cache. */
  providerRevision?: string;
  /**
   * A non-native transport (issue #8), present only when the run was forced off
   * the provider's default route (e.g. a flex host over plain chat). Undefined
   * for a native-transport run, which keeps the pre-existing identity so warm
   * caches still hit.
   */
  transportOverride?: string;
  /**
   * Agentic trajectory identity (issue #23), present only for a multi-turn run.
   * The whole trajectory is cached under its INITIAL request, so the replay
   * policy and scripted tool results — which steer the trajectory to a different
   * outcome — must join the fingerprint. Absent for a single-call run, whose
   * identity (and warm cache) is therefore unchanged.
   */
  agentic?: unknown;
}

/**
 * Fingerprint over {provider, model, params, input, provider_revision}.
 *
 * A `trial` discriminator lets repeated stochastic trials of the same request
 * be distinct paid calls, mirroring the benchmark engine's trial handling.
 */
export function completionFingerprint(input: FingerprintInput, trial = 0): string {
  const subject = {
    provider: input.provider,
    provider_revision: input.providerRevision ?? null,
    model: input.request.model,
    messages: input.request.messages,
    tools: input.request.tools ?? null,
    params: input.request.params ?? null,
    // Present only for a forced non-native transport, so a native run's identity
    // (and its warm cache) is unchanged; a chat-vs-flex override never collides.
    ...(input.transportOverride !== undefined ? { transport: input.transportOverride } : {}),
    // Present only for an agentic run, so a single-call fingerprint is unchanged.
    ...(input.agentic !== undefined ? { agentic: input.agentic } : {}),
    // Trial 0 keeps the base identity so it reuses any legacy cache entry.
    ...(trial > 0 ? { trial } : {}),
  };
  return createHash("sha256").update(canonicalJson(subject)).digest("hex");
}

export interface TokenPrice {
  /** USD per million input tokens. */
  input: number;
  /** USD per million output tokens. */
  output: number;
}

/**
 * Cost from usage and a price table. Reasoning tokens are billed at the output
 * rate (they are generated tokens), matching the benchmark engine.
 */
export function costFromUsage(usage: CompletionUsage | null, price: TokenPrice): number {
  if (usage === null) return 0;
  const inputCost = (usage.input_tokens / 1_000_000) * price.input;
  const outputCost = (usage.output_tokens / 1_000_000) * price.output;
  return inputCost + outputCost;
}

/**
 * A conservative pre-call cost estimate: assume the full output-token budget is
 * consumed, and count BOTH the prompt messages and the tool schemas sent with
 * the request. Used to reserve budget headroom BEFORE a call. It is a rough
 * (~4 chars/token) projection, not an exact ceiling — the true billed prompt can
 * still exceed it — but it must never under-count a component of the request, so
 * the tool schemas (often larger than the user message on an agentic call) are
 * included rather than silently dropped (#4).
 */
export function estimateCost(
  request: CompletionRequest,
  price: TokenPrice,
  defaultMaxTokens = 4096,
): number {
  const promptTokens = estimatePromptTokens(request.messages) + estimateToolTokens(request.tools);
  const maxOutput =
    typeof request.params?.max_tokens === "number"
      ? (request.params.max_tokens as number)
      : defaultMaxTokens;
  return (promptTokens / 1_000_000) * price.input + (maxOutput / 1_000_000) * price.output;
}

/** Rough token estimate (~4 chars/token) for pre-call budgeting only. */
export function estimatePromptTokens(messages: CompletionRequest["messages"]): number {
  let chars = 0;
  for (const message of messages) {
    if (typeof message.content === "string") chars += message.content.length;
    else if (Array.isArray(message.content)) {
      for (const part of message.content) if (part.type === "text") chars += part.text.length;
    }
    for (const call of message.tool_calls ?? []) {
      chars += call.name.length + JSON.stringify(call.arguments).length;
    }
  }
  return Math.ceil(chars / 4) + 8 * messages.length;
}

/**
 * Rough token estimate for the tool schemas on a request (#4). Tool JSON Schemas
 * are billed as prompt tokens and can dwarf a short user message, so omitting
 * them from the pre-call estimate lets the first paid agentic call cross the cap
 * despite passing the headroom check. Serialized length over ~4 chars/token.
 */
export function estimateToolTokens(tools: CompletionRequest["tools"]): number {
  if (tools === undefined || tools.length === 0) return 0;
  let chars = 0;
  for (const tool of tools) chars += JSON.stringify(tool).length;
  return Math.ceil(chars / 4);
}

/**
 * The cost to LEDGER for a completed paid call. Real cost when the provider
 * reported usage; the conservative pre-call estimate when it returned `usage:
 * null` (#3). A real call whose token counts we cannot read must never ledger
 * $0 — that silently leaks the cap and the global limit — so we fall back to the
 * same estimate the pre-call check reserved. `usageKnown` lets the caller flag a
 * completion whose cost is estimated rather than measured.
 */
export function chargeableCost(
  usage: CompletionUsage | null,
  request: CompletionRequest,
  price: TokenPrice,
): { costUsd: number; usageKnown: boolean } {
  if (usage !== null) return { costUsd: costFromUsage(usage, price), usageKnown: true };
  return { costUsd: estimateCost(request, price), usageKnown: false };
}
