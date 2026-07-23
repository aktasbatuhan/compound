/**
 * Usage and cost mapping.
 *
 * Langfuse `usageDetails` buckets are **mutually exclusive** (`input` excludes
 * tokens counted under `input_*`); the contract's `input_tokens` is the
 * **inclusive** total prompt count. This module does the summing, per
 * "Usage and cost" in docs/langfuse-import-mapping.md.
 */
import type { Usage } from "@compound/contract";
import { type Collector, DIALECTS } from "./diagnostics";
import { asFiniteNumber, isRecord } from "./values";

/**
 * Prompt-side keys that are not prefixed `input_`.
 *
 * Anthropic reports `cache_read_input_tokens` and `cache_creation_input_tokens`.
 * Both are prompt buckets: the mapping doc requires `cache_creation_input_tokens`
 * to count toward `input_tokens`, and `cached_input_tokens` must remain a subset
 * of `input_tokens`, so `cache_read_input_tokens` must count toward it too.
 */
const PROMPT_SUFFIX = "_input_tokens";

/** Keys whose sum is the contract's `cached_input_tokens` (informational subset). */
const CACHED_INPUT_KEYS = ["input_cached_tokens", "cache_read_input_tokens"];

const REASONING_KEY = "output_reasoning_tokens";

function isPromptKey(key: string): boolean {
  return key === "input" || key.startsWith("input_") || key.endsWith(PROMPT_SUFFIX);
}

function isCompletionKey(key: string): boolean {
  return key === "output" || key.startsWith("output_");
}

/**
 * Convert Langfuse usage into contract `Usage`.
 *
 * `usageDetails` is canonical. The deprecated `usage` object is read **only**
 * when `usageDetails` carries no numeric entries (old exports), in either of its
 * two documented shapes: `{input, output, total}` and
 * `{promptTokens, completionTokens, totalTokens}`.
 */
export function normalizeUsage(
  usageDetails: unknown,
  deprecatedUsage: unknown,
  collector: Collector,
): Usage | null {
  const details = fromUsageDetails(usageDetails);
  if (details !== null) {
    collector.dialect(DIALECTS.usageDetails);
    return details;
  }
  const legacy = fromDeprecatedUsage(deprecatedUsage);
  if (legacy !== null) collector.dialect(DIALECTS.deprecatedUsage);
  return legacy;
}

function fromUsageDetails(raw: unknown): Usage | null {
  if (!isRecord(raw)) return null;

  let inputTokens = 0;
  let outputTokens = 0;
  let cachedInput = 0;
  let hasCachedInput = false;
  let reasoning: number | null = null;
  let total: number | null = null;
  let sawNumeric = false;

  for (const [key, value] of Object.entries(raw)) {
    const count = asFiniteNumber(value);
    if (count === null) continue;
    sawNumeric = true;

    if (key === "total") {
      total = count;
      continue;
    }
    if (isPromptKey(key)) inputTokens += count;
    else if (isCompletionKey(key)) outputTokens += count;

    if (CACHED_INPUT_KEYS.includes(key)) {
      cachedInput += count;
      hasCachedInput = true;
    }
    if (key === REASONING_KEY) reasoning = count;
  }

  if (!sawNumeric) return null;

  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    reasoning_tokens: reasoning,
    cached_input_tokens: hasCachedInput ? cachedInput : null,
    total_tokens: total ?? inputTokens + outputTokens,
  };
}

function fromDeprecatedUsage(raw: unknown): Usage | null {
  if (!isRecord(raw)) return null;
  const input = asFiniteNumber(raw.input) ?? asFiniteNumber(raw.promptTokens);
  const output = asFiniteNumber(raw.output) ?? asFiniteNumber(raw.completionTokens);
  const total = asFiniteNumber(raw.total) ?? asFiniteNumber(raw.totalTokens);
  if (input === null && output === null && total === null) return null;

  const inputTokens = input ?? 0;
  const outputTokens = output ?? 0;
  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    reasoning_tokens: null,
    cached_input_tokens: null,
    total_tokens: total ?? inputTokens + outputTokens,
  };
}

/**
 * `costDetails.total` in USD. Never recomputed at import time, and the
 * deprecated `calculated*Cost` fields are deliberately ignored.
 */
export function normalizeCost(costDetails: unknown): number | null {
  if (!isRecord(costDetails)) return null;
  return asFiniteNumber(costDetails.total);
}
