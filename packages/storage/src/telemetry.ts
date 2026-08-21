/**
 * Telemetry rollup: the operational side of a routing decision.
 *
 * Every completion already carries latency, cost, and token usage in the
 * content-addressed cache; this aggregates them per task x model x provider so
 * quality (the gate) and speed/cost can sit side by side. Pure reads — no
 * provider calls, no new instrumentation.
 *
 * Each COMPLETION is one observation: the same cached completion served to five
 * re-runs is still one measurement of the provider, so rows are deduplicated by
 * fingerprint within a group before aggregating.
 */
import { eq } from "drizzle-orm";
import type { CompoundDatabase } from "./db";
import { completions, experimentResults, experiments } from "./schema";

export interface TelemetryGroup {
  taskKey: string;
  model: string;
  provider: string;
  /** Distinct completions observed for this group. */
  completions: number;
  latencyP50Ms: number;
  latencyP95Ms: number;
  /**
   * Async-queue and decode split of latency for a flex/background route (#8).
   * For synchronous providers there is no queue, so queue is 0 and decode
   * equals latency — the columns still line up for a side-by-side comparison.
   */
  queueP50Ms: number;
  queueP95Ms: number;
  decodeP50Ms: number;
  decodeP95Ms: number;
  meanCostUsd: number;
  totalCostUsd: number;
  meanInputTokens: number;
  meanOutputTokens: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  /**
   * Median output tokens/second, computed per completion as output tokens over
   * FULL request latency — it deliberately includes provider queueing and
   * time-to-first-token; it is not a server-only decode benchmark.
   */
  outputTps: number;
  /**
   * Total provider-reported cached prompt tokens over the group (#34). `null`
   * when NO completion in the group reported a cached-token field — the
   * provider is "unreported", which must never render as 0.
   */
  totalCachedInputTokens: number | null;
  /**
   * Provider cache hit rate: cached / input tokens, summed over the completions
   * that REPORTED a cached figure (a reported 0 counts; an unreported one is
   * excluded from both sides). `null` when nothing was reported.
   */
  cacheHitRate: number | null;
}

interface Observation {
  latencyMs: number | null;
  /** Async-queue portion of latency (flex route), or null for sync providers. */
  queueMs: number | null;
  costUsd: number;
  inputTokens: number;
  outputTokens: number;
  /** Provider-reported cached prompt tokens; null when unreported (#34). */
  cachedInputTokens: number | null;
}

/** Nearest-rank quantile over an unsorted sample; 0 for an empty sample. */
function quantile(values: number[], q: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.ceil(q * sorted.length) - 1);
  return sorted[Math.max(0, index)] as number;
}

/**
 * The service tier a completion ran under (Doubleword flex/default/scale), read
 * from its stored request params; null when none was set (#9 speed axis). Lets
 * one provider's tiers be compared side by side instead of collapsing to a row.
 */
export function serviceTierOf(paramsJson: unknown): string | null {
  const params = (paramsJson ?? {}) as { service_tier?: unknown };
  return typeof params.service_tier === "string" ? params.service_tier : null;
}

/**
 * The comparable "route" label for a completion: the provider, plus the upstream
 * host a broker dispatched to (OpenRouter) and/or the service tier — so each
 * distinct route is its own telemetry row rather than collapsing by provider.
 */
export function routeLabel(
  provider: string,
  upstreamProvider: string | null,
  serviceTier: string | null,
): string {
  return (
    provider +
    (upstreamProvider != null ? `/${upstreamProvider}` : "") +
    (serviceTier != null ? `/${serviceTier}` : "")
  );
}

/** Input/output token counts from a completion's stored usage, 0 when absent. */
export function usageTokens(usageJson: unknown): { input: number; output: number } {
  const usage = (usageJson ?? {}) as { input_tokens?: unknown; output_tokens?: unknown };
  return {
    input: typeof usage.input_tokens === "number" ? usage.input_tokens : 0,
    output: typeof usage.output_tokens === "number" ? usage.output_tokens : 0,
  };
}

/**
 * The cached-prompt token count from a completion's stored usage (#34), or
 * `null` when the provider reported no cached-token field — NOT a defaulted 0,
 * because "unreported" and "reported no hit" must stay distinguishable all the
 * way to the views. This is what `cacheCompletion` projects into the
 * `cached_input_tokens` column.
 */
export function cachedTokensOf(usageJson: unknown): number | null {
  const usage = (usageJson ?? {}) as { cached_input_tokens?: unknown };
  return typeof usage.cached_input_tokens === "number" ? usage.cached_input_tokens : null;
}

function sum(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0);
}

interface GroupKey {
  taskKey: string;
  model: string;
  provider: string;
}

/**
 * Aggregate telemetry per task x model x provider over every completion that a
 * graded experiment case produced. Sorted by task, then model, then provider.
 */
export function telemetryRollup(handle: CompoundDatabase, taskKey?: string): TelemetryGroup[] {
  const base = handle.db
    .select({
      taskKey: experiments.taskKey,
      model: completions.model,
      provider: completions.provider,
      upstreamProvider: completions.upstreamProvider,
      paramsJson: completions.paramsJson,
      fingerprint: completions.fingerprint,
      latencyMs: completions.latencyMs,
      queueMs: completions.queueMs,
      costUsd: completions.costUsd,
      usageJson: completions.usageJson,
      cachedInputTokens: completions.cachedInputTokens,
    })
    .from(experimentResults)
    .innerJoin(experiments, eq(experiments.id, experimentResults.experimentId))
    .innerJoin(completions, eq(completions.fingerprint, experimentResults.completionFingerprint))
    .$dynamic();
  const rows =
    taskKey !== undefined ? base.where(eq(experiments.taskKey, taskKey)).all() : base.all();

  // One observation per completion per group, however many runs reused it.
  const keys = new Map<string, GroupKey>();
  const groups = new Map<string, Map<string, Observation>>();
  for (const row of rows) {
    // Each distinct route is its own row: the OpenRouter upstream (→ "Fireworks")
    // AND the service tier (Doubleword flex/default/scale) fold into the provider
    // label, so one model id splits host-by-host and tier-by-tier (#9).
    const provider = routeLabel(row.provider, row.upstreamProvider, serviceTierOf(row.paramsJson));
    const key = JSON.stringify([row.taskKey, row.model, provider]);
    keys.set(key, { taskKey: row.taskKey, model: row.model, provider });
    let byFingerprint = groups.get(key);
    if (byFingerprint === undefined) {
      byFingerprint = new Map();
      groups.set(key, byFingerprint);
    }
    if (byFingerprint.has(row.fingerprint)) continue;
    const tokens = usageTokens(row.usageJson);
    byFingerprint.set(row.fingerprint, {
      latencyMs: row.latencyMs,
      queueMs: row.queueMs,
      costUsd: row.costUsd,
      inputTokens: tokens.input,
      outputTokens: tokens.output,
      cachedInputTokens: row.cachedInputTokens,
    });
  }

  const result: TelemetryGroup[] = [];
  for (const [key, byFingerprint] of groups) {
    const group = keys.get(key) as GroupKey;
    const observations = [...byFingerprint.values()];
    const n = observations.length;
    const latencies = observations
      .map((o) => o.latencyMs)
      .filter((v): v is number => v !== null && v > 0);
    // Queue is measured only on the async route; decode = latency - queue, so a
    // sync provider (no queue) reports decode == latency. Paired per observation.
    const queues = observations
      .map((o) => o.queueMs)
      .filter((v): v is number => v !== null && v >= 0);
    const decodes = observations
      .filter((o) => o.latencyMs !== null && o.latencyMs > 0)
      .map((o) => Math.max(0, (o.latencyMs as number) - (o.queueMs ?? 0)));
    const tpsSamples = observations
      .filter((o) => o.latencyMs !== null && o.latencyMs > 0 && o.outputTokens > 0)
      .map((o) => o.outputTokens / ((o.latencyMs as number) / 1000));
    const totalCost = sum(observations.map((o) => o.costUsd));
    const totalInput = sum(observations.map((o) => o.inputTokens));
    const totalOutput = sum(observations.map((o) => o.outputTokens));
    // Hit rate over the completions that REPORTED a cached figure only (#34):
    // mixing unreported observations into the denominator would understate a
    // provider whose reporting is partial. No reports at all → null, so an
    // unreported provider surfaces as "—", never a false 0%.
    const reporting = observations.filter((o) => o.cachedInputTokens !== null);
    const totalCached =
      reporting.length > 0 ? sum(reporting.map((o) => o.cachedInputTokens as number)) : null;
    const reportedInput = sum(reporting.map((o) => o.inputTokens));
    result.push({
      taskKey: group.taskKey,
      model: group.model,
      provider: group.provider,
      completions: n,
      latencyP50Ms: quantile(latencies, 0.5),
      latencyP95Ms: quantile(latencies, 0.95),
      queueP50Ms: quantile(queues, 0.5),
      queueP95Ms: quantile(queues, 0.95),
      decodeP50Ms: quantile(decodes, 0.5),
      decodeP95Ms: quantile(decodes, 0.95),
      meanCostUsd: n > 0 ? totalCost / n : 0,
      totalCostUsd: totalCost,
      meanInputTokens: n > 0 ? totalInput / n : 0,
      meanOutputTokens: n > 0 ? totalOutput / n : 0,
      totalInputTokens: totalInput,
      totalOutputTokens: totalOutput,
      outputTps: quantile(tpsSamples, 0.5),
      totalCachedInputTokens: totalCached,
      cacheHitRate: totalCached !== null && reportedInput > 0 ? totalCached / reportedInput : null,
    });
  }
  result.sort(
    (a, b) =>
      a.taskKey.localeCompare(b.taskKey) ||
      a.model.localeCompare(b.model) ||
      a.provider.localeCompare(b.provider),
  );
  return result;
}
