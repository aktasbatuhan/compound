import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  type CompoundDatabase,
  cacheCompletion,
  createDatabase,
  createExperiment,
  getCachedCompletion,
  insertTraces,
  migrate,
  recordCaseResults,
  taskTrafficVolume,
  telemetryRollup,
} from "../src/index";
import { newBatch, recordFromFixture } from "./helpers";

let db: CompoundDatabase;

beforeEach(() => {
  db = createDatabase();
  migrate(db);
});

afterEach(() => {
  db.close();
});

interface SeedCall {
  fingerprint: string;
  latencyMs: number;
  costUsd: number;
  outputTokens: number;
  /** Async-queue portion of latency for a flex route; omit for a sync provider. */
  queueMs?: number;
  /** Upstream host a broker (OpenRouter) served this from; omit for a direct provider (#9). */
  upstreamProvider?: string;
  /** Doubleword service tier (flex/default/scale); omit for the default route (#9). */
  serviceTier?: string;
  /** Provider-reported cached prompt tokens; omit to model an UNREPORTING provider (#34). */
  cachedTokens?: number;
}

describe("taskTrafficVolume", () => {
  test("returns null when a task has no ingested traffic basis", () => {
    expect(taskTrafficVolume(db, "support")).toBeNull();
  });

  test("derives a monthly trace rate from the task's observed timestamp window", () => {
    const batch = newBatch(db);
    insertTraces(db, batch.id, [
      recordFromFixture("eval-ready-full.json", (trace) => {
        trace.trace_id = "traffic-1";
        trace.task_key = "support";
        trace.started_at = "2026-07-01T00:00:00.000Z";
      }),
      recordFromFixture("eval-ready-minimal.json", (trace) => {
        trace.trace_id = "traffic-2";
        trace.task_key = "support";
        trace.started_at = "2026-07-11T00:00:00.000Z";
      }),
      recordFromFixture("diagnostic-missing-focal.json", (trace) => {
        trace.trace_id = "other-task";
        trace.task_key = "billing";
        trace.started_at = "2026-07-05T00:00:00.000Z";
      }),
    ]);

    const traffic = taskTrafficVolume(db, "support");
    expect(traffic?.traceCount).toBe(2);
    expect(traffic?.observedSpanDays).toBe(10);
    expect(traffic?.rateWindowDays).toBe(10);
    expect(traffic?.tracesPerDay).toBeCloseTo(0.2, 10);
    expect(traffic?.projectedMonthlyTraces).toBeCloseTo(6.0875, 10);
  });

  test("uses a labelled one-day minimum for a sub-day sample", () => {
    const batch = newBatch(db);
    insertTraces(db, batch.id, [
      recordFromFixture("eval-ready-full.json", (trace) => {
        trace.trace_id = "one-trace";
        trace.task_key = "support";
        trace.started_at = "2026-07-01T12:00:00.000Z";
      }),
    ]);

    const traffic = taskTrafficVolume(db, "support");
    expect(traffic?.observedSpanDays).toBe(0);
    expect(traffic?.rateWindowDays).toBe(1);
    expect(traffic?.projectedMonthlyTraces).toBeCloseTo(30.4375, 10);
  });
});

/** One experiment whose graded cases each point at a cached completion. */
function seedExperiment(taskKey: string, model: string, provider: string, calls: SeedCall[]): void {
  const experiment = createExperiment(db, {
    taskKey,
    candidateModel: model,
    provider,
    partition: "optimizer_validation",
    paid: true,
  });
  for (const call of calls) {
    cacheCompletion(db, {
      fingerprint: call.fingerprint,
      provider,
      model,
      params: call.serviceTier !== undefined ? { service_tier: call.serviceTier } : null,
      output: { role: "assistant", content: "x" },
      usage: {
        input_tokens: 100,
        output_tokens: call.outputTokens,
        ...(call.cachedTokens !== undefined ? { cached_input_tokens: call.cachedTokens } : {}),
      },
      latencyMs: call.latencyMs,
      ...(call.queueMs !== undefined ? { queueMs: call.queueMs } : {}),
      ...(call.upstreamProvider !== undefined ? { upstreamProvider: call.upstreamProvider } : {}),
      costUsd: call.costUsd,
    });
  }
  recordCaseResults(
    db,
    experiment.id,
    calls.map((call, i) => ({
      caseId: `case-${i}`,
      status: "graded" as const,
      passed: true,
      score: 1,
      completionFingerprint: call.fingerprint,
    })),
  );
}

describe("telemetryRollup", () => {
  test("aggregates latency percentiles, cost, tokens, and TPS per group", () => {
    seedExperiment("support", "cheap-model", "mock", [
      { fingerprint: "f1", latencyMs: 100, costUsd: 0.01, outputTokens: 50 },
      { fingerprint: "f2", latencyMs: 200, costUsd: 0.03, outputTokens: 100 },
      { fingerprint: "f3", latencyMs: 1000, costUsd: 0.02, outputTokens: 100 },
    ]);

    const [group] = telemetryRollup(db);
    expect(group?.taskKey).toBe("support");
    expect(group?.model).toBe("cheap-model");
    expect(group?.provider).toBe("mock");
    expect(group?.completions).toBe(3);
    expect(group?.latencyP50Ms).toBe(200);
    expect(group?.latencyP95Ms).toBe(1000);
    expect(group?.meanCostUsd).toBeCloseTo(0.02, 10);
    expect(group?.totalCostUsd).toBeCloseTo(0.06, 10);
    expect(group?.meanInputTokens).toBe(100);
    expect(group?.totalOutputTokens).toBe(250);
    // Per-call TPS: 500, 500, 100 → median 500.
    expect(group?.outputTps).toBe(500);
  });

  test("a completion reused by many runs is ONE observation, not several", () => {
    const calls = [{ fingerprint: "shared", latencyMs: 100, costUsd: 0.01, outputTokens: 10 }];
    seedExperiment("support", "cheap-model", "mock", calls);
    seedExperiment("support", "cheap-model", "mock", calls); // cache-hit re-run

    const [group] = telemetryRollup(db);
    expect(group?.completions).toBe(1);
    expect(group?.totalCostUsd).toBeCloseTo(0.01, 10);
  });

  test("groups split by provider — the routing comparison", () => {
    seedExperiment("support", "same-model", "provider-a", [
      { fingerprint: "a1", latencyMs: 100, costUsd: 0.01, outputTokens: 10 },
    ]);
    seedExperiment("support", "same-model", "provider-b", [
      { fingerprint: "b1", latencyMs: 900, costUsd: 0.002, outputTokens: 10 },
    ]);

    const groups = telemetryRollup(db);
    expect(groups).toHaveLength(2);
    expect(groups.map((g) => g.provider)).toEqual(["provider-a", "provider-b"]);
    expect(groups[0]?.latencyP50Ms).toBe(100);
    expect(groups[1]?.latencyP50Ms).toBe(900);
  });

  test("one model id splits by OpenRouter upstream host (#9)", () => {
    // Same model, same broker (openrouter), two upstreams → two comparable rows.
    seedExperiment("support", "kimi-k3", "openrouter", [
      {
        fingerprint: "fw",
        latencyMs: 800,
        costUsd: 0.003,
        outputTokens: 40,
        upstreamProvider: "Fireworks",
      },
    ]);
    seedExperiment("support", "kimi-k3", "openrouter", [
      {
        fingerprint: "tg",
        latencyMs: 1500,
        costUsd: 0.003,
        outputTokens: 40,
        upstreamProvider: "Together",
      },
    ]);

    const groups = telemetryRollup(db);
    expect(groups).toHaveLength(2);
    expect(groups.map((g) => g.provider).sort()).toEqual([
      "openrouter/Fireworks",
      "openrouter/Together",
    ]);
  });

  test("one provider splits by service tier — the Doubleword tier comparison (#9)", () => {
    // Same model + provider, different async tiers → separate comparable rows.
    seedExperiment("support", "kimi-k3", "doubleword", [
      {
        fingerprint: "flex1",
        latencyMs: 8000,
        queueMs: 7000,
        costUsd: 0.003,
        outputTokens: 40,
        serviceTier: "flex",
      },
    ]);
    seedExperiment("support", "kimi-k3", "doubleword", [
      {
        fingerprint: "scale1",
        latencyMs: 900,
        queueMs: 0,
        costUsd: 0.003,
        outputTokens: 40,
        serviceTier: "scale",
      },
    ]);

    const groups = telemetryRollup(db);
    expect(groups).toHaveLength(2);
    expect(groups.map((g) => g.provider).sort()).toEqual(["doubleword/flex", "doubleword/scale"]);
    // The flex tier's latency is mostly queue; scale has none.
    const flex = groups.find((g) => g.provider === "doubleword/flex");
    expect(flex?.queueP50Ms).toBe(7000);
  });

  test("splits queue vs decode for a flex route (#8)", () => {
    // Three flex completions: queue is most of the latency (async wait).
    seedExperiment("support", "glm-flex", "doubleword", [
      { fingerprint: "q1", latencyMs: 1000, queueMs: 900, costUsd: 0.001, outputTokens: 50 },
      { fingerprint: "q2", latencyMs: 1200, queueMs: 1000, costUsd: 0.001, outputTokens: 50 },
      { fingerprint: "q3", latencyMs: 1400, queueMs: 1100, costUsd: 0.001, outputTokens: 50 },
    ]);

    const [group] = telemetryRollup(db);
    expect(group?.latencyP50Ms).toBe(1200);
    expect(group?.queueP50Ms).toBe(1000);
    // decode = latency - queue → {100, 200, 300}, median 200.
    expect(group?.decodeP50Ms).toBe(200);
  });

  test("a synchronous provider reports no queue and decode equal to latency", () => {
    // No queueMs seeded → null in storage → queue is 0, decode == latency.
    seedExperiment("support", "gpt-4o-mini", "openai", [
      { fingerprint: "s1", latencyMs: 500, costUsd: 0.0001, outputTokens: 20 },
      { fingerprint: "s2", latencyMs: 700, costUsd: 0.0001, outputTokens: 20 },
    ]);

    const [group] = telemetryRollup(db);
    expect(group?.queueP50Ms).toBe(0);
    expect(group?.queueP95Ms).toBe(0);
    expect(group?.decodeP50Ms).toBe(group?.latencyP50Ms);
  });

  test("filters by task and skips results with no completion fingerprint", () => {
    seedExperiment("support", "m", "p", [
      { fingerprint: "s1", latencyMs: 100, costUsd: 0.01, outputTokens: 10 },
    ]);
    seedExperiment("billing", "m", "p", [
      { fingerprint: "b1", latencyMs: 100, costUsd: 0.01, outputTokens: 10 },
    ]);
    const skipped = createExperiment(db, {
      taskKey: "support",
      candidateModel: "m",
      provider: "p",
      partition: "optimizer_validation",
      paid: false,
    });
    recordCaseResults(db, skipped.id, [{ caseId: "c", status: "cache_miss_dry_run" }]);

    expect(telemetryRollup(db, "support")).toHaveLength(1);
    expect(telemetryRollup(db, "support")[0]?.taskKey).toBe("support");
    expect(telemetryRollup(db)).toHaveLength(2);
  });

  test("cache hit rate is cached / input over the reporting completions (#34)", () => {
    // 100 input tokens per call: 90 + 50 cached over 200 input → 70%.
    seedExperiment("support", "kimi-k3", "zai", [
      { fingerprint: "z1", latencyMs: 100, costUsd: 0.01, outputTokens: 10, cachedTokens: 90 },
      { fingerprint: "z2", latencyMs: 100, costUsd: 0.01, outputTokens: 10, cachedTokens: 50 },
    ]);

    const [group] = telemetryRollup(db);
    expect(group?.totalCachedInputTokens).toBe(140);
    expect(group?.cacheHitRate).toBeCloseTo(0.7, 10);
  });

  test("an unreporting provider yields NULL hit rate — never a false 0% (#34)", () => {
    // No cachedTokens seeded → the stored column is NULL, not 0.
    seedExperiment("support", "kimi-k3", "doubleword", [
      { fingerprint: "d1", latencyMs: 100, costUsd: 0.01, outputTokens: 10 },
    ]);
    // A provider that REPORTS 0 is a real 0% — distinct from unreported.
    seedExperiment("support", "kimi-k3", "parasail", [
      { fingerprint: "p1", latencyMs: 100, costUsd: 0.01, outputTokens: 10, cachedTokens: 0 },
    ]);

    const groups = telemetryRollup(db);
    const unreported = groups.find((g) => g.provider === "doubleword");
    expect(unreported?.totalCachedInputTokens).toBeNull();
    expect(unreported?.cacheHitRate).toBeNull();
    const zero = groups.find((g) => g.provider === "parasail");
    expect(zero?.totalCachedInputTokens).toBe(0);
    expect(zero?.cacheHitRate).toBe(0);
  });

  test("mixed reporting: unreported completions are excluded from both sides (#34)", () => {
    // One reporting call (80/100 cached), one unreported → 80%, not 40%.
    seedExperiment("support", "kimi-k3", "fireworks", [
      { fingerprint: "m1", latencyMs: 100, costUsd: 0.01, outputTokens: 10, cachedTokens: 80 },
      { fingerprint: "m2", latencyMs: 100, costUsd: 0.01, outputTokens: 10 },
    ]);

    const [group] = telemetryRollup(db);
    expect(group?.totalCachedInputTokens).toBe(80);
    expect(group?.cacheHitRate).toBeCloseTo(0.8, 10);
  });
});

describe("cached-token column round-trip (#34)", () => {
  test("a reported figure is projected into the column; unreported stays NULL", () => {
    cacheCompletion(db, {
      fingerprint: "rt-reported",
      provider: "zai",
      model: "kimi-k3",
      params: null,
      output: { role: "assistant", content: "x" },
      usage: { input_tokens: 100, output_tokens: 10, cached_input_tokens: 93 },
      costUsd: 0.01,
    });
    cacheCompletion(db, {
      fingerprint: "rt-zero",
      provider: "parasail",
      model: "kimi-k3",
      params: null,
      output: { role: "assistant", content: "x" },
      usage: { input_tokens: 100, output_tokens: 10, cached_input_tokens: 0 },
      costUsd: 0.01,
    });
    cacheCompletion(db, {
      fingerprint: "rt-unreported",
      provider: "doubleword",
      model: "kimi-k3",
      params: null,
      output: { role: "assistant", content: "x" },
      usage: { input_tokens: 100, output_tokens: 10 },
      costUsd: 0.01,
    });

    expect(getCachedCompletion(db, "rt-reported")?.cachedInputTokens).toBe(93);
    // A reported 0 round-trips as 0; a missing field round-trips as NULL.
    expect(getCachedCompletion(db, "rt-zero")?.cachedInputTokens).toBe(0);
    expect(getCachedCompletion(db, "rt-unreported")?.cachedInputTokens).toBeNull();
  });
});
