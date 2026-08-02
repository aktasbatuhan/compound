import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  type CompoundDatabase,
  cacheCompletion,
  createDatabase,
  createExperiment,
  migrate,
  recordCaseResults,
  telemetryRollup,
} from "../src/index";

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
}

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
      usage: { input_tokens: 100, output_tokens: call.outputTokens },
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
});
