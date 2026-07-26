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
      params: null,
      output: { role: "assistant", content: "x" },
      usage: { input_tokens: 100, output_tokens: call.outputTokens },
      latencyMs: call.latencyMs,
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
