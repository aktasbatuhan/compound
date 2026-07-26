import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  type CompoundDatabase,
  cacheCompletion,
  createExperiment,
  recordCaseResults,
} from "@compound/storage";
import { freshDatabase, getJson, testApp } from "./helpers";

let db: CompoundDatabase;

beforeEach(() => {
  db = freshDatabase();
});

afterEach(() => {
  db.close();
});

function seedOneCompletion(taskKey: string): void {
  const experiment = createExperiment(db, {
    taskKey,
    candidateModel: "cheap-model",
    provider: "mock",
    partition: "optimizer_validation",
    paid: true,
  });
  cacheCompletion(db, {
    fingerprint: `fp-${taskKey}`,
    provider: "mock",
    model: "cheap-model",
    params: null,
    output: { role: "assistant", content: "x" },
    usage: { input_tokens: 100, output_tokens: 50 },
    latencyMs: 250,
    costUsd: 0.01,
  });
  recordCaseResults(db, experiment.id, [
    {
      caseId: "c1",
      status: "graded",
      passed: true,
      score: 1,
      completionFingerprint: `fp-${taskKey}`,
    },
  ]);
}

describe("GET /api/telemetry", () => {
  test("is empty before any experiment", async () => {
    const { status, body } = await getJson(testApp(db), "/api/telemetry");
    expect(status).toBe(200);
    expect(body.items).toEqual([]);
  });

  test("returns the rollup with latency, cost, tokens, and TPS", async () => {
    seedOneCompletion("support");
    const { status, body } = await getJson(testApp(db), "/api/telemetry");
    expect(status).toBe(200);
    expect(body.items).toHaveLength(1);
    const row = body.items[0];
    expect(row.task_key).toBe("support");
    expect(row.model).toBe("cheap-model");
    expect(row.provider).toBe("mock");
    expect(row.completions).toBe(1);
    expect(row.latency_p50_ms).toBe(250);
    expect(row.mean_cost_usd).toBeCloseTo(0.01, 10);
    expect(row.output_tps).toBeCloseTo(200, 5);
  });

  test("filters by task_key", async () => {
    seedOneCompletion("support");
    seedOneCompletion("billing");
    const { body } = await getJson(testApp(db), "/api/telemetry?task_key=billing");
    expect(body.items).toHaveLength(1);
    expect(body.items[0].task_key).toBe("billing");
  });
});
