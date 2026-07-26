import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { type CompoundDatabase, recordOptimizationRun } from "@compound/storage";
import { freshDatabase, getJson, testApp } from "./helpers";

let db: CompoundDatabase;

beforeEach(() => {
  db = freshDatabase();
});

afterEach(() => {
  db.close();
});

describe("GET /api/optimizations", () => {
  test("is empty before any run", async () => {
    const { status, body } = await getJson(testApp(db), "/api/optimizations");
    expect(status).toBe(200);
    expect(body.items).toEqual([]);
  });

  test("returns a run with its before/after validation scores", async () => {
    recordOptimizationRun(db, {
      taskKey: "finance.dispute_charge",
      candidateModel: "openai/gpt-4o-mini",
      seedPrompt: "bad seed",
      optimizedPrompt: "call the dispute_charge tool",
      beforeValScore: 0,
      afterValScore: 1,
      valCases: 7,
      reflectionCalls: 1,
      eligibilityReason: "forced",
    });
    const { status, body } = await getJson(testApp(db), "/api/optimizations");
    expect(status).toBe(200);
    expect(body.items).toHaveLength(1);
    const run = body.items[0];
    expect(run.before_val_score).toBe(0);
    expect(run.after_val_score).toBe(1);
    expect(run.optimized_prompt).toContain("dispute_charge");
    expect(run.task_key).toBe("finance.dispute_charge");
  });
});
