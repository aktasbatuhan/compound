import { describe, expect, test } from "bun:test";
import type { Message } from "@compound/contract";
import { type CompoundDatabase, insertCases, reviewCase } from "@compound/storage";
import { calibrateJudge } from "../src/calibrate";
import { judgeGradeBatch } from "../src/grade-batch";
import { freshDb, judgeConfig, judgeCtx, scriptedJudge } from "./helpers";

/** A judge that agrees with humans: "good" outputs score high, "bad" low. */
const agreeingJudge = scriptedJudge((text) => (text.includes("good") ? 0.9 : 0.1));

function seedCalibration(db: CompoundDatabase, good: number, bad: number): void {
  const records = [];
  for (let i = 0; i < good + bad; i++) {
    const isGood = i < good;
    records.push({
      caseId: `cal-${i}`,
      taskKey: "support",
      sourceTraceId: `tr-${i}`,
      contentHash: `hash-${i}`,
      provenance: "observed_output" as const,
      partition: "judge_calibration" as const,
      input: { input: [{ role: "user", content: "help" }] },
      expected: { role: "assistant", content: isGood ? "good answer" : "bad answer" } as Message,
    });
  }
  insertCases(db, records);
  for (let i = 0; i < good + bad; i++) {
    reviewCase(db, `cal-${i}`, { reviewState: i < good ? "approved" : "rejected" });
  }
}

/** Calibrate the pin above threshold so the judge is trusted for grading. */
async function calibrate(db: CompoundDatabase): Promise<void> {
  seedCalibration(db, 8, 8);
  const result = await calibrateJudge(judgeCtx(db, agreeingJudge), judgeConfig());
  expect(result.calibration.calibrated).toBe(true);
}

const items = [
  { caseId: "a", responseText: "a good answer" },
  { caseId: "b", responseText: "a bad answer" },
];

describe("judgeGradeBatch — earned trust", () => {
  test("an UNCALIBRATED judge grades nothing and returns no verdicts", async () => {
    const db = freshDb();
    // No calibration seeded → the pin is untrusted.
    const result = await judgeGradeBatch(judgeCtx(db, agreeingJudge), judgeConfig(), items);
    expect(result.trust.calibrated).toBe(false);
    expect(result.verdicts).toEqual([]);
    expect(result.totalCostUsd).toBe(0);
    db.close();
  });

  test("a CALIBRATED judge grades each output with its score and reasoning", async () => {
    const db = freshDb();
    await calibrate(db);
    const result = await judgeGradeBatch(judgeCtx(db, agreeingJudge), judgeConfig(), items);
    expect(result.trust.calibrated).toBe(true);
    expect(result.verdicts.map((v) => v.status)).toEqual(["graded", "graded"]);
    const byCase = new Map(result.verdicts.map((v) => [v.caseId, v]));
    expect(byCase.get("a")?.score).toBeCloseTo(0.9, 6);
    expect(byCase.get("b")?.score).toBeCloseTo(0.1, 6);
    expect(result.totalCostUsd).toBeGreaterThan(0);
    db.close();
  });

  test("a re-grade of the same outputs is served from cache at $0", async () => {
    const db = freshDb();
    await calibrate(db);
    await judgeGradeBatch(judgeCtx(db, agreeingJudge), judgeConfig(), items);
    const again = await judgeGradeBatch(judgeCtx(db, agreeingJudge), judgeConfig(), items);
    expect(again.verdicts.map((v) => v.status)).toEqual(["graded", "graded"]);
    expect(again.totalCostUsd).toBe(0);
    db.close();
  });

  test("a dry run over never-judged outputs reports cache misses, spends nothing", async () => {
    const db = freshDb();
    await calibrate(db);
    const result = await judgeGradeBatch(
      judgeCtx(db, agreeingJudge, { paid: false }),
      judgeConfig(),
      items,
    );
    expect(result.trust.calibrated).toBe(true);
    expect(result.verdicts.map((v) => v.status)).toEqual([
      "cache_miss_dry_run",
      "cache_miss_dry_run",
    ]);
    expect(result.totalCostUsd).toBe(0);
    db.close();
  });
});
