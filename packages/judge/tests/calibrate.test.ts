import { describe, expect, test } from "bun:test";
import type { Message } from "@compound/contract";
import { type CompoundDatabase, insertCases, reviewCase } from "@compound/storage";
import { calibrateJudge } from "../src/calibrate";
import { freshDb, judgeConfig, judgeCtx, scriptedJudge } from "./helpers";

/** Seed n calibration cases; the first `good` are good outputs, rest are bad. */
function seedCalibrationCases(db: CompoundDatabase, good: number, bad: number) {
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

// A judge that agrees with humans: "good" scores high, "bad" scores low.
const agreeingJudge = scriptedJudge((text) => (text.includes("good") ? 0.9 : 0.1));

describe("calibrateJudge", () => {
  test("a judge that agrees with humans calibrates above threshold", async () => {
    const db = freshDb();
    seedCalibrationCases(db, 8, 8);
    const result = await calibrateJudge(judgeCtx(db, agreeingJudge), judgeConfig());
    expect(result.casesLabelled).toBe(16);
    expect(result.casesGraded).toBe(16);
    expect(result.calibration.agreementKappa).toBeCloseTo(1, 6);
    expect(result.calibration.calibrated).toBe(true);
    db.close();
  });

  test("a coin-flip judge does not calibrate", async () => {
    const db = freshDb();
    seedCalibrationCases(db, 8, 8);
    // Always says 'good' regardless of the actual output → no better than chance.
    const flip = scriptedJudge(() => 0.9);
    const result = await calibrateJudge(judgeCtx(db, flip), judgeConfig());
    expect(result.calibration.calibrated).toBe(false);
    db.close();
  });

  test("too few labelled cases cannot calibrate even with perfect agreement", async () => {
    const db = freshDb();
    seedCalibrationCases(db, 2, 2);
    const result = await calibrateJudge(judgeCtx(db, agreeingJudge), judgeConfig());
    expect(result.casesGraded).toBe(4);
    expect(result.calibration.calibrated).toBe(false); // below MIN_CALIBRATION_CASES
    db.close();
  });

  test("a dry run (no paid, no cache) grades nothing and stays uncalibrated", async () => {
    const db = freshDb();
    seedCalibrationCases(db, 8, 8);
    const result = await calibrateJudge(
      judgeCtx(db, agreeingJudge, { paid: false }),
      judgeConfig(),
    );
    expect(result.casesGraded).toBe(0);
    expect(result.skippedNoCache).toBe(16);
    expect(result.calibration.calibrated).toBe(false);
    db.close();
  });
});
