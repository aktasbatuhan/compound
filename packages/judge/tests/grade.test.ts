import { describe, expect, test } from "bun:test";
import type { Message } from "@compound/contract";
import {
  type CompoundDatabase,
  cacheCompletion,
  createExperiment,
  getExperimentResults,
  recordCaseResults,
  recordJudgeCalibration,
} from "@compound/storage";
import { gradeExperimentWithJudge } from "../src/grade";
import { hashRubric } from "../src/prompt";
import { freshDb, judgeConfig, judgeCtx, scriptedJudge } from "./helpers";

function makeCalibrated(db: CompoundDatabase, judge = judgeConfig()) {
  recordJudgeCalibration(db, {
    taskKey: judge.taskKey,
    judgeModel: judge.model,
    promptVersion: judge.promptVersion,
    rubricHash: hashRubric(judge.rubric),
    mode: "pointwise",
    agreementKappa: 0.8,
    kappaCiLo: 0.7,
    kappaCiHi: 0.9,
    n: 20,
    positionBiasRate: 0,
    threshold: 0.6,
    calibrated: true,
  });
}

/** An experiment whose passing cases have cached outputs the judge can fetch. */
function seedExperiment(db: CompoundDatabase, outputs: string[]) {
  const exp = createExperiment(db, {
    taskKey: "support",
    candidateModel: "cand",
    provider: "openrouter",
    partition: "decision_test",
    paid: false,
  });
  const results = outputs.map((text, i) => {
    const fp = `cand-fp-${i}`;
    cacheCompletion(db, {
      fingerprint: fp,
      provider: "openrouter",
      model: "cand",
      params: null,
      output: { role: "assistant", content: text } as Message,
      costUsd: 0,
    });
    return {
      caseId: `case-${i}`,
      status: "graded" as const,
      passed: true,
      score: 1,
      completionFingerprint: fp,
    };
  });
  recordCaseResults(db, exp.id, results);
  return exp;
}

const judge = scriptedJudge((text) => (text.includes("great") ? 0.9 : 0.2));

describe("gradeExperimentWithJudge", () => {
  test("a calibrated judge writes per-case scores into experiment_results", async () => {
    const db = freshDb();
    makeCalibrated(db);
    const exp = seedExperiment(db, ["a great answer", "a weak answer", "a great answer"]);

    const summary = await gradeExperimentWithJudge(judgeCtx(db, judge), judgeConfig(), exp.id);
    expect(summary.trust.calibrated).toBe(true);
    expect(summary.judged).toBe(3);
    expect(summary.abstained).toBe(0);

    const rows = getExperimentResults(db, exp.id);
    const byCase = Object.fromEntries(rows.map((r) => [r.caseId, r]));
    expect(byCase["case-0"]?.score).toBeCloseTo(0.9, 6);
    expect(byCase["case-0"]?.passed).toBe(true);
    expect(byCase["case-1"]?.score).toBeCloseTo(0.2, 6);
    expect(byCase["case-1"]?.passed).toBe(false); // below decision_point 0.5
    expect(byCase["case-1"]?.judgeAbstained).toBe(false);
    db.close();
  });

  test("an uncalibrated judge abstains on every case and makes no provider calls", async () => {
    const db = freshDb();
    // No calibration recorded → uncalibrated.
    const exp = seedExperiment(db, ["a great answer", "a weak answer"]);
    const throwingJudge = scriptedJudge(() => {
      throw new Error("judge must not be called when uncalibrated");
    });

    const summary = await gradeExperimentWithJudge(
      judgeCtx(db, throwingJudge),
      judgeConfig(),
      exp.id,
    );
    expect(summary.trust.calibrated).toBe(false);
    expect(summary.judged).toBe(0);
    expect(summary.abstained).toBe(2);

    const rows = getExperimentResults(db, exp.id);
    expect(rows.every((r) => r.judgeAbstained)).toBe(true);
    db.close();
  });

  test("refuses an experiment whose task does not match the judge", async () => {
    const db = freshDb();
    makeCalibrated(db);
    const exp = createExperiment(db, {
      taskKey: "billing",
      candidateModel: "cand",
      provider: "openrouter",
      partition: "decision_test",
      paid: false,
    });
    await expect(
      gradeExperimentWithJudge(judgeCtx(db, judge), judgeConfig({ taskKey: "support" }), exp.id),
    ).rejects.toThrow(/does not match/);
    db.close();
  });
});
