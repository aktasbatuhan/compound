import { describe, expect, test } from "bun:test";
import {
  createExperiment,
  getExperimentResults,
  latestCalibrationForPin,
  latestCalibrationForTask,
  listLatestCalibrations,
  recordCaseResults,
  recordJudgeCalibration,
  setCaseJudgment,
} from "../src/index";
import { freshDatabase } from "./helpers";

const pin = {
  taskKey: "support",
  judgeModel: "anthropic/claude-opus-4.8",
  promptVersion: "v1",
  rubricHash: "sha256:rubricA",
};

const calibration = {
  ...pin,
  mode: "pointwise" as const,
  agreementKappa: 0.72,
  kappaCiLo: 0.6,
  kappaCiHi: 0.84,
  n: 40,
  positionBiasRate: 0,
  threshold: 0.6,
  calibrated: true,
};

describe("judge calibration storage", () => {
  test("records and reads the latest calibration for an exact pin", () => {
    const handle = freshDatabase();
    recordJudgeCalibration(handle, calibration);
    const found = latestCalibrationForPin(handle, pin);
    expect(found?.agreementKappa).toBe(0.72);
    expect(found?.calibrated).toBe(true);
    handle.close();
  });

  test("a different pin has no calibration (trust is never inherited)", () => {
    const handle = freshDatabase();
    recordJudgeCalibration(handle, calibration);
    expect(latestCalibrationForPin(handle, { ...pin, promptVersion: "v2" })).toBeNull();
    expect(latestCalibrationForPin(handle, { ...pin, rubricHash: "sha256:rubricB" })).toBeNull();
    handle.close();
  });

  test("latest measurement wins for a pin", () => {
    const handle = freshDatabase();
    recordJudgeCalibration(handle, { ...calibration, agreementKappa: 0.5, calibrated: false });
    // A later, higher measurement supersedes.
    recordJudgeCalibration(handle, {
      ...calibration,
      agreementKappa: 0.8,
      calibrated: true,
      measuredAt: undefined,
    } as typeof calibration);
    const found = latestCalibrationForPin(handle, pin);
    expect(found?.agreementKappa).toBe(0.8);
    handle.close();
  });

  test("task-level lookup and latest-per-task listing", () => {
    const handle = freshDatabase();
    recordJudgeCalibration(handle, calibration);
    recordJudgeCalibration(handle, { ...calibration, taskKey: "billing", agreementKappa: 0.4 });
    expect(latestCalibrationForTask(handle, "support")?.agreementKappa).toBe(0.72);
    const all = listLatestCalibrations(handle);
    expect(new Set(all.map((c) => c.taskKey))).toEqual(new Set(["support", "billing"]));
    handle.close();
  });
});

describe("setCaseJudgment", () => {
  test("overwrites a case's grade with the judge's verdict", () => {
    const handle = freshDatabase();
    const exp = createExperiment(handle, {
      taskKey: "support",
      candidateModel: "cand",
      provider: "openrouter",
      partition: "optimizer_validation",
      paid: false,
    });
    recordCaseResults(handle, exp.id, [
      { caseId: "c1", status: "graded", passed: true, score: 1, completionFingerprint: "fp1" },
    ]);
    setCaseJudgment(handle, exp.id, "c1", { score: 0.3, passed: false, judgeAbstained: false });
    const [row] = getExperimentResults(handle, exp.id);
    expect(row?.score).toBe(0.3);
    expect(row?.passed).toBe(false);
    expect(row?.judgeAbstained).toBe(false);
    // The completion fingerprint is preserved for auditability.
    expect(row?.completionFingerprint).toBe("fp1");
    handle.close();
  });

  test("an abstaining judge marks the case abstained", () => {
    const handle = freshDatabase();
    const exp = createExperiment(handle, {
      taskKey: "support",
      candidateModel: "cand",
      provider: "openrouter",
      partition: "optimizer_validation",
      paid: false,
    });
    recordCaseResults(handle, exp.id, [{ caseId: "c1", status: "graded", passed: true, score: 1 }]);
    setCaseJudgment(handle, exp.id, "c1", { score: 0.9, passed: true, judgeAbstained: true });
    const [row] = getExperimentResults(handle, exp.id);
    expect(row?.judgeAbstained).toBe(true);
    handle.close();
  });
});
