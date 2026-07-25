import { describe, expect, test } from "bun:test";
import { recordJudgeCalibration } from "@compound/storage";
import { hashRubric } from "../src/prompt";
import { judgeTrust } from "../src/trust";
import { freshDb, judgeConfig } from "./helpers";

function record(db: ReturnType<typeof freshDb>, judge = judgeConfig(), over = {}) {
  return recordJudgeCalibration(db, {
    taskKey: judge.taskKey,
    judgeModel: judge.model,
    promptVersion: judge.promptVersion,
    rubricHash: hashRubric(judge.rubric),
    mode: "pointwise",
    agreementKappa: 0.75,
    kappaCiLo: 0.6,
    kappaCiHi: 0.9,
    n: 30,
    positionBiasRate: 0,
    threshold: 0.6,
    calibrated: true,
    ...over,
  });
}

describe("judgeTrust", () => {
  test("uncalibrated when there is no calibration for the pin", () => {
    const db = freshDb();
    const t = judgeTrust(db, judgeConfig());
    expect(t.calibrated).toBe(false);
    expect(t.reason).toContain("no calibration");
    db.close();
  });

  test("calibrated when the latest measurement clears the threshold on enough cases", () => {
    const db = freshDb();
    record(db);
    const t = judgeTrust(db, judgeConfig());
    expect(t.calibrated).toBe(true);
    expect(t.kappa).toBe(0.75);
    db.close();
  });

  test("uncalibrated when agreement is below the CURRENT threshold", () => {
    const db = freshDb();
    record(db, judgeConfig(), { agreementKappa: 0.5 });
    // Config threshold 0.6 > measured 0.5 → not trusted, even though the row was
    // stored as calibrated against a looser threshold.
    const t = judgeTrust(db, judgeConfig({ calibrationThreshold: 0.6 }));
    expect(t.calibrated).toBe(false);
    expect(t.reason).toContain("kappa");
    db.close();
  });

  test("uncalibrated with too few labelled cases", () => {
    const db = freshDb();
    record(db, judgeConfig(), { n: 4 });
    const t = judgeTrust(db, judgeConfig());
    expect(t.calibrated).toBe(false);
    expect(t.reason).toContain("need");
    db.close();
  });

  test("a changed pin (prompt_version) has no calibration — trust not inherited", () => {
    const db = freshDb();
    record(db);
    const t = judgeTrust(db, judgeConfig({ promptVersion: "v2" }));
    expect(t.calibrated).toBe(false);
    db.close();
  });
});
