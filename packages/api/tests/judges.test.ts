import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { type CompoundDatabase, recordJudgeCalibration } from "@compound/storage";
import { freshDatabase, getJson, testApp } from "./helpers";

let db: CompoundDatabase;

beforeEach(() => {
  db = freshDatabase();
});

afterEach(() => {
  db.close();
});

function calibration(
  handle: CompoundDatabase,
  taskKey: string,
  calibrated: boolean,
  kappa: number,
) {
  return recordJudgeCalibration(handle, {
    taskKey,
    judgeModel: "anthropic/claude-opus-4.8",
    promptVersion: "v1",
    rubricHash: "sha256:r",
    mode: "pointwise",
    agreementKappa: kappa,
    kappaCiLo: kappa - 0.1,
    kappaCiHi: kappa + 0.1,
    n: 30,
    positionBiasRate: 0,
    threshold: 0.6,
    calibrated,
  });
}

describe("GET /api/judges", () => {
  test("is empty before any calibration", async () => {
    const { status, body } = await getJson(testApp(db), "/api/judges");
    expect(status).toBe(200);
    expect(body.items).toEqual([]);
  });

  test("reports the latest calibration per task with its status", async () => {
    calibration(db, "support", true, 0.75);
    calibration(db, "billing", false, 0.3);
    const { status, body } = await getJson(testApp(db), "/api/judges");
    expect(status).toBe(200);
    expect(body.items).toHaveLength(2);
    const byTask = Object.fromEntries(body.items.map((i: { task_key: string }) => [i.task_key, i]));
    expect(byTask.support.calibrated).toBe(true);
    expect(byTask.support.agreement_kappa).toBe(0.75);
    expect(byTask.support.kappa_ci).toEqual([0.65, 0.85]);
    expect(byTask.billing.calibrated).toBe(false);
  });
});
