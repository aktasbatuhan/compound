/**
 * The trust rule (docs/judges-v1.md, "The trust rule").
 *
 * A judge may feed a gate only when the LATEST calibration for its EXACT pin
 * (task, model, prompt_version, rubric_hash) meets the threshold on enough
 * human-labelled cases. No calibration for the pin, too few cases, or agreement
 * below threshold all leave the judge uncalibrated — and an uncalibrated judge
 * abstains. Trust is never inherited across a pin change.
 */
import { type CompoundDatabase, latestCalibrationForPin } from "@compound/storage";
import { hashRubric } from "./prompt";
import { type JudgeConfig, type JudgeTrust, MIN_CALIBRATION_CASES } from "./types";

export function judgeTrust(
  db: CompoundDatabase,
  judge: JudgeConfig,
  minCases = MIN_CALIBRATION_CASES,
): JudgeTrust {
  const row = latestCalibrationForPin(db, {
    taskKey: judge.taskKey,
    judgeModel: judge.model,
    promptVersion: judge.promptVersion,
    rubricHash: hashRubric(judge.rubric),
  });

  if (row === null) {
    return { calibrated: false, reason: "no calibration for this judge pin — run calibrate first" };
  }
  const ci: [number, number] = [row.kappaCiLo, row.kappaCiHi];
  if (row.n < minCases) {
    return {
      calibrated: false,
      reason: `only ${row.n} human-labelled cases (need ${minCases})`,
      kappa: row.agreementKappa,
      ci,
      n: row.n,
    };
  }
  // Re-evaluate against the CURRENT threshold, not the stored one — the config
  // threshold is authoritative and may have been tightened since measurement.
  if (row.agreementKappa < judge.calibrationThreshold) {
    return {
      calibrated: false,
      reason: `agreement kappa ${row.agreementKappa.toFixed(2)} < threshold ${judge.calibrationThreshold}`,
      kappa: row.agreementKappa,
      ci,
      n: row.n,
    };
  }
  return {
    calibrated: true,
    reason: `kappa ${row.agreementKappa.toFixed(2)} ≥ ${judge.calibrationThreshold} on ${row.n} cases`,
    kappa: row.agreementKappa,
    ci,
    n: row.n,
  };
}
