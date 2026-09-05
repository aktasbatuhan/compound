/**
 * Calibrate a judge against human labels (docs/judges-v1.md, "Calibration set").
 *
 * Uses the `judge_calibration` partition — cases a human reviewed through the
 * labeling UI. The human label is their verdict on the case's observed output
 * (approved → good, rejected → bad); the judge scores the same output blind. We
 * measure Cohen's kappa between the two with a bootstrap CI, pin the result to
 * (task, model, prompt_version, rubric_hash), and mark it calibrated only if it
 * clears the threshold on enough cases. Below that the judge stays uncalibrated.
 */
import {
  type CaseRow,
  type JudgeCalibrationRow,
  listCases,
  recordJudgeCalibration,
} from "@compound/storage";
import { judgeGradeOutput } from "./execute";
import { hashRubric, messageText } from "./prompt";
import { bootstrapKappaCi, seedFromString } from "./statistics";
import { type JudgeConfig, type JudgeExecutionContext, MIN_CALIBRATION_CASES } from "./types";

export interface CalibrationResult {
  calibration: JudgeCalibrationRow;
  casesLabelled: number;
  casesGraded: number;
  skippedNoCache: number;
  skippedUnparseable: number;
}

function expectedText(expected: unknown): string {
  if (typeof expected === "string") return expected;
  if (expected !== null && typeof expected === "object") {
    return messageText(expected as Parameters<typeof messageText>[0]);
  }
  return expected === undefined ? "" : JSON.stringify(expected);
}

export async function calibrateJudge(
  ctx: JudgeExecutionContext,
  judge: JudgeConfig,
  options: { confidence?: number; minCases?: number } = {},
): Promise<CalibrationResult> {
  const confidence = options.confidence ?? 0.95;
  const minCases = options.minCases ?? MIN_CALIBRATION_CASES;

  // Human-reviewed calibration cases only. judge_calibration is not sealed, so
  // no firewall token is needed.
  const cases: CaseRow[] = listCases(ctx.db, {
    taskKey: judge.taskKey,
    partition: "judge_calibration",
    reviewState: ["approved", "rejected"],
    limit: 1000,
  });

  const humanLabels: number[] = [];
  const judgeLabels: number[] = [];
  let skippedNoCache = 0;
  let skippedUnparseable = 0;

  for (const c of cases) {
    const outcome = await judgeGradeOutput(ctx, judge.rubric, expectedText(c.expected));
    if (outcome.status === "cache_miss_dry_run") {
      skippedNoCache += 1;
      continue;
    }
    if (outcome.status === "unparseable") {
      skippedUnparseable += 1;
      continue;
    }
    humanLabels.push(c.reviewState === "approved" ? 1 : 0);
    judgeLabels.push(outcome.verdict.score >= judge.decisionPoint ? 1 : 0);
  }

  const rubricHash = hashRubric(judge.rubric);
  const seed = seedFromString(`${judge.model}:${judge.promptVersion}:${rubricHash}`);
  const ci = bootstrapKappaCi(humanLabels, judgeLabels, confidence, seed);
  const n = ci.n;
  const calibrated = n >= minCases && ci.kappa >= judge.calibrationThreshold;

  const calibration = recordJudgeCalibration(ctx.db, {
    taskKey: judge.taskKey,
    judgeModel: judge.model,
    promptVersion: judge.promptVersion,
    rubricHash,
    mode: judge.mode,
    agreementKappa: ci.kappa,
    kappaCiLo: ci.lo,
    kappaCiHi: ci.hi,
    n,
    positionBiasRate: 0, // pointwise: no ordering, so no position bias to measure
    threshold: judge.calibrationThreshold,
    calibrated,
  });

  return {
    calibration,
    casesLabelled: cases.length,
    casesGraded: n,
    skippedNoCache,
    skippedUnparseable,
  };
}
