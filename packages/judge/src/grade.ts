/**
 * Grade an experiment's outputs with the judge and write the verdicts into
 * experiment_results — the exact rows the gate reads (docs/judges-v1.md +
 * docs/gate-decision-v1.md). This is what makes a judge-graded task gateable.
 *
 * Two rules from the design shape this:
 * - Only outputs that PASSED the cheap deterministic assertions are sent to the
 *   judge; a broken output is already a fail and never costs a judge token.
 * - If the judge is UNCALIBRATED it abstains: every graded case is marked
 *   `judgeAbstained`, so the gate excludes them and returns "judge abstained"
 *   rather than trusting an untrusted opinion. No judge calls are made then, so
 *   an untrusted judge never spends money.
 */

import type { CompletionResponse } from "@compound/execution";
import {
  getCachedCompletion,
  getExperiment,
  getExperimentResults,
  setCaseJudgment,
} from "@compound/storage";
import { judgeGradeOutput } from "./execute";
import { messageText } from "./prompt";
import { judgeTrust } from "./trust";
import type { JudgeConfig, JudgeExecutionContext, JudgeTrust } from "./types";

export interface JudgeGradeSummary {
  experimentId: string;
  trust: JudgeTrust;
  casesConsidered: number;
  judged: number;
  abstained: number;
  skippedNoCache: number;
  skippedUnparseable: number;
  meanJudgeScore: number | null;
}

export class JudgeGradeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "JudgeGradeError";
  }
}

export async function gradeExperimentWithJudge(
  ctx: JudgeExecutionContext,
  judge: JudgeConfig,
  experimentId: string,
  minCalibrationCases?: number,
): Promise<JudgeGradeSummary> {
  const experiment = getExperiment(ctx.db, experimentId);
  if (experiment === null) throw new JudgeGradeError(`experiment ${experimentId} not found`);
  if (experiment.taskKey !== judge.taskKey) {
    throw new JudgeGradeError(
      `experiment task '${experiment.taskKey}' does not match judge task '${judge.taskKey}'`,
    );
  }

  const trust = judgeTrust(ctx.db, judge, minCalibrationCases);
  const rows = getExperimentResults(ctx.db, experimentId);
  const graded = rows.filter((r) => r.status === "graded");

  // Uncalibrated: abstain on every graded case, make no judge calls.
  if (!trust.calibrated) {
    for (const r of graded) {
      setCaseJudgment(ctx.db, experimentId, r.caseId, {
        score: r.score ?? 0,
        passed: r.passed ?? false,
        judgeAbstained: true,
      });
    }
    return {
      experimentId,
      trust,
      casesConsidered: graded.length,
      judged: 0,
      abstained: graded.length,
      skippedNoCache: 0,
      skippedUnparseable: 0,
      meanJudgeScore: null,
    };
  }

  // Calibrated: judge the outputs that passed assertions; leave failures as-is.
  let judged = 0;
  let skippedNoCache = 0;
  let skippedUnparseable = 0;
  let scoreSum = 0;

  for (const r of graded) {
    if (r.passed !== true || r.completionFingerprint === null) continue;
    const completion = getCachedCompletion(ctx.db, r.completionFingerprint);
    if (completion === null) {
      skippedNoCache += 1;
      continue;
    }
    const responseText = messageText(completion.outputJson as CompletionResponse["output"]);
    const outcome = await judgeGradeOutput(ctx, judge.rubric, responseText);
    if (outcome.status === "cache_miss_dry_run") {
      skippedNoCache += 1;
      continue;
    }
    if (outcome.status === "unparseable") {
      skippedUnparseable += 1;
      continue;
    }
    setCaseJudgment(ctx.db, experimentId, r.caseId, {
      score: outcome.verdict.score,
      passed: outcome.verdict.score >= judge.decisionPoint,
      judgeAbstained: false,
    });
    judged += 1;
    scoreSum += outcome.verdict.score;
  }

  return {
    experimentId,
    trust,
    casesConsidered: graded.length,
    judged,
    abstained: 0,
    skippedNoCache,
    skippedUnparseable,
    meanJudgeScore: judged > 0 ? scoreSum / judged : null,
  };
}
