/**
 * `compound judge calibrate <task>` and `compound judge grade <task> <experiment>`.
 *
 * Calibrate measures the judge against the human labels in the judge_calibration
 * partition and records whether it may be trusted. Grade runs a calibrated judge
 * over an experiment's outputs and writes the verdicts into the rows the gate
 * reads; an uncalibrated judge abstains (docs/judges-v1.md). Both are money-safe:
 * without --paid they make no provider calls and only use cached judge grades.
 */
import { loadConfig } from "@compound/config";
import { ExecutionConfigError, moneyControls, resolveModel } from "@compound/execution";
import {
  calibrateJudge,
  gradeExperimentWithJudge,
  type JudgeConfig,
  JudgeGradeError,
} from "@compound/judge";
import type { CommandEnvironment, CommandResult, ParsedArgs } from "./commands";
import { DEFAULT_CONFIG_PATH, DEFAULT_DATABASE_PATH } from "./commands";

function stringFlag(flags: ParsedArgs["flags"], name: string): string | undefined {
  const value = flags[name];
  return typeof value === "string" ? value : undefined;
}

function resolveJudgeConfig(
  config: ReturnType<typeof loadConfig>,
  taskKey: string,
): JudgeConfig | null {
  const raw = config.judges?.[taskKey];
  if (raw === undefined) return null;
  return {
    taskKey,
    model: raw.model,
    promptVersion: raw.prompt_version,
    rubric: raw.rubric,
    mode: raw.mode ?? "pointwise",
    calibrationThreshold: raw.calibration_threshold,
    decisionPoint: raw.decision_point ?? 0.5,
  };
}

interface Money {
  paidRunsEnabled: boolean;
  globalHardLimitUsd: number;
  experimentCapUsd: number;
  wantsPaid: boolean;
}

function money(args: ParsedArgs, config: ReturnType<typeof loadConfig>): Money | { error: string } {
  const controls = moneyControls(config);
  const wantsPaid = args.flags.paid === true;
  const cap = stringFlag(args.flags, "cap");
  const experimentCapUsd = cap !== undefined ? Number.parseFloat(cap) : 0;
  if (wantsPaid) {
    if (!controls.paidRunsEnabled) {
      return { error: "--paid requires budget.paid_runs_enabled: true in compound.yaml" };
    }
    if (controls.globalHardLimitUsd <= 0) {
      return { error: "--paid requires a positive budget.hard_limit_usd in compound.yaml" };
    }
    if (!(experimentCapUsd > 0)) {
      return { error: "--paid requires a positive --cap <usd> per-run ceiling" };
    }
  }
  return { ...controls, experimentCapUsd, wantsPaid };
}

export async function runJudgeCommand(
  args: ParsedArgs,
  env: CommandEnvironment,
): Promise<CommandResult> {
  const sub = args.positional[0];
  const taskKey = args.positional[1];
  if ((sub !== "calibrate" && sub !== "grade") || taskKey === undefined) {
    env.write(
      "error: usage: compound judge calibrate <task> [--paid --cap USD]\n" +
        "              compound judge grade <task> <experiment_id> [--paid --cap USD]",
    );
    return { exitCode: 2 };
  }

  const config = loadConfig(stringFlag(args.flags, "config") ?? DEFAULT_CONFIG_PATH);
  const judge = resolveJudgeConfig(config, taskKey);
  if (judge === null) {
    env.write(`error: no judge configured for task '${taskKey}' (add judges.${taskKey})`);
    return { exitCode: 1 };
  }

  let resolved: ReturnType<typeof resolveModel>;
  try {
    resolved = resolveModel(config, judge.model);
  } catch (error) {
    if (error instanceof ExecutionConfigError) {
      env.write(`error: ${error.message}`);
      return { exitCode: 1 };
    }
    throw error;
  }

  const m = money(args, config);
  if ("error" in m) {
    env.write(`error: ${m.error}`);
    return { exitCode: 1 };
  }

  const db = env.openDatabase(stringFlag(args.flags, "db") ?? DEFAULT_DATABASE_PATH);
  const ctx = {
    db,
    provider: resolved.provider,
    providerName: resolved.providerName,
    judgeModel: judge.model,
    price: resolved.price,
    paid: m.wantsPaid,
    experimentCapUsd: m.experimentCapUsd,
    globalHardLimitUsd: m.globalHardLimitUsd,
  };

  try {
    if (sub === "calibrate") {
      const r = await calibrateJudge(ctx, judge);
      const c = r.calibration;
      env.write(`judge calibration: ${taskKey} (${judge.model}, prompt ${judge.promptVersion})`);
      env.write(`  human-labelled cases: ${r.casesLabelled}`);
      env.write(`  judge-graded cases:   ${r.casesGraded}`);
      if (r.skippedNoCache > 0) {
        env.write(`  skipped (no cache):   ${r.skippedNoCache} — run with --paid --cap to grade`);
      }
      if (r.skippedUnparseable > 0) env.write(`  skipped (bad reply):  ${r.skippedUnparseable}`);
      env.write(
        `  agreement kappa:      ${c.agreementKappa.toFixed(3)} ` +
          `[${c.kappaCiLo.toFixed(3)}, ${c.kappaCiHi.toFixed(3)}]`,
      );
      env.write(`  threshold:            ${c.threshold}`);
      env.write(
        `  => ${c.calibrated ? "CALIBRATED (may feed a gate)" : "UNCALIBRATED (abstains)"}`,
      );
      return { exitCode: 0 };
    }

    const experimentId = args.positional[2];
    if (experimentId === undefined) {
      env.write("error: usage: compound judge grade <task> <experiment_id>");
      return { exitCode: 2 };
    }
    const s = await gradeExperimentWithJudge(ctx, judge, experimentId);
    env.write(`judge grade: ${taskKey} on experiment ${experimentId}`);
    env.write(`  trust: ${s.trust.calibrated ? "calibrated" : "UNCALIBRATED"} — ${s.trust.reason}`);
    env.write(`  cases considered: ${s.casesConsidered}`);
    env.write(`  judged:           ${s.judged}`);
    env.write(`  abstained:        ${s.abstained}`);
    if (s.skippedNoCache > 0) {
      env.write(`  skipped (no cache): ${s.skippedNoCache} — run with --paid --cap to grade`);
    }
    if (s.skippedUnparseable > 0) env.write(`  skipped (bad reply): ${s.skippedUnparseable}`);
    if (s.meanJudgeScore !== null) env.write(`  mean judge score: ${s.meanJudgeScore.toFixed(3)}`);
    return { exitCode: 0 };
  } catch (error) {
    if (error instanceof JudgeGradeError) {
      env.write(`error: ${error.message}`);
      return { exitCode: 1 };
    }
    env.write(`error: judge failed: ${error instanceof Error ? error.message : error}`);
    return { exitCode: 1 };
  } finally {
    db.close();
  }
}
