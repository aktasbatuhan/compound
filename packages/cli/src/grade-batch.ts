/**
 * `compound grade-batch <file>` — grade a batch of model outputs and print JSON.
 * This is the SINGLE grader the Python GEPA adapter calls back into
 * (docs/optimization-v1.md), so grading never forks across languages: GEPA
 * proposes prompts, this scores them with the exact same `@compound/assertions`
 * (and, for fuzzy tasks, the same calibrated judge) the gate uses.
 *
 * Two tiers, matching the gate (docs/judges-v1.md):
 * - Assertions run first and for free. A structurally broken output fails here
 *   and never costs a judge token; its feedback is the failing checks.
 * - With `--judge`, outputs that PASS assertions are then scored by the task's
 *   judge for quality an assertion can't capture. The judge grades ONLY on
 *   earned trust: if it is uncalibrated, grade-batch REFUSES the whole batch
 *   (non-zero exit) rather than let optimization chase an untrusted opinion.
 *
 * Input  JSON: { "task_key": "...", "items": [ { "case_id": "...", "output": <Message> } ] }
 * Output JSON: { "items": [ { "case_id", "passed", "score", "feedback" } ] }
 */
import { readFileSync } from "node:fs";
import { type Assertion, evaluateAssertions } from "@compound/assertions";
import { loadConfig } from "@compound/config";
import type { Message } from "@compound/contract";
import { ExecutionConfigError, moneyControls, resolveModel } from "@compound/execution";
import {
  type JudgeConfig,
  type JudgeExecutionContext,
  judgeGradeBatch,
  messageText,
} from "@compound/judge";
import type { CommandEnvironment, CommandResult, ParsedArgs } from "./commands";
import { DEFAULT_CONFIG_PATH, DEFAULT_DATABASE_PATH } from "./commands";

interface GradeItem {
  case_id: string;
  output: Message | null;
}

interface GradedRow {
  case_id: string;
  passed: boolean;
  score: number;
  feedback: string;
}

function stringFlag(flags: ParsedArgs["flags"], name: string): string | undefined {
  const value = flags[name];
  return typeof value === "string" ? value : undefined;
}

function assertionFeedback(report: ReturnType<typeof evaluateAssertions>): string {
  const failed = report.results
    .filter((r) => !r.passed)
    .map((r) => `${r.type}: ${r.detail}`)
    .join("; ");
  return failed.length > 0 ? failed : "all checks passed";
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

export async function runGradeBatchCommand(
  args: ParsedArgs,
  env: CommandEnvironment,
): Promise<CommandResult> {
  const path = args.positional[0];
  if (path === undefined) {
    env.write(
      "error: usage: compound grade-batch <file|-> [--judge] (reads {task_key, items:[{case_id, output}]})",
    );
    return { exitCode: 2 };
  }

  let raw: string;
  try {
    raw = readFileSync(path === "-" ? 0 : path, "utf8");
  } catch (error) {
    env.write(`error: could not read ${path}: ${error instanceof Error ? error.message : error}`);
    return { exitCode: 1 };
  }

  let request: { task_key?: string; items?: GradeItem[] };
  try {
    request = JSON.parse(raw);
  } catch {
    env.write("error: input must be a JSON object");
    return { exitCode: 1 };
  }
  const taskKey = request.task_key;
  const items = request.items;
  if (typeof taskKey !== "string" || !Array.isArray(items)) {
    env.write('error: input needs a string "task_key" and an "items" array');
    return { exitCode: 1 };
  }

  const config = loadConfig(stringFlag(args.flags, "config") ?? DEFAULT_CONFIG_PATH);
  const assertions = (config.assertions?.[taskKey] ?? []) as Assertion[];

  // Tier 1: assertions, for every item, for free.
  const assessed = items.map((item) => ({
    item,
    report: evaluateAssertions(assertions, { output: item.output ?? null }),
  }));

  if (args.flags.judge !== true) {
    const graded: GradedRow[] = assessed.map(({ item, report }) => ({
      case_id: item.case_id,
      passed: report.passed,
      score: report.score,
      feedback: assertionFeedback(report),
    }));
    env.write(JSON.stringify({ items: graded }));
    return { exitCode: 0 };
  }

  // Tier 2: the judge, on outputs that passed assertions and on earned trust.
  const judge = resolveJudgeConfig(config, taskKey);
  if (judge === null) {
    env.write(
      `error: --judge given but no judge configured for task '${taskKey}' (add judges.${taskKey})`,
    );
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

  const controls = moneyControls(config);
  const wantsPaid = args.flags.paid === true;
  const cap = stringFlag(args.flags, "cap");
  const experimentCapUsd = cap !== undefined ? Number.parseFloat(cap) : 0;
  if (
    wantsPaid &&
    (!controls.paidRunsEnabled || controls.globalHardLimitUsd <= 0 || !(experimentCapUsd > 0))
  ) {
    env.write(
      "error: --paid needs budget.paid_runs_enabled, a positive budget.hard_limit_usd, and --cap <usd>",
    );
    return { exitCode: 1 };
  }

  const db = env.openDatabase(stringFlag(args.flags, "db") ?? DEFAULT_DATABASE_PATH);
  const minCalCases = stringFlag(args.flags, "min-calibration-cases");
  try {
    const ctx: JudgeExecutionContext = {
      db,
      provider: resolved.provider,
      providerName: resolved.providerName,
      judgeModel: judge.model,
      price: resolved.price,
      paid: wantsPaid,
      experimentCapUsd,
      globalHardLimitUsd: controls.globalHardLimitUsd,
    };

    const toJudge = assessed
      .filter((a) => a.report.passed)
      .map((a) => ({ caseId: a.item.case_id, responseText: messageText(a.item.output ?? null) }));

    const batch = await judgeGradeBatch(
      ctx,
      judge,
      toJudge,
      minCalCases !== undefined ? Number.parseInt(minCalCases, 10) : undefined,
    );

    // Earned trust: an uncalibrated judge grades nothing. Refuse the batch so
    // optimization never chases an untrusted opinion.
    if (!batch.trust.calibrated) {
      env.write(`error: judge for '${taskKey}' is not calibrated — ${batch.trust.reason}.`);
      env.write(
        "       label more judge_calibration cases and run `compound judge calibrate` first.",
      );
      return { exitCode: 3 };
    }

    // A verdict is required for every judged output; a cache miss (dry run) or an
    // unparseable judge reply leaves a hole that would corrupt the score vector.
    const misses = batch.verdicts.filter((v) => v.status === "cache_miss_dry_run").length;
    const unparseable = batch.verdicts.filter((v) => v.status === "unparseable").length;
    if (misses > 0) {
      env.write(
        `error: ${misses} output(s) not yet judged (cache miss) — run judge grading with --paid --cap.`,
      );
      return { exitCode: 3 };
    }
    if (unparseable > 0) {
      env.write(`error: the judge returned an unparseable reply on ${unparseable} output(s).`);
      return { exitCode: 3 };
    }

    const byCase = new Map(batch.verdicts.map((v) => [v.caseId, v]));
    const graded: GradedRow[] = assessed.map(({ item, report }) => {
      if (!report.passed) {
        // Failed the cheap gate — a fail, no judge token spent.
        return {
          case_id: item.case_id,
          passed: false,
          score: report.score,
          feedback: assertionFeedback(report),
        };
      }
      const verdict = byCase.get(item.case_id);
      const score = verdict?.score ?? 0;
      return {
        case_id: item.case_id,
        passed: score >= judge.decisionPoint,
        score,
        feedback: verdict?.reasoning?.trim() || "judged",
      };
    });

    env.write(JSON.stringify({ items: graded }));
    return { exitCode: 0 };
  } finally {
    db.close();
  }
}
