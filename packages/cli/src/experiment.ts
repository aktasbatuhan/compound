/**
 * `compound experiment <task_key> <model>` — run a candidate against a task.
 *
 * Money-safe by default: without `--paid`, it is a dry run that makes zero
 * provider calls and reports what a paid run would cost. `--paid` requires the
 * config to have `budget.paid_runs_enabled: true` and a positive hard limit,
 * and a `--cap` per-run ceiling.
 */
import type { Assertion } from "@compound/assertions";
import { loadConfig } from "@compound/config";
import {
  DecisionPartitionRefusedError,
  ExecutionConfigError,
  moneyControls,
  resolveModel,
  runExperiment,
} from "@compound/execution";
import type { CasePartition } from "@compound/storage";
import type { CommandEnvironment, CommandResult, ParsedArgs } from "./commands";
import { DEFAULT_CONFIG_PATH, DEFAULT_DATABASE_PATH } from "./commands";

function stringFlag(flags: ParsedArgs["flags"], name: string): string | undefined {
  const value = flags[name];
  return typeof value === "string" ? value : undefined;
}

const PARTITIONS: CasePartition[] = [
  "optimization_train",
  "optimizer_validation",
  "judge_calibration",
  "decision_test",
];

export async function runExperimentCommand(
  args: ParsedArgs,
  env: CommandEnvironment,
): Promise<CommandResult> {
  const taskKey = args.positional[0];
  const modelId = args.positional[1];
  if (taskKey === undefined || modelId === undefined) {
    env.write(
      "error: usage: compound experiment <task_key> <model> [--provider P] [--partition P] [--paid --cap N]",
    );
    return { exitCode: 2 };
  }

  const providerOverride = stringFlag(args.flags, "provider");

  const partition = (stringFlag(args.flags, "partition") ??
    "optimizer_validation") as CasePartition;
  if (!PARTITIONS.includes(partition)) {
    env.write(`error: unknown partition '${partition}'; one of ${PARTITIONS.join(", ")}`);
    return { exitCode: 2 };
  }

  const config = loadConfig(stringFlag(args.flags, "config") ?? DEFAULT_CONFIG_PATH);

  let resolved: ReturnType<typeof resolveModel>;
  try {
    resolved = resolveModel(config, modelId, { provider: providerOverride });
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

  if (wantsPaid) {
    if (!controls.paidRunsEnabled) {
      env.write("error: --paid requires budget.paid_runs_enabled: true in compound.yaml");
      return { exitCode: 1 };
    }
    if (controls.globalHardLimitUsd <= 0) {
      env.write("error: --paid requires a positive budget.hard_limit_usd in compound.yaml");
      return { exitCode: 1 };
    }
    if (!(experimentCapUsd > 0)) {
      env.write("error: --paid requires a positive --cap <usd> per-run ceiling");
      return { exitCode: 1 };
    }
  }

  const db = env.openDatabase(stringFlag(args.flags, "db") ?? DEFAULT_DATABASE_PATH);
  try {
    const { report } = await runExperiment(db, {
      taskKey,
      candidateModel: modelId,
      provider: resolved.provider,
      providerName: resolved.providerName,
      price: resolved.price,
      // The config schema is loose on assertion params (the engine owns the
      // exact shapes); the engine validates per-type at evaluation.
      assertions: (config.assertions?.[taskKey] ?? []) as Assertion[],
      partition,
      paidRunsEnabled: wantsPaid,
      experimentCapUsd,
      globalHardLimitUsd: controls.globalHardLimitUsd,
      dryRun: !wantsPaid,
      ...(stringFlag(args.flags, "max") !== undefined
        ? { maxCases: Number.parseInt(stringFlag(args.flags, "max") as string, 10) }
        : {}),
    });

    const paid = wantsPaid;
    env.write(`experiment: ${modelId} on task ${taskKey} (${partition})`);
    env.write(`  provider:     ${resolved.providerName}`);
    env.write(`  mode:         ${paid ? "PAID" : "dry run (no provider calls)"}`);
    env.write(`  cases:        ${report.cases_total ?? 0}`);
    env.write(`  graded:       ${report.cases_graded ?? 0}`);
    env.write(`  cache hits:   ${report.cache_hits ?? 0}`);
    env.write(`  provider calls: ${report.provider_calls ?? 0}`);
    if ((report.cases_graded ?? 0) > 0) {
      env.write(`  pass rate:    ${((report.pass_rate ?? 0) * 100).toFixed(1)}%`);
      env.write(`  mean score:   ${(report.mean_score ?? 0).toFixed(3)}`);
    } else if (!paid) {
      env.write("  (dry run: no cached outputs to grade yet — run with --paid to execute)");
    }
    env.write(
      paid
        ? `  actual cost:  $${(report.actual_cost_usd ?? 0).toFixed(6)}`
        : `  est. cost:    $${(report.estimated_cost_usd ?? 0).toFixed(6)} if run with --paid`,
    );
    return { exitCode: 0 };
  } catch (error) {
    if (error instanceof DecisionPartitionRefusedError) {
      env.write(`error: ${error.message}`);
      return { exitCode: 1 };
    }
    env.write(`error: experiment failed: ${error instanceof Error ? error.message : error}`);
    return { exitCode: 1 };
  } finally {
    db.close();
  }
}
