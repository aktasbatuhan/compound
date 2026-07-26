/**
 * `compound gate <task_key> --candidate M --reference M --reason "..."` — the
 * verdict: does the candidate meet a pre-declared non-inferiority bar against the
 * reference on the task's SEALED decision partition (docs/gate-decision-v1.md)?
 *
 * Opening the sealed partition requires a stated `--reason` (the firewall). The
 * two runs go through the money-safe runner: without `--paid` they are dry runs,
 * so the gate can only decide when both runs' completions are already cached
 * (a re-decision then costs $0).
 */
import { createHash } from "node:crypto";
import type { Assertion } from "@compound/assertions";
import { loadConfig } from "@compound/config";
import {
  ExecutionConfigError,
  moneyControls,
  resolveModel,
  runExperiment,
} from "@compound/execution";
import { decideGate, GateInputError } from "@compound/gate";
import { type GateMetric, type GateMode, getOptimizationRun } from "@compound/storage";
import type { CommandEnvironment, CommandResult, ParsedArgs } from "./commands";
import { DEFAULT_CONFIG_PATH, DEFAULT_DATABASE_PATH } from "./commands";

function stringFlag(flags: ParsedArgs["flags"], name: string): string | undefined {
  const value = flags[name];
  return typeof value === "string" ? value : undefined;
}

function numberFlag(flags: ParsedArgs["flags"], name: string, fallback: number): number {
  const value = stringFlag(flags, name);
  if (value === undefined) return fallback;
  const n = Number.parseFloat(value);
  return Number.isFinite(n) ? n : fallback;
}

const OUTCOME_LABEL: Record<string, string> = {
  meets_gate: "MEETS GATE",
  fails_gate: "FAILS GATE",
  insufficient_data: "INSUFFICIENT DATA",
  judge_abstained: "JUDGE ABSTAINED",
  no_reliable_improvement: "NO RELIABLE IMPROVEMENT",
};

export async function runGateCommand(
  args: ParsedArgs,
  env: CommandEnvironment,
): Promise<CommandResult> {
  const taskKey = args.positional[0];
  const candidateModel = stringFlag(args.flags, "candidate");
  const referenceModel = stringFlag(args.flags, "reference");
  const reason = stringFlag(args.flags, "reason");

  if (taskKey === undefined || candidateModel === undefined || referenceModel === undefined) {
    env.write(
      'error: usage: compound gate <task_key> --candidate M --reference M --reason "..." ' +
        "[--margin 0.05] [--confidence 0.95] [--min-cases 20] [--metric pass_rate|mean_score] " +
        "[--mode non_inferiority|superiority] [--prompt-artifact <optimization_run_id>] " +
        "[--paid --cap USD]",
    );
    return { exitCode: 2 };
  }
  if (reason === undefined || reason.trim().length === 0) {
    env.write(
      "error: --reason is required — the sealed decision set is only opened with a stated reason",
    );
    return { exitCode: 2 };
  }

  const metric = (stringFlag(args.flags, "metric") ?? "pass_rate") as GateMetric;
  const mode = (stringFlag(args.flags, "mode") ?? "non_inferiority") as GateMode;
  const margin = numberFlag(args.flags, "margin", mode === "superiority" ? 0 : 0.05);
  const confidence = numberFlag(args.flags, "confidence", 0.95);
  const minCases = numberFlag(args.flags, "min-cases", 20);
  const judgeAbstainMax = numberFlag(args.flags, "judge-abstain-max", 0);

  const config = loadConfig(stringFlag(args.flags, "config") ?? DEFAULT_CONFIG_PATH);
  const assertions = (config.assertions?.[taskKey] ?? []) as Assertion[];

  let candidate: ReturnType<typeof resolveModel>;
  let reference: ReturnType<typeof resolveModel>;
  try {
    candidate = resolveModel(config, candidateModel);
    reference = resolveModel(config, referenceModel);
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

  // Adoption re-gate: run the candidate WITH an optimized prompt from a stored
  // optimization artifact. The prompt joins the pre-declared rule (its hash),
  // and the reference always runs untouched.
  const promptArtifactId = stringFlag(args.flags, "prompt-artifact");
  let promptOverride: string | undefined;
  let candidatePromptHash: string | undefined;
  if (promptArtifactId !== undefined) {
    const artifact = getOptimizationRun(db, promptArtifactId);
    if (artifact === null) {
      env.write(`error: optimization artifact '${promptArtifactId}' not found`);
      db.close();
      return { exitCode: 1 };
    }
    if (artifact.taskKey !== taskKey) {
      env.write(
        `error: optimization artifact '${promptArtifactId}' belongs to task ` +
          `'${artifact.taskKey}', not '${taskKey}'`,
      );
      db.close();
      return { exitCode: 1 };
    }
    if (artifact.candidateModel !== candidateModel) {
      env.write(
        `warning: artifact prompt was optimized for '${artifact.candidateModel}', ` +
          `gating '${candidateModel}' with it anyway`,
      );
    }
    promptOverride = artifact.optimizedPrompt;
    candidatePromptHash = `sha256:${createHash("sha256").update(artifact.optimizedPrompt).digest("hex")}`;
  }

  const maxCases = stringFlag(args.flags, "max");
  const runOne = (
    resolved: ReturnType<typeof resolveModel>,
    model: string,
    systemPromptOverride?: string,
  ) =>
    runExperiment(db, {
      taskKey,
      candidateModel: model,
      provider: resolved.provider,
      providerName: resolved.providerName,
      price: resolved.price,
      assertions,
      partition: "decision_test",
      allowDecisionTest: true,
      paidRunsEnabled: wantsPaid,
      experimentCapUsd,
      globalHardLimitUsd: controls.globalHardLimitUsd,
      dryRun: !wantsPaid,
      ...(systemPromptOverride !== undefined ? { systemPromptOverride } : {}),
      ...(maxCases !== undefined ? { maxCases: Number.parseInt(maxCases, 10) } : {}),
    });

  try {
    env.write(`opening the sealed decision set for '${taskKey}': ${reason}`);
    if (promptArtifactId !== undefined) {
      env.write(`optimized prompt under test: artifact ${promptArtifactId}`);
    }
    const candidateRun = await runOne(candidate, candidateModel, promptOverride);
    const referenceRun = await runOne(reference, referenceModel);

    if ((candidateRun.report.cases_graded ?? 0) === 0 && !wantsPaid) {
      env.write(
        "note: no cached completions on the decision set yet — run with --paid --cap to execute, " +
          "then re-run this gate ($0 from cache).",
      );
    }

    const { result, pairs } = decideGate(db, {
      taskKey,
      candidateModel,
      referenceModel,
      metric,
      mode,
      margin,
      confidence,
      minCases,
      judgeAbstainMax,
      firewallReason: reason,
      candidateExperimentId: candidateRun.experimentId,
      referenceExperimentId: referenceRun.experimentId,
      ...(candidatePromptHash !== undefined ? { candidatePromptHash } : {}),
      ...(promptArtifactId !== undefined ? { optimizationRunId: promptArtifactId } : {}),
    });

    const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
    env.write("");
    env.write(`GATE: ${OUTCOME_LABEL[result.outcome] ?? result.outcome}`);
    env.write(`  task:        ${taskKey}   metric: ${metric}   mode: ${mode}`);
    env.write(`  candidate:   ${candidateModel}   ${pct(result.candidateRate)}`);
    env.write(`  reference:   ${referenceModel}   ${pct(result.referenceRate)}`);
    env.write(
      `  delta:       ${result.delta >= 0 ? "+" : ""}${(result.delta * 100).toFixed(1)}pp ` +
        `(candidate − reference)`,
    );
    env.write(
      `  ${Math.round(confidence * 100)}% CI:      ` +
        `[${(result.ciLo * 100).toFixed(1)}pp, ${(result.ciHi * 100).toFixed(1)}pp]`,
    );
    env.write(`  cases:       ${result.n} paired (min ${minCases})`);
    if (mode === "non_inferiority") {
      env.write(
        `  margin:      ${(-margin * 100).toFixed(1)}pp (candidate may be this much worse)`,
      );
    }

    const disagree = pairs.filter((p) => p.candidatePassed !== p.referencePassed).slice(0, 5);
    if (disagree.length > 0) {
      env.write("\ndisagreements (candidate vs reference):");
      for (const p of disagree) {
        env.write(
          `  ${p.caseId}: candidate ${p.candidatePassed ? "pass" : "fail"}, ` +
            `reference ${p.referencePassed ? "pass" : "fail"}`,
        );
      }
    }
    return { exitCode: 0 };
  } catch (error) {
    if (error instanceof GateInputError) {
      env.write(`error: ${error.message}`);
      return { exitCode: 1 };
    }
    env.write(`error: gate failed: ${error instanceof Error ? error.message : error}`);
    return { exitCode: 1 };
  } finally {
    db.close();
  }
}
