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

/**
 * CI exit code for a verdict, used only in `eval` mode (see runEvalCommand).
 * 0 = the candidate cleared the bar; 1 = it did not; 2 = the gate could not
 * decide (too few paired cases, or the judge abstained) — a distinct code so
 * CI can tell "regressed" from "couldn't tell yet".
 */
export function verdictExitCode(outcome: string): number {
  switch (outcome) {
    case "meets_gate":
      return 0;
    case "fails_gate":
    case "no_reliable_improvement":
      return 1;
    default:
      // insufficient_data, judge_abstained, or anything unrecognized.
      return 2;
  }
}

/** The config's free-form gate metric ("task_success") mapped to a decidable one. */
export function configGateMetric(raw: string | undefined): GateMetric | undefined {
  if (raw === "pass_rate" || raw === "mean_score") return raw;
  // "task_success" and other binary-success labels decide on the pass rate.
  if (raw !== undefined) return "pass_rate";
  return undefined;
}

export interface GateCommandOptions {
  /**
   * In `eval` mode the exit code reflects the VERDICT (0 meets / 1 fails /
   * 2 undecidable) so a gate can sit in CI. The plain `gate` command instead
   * exits 0 whenever the decision was produced — the verdict is the output,
   * not the process status.
   */
  exitOnVerdict?: boolean;
  /** Fall back to `config.gate` (metric, max_regression) for unset flags. */
  useConfigDefaults?: boolean;
  /** A standing reason to open the seal with when `--reason` is omitted. */
  defaultReason?: string;
}

export async function runGateCommand(
  args: ParsedArgs,
  env: CommandEnvironment,
  opts: GateCommandOptions = {},
): Promise<CommandResult> {
  const taskKey = args.positional[0];
  const candidateModel = stringFlag(args.flags, "candidate");
  const referenceModel = stringFlag(args.flags, "reference");
  const command = opts.exitOnVerdict ? "eval" : "gate";

  if (taskKey === undefined || candidateModel === undefined || referenceModel === undefined) {
    env.write(
      `error: usage: compound ${command} <task_key> --candidate M --reference M --reason "..." ` +
        "[--margin 0.05] [--confidence 0.95] [--min-cases 20] [--metric pass_rate|mean_score] " +
        "[--mode non_inferiority|superiority] [--prompt-artifact <optimization_run_id>] " +
        "[--provider P | --candidate-provider P --reference-provider P] [--force] [--paid --cap USD]",
    );
    return { exitCode: 2 };
  }

  // The seal is only opened with a stated reason. In `eval` mode a standing
  // reason (defaultReason) stands in so a CI job need not invent one each run.
  const reason = stringFlag(args.flags, "reason") ?? opts.defaultReason;
  if (reason === undefined || reason.trim().length === 0) {
    env.write(
      "error: --reason is required — the sealed decision set is only opened with a stated reason",
    );
    return { exitCode: 2 };
  }

  const config = loadConfig(stringFlag(args.flags, "config") ?? DEFAULT_CONFIG_PATH);
  const assertions = (config.assertions?.[taskKey] ?? []) as Assertion[];

  // In `eval` mode, unset metric/margin fall back to the declared gate policy
  // in compound.yaml, so the CI bar lives in config rather than the command line.
  const configGate = opts.useConfigDefaults ? config.gate : undefined;
  const metric = (stringFlag(args.flags, "metric") ??
    configGateMetric(configGate?.metric) ??
    "pass_rate") as GateMetric;
  const mode = (stringFlag(args.flags, "mode") ?? "non_inferiority") as GateMode;
  const marginDefault = configGate?.max_regression ?? (mode === "superiority" ? 0 : 0.05);
  const margin = numberFlag(args.flags, "margin", marginDefault);
  const confidence = numberFlag(args.flags, "confidence", 0.95);
  const minCases = numberFlag(args.flags, "min-cases", 20);
  const judgeAbstainMax = numberFlag(args.flags, "judge-abstain-max", 0);

  // The provider axis: --provider applies to both sides; per-side flags win.
  const bothProvider = stringFlag(args.flags, "provider");
  const candidateProvider = stringFlag(args.flags, "candidate-provider") ?? bothProvider;
  const referenceProvider = stringFlag(args.flags, "reference-provider") ?? bothProvider;

  let candidate: ReturnType<typeof resolveModel>;
  let reference: ReturnType<typeof resolveModel>;
  try {
    candidate = resolveModel(config, candidateModel, { provider: candidateProvider });
    reference = resolveModel(config, referenceModel, { provider: referenceProvider });
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
      wireModel: resolved.wireModel,
      provider: resolved.provider,
      providerName: resolved.providerName,
      price: resolved.price,
      transport: resolved.transport,
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
    // A dry run previews the verdict WITHOUT opening the seal or recording it;
    // only a deliberate (paid) run opens the sealed set and persists a decision.
    if (wantsPaid) {
      env.write(`opening the sealed decision set for '${taskKey}': ${reason}`);
    } else {
      env.write(
        `preview (dry run) for '${taskKey}': ${reason} — computing the gate without ` +
          "opening the seal or recording a verdict; re-run with --paid --cap to decide.",
      );
    }
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

    const {
      result,
      pairs,
      priorDecisions: prior,
    } = decideGate(db, {
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
      // Only a paid, deliberate run persists the spec + verdict; a dry run is a
      // side-effect-free preview (issue #20).
      persist: wantsPaid,
      // The peeking guard (#22): block a repeat decision after an adoption when
      // the task's gate policy opts in; --force overrides with a stated reason.
      blockRepeatAfterAdoption: config.gate?.block_repeat_after_adoption === true,
      force: args.flags.force === true,
      candidateExperimentId: candidateRun.experimentId,
      referenceExperimentId: referenceRun.experimentId,
      ...(candidatePromptHash !== undefined ? { candidatePromptHash } : {}),
      ...(promptArtifactId !== undefined ? { optimizationRunId: promptArtifactId } : {}),
      ...(candidateProvider !== undefined ? { candidateProvider } : {}),
      ...(referenceProvider !== undefined ? { referenceProvider } : {}),
    });

    // Peeking warning (#22): this sealed set has been decided before. On a paid
    // decision it means a real re-examination; on a preview it's a heads-up.
    if (prior.count > 0) {
      const first = prior.firstDecidedAt?.toISOString() ?? "an earlier run";
      const lead = wantsPaid ? "warning" : "note";
      env.write(
        `${lead}: the sealed decision set for '${taskKey}' has already been decided ` +
          `${prior.count}× (first ${first}` +
          (prior.adoptionCount > 0 ? `, ${prior.adoptionCount} adoption` : "") +
          "); each re-decision on the same held-out set weakens its guarantee.",
      );
    }

    const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
    env.write("");
    const verdictLabel = wantsPaid ? "GATE" : "GATE (preview)";
    env.write(`${verdictLabel}: ${OUTCOME_LABEL[result.outcome] ?? result.outcome}`);
    env.write(`  task:        ${taskKey}   metric: ${metric}   mode: ${mode}`);
    const onProvider = (r: ReturnType<typeof resolveModel>) => ` @${r.providerName}`;
    env.write(
      `  candidate:   ${candidateModel}${onProvider(candidate)}   ${pct(result.candidateRate)}`,
    );
    env.write(
      `  reference:   ${referenceModel}${onProvider(reference)}   ${pct(result.referenceRate)}`,
    );
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

    if (opts.exitOnVerdict) {
      const code = verdictExitCode(result.outcome);
      env.write(
        `\neval verdict: ${OUTCOME_LABEL[result.outcome] ?? result.outcome} (exit ${code})`,
      );
      return { exitCode: code };
    }
    return { exitCode: 0 };
  } catch (error) {
    if (error instanceof GateInputError) {
      env.write(`error: ${error.message}`);
      return { exitCode: 1 };
    }
    env.write(`error: ${command} failed: ${error instanceof Error ? error.message : error}`);
    return { exitCode: 1 };
  } finally {
    db.close();
  }
}

/**
 * `compound eval <task_key> --candidate M --reference M` — the CI entry point.
 *
 * It is the SAME sealed non-inferiority decision as `gate`, with two changes for
 * automation: the process exit code reflects the verdict (0 meets / 1 regresses /
 * 2 undecidable), and the metric and margin default from the `gate:` policy in
 * compound.yaml so the bar lives in version control, not in a CI script.
 *
 * A note on the seal: running this on every PR does re-examine the held-out set,
 * which weakens its statistical guarantee over time. It is meant as a release
 * gate on a candidate you intend to adopt, not a per-commit check.
 */
export async function runEvalCommand(
  args: ParsedArgs,
  env: CommandEnvironment,
): Promise<CommandResult> {
  return runGateCommand(args, env, {
    exitOnVerdict: true,
    useConfigDefaults: true,
    defaultReason: "CI gate check",
  });
}
