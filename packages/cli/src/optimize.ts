/**
 * `compound optimize <task> --candidate M [--reflection M] [--max-calls N] [--force]`
 *
 * Orchestrates GEPA prompt optimization (docs/optimization-v1.md): loads the
 * task's optimizer_train / optimizer_validation cases (NEVER the sealed set),
 * writes a job, invokes the Python `compound.optimize_product` entrypoint (which
 * drives the real gepa library and grades via `compound grade-batch`), and
 * stores the optimized prompt as an artifact. The optimized prompt is a
 * PROPOSAL — adopting it means re-gating it on the sealed set, a separate step.
 *
 * Eligibility: if a gate has been decided for the task, we only optimize when
 * the gap is worth closing (unless --force).
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadConfig } from "@compound/config";
import { type JudgeConfig, judgeTrust } from "@compound/judge";
import { assessEligibility } from "@compound/optimize";
import { listCases, listGateResults, recordOptimizationRun } from "@compound/storage";
import type { CommandEnvironment, CommandResult, ParsedArgs } from "./commands";
import { DEFAULT_CONFIG_PATH, DEFAULT_DATABASE_PATH } from "./commands";

function stringFlag(flags: ParsedArgs["flags"], name: string): string | undefined {
  const v = flags[name];
  return typeof v === "string" ? v : undefined;
}

interface ResolvedEndpoint {
  base_url: string;
  api_key_env: string;
}

function resolveEndpoint(
  config: ReturnType<typeof loadConfig>,
  modelId: string,
): ResolvedEndpoint | { error: string } {
  const all = [...(config.models?.frontier ?? []), ...(config.models?.candidates ?? [])];
  const entry = all.find((m) => m.id === modelId);
  if (entry === undefined) return { error: `model '${modelId}' is not in models` };
  const provider = config.providers?.[entry.provider];
  if (provider === undefined) return { error: `provider '${entry.provider}' is not configured` };
  return { base_url: provider.base_url, api_key_env: provider.api_key_env };
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

function toOpenAiTools(tools: unknown[] | null | undefined): unknown[] {
  return (tools ?? []).map((t) =>
    t !== null && typeof t === "object" && "function" in t ? t : { type: "function", function: t },
  );
}

/** The system prompt to seed from: an explicit flag, else the cases' own system message. */
function seedPromptFrom(cases: ReturnType<typeof listCases>, override?: string): string | null {
  if (override !== undefined) return override;
  for (const c of cases) {
    const input = c.input as { input?: Array<{ role?: string; content?: string }> };
    const system = input.input?.find((m) => m.role === "system");
    if (system?.content) return system.content;
  }
  return null;
}

export function runOptimizeCommand(args: ParsedArgs, env: CommandEnvironment): CommandResult {
  const taskKey = args.positional[0];
  const candidateModel = stringFlag(args.flags, "candidate");
  if (taskKey === undefined || candidateModel === undefined) {
    env.write(
      "error: usage: compound optimize <task> --candidate M [--reflection M] [--max-calls N] [--force]",
    );
    return { exitCode: 2 };
  }
  const reflectionModel = stringFlag(args.flags, "reflection") ?? candidateModel;
  const maxCalls = Number.parseInt(stringFlag(args.flags, "max-calls") ?? "30", 10);
  const config = loadConfig(stringFlag(args.flags, "config") ?? DEFAULT_CONFIG_PATH);

  const candidateEp = resolveEndpoint(config, candidateModel);
  const reflectionEp = resolveEndpoint(config, reflectionModel);
  if ("error" in candidateEp) {
    env.write(`error: ${candidateEp.error}`);
    return { exitCode: 1 };
  }
  if ("error" in reflectionEp) {
    env.write(`error: ${reflectionEp.error}`);
    return { exitCode: 1 };
  }

  const configPath = stringFlag(args.flags, "config") ?? DEFAULT_CONFIG_PATH;
  const dbPath = stringFlag(args.flags, "db") ?? DEFAULT_DATABASE_PATH;
  const db = env.openDatabase(dbPath);
  try {
    // Fuzzy task? If a judge is configured, GEPA must grade through it — and it
    // may only do so on earned trust. Refuse up front (with exactly what unblocks
    // it) rather than launch a run that grade-batch would reject mid-flight.
    const judge = resolveJudgeConfig(config, taskKey);
    const cap = stringFlag(args.flags, "cap");
    if (judge !== null) {
      const trust = judgeTrust(db, judge);
      if (!trust.calibrated) {
        env.write(
          `not optimizing: task '${taskKey}' is judge-graded but the judge is not calibrated.`,
        );
        env.write(`  ${trust.reason}`);
        env.write(
          "  label more judge_calibration cases, run `compound judge calibrate`, then retry.",
        );
        return { exitCode: 1 };
      }
      if (args.flags.paid !== true || cap === undefined || !(Number.parseFloat(cap) > 0)) {
        env.write(
          `error: judge-graded optimization spends judge tokens — pass --paid --cap <usd> ` +
            `(judge: ${judge.model}).`,
        );
        return { exitCode: 1 };
      }
      env.write(
        `judge-graded task: grading via calibrated judge ${judge.model} (${trust.reason}).`,
      );
    }

    // Eligibility: only optimize a gap worth closing (unless forced).
    let eligibilityReason = "forced";
    if (args.flags.force !== true) {
      const gate = listGateResults(db).find(
        (g) => g.spec.taskKey === taskKey && g.spec.candidateModel === candidateModel,
      );
      if (gate !== undefined) {
        const e = assessEligibility({ outcome: gate.result.outcome, delta: gate.result.delta });
        eligibilityReason = e.reason;
        if (!e.eligible) {
          env.write(`not optimizing: ${e.detail}. Use --force to override.`);
          return { exitCode: 0 };
        }
        env.write(`eligible: ${e.detail}`);
      }
    }

    const train = listCases(db, { taskKey, partition: "optimization_train", limit: 1000 });
    const val = listCases(db, { taskKey, partition: "optimizer_validation", limit: 1000 });
    if (train.length === 0 || val.length === 0) {
      env.write(
        `error: need optimization_train and optimizer_validation cases (have ${train.length}/${val.length}). Curate first.`,
      );
      return { exitCode: 1 };
    }
    const seedPrompt = seedPromptFrom([...train, ...val], stringFlag(args.flags, "seed-prompt"));
    if (seedPrompt === null) {
      env.write("error: no seed system prompt found in the cases; pass --seed-prompt");
      return { exitCode: 1 };
    }

    const toJobCase = (c: (typeof train)[number]) => {
      const input = c.input as { input?: unknown[]; tools_available?: unknown[] | null };
      return {
        case_id: c.caseId,
        messages: input.input ?? [],
        tools: toOpenAiTools(input.tools_available),
      };
    };

    const workDir = mkdtempSync(join(tmpdir(), "compound-optimize-"));
    const jobPath = join(workDir, "job.json");
    const outPath = join(workDir, "result.json");
    const job = {
      task_key: taskKey,
      candidate_model: candidateModel,
      candidate: candidateEp,
      reflection_model: reflectionModel,
      reflection: reflectionEp,
      seed_prompt: seedPrompt,
      trainset: train.map(toJobCase),
      valset: val.map(toJobCase),
      max_metric_calls: maxCalls,
      reflection_minibatch_size: 3,
      // The single grader the Python adapter shells back to. For a judge-graded
      // task it grades on the calibrated judge, so pass the money-safe judge
      // controls and the same db that holds its calibration + cache.
      grade_cmd: [
        "bun",
        "run",
        "packages/cli/src/main.ts",
        "grade-batch",
        "-",
        "--config",
        configPath,
        ...(judge !== null ? ["--judge", "--db", dbPath, "--paid", "--cap", cap as string] : []),
      ],
      run_dir: join(workDir, "gepa"),
      output: outPath,
    };
    writeFileSync(jobPath, JSON.stringify(job));

    env.write(
      `optimizing ${candidateModel} on '${taskKey}' (${train.length} train, ${val.length} val)...`,
    );
    try {
      execFileSync("uv", ["run", "python", "-m", "compound.optimize_product", jobPath], {
        stdio: ["ignore", "ignore", "inherit"],
      });
    } catch (error) {
      env.write(`error: optimization failed: ${error instanceof Error ? error.message : error}`);
      return { exitCode: 1 };
    }

    const result = JSON.parse(readFileSync(outPath, "utf8")) as {
      optimized_prompt: string;
      before_val_score: number;
      after_val_score: number;
      val_cases: number;
      reflection_calls: number;
    };
    const run = recordOptimizationRun(db, {
      taskKey,
      candidateModel,
      seedPrompt,
      optimizedPrompt: result.optimized_prompt,
      beforeValScore: result.before_val_score,
      afterValScore: result.after_val_score,
      valCases: result.val_cases,
      reflectionCalls: result.reflection_calls,
      eligibilityReason,
    });

    const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
    env.write(`\noptimization artifact ${run.id}`);
    env.write(
      `  validation score: ${pct(result.before_val_score)} -> ${pct(result.after_val_score)} (${result.val_cases} cases)`,
    );
    env.write(`  reflection calls: ${result.reflection_calls}`);
    env.write("  the optimized prompt is a PROPOSAL — re-gate it before adopting:");
    env.write(
      `  compound gate ${taskKey} --candidate ${candidateModel} --reference <M> ` +
        `--prompt-artifact ${run.id} --reason "adoption re-gate"`,
    );
    return { exitCode: 0 };
  } finally {
    db.close();
  }
}
