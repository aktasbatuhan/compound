/**
 * `compound init` — scaffold a starter compound.yaml (issue #5).
 *
 * Money-safe by construction: paid runs are OFF in the scaffold, so a fresh
 * checkout can import, curate, and dry-run without any chance of spend until the
 * user deliberately turns it on. If a database with imported traces is present,
 * the discovered task keys are pre-filled so the file matches the user's data
 * rather than a generic placeholder.
 *
 * It refuses to overwrite an existing config without --force: the config is
 * hand-tuned evidence policy, not a throwaway.
 */
import { existsSync, writeFileSync } from "node:fs";
import { countTracesByTaskKey } from "@compound/storage";
import type { CommandEnvironment, CommandResult, ParsedArgs } from "./commands";
import { DEFAULT_CONFIG_PATH, DEFAULT_DATABASE_PATH } from "./commands";

function stringFlag(flags: ParsedArgs["flags"], name: string): string | undefined {
  const value = flags[name];
  return typeof value === "string" ? value : undefined;
}

/** Task keys discovered in the database, or an empty list if none/unavailable. */
function discoverTaskKeys(env: CommandEnvironment, dbPath: string): string[] {
  if (!existsSync(dbPath)) return [];
  const db = env.openDatabase(dbPath);
  try {
    return countTracesByTaskKey(db)
      .map((row) => row.taskKey)
      .filter((key): key is string => key !== null && key.length > 0);
  } finally {
    db.close();
  }
}

function taskKeysSection(taskKeys: string[]): string {
  // `recorded` replays each tool's captured output — the safe default for an
  // eval, since it never touches a live tool.
  if (taskKeys.length === 0) {
    return `task_keys:
  # Name each task you evaluate. A task groups traces that do the same job.
  your_task:
    description: What this task does
    replay:
      default_tool_policy: recorded

# Assertions are free, deterministic checks. Get suggestions from your own
# accepted outputs with:  compound suggest-assertions <task_key>
assertions:
  your_task: []
`;
  }
  const keys = taskKeys
    .map(
      (key) => `  ${key}:\n    description: TODO\n    replay:\n      default_tool_policy: recorded`,
    )
    .join("\n");
  const asserts = taskKeys.map((key) => `  ${key}: []`).join("\n");
  return `task_keys:
  # Discovered from your imported traces.
${keys}

# Assertions are free, deterministic checks. Get suggestions from your own
# accepted outputs with:  compound suggest-assertions <task_key>
assertions:
${asserts}
`;
}

function scaffold(taskKeys: string[]): string {
  return `version: 1

artifacts_dir: artifacts
manifests_dir: benchmarks/manifests

# Public benchmark suites (optional). Empty here — Compound works entirely from
# your own imported traces; add benchmark sections only if you run them.
benchmarks: {}

# Money-safety: paid runs are OFF until you flip this on. Every experiment and
# gate is a dry run (zero provider calls) until paid_runs_enabled is true AND
# you pass --paid --cap <usd> on the command.
budget:
  paid_runs_enabled: false
  hard_limit_usd: 5.00

# Where completions are executed. Any OpenAI-compatible host is one block; set
# its wire protocol with \`type\`. Keep API keys in the environment, not here.
providers:
  openrouter:
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    type: openai_compatible

models:
  frontier:
    - id: anthropic/claude-opus-4.8
      provider: openrouter
      role: reference
  candidates:
    - id: openai/gpt-4o-mini
      provider: openrouter
      role: candidate

# A price is required to run a model — Compound refuses to spend blind.
pricing_usd_per_million_tokens:
  anthropic/claude-opus-4.8:
    input: 5.00
    output: 25.00
  openai/gpt-4o-mini:
    input: 0.15
    output: 0.60

# Import production traces, then curate them into sealed eval splits.
ingest:
  default_permissions:
    judging: true
    optimization: true
    fine_tuning: false
  sources:
    - name: langfuse-prod
      importer: langfuse
      path: exports/langfuse.jsonl

${taskKeysSection(taskKeys)}
# The CI gate policy (compound eval reads these): the candidate may regress the
# reference by at most max_regression on the sealed decision set.
gate:
  metric: task_success
  max_regression: 0.02
  require_decision_test: true
  # Refuse a further paid decision that reuses already-inspected sealed labels.
  # This is the default; it is written out so the choice is visible.
  block_repeat_decision: true
`;
}

export function runInitCommand(args: ParsedArgs, env: CommandEnvironment): CommandResult {
  const outPath = stringFlag(args.flags, "config") ?? DEFAULT_CONFIG_PATH;
  const force = args.flags.force === true;

  if (existsSync(outPath) && !force) {
    env.write(`error: ${outPath} already exists. Pass --force to overwrite it.`);
    return { exitCode: 1 };
  }

  const taskKeys = discoverTaskKeys(env, stringFlag(args.flags, "db") ?? DEFAULT_DATABASE_PATH);

  try {
    writeFileSync(outPath, scaffold(taskKeys));
  } catch (error) {
    env.write(
      `error: could not write ${outPath}: ${error instanceof Error ? error.message : error}`,
    );
    return { exitCode: 1 };
  }

  env.write(`wrote ${outPath}${force ? " (overwritten)" : ""}`);
  if (taskKeys.length > 0) {
    env.write(`  pre-filled ${taskKeys.length} discovered task key(s): ${taskKeys.join(", ")}`);
  } else {
    env.write("  no imported traces found — scaffolded a placeholder task.");
  }
  env.write("");
  env.write("next steps:");
  env.write("  1. compound import <export.jsonl>        # bring in production traces");
  env.write("  2. compound curate <task_key>            # build sealed eval splits");
  env.write("  3. compound suggest-assertions <task_key>  # propose assertions to add");
  env.write("  4. compound experiment <task_key> <model>  # dry-run a candidate");
  env.write("  5. compound eval <task_key> --candidate M --reference M  # the CI gate");
  return { exitCode: 0 };
}
