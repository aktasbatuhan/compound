/**
 * CLI commands, written as pure functions over an injected environment so they
 * are testable without spawning a process or binding a port.
 *
 * Every command that touches paid work or writes data reports what it did in
 * concrete numbers. "No traces were eval_ready" is a legitimate, clearly
 * reported outcome, not a silent success.
 */
import { readFileSync } from "node:fs";
import { loadConfig } from "@compound/config";
import { curateTask, type PartitionRatios } from "@compound/curation";
import {
  casesForDetectableEffect,
  DEFAULT_TARGET_POWER,
  powerEstimate,
  RECOMMENDED_MIN_DECISION_CASES,
} from "@compound/gate";
import { runImport, SUPPORTED_IMPORTERS } from "@compound/pipeline";
import {
  type CompoundDatabase,
  countCasesByPartition,
  countCasesByProvenance,
  countTracesByDiagnosticReason,
  countTracesByTaskKey,
  countTracesByValidationClass,
  createDatabase,
  migrate,
} from "@compound/storage";
import { runExperimentCommand } from "./experiment";
import { runEvalCommand, runGateCommand } from "./gate";
import { runGradeBatchCommand } from "./grade-batch";
import { runInitCommand } from "./init";
import { runJudgeCommand } from "./judge";
import { runOptimizeCommand } from "./optimize";
import { runProvidersCommand } from "./providers";
import { runSuggestAssertionsCommand } from "./suggest";
import { runTelemetryCommand } from "./telemetry";
import { runValidateCommand } from "./validate";
import { runViewCommand } from "./view";

export interface CommandEnvironment {
  /** Where output goes; injected so tests can capture it. */
  write: (line: string) => void;
  /** Opens the database. Injected so tests can use an in-memory one. */
  openDatabase: (path: string) => CompoundDatabase;
  cwd: string;
}

export interface CommandResult {
  exitCode: number;
}

export const DEFAULT_DATABASE_PATH = "compound.db";
export const DEFAULT_CONFIG_PATH = "compound.yaml";

export function defaultEnvironment(): CommandEnvironment {
  return {
    write: (line) => console.log(line),
    openDatabase: (path) => {
      const db = createDatabase({ path });
      migrate(db);
      return db;
    },
    cwd: process.cwd(),
  };
}

export interface ParsedArgs {
  command: string | undefined;
  positional: string[];
  flags: Record<string, string | boolean>;
}

export function parseArgs(argv: readonly string[]): ParsedArgs {
  const [command, ...rest] = argv;
  const positional: string[] = [];
  const flags: Record<string, string | boolean> = {};

  for (let index = 0; index < rest.length; index += 1) {
    const token = rest[index] as string;
    if (!token.startsWith("--")) {
      positional.push(token);
      continue;
    }
    const name = token.slice(2);
    const next = rest[index + 1];
    if (next !== undefined && !next.startsWith("--")) {
      flags[name] = next;
      index += 1;
    } else {
      flags[name] = true;
    }
  }

  return { command, positional, flags };
}

function stringFlag(flags: ParsedArgs["flags"], name: string): string | undefined {
  const value = flags[name];
  return typeof value === "string" ? value : undefined;
}

export const HELP_TEXT = `compound — turn production traces into gated optimization evidence

Usage:
  compound init [--config PATH] [--db PATH] [--force]
  compound validate [--config PATH]
  compound providers [name]                        (known providers; a name prints a paste-ready block)
  compound import <file> [--importer langfuse|json|otel] [--db PATH] [--config PATH] [--project-id ID] [--unsafe-no-redaction]
  compound curate <task_key> [--split train:val:cal:dec] [--db PATH]
  compound suggest-assertions <task_key> [--db PATH] [--config PATH]
  compound experiment <task_key> <model> [--partition P] [--paid --cap USD]
  compound gate <task_key> --candidate M --reference M --reason "..." [--margin 0.05] [--monthly-volume N] [--paid --cap USD]
  compound eval <task_key> --candidate M --reference M [--reason "..."]   (CI gate: exit 0 meets / 1 regresses / 2 undecidable)
  compound judge calibrate <task_key> [--paid --cap USD]
  compound judge grade <task_key> <experiment_id> [--paid --cap USD]
  compound optimize <task_key> --candidate M [--reflection M] [--max-calls N] [--force]
  compound telemetry [task_key] [--json] [--db PATH]
  compound view [gate|case|trace|experiment] [id] [--full] [--db PATH]   (read-only browser; overview if no args)
  compound view compare [task_key] [--priority quality=0.5,cost=0.3,latency=0.2] [--monthly-volume N] [--db PATH]
                                                                         (cost vs score per model; --priority adds a weighted
                                                                          ranking + Pareto frontier; axes: quality, cost,
                                                                          latency, throughput; per-task default via
                                                                          compound.yaml task_keys.<task>.priority)
  compound status [--db PATH]
  compound serve [--port N] [--host HOST] [--db PATH] [--config PATH]
  compound help

Importers: ${SUPPORTED_IMPORTERS.join(", ")}
`;

export function runImportCommand(args: ParsedArgs, env: CommandEnvironment): CommandResult {
  const file = args.positional[0];
  if (file === undefined) {
    env.write(`error: a file to import is required\n\n${HELP_TEXT}`);
    return { exitCode: 2 };
  }

  let content: string;
  try {
    content = readFileSync(file, "utf8");
  } catch (error) {
    env.write(`error: could not read ${file}: ${error instanceof Error ? error.message : error}`);
    return { exitCode: 1 };
  }

  const configPath = stringFlag(args.flags, "config") ?? DEFAULT_CONFIG_PATH;
  let config: ReturnType<typeof loadConfig> | undefined;
  try {
    config = loadConfig(configPath);
  } catch (error) {
    // Fail closed (#53). Without config there are no redaction rules and no
    // ingest permissions, so a YAML typo would otherwise turn a redacted import
    // into raw persistence of production traces with only a warning. Importing
    // unredacted is allowed only when the user says so in the command itself.
    const reason = error instanceof Error ? error.message : String(error);
    if (args.flags["unsafe-no-redaction"] !== true) {
      env.write(`error: could not load ${configPath} (${reason}).`);
      env.write(
        "error: refusing to import without redaction rules or configured permissions. " +
          "Fix the config, or pass --unsafe-no-redaction to persist these traces UNREDACTED.",
      );
      return { exitCode: 1 };
    }
    env.write(`warning: could not load ${configPath} (${reason}).`);
    env.write(
      "warning: --unsafe-no-redaction given: importing WITHOUT redaction rules or configured permissions.",
    );
  }

  const db = env.openDatabase(stringFlag(args.flags, "db") ?? DEFAULT_DATABASE_PATH);
  try {
    const { batch, report } = runImport(db, {
      importer: stringFlag(args.flags, "importer") ?? "langfuse",
      content,
      config,
      projectId: stringFlag(args.flags, "project-id"),
    });

    const counts = report.counts ?? {};
    env.write(`imported batch ${batch.id}`);
    env.write(`  eval_ready:  ${counts.eval_ready ?? 0}`);
    env.write(`  diagnostic:  ${counts.diagnostic ?? 0}`);
    env.write(`  rejected:    ${counts.rejected ?? 0} (malformed source records, not stored)`);
    env.write(`  duplicate:   ${counts.duplicate ?? 0} (already imported)`);

    const internal = (counts as Record<string, number>).internal_normalization_errors ?? 0;
    if (internal > 0) {
      // Loud on purpose: this is a bug in Compound, not in the user's export.
      env.write(`  INTERNAL ERRORS: ${internal} — traces we produced failed our own contract.`);
      env.write("  This is a Compound bug. Please report it with the batch id.");
    }

    const redactions = report.redactions_by_rule as Record<string, number> | undefined;
    const redactionTotal = Object.values(redactions ?? {}).reduce((sum, n) => sum + n, 0);
    env.write(`  redactions:  ${redactionTotal}`);
    if (redactionTotal === 0 && config?.redaction === undefined) {
      env.write("  note: no redaction rules configured; nothing was redacted.");
    }

    const reasons = report.diagnostic_reasons ?? {};
    const ranked = Object.entries(reasons).sort(([, a], [, b]) => b - a);
    if (ranked.length > 0) {
      env.write("\ndiagnostic reasons:");
      for (const [reason, count] of ranked) env.write(`  ${count}  ${reason}`);
    }

    return { exitCode: 0 };
  } catch (error) {
    env.write(`error: import failed: ${error instanceof Error ? error.message : error}`);
    return { exitCode: 1 };
  } finally {
    db.close();
  }
}

/** Parse `--split train:val:cal:dec` into partition ratios (any non-negative numbers). */
function parseSplit(value: string): PartitionRatios {
  const parts = value.split(":").map((p) => Number.parseFloat(p.trim()));
  if (parts.length !== 4 || parts.some((n) => !Number.isFinite(n) || n < 0)) {
    throw new Error(
      `--split must be four non-negative numbers "train:val:cal:dec", got "${value}"`,
    );
  }
  return {
    optimization_train: parts[0] as number,
    optimizer_validation: parts[1] as number,
    judge_calibration: parts[2] as number,
    decision_test: parts[3] as number,
  };
}

/**
 * A pre-run precision read on a task's sealed decision set (#24): the minimum
 * regression a gate could DETECT with 80% power at the current case count,
 * with a nudge to curate more when the set is under the recommended floor.
 * Reported at curate time (and on a gate dry run) so a user learns a tiny set
 * is underpowered BEFORE paying to run both sides. The figure assumes a
 * per-case difference SD ≈ 0.5; a higher real SD widens it (worst case,
 * SD = 1.0, roughly doubles it) — see #7. `margin` sizes the suggested-n line:
 * a true regression near the margin is what the gate must resolve, and a CI
 * that straddles −margin comes back as insufficient_data (rule.ts).
 */
export function decisionPowerLines(
  sealedCases: number,
  opts: { confidence?: number; margin?: number } = {},
): string[] {
  const confidence = opts.confidence ?? 0.95;
  const margin = opts.margin ?? 0.05;
  const power = DEFAULT_TARGET_POWER;
  if (sealedCases < 2) {
    return [
      `  decision power: ${sealedCases} sealed case(s) — far too few to decide; ` +
        `curate ≥ ${RECOMMENDED_MIN_DECISION_CASES} decision_test cases.`,
    ];
  }
  const est = powerEstimate(sealedCases, confidence, power);
  const suggested = casesForDetectableEffect(margin, confidence, power);
  const lines = [
    `  decision power: ~${Math.round(est.minDetectableEffect * 100)}pp min. detectable ` +
      `regression (${Math.round(confidence * 100)}% gate, ${Math.round(power * 100)}% power, ` +
      `${sealedCases} sealed cases, assumes per-case SD≈${est.sd}).`,
    "  → smaller true deltas will likely come back INSUFFICIENT DATA" +
      (Number.isFinite(suggested)
        ? `; resolving one at the ${(margin * 100).toFixed(1)}pp margin needs ` +
          `~${suggested} sealed cases.`
        : "."),
  ];
  if (sealedCases < RECOMMENDED_MIN_DECISION_CASES) {
    lines.push(
      `  → below the recommended ${RECOMMENDED_MIN_DECISION_CASES}+; curate more ` +
        "decision_test cases to tighten the gate.",
    );
  }
  return lines;
}

export function runCurateCommand(args: ParsedArgs, env: CommandEnvironment): CommandResult {
  const taskKey = args.positional[0];
  if (taskKey === undefined) {
    env.write(`error: a task key to curate is required\n\n${HELP_TEXT}`);
    return { exitCode: 2 };
  }

  const splitFlag = stringFlag(args.flags, "split");
  let ratios: PartitionRatios | undefined;
  if (splitFlag !== undefined) {
    try {
      ratios = parseSplit(splitFlag);
    } catch (error) {
      env.write(`error: ${error instanceof Error ? error.message : error}`);
      return { exitCode: 2 };
    }
  }

  const db = env.openDatabase(stringFlag(args.flags, "db") ?? DEFAULT_DATABASE_PATH);
  try {
    const report = curateTask(db, { taskKey, ...(ratios ? { ratios } : {}) });
    env.write(`curated task ${taskKey}`);
    env.write(`  traces scanned:  ${report.tracesScanned}`);
    env.write(`  cases created:   ${report.casesCreated}`);
    env.write(`  duplicates:      ${report.duplicates} (same work, already a case)`);
    if (report.skippedNotExtractable > 0) {
      env.write(
        `  not extractable: ${report.skippedNotExtractable} (not a replayable single call)`,
      );
    }

    const partitions = Object.entries(report.byPartition).sort();
    if (partitions.length > 0) {
      env.write("\npartitions (new cases):");
      for (const [partition, count] of partitions) env.write(`  ${count}  ${partition}`);
      // Name the seal explicitly so it is never a surprise later.
      const sealed = report.byPartition.decision_test ?? 0;
      env.write(`  (${sealed} sealed into decision_test — never seen by optimization)`);
    }

    // The cumulative sealed size (not just this run's new cases) is what a gate
    // will decide on, so estimate its power against that total.
    const totalSealed =
      countCasesByPartition(db, taskKey).find((p) => p.partition === "decision_test")?.count ?? 0;
    if (totalSealed > 0) {
      env.write("");
      for (const line of decisionPowerLines(totalSealed)) env.write(line);
    }
    return { exitCode: 0 };
  } catch (error) {
    env.write(`error: curation failed: ${error instanceof Error ? error.message : error}`);
    return { exitCode: 1 };
  } finally {
    db.close();
  }
}

export function runStatusCommand(args: ParsedArgs, env: CommandEnvironment): CommandResult {
  const db = env.openDatabase(stringFlag(args.flags, "db") ?? DEFAULT_DATABASE_PATH);
  try {
    const byClass = countTracesByValidationClass(db);
    const total = byClass.eval_ready + byClass.diagnostic;

    env.write(`traces: ${total}`);
    env.write(`  eval_ready: ${byClass.eval_ready}`);
    env.write(`  diagnostic: ${byClass.diagnostic}`);

    const byTaskKey = countTracesByTaskKey(db);
    if (byTaskKey.length > 0) {
      env.write("\nby task key:");
      for (const row of byTaskKey) {
        env.write(`  ${row.count}  ${row.taskKey ?? "(unassigned)"}`);
      }
    }

    const byReason = countTracesByDiagnosticReason(db);
    if (byReason.length > 0) {
      env.write("\ndiagnostic queue:");
      for (const row of byReason) env.write(`  ${row.count}  ${row.reason}`);
    }

    const byPartition = countCasesByPartition(db);
    const caseTotal = byPartition.reduce((sum, row) => sum + row.count, 0);
    if (caseTotal > 0) {
      env.write(`\ncases: ${caseTotal}`);
      for (const row of [...byPartition].sort((a, b) => a.partition.localeCompare(b.partition))) {
        env.write(`  ${row.count}  ${row.partition}`);
      }
      env.write("\nby provenance:");
      for (const row of countCasesByProvenance(db)) {
        env.write(`  ${row.count}  ${row.provenance}`);
      }
    }

    return { exitCode: 0 };
  } finally {
    db.close();
  }
}

export async function runCommand(
  argv: readonly string[],
  env: CommandEnvironment,
): Promise<CommandResult> {
  const args = parseArgs(argv);

  switch (args.command) {
    case undefined:
    case "help":
    case "--help":
    case "-h":
      env.write(HELP_TEXT);
      return { exitCode: args.command === undefined ? 2 : 0 };
    case "init":
      return runInitCommand(args, env);
    case "validate":
      return runValidateCommand(args, env);
    case "providers":
      return runProvidersCommand(args, env);
    case "import":
      return runImportCommand(args, env);
    case "curate":
      return runCurateCommand(args, env);
    case "status":
      return runStatusCommand(args, env);
    case "experiment":
      return runExperimentCommand(args, env);
    case "gate":
      return runGateCommand(args, env);
    case "eval":
      return runEvalCommand(args, env);
    case "judge":
      return runJudgeCommand(args, env);
    case "grade-batch":
      return runGradeBatchCommand(args, env);
    case "optimize":
      return runOptimizeCommand(args, env);
    case "telemetry":
      return runTelemetryCommand(args, env);
    case "view":
      return runViewCommand(args, env);
    case "suggest-assertions":
      return runSuggestAssertionsCommand(args, env);
    default:
      env.write(`error: unknown command '${args.command}'\n\n${HELP_TEXT}`);
      return { exitCode: 2 };
  }
}
