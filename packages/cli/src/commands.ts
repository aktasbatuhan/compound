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
import { curateTask } from "@compound/curation";
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
  compound import <file> [--importer langfuse] [--db PATH] [--config PATH] [--project-id ID]
  compound curate <task_key> [--db PATH]
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
    // An import can proceed without config, but the user must know that the
    // redaction rules and ingest permissions they configured are NOT applied.
    env.write(
      `warning: could not load ${configPath} (${error instanceof Error ? error.message : error}).`,
    );
    env.write("warning: importing WITHOUT redaction rules or configured permissions.");
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

export function runCurateCommand(args: ParsedArgs, env: CommandEnvironment): CommandResult {
  const taskKey = args.positional[0];
  if (taskKey === undefined) {
    env.write(`error: a task key to curate is required\n\n${HELP_TEXT}`);
    return { exitCode: 2 };
  }

  const db = env.openDatabase(stringFlag(args.flags, "db") ?? DEFAULT_DATABASE_PATH);
  try {
    const report = curateTask(db, { taskKey });
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

export function runCommand(argv: readonly string[], env: CommandEnvironment): CommandResult {
  const args = parseArgs(argv);

  switch (args.command) {
    case undefined:
    case "help":
    case "--help":
    case "-h":
      env.write(HELP_TEXT);
      return { exitCode: args.command === undefined ? 2 : 0 };
    case "import":
      return runImportCommand(args, env);
    case "curate":
      return runCurateCommand(args, env);
    case "status":
      return runStatusCommand(args, env);
    default:
      env.write(`error: unknown command '${args.command}'\n\n${HELP_TEXT}`);
      return { exitCode: 2 };
  }
}
