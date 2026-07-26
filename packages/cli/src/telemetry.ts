/**
 * `compound telemetry [task_key] [--json]` — the operational rollup next to the
 * gate's quality verdict: per task x model x provider, latency percentiles,
 * cost per case, token volumes, and output TPS, aggregated from the completion
 * cache. Read-only: it never makes a provider call.
 */
import { telemetryRollup } from "@compound/storage";
import type { CommandEnvironment, CommandResult, ParsedArgs } from "./commands";
import { DEFAULT_DATABASE_PATH } from "./commands";

function stringFlag(flags: ParsedArgs["flags"], name: string): string | undefined {
  const value = flags[name];
  return typeof value === "string" ? value : undefined;
}

const HEADERS = ["task", "model", "provider", "n", "p50 ms", "p95 ms", "$/case", "tps"] as const;

export function runTelemetryCommand(args: ParsedArgs, env: CommandEnvironment): CommandResult {
  const taskKey = args.positional[0];
  const db = env.openDatabase(stringFlag(args.flags, "db") ?? DEFAULT_DATABASE_PATH);
  try {
    const groups = telemetryRollup(db, taskKey);
    if (args.flags.json === true) {
      env.write(JSON.stringify({ items: groups }, null, 2));
      return { exitCode: 0 };
    }
    if (groups.length === 0) {
      env.write(
        taskKey !== undefined
          ? `no telemetry for task '${taskKey}' yet — run an experiment first`
          : "no telemetry yet — run an experiment first",
      );
      return { exitCode: 0 };
    }

    const rows = groups.map((g) => [
      g.taskKey,
      g.model,
      g.provider,
      String(g.completions),
      String(Math.round(g.latencyP50Ms)),
      String(Math.round(g.latencyP95Ms)),
      g.meanCostUsd.toFixed(4),
      g.outputTps.toFixed(1),
    ]);
    const widths = HEADERS.map((h, i) =>
      Math.max(h.length, ...rows.map((r) => (r[i] as string).length)),
    );
    const line = (cells: readonly string[]) =>
      cells.map((c, i) => c.padEnd(widths[i] as number)).join("  ");
    env.write(line(HEADERS));
    for (const row of rows) env.write(line(row));
    env.write(
      "\ntps = output tokens / full request latency (includes queueing and TTFT); " +
        "each cached completion counts once.",
    );
    return { exitCode: 0 };
  } finally {
    db.close();
  }
}
