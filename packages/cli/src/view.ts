/**
 * `compound view [gate|case|trace|experiment] [id]` — a read-only browser for a
 * run's results and data. Zero side effects: every subview is a formatted read
 * over the storage layer, so a reviewer can drill from the overview into a gate's
 * disagreements, a case's competing outputs, or a raw trace without SQL.
 *
 * Ids may be given as an 8+ char prefix; a unique match resolves. `--full`
 * disables the output truncation that keeps the default views scannable.
 */
import {
  type CaseRow,
  countCasesByPartition,
  countCasesByTaskKey,
  countTracesByTaskKey,
  type ExperimentRow,
  experimentScoreCost,
  getCase,
  getExperiment,
  getExperimentResults,
  getGateResult,
  getGateSpec,
  getTraceByTraceId,
  listCaseOutcomes,
  listExperiments,
  listGateResults,
  routeLabel,
  telemetryRollup,
} from "@compound/storage";
import type { CommandEnvironment, CommandResult, ParsedArgs } from "./commands";
import { DEFAULT_DATABASE_PATH } from "./commands";

function stringFlag(flags: ParsedArgs["flags"], name: string): string | undefined {
  const value = flags[name];
  return typeof value === "string" ? value : undefined;
}

const short = (id: string): string => (id.length > 8 ? id.slice(0, 8) : id);
const pct = (x: number): string => `${(x * 100).toFixed(1)}%`;
const when = (d: Date): string => d.toISOString().replace("T", " ").slice(0, 19);
const truncate = (s: string, full: boolean, n = 400): string =>
  full || s.length <= n ? s : `${s.slice(0, n)}… (+${s.length - n} chars — pass --full)`;

/** Render an assistant/user message (or any output shape) to readable lines. */
function renderMessage(value: unknown, full: boolean, indent = "    "): string[] {
  if (value == null) return [`${indent}(no output — never executed)`];
  if (typeof value === "string")
    return truncate(value, full)
      .split("\n")
      .map((l) => indent + l);
  if (typeof value !== "object") return [`${indent}${String(value)}`];
  const msg = value as { role?: unknown; content?: unknown; tool_calls?: unknown };
  const lines: string[] = [];
  const role = typeof msg.role === "string" ? msg.role : undefined;
  if (typeof msg.content === "string" && msg.content.length > 0) {
    const body = truncate(msg.content, full);
    for (const l of body.split("\n")) lines.push(indent + l);
  }
  if (Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
    for (const call of msg.tool_calls as Array<{
      name?: unknown;
      function?: { name?: unknown; arguments?: unknown };
      arguments?: unknown;
    }>) {
      const name = call.name ?? call.function?.name ?? "?";
      const rawArgs = call.arguments ?? call.function?.arguments;
      const argStr = typeof rawArgs === "string" ? rawArgs : JSON.stringify(rawArgs ?? {});
      lines.push(`${indent}→ tool_call ${String(name)}(${truncate(argStr, full, 200)})`);
    }
  }
  if (lines.length === 0) lines.push(`${indent}${truncate(JSON.stringify(value), full)}`);
  const first = lines[0];
  if (role !== undefined && first !== undefined) {
    lines[0] = `${indent}[${role}] ${first.slice(indent.length)}`;
  }
  return lines;
}

// --- overview --------------------------------------------------------------

function overview(
  db: ReturnType<CommandEnvironment["openDatabase"]>,
  env: CommandEnvironment,
): CommandResult {
  env.write("COMPOUND — overview\n");

  // A task shows up if it has traces OR curated cases — cases can outlive the
  // traces they came from, and a reviewer still needs to see them.
  const traceCounts = new Map<string, number>();
  for (const t of countTracesByTaskKey(db)) {
    if (t.taskKey !== null) traceCounts.set(t.taskKey, t.count);
  }
  const taskKeys = new Set<string>(traceCounts.keys());
  for (const c of countCasesByTaskKey(db)) taskKeys.add(c.taskKey);
  if (taskKeys.size > 0) {
    env.write("tasks:");
    for (const taskKey of taskKeys) {
      const parts = countCasesByPartition(db, taskKey);
      const partStr = parts.map((p) => `${p.count} ${p.partition}`).join(", ");
      env.write(
        `  ${taskKey}  ·  ${traceCounts.get(taskKey) ?? 0} traces  ·  cases: ${partStr || "none"}`,
      );
    }
  } else {
    env.write("tasks: (none — import traces first)");
  }

  const gates = listGateResults(db, 5);
  if (gates.length > 0) {
    env.write("\nlatest gate decisions:");
    for (const g of gates) {
      env.write(
        `  ${short(g.result.id)}  ${g.spec.taskKey}  ${g.result.outcome.toUpperCase()}  ` +
          `cand ${pct(g.result.candidateRate)} vs ref ${pct(g.result.referenceRate)}  ` +
          `(view gate ${short(g.result.id)})`,
      );
    }
  }

  const tel = telemetryRollup(db);
  if (tel.length > 0) {
    env.write("\ntelemetry (p50 latency · queue/decode · $·per-case):");
    for (const r of tel) {
      env.write(
        `  ${r.taskKey}  ${r.model} @${r.provider}  ` +
          `${r.latencyP50Ms}ms (q ${r.queueP50Ms} / d ${r.decodeP50Ms})  ` +
          `$${r.meanCostUsd.toFixed(5)}  ${r.outputTps.toFixed(1)} tps`,
      );
    }
  }

  env.write(
    "\ndrill in:  compound view gate|case|trace|experiment <id>   ·   compare cost vs score:  compound view compare [task_key]",
  );
  return { exitCode: 0 };
}

// --- gate ------------------------------------------------------------------

function gateList(
  db: ReturnType<CommandEnvironment["openDatabase"]>,
  env: CommandEnvironment,
): CommandResult {
  const gates = listGateResults(db, 50);
  if (gates.length === 0) {
    env.write("no gate decisions recorded yet.");
    return { exitCode: 0 };
  }
  env.write("gate decisions (newest first):\n");
  for (const g of gates) {
    env.write(
      `  ${short(g.result.id)}  ${when(g.result.decidedAt)}  ${g.spec.taskKey}  ` +
        `${g.result.outcome.toUpperCase()}`,
    );
    env.write(
      `     ${g.spec.candidateModel} ${pct(g.result.candidateRate)} vs ` +
        `${g.spec.referenceModel} ${pct(g.result.referenceRate)}  ` +
        `Δ${(g.result.delta * 100).toFixed(1)}pp  n=${g.result.n}`,
    );
  }
  env.write("\nview a decision:  compound view gate <id>");
  return { exitCode: 0 };
}

function gateDetail(
  db: ReturnType<CommandEnvironment["openDatabase"]>,
  env: CommandEnvironment,
  id: string,
  full: boolean,
): CommandResult {
  const resolved = resolvePrefix(
    id,
    listGateResults(db, 500).map((g) => g.result.id),
  );
  if (resolved === null) return ambiguous(env, id, "gate");
  const result = getGateResult(db, resolved);
  if (result === null) {
    env.write(`error: no gate result '${id}'`);
    return { exitCode: 1 };
  }
  const spec = getGateSpec(db, result.gateSpecId);

  env.write(`GATE ${result.outcome.toUpperCase()}   (${short(result.id)})`);
  if (spec !== null) {
    env.write(`  task:        ${spec.taskKey}   metric: ${spec.metric}   mode: ${spec.mode}`);
    env.write(`  reason:      ${spec.firewallReason}`);
  }
  env.write(
    `  candidate:   ${spec?.candidateModel ?? "?"}` +
      `${spec?.candidateProvider ? ` @${spec.candidateProvider}` : ""}   ${pct(result.candidateRate)}`,
  );
  env.write(
    `  reference:   ${spec?.referenceModel ?? "?"}` +
      `${spec?.referenceProvider ? ` @${spec.referenceProvider}` : ""}   ${pct(result.referenceRate)}`,
  );
  env.write(
    `  delta:       ${result.delta >= 0 ? "+" : ""}${(result.delta * 100).toFixed(1)}pp   ` +
      `95% CI [${(result.ciLo * 100).toFixed(1)}pp, ${(result.ciHi * 100).toFixed(1)}pp]`,
  );
  if (spec !== null && spec.mode === "non_inferiority") {
    env.write(
      `  margin:      -${(spec.margin * 100).toFixed(1)}pp   n=${result.n} (min ${spec.minCases})`,
    );
  } else {
    env.write(`  n:           ${result.n}`);
  }
  env.write(`  decided:     ${when(result.decidedAt)}`);

  // Reconstruct disagreements by pairing the two experiments' per-case grades.
  const candByCase = new Map(
    getExperimentResults(db, result.candidateExperimentId).map((r) => [r.caseId, r.passed]),
  );
  const disagreements: string[] = [];
  for (const ref of getExperimentResults(db, result.referenceExperimentId)) {
    const cand = candByCase.get(ref.caseId);
    if (cand !== undefined && cand !== ref.passed) {
      disagreements.push(
        `  ${ref.caseId}: candidate ${cand ? "PASS" : "FAIL"} / reference ${ref.passed ? "PASS" : "FAIL"}`,
      );
    }
  }
  if (disagreements.length > 0) {
    env.write(`\ndisagreements (${disagreements.length}):`);
    for (const line of full ? disagreements : disagreements.slice(0, 10)) env.write(line);
    if (!full && disagreements.length > 10) {
      env.write(`  … +${disagreements.length - 10} more (pass --full)`);
    }
    env.write("\ninspect a case:  compound view case <case_id>");
  }
  return { exitCode: 0 };
}

// --- case ------------------------------------------------------------------

function caseDetail(
  db: ReturnType<CommandEnvironment["openDatabase"]>,
  env: CommandEnvironment,
  id: string,
  full: boolean,
): CommandResult {
  const row: CaseRow | null = getCase(db, id) ?? getCase(db, `case:${id}`);
  if (row === null) {
    env.write(`error: no case '${id}'`);
    return { exitCode: 1 };
  }
  env.write(`CASE ${row.caseId}`);
  env.write(`  task: ${row.taskKey}   partition: ${row.partition}   provenance: ${row.provenance}`);
  env.write(`  source trace: ${row.sourceTraceId}   (view trace ${row.sourceTraceId})`);

  const input = row.input as { input?: unknown } | null;
  const messages = input && Array.isArray(input.input) ? input.input : null;
  if (messages !== null) {
    env.write("\ninput:");
    for (const m of messages) for (const l of renderMessage(m, full)) env.write(l);
  }
  if (row.expected != null) {
    env.write("\nexpected:");
    for (const l of renderMessage(row.expected, full)) env.write(l);
  }

  const outcomes = listCaseOutcomes(db, row.caseId);
  if (outcomes.length > 0) {
    env.write("\nmodel outputs on this case (newest first):");
    for (const o of outcomes) {
      const verdict = o.passed === null ? "—" : o.passed ? "PASS" : "FAIL";
      env.write(
        `\n  ${o.candidateModel}  [${verdict}]` +
          `${o.score === null ? "" : ` score ${o.score.toFixed(2)}`}  ` +
          `${o.partition}  ${when(o.startedAt)}`,
      );
      for (const l of renderMessage(o.output, full, "      ")) env.write(l);
    }
  } else {
    env.write("\n(no experiment has produced an output for this case yet)");
  }
  return { exitCode: 0 };
}

// --- trace -----------------------------------------------------------------

function traceDetail(
  db: ReturnType<CommandEnvironment["openDatabase"]>,
  env: CommandEnvironment,
  id: string,
  full: boolean,
): CommandResult {
  const trace = getTraceByTraceId(db, id);
  if (trace === null) {
    env.write(`error: no trace '${id}'`);
    return { exitCode: 1 };
  }
  const payload = trace.trace as unknown as {
    task_key?: string | null;
    started_at?: string;
    source?: unknown;
    focal_step_id?: string | null;
    steps?: Array<{
      step_id?: string;
      provider?: string;
      model?: string;
      input?: unknown;
      output?: unknown;
      finish_reason?: string;
      cost_usd?: number;
    }>;
  };
  env.write(`TRACE ${trace.traceId}`);
  env.write(
    `  task: ${payload.task_key ?? "(unassigned)"}   started: ${payload.started_at ?? "?"}`,
  );
  const steps = Array.isArray(payload.steps) ? payload.steps : [];
  const focal = steps.find((s) => s.step_id === payload.focal_step_id) ?? steps[0];
  env.write(
    `  steps: ${steps.length}${focal ? `   focal model: ${focal.model ?? "?"} @${focal.provider ?? "?"}` : ""}`,
  );
  if (focal) {
    const inputMsgs = Array.isArray(focal.input) ? focal.input : null;
    if (inputMsgs !== null) {
      env.write("\nfocal step input:");
      for (const m of inputMsgs) for (const l of renderMessage(m, full)) env.write(l);
    }
    if (focal.output != null) {
      env.write("\nfocal step output:");
      for (const l of renderMessage(focal.output, full)) env.write(l);
    }
    if (focal.finish_reason !== undefined) env.write(`\n  finish_reason: ${focal.finish_reason}`);
  }
  return { exitCode: 0 };
}

// --- experiment ------------------------------------------------------------

function experimentList(
  db: ReturnType<CommandEnvironment["openDatabase"]>,
  env: CommandEnvironment,
): CommandResult {
  const rows = listExperiments(db, { limit: 50 });
  if (rows.length === 0) {
    env.write("no experiments recorded yet.");
    return { exitCode: 0 };
  }
  env.write("experiments (newest first):\n");
  for (const e of rows) {
    env.write(
      `  ${short(e.id)}  ${when(e.startedAt)}  ${e.taskKey}  ${e.candidateModel}  ` +
        `${e.partition}  ${e.paid ? "PAID" : "dry-run"}`,
    );
  }
  env.write("\nview an experiment:  compound view experiment <id>");
  return { exitCode: 0 };
}

function experimentDetail(
  db: ReturnType<CommandEnvironment["openDatabase"]>,
  env: CommandEnvironment,
  id: string,
): CommandResult {
  const resolved = resolvePrefix(
    id,
    listExperiments(db, { limit: 1000 }).map((e) => e.id),
  );
  if (resolved === null) return ambiguous(env, id, "experiment");
  const exp: ExperimentRow | null = getExperiment(db, resolved);
  if (exp === null) {
    env.write(`error: no experiment '${id}'`);
    return { exitCode: 1 };
  }
  const results = getExperimentResults(db, exp.id);
  const graded = results.filter((r) => r.passed !== null);
  const passed = graded.filter((r) => r.passed === true).length;
  env.write(`EXPERIMENT ${short(exp.id)}   ${exp.paid ? "PAID" : "dry-run"}`);
  env.write(`  task: ${exp.taskKey}   model: ${exp.candidateModel}   partition: ${exp.partition}`);
  env.write(`  started: ${when(exp.startedAt)}`);
  env.write(
    `  cases: ${results.length}   graded: ${graded.length}   ` +
      `pass rate: ${graded.length > 0 ? pct(passed / graded.length) : "—"}`,
  );
  const fails = graded.filter((r) => r.passed === false);
  if (fails.length > 0) {
    env.write(`\nfailing cases (${fails.length}):`);
    for (const r of fails.slice(0, 20)) env.write(`  ${r.caseId}   (view case ${r.caseId})`);
    if (fails.length > 20) env.write(`  … +${fails.length - 20} more`);
  }
  return { exitCode: 0 };
}

// --- compare (cost vs score) -----------------------------------------------

interface CompareRow {
  taskKey: string;
  model: string;
  provider: string;
  partition: string;
  graded: number;
  passed: number;
  scoreSum: number;
  totalCost: number;
  latencySum: number;
  latencyCount: number;
  tpsSum: number;
  tpsCount: number;
}

/** Render an aligned table (telemetry's column-padding style). */
function renderTable(env: CommandEnvironment, headers: readonly string[], rows: string[][]): void {
  const widths = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => (r[i] ?? "").length)));
  const line = (cells: readonly string[]) =>
    cells.map((c, i) => (c ?? "").padEnd(widths[i] ?? 0)).join("  ");
  env.write(`  ${line(headers)}`);
  for (const row of rows) env.write(`  ${line(row)}`);
}

const rate = (passed: number, graded: number): string =>
  graded > 0 ? `${((passed / graded) * 100).toFixed(1)}%` : "—";
const perCase = (total: number, graded: number): string =>
  graded > 0 ? `$${(total / graded).toFixed(5)}` : "—";
const avgMs = (r: CompareRow): string =>
  r.latencyCount > 0 ? String(Math.round(r.latencySum / r.latencyCount)) : "—";
const avgTps = (r: CompareRow): string =>
  r.tpsCount > 0 ? (r.tpsSum / r.tpsCount).toFixed(1) : "—";
/** Best-first: highest pass rate, then cheapest per case. */
const byQualityThenCost = (a: CompareRow, b: CompareRow): number =>
  b.passed / b.graded - a.passed / a.graded || a.totalCost / a.graded - b.totalCost / b.graded;

function compareView(
  db: ReturnType<CommandEnvironment["openDatabase"]>,
  env: CommandEnvironment,
  taskKey: string | undefined,
): CommandResult {
  const experiments = listExperiments(
    db,
    taskKey === undefined ? { limit: 5000 } : { taskKey, limit: 5000 },
  );
  // listExperiments is newest-first. Dedup to the latest run of each distinct
  // (task, model, partition, host), where host folds in the OpenRouter upstream
  // (#9): one model id fanned across nine hosts is nine rows, while re-runs of
  // the same host collapse to the newest.
  const seen = new Set<string>();
  const rows: CompareRow[] = [];
  for (const e of experiments) {
    const sc = experimentScoreCost(db, e.id);
    if (sc.graded === 0) continue; // a comparison needs graded outcomes
    const provider = routeLabel(sc.provider ?? "?", sc.upstreamProvider, sc.serviceTier);
    const key = `${e.taskKey}|${e.candidateModel}|${e.partition}|${provider}`;
    if (seen.has(key)) continue;
    seen.add(key);
    rows.push({
      taskKey: e.taskKey,
      model: e.candidateModel,
      provider,
      partition: e.partition,
      graded: sc.graded,
      passed: sc.passed,
      scoreSum: (sc.meanScore ?? 0) * sc.graded,
      totalCost: sc.totalCostUsd,
      latencySum: (sc.meanLatencyMs ?? 0) * sc.latencyCount,
      latencyCount: sc.latencyCount,
      tpsSum: (sc.meanTps ?? 0) * sc.tpsCount,
      tpsCount: sc.tpsCount,
    });
  }
  if (rows.length === 0) {
    env.write(
      taskKey === undefined
        ? "no graded experiments to compare yet — run some, then gate or grade them."
        : `no graded experiments for task '${taskKey}' yet.`,
    );
    return { exitCode: 0 };
  }

  const perTaskHeaders = [
    "model",
    "provider",
    "partition",
    "n",
    "pass%",
    "score",
    "avg ms",
    "tps",
    "$/case",
    "total $",
  ] as const;
  const perTaskRow = (r: CompareRow): string[] => [
    r.model,
    r.provider,
    r.partition,
    String(r.graded),
    rate(r.passed, r.graded),
    (r.scoreSum / r.graded).toFixed(3),
    avgMs(r),
    avgTps(r),
    perCase(r.totalCost, r.graded),
    `$${r.totalCost.toFixed(5)}`,
  ];

  if (taskKey !== undefined) {
    env.write(`cost / score — ${taskKey}\n`);
    renderTable(env, perTaskHeaders, [...rows].sort(byQualityThenCost).map(perTaskRow));
    env.write(
      "\nlatest run per model+partition; cost is per graded case. " +
        "avg ms = full request latency (flex queue included); tps = output tokens / full latency.",
    );
    return { exitCode: 0 };
  }

  // Aggregated across tasks and partitions, one row per model+provider.
  const agg = new Map<string, CompareRow & { tasks: Set<string> }>();
  for (const r of rows) {
    const key = `${r.model} ${r.provider}`;
    const acc = agg.get(key);
    if (acc === undefined) {
      agg.set(key, { ...r, partition: "*", tasks: new Set([r.taskKey]) });
    } else {
      acc.graded += r.graded;
      acc.passed += r.passed;
      acc.scoreSum += r.scoreSum;
      acc.totalCost += r.totalCost;
      acc.latencySum += r.latencySum;
      acc.latencyCount += r.latencyCount;
      acc.tpsSum += r.tpsSum;
      acc.tpsCount += r.tpsCount;
      acc.tasks.add(r.taskKey);
    }
  }
  const aggHeaders = [
    "model",
    "provider",
    "tasks",
    "n",
    "pass%",
    "score",
    "avg ms",
    "tps",
    "$/case",
    "total $",
  ] as const;
  const aggRows = [...agg.values()]
    .sort(byQualityThenCost)
    .map((r) => [
      r.model,
      r.provider,
      String(r.tasks.size),
      String(r.graded),
      rate(r.passed, r.graded),
      (r.scoreSum / r.graded).toFixed(3),
      avgMs(r),
      avgTps(r),
      perCase(r.totalCost, r.graded),
      `$${r.totalCost.toFixed(5)}`,
    ]);
  env.write("cost / score — aggregated (per model, across all tasks)\n");
  renderTable(env, aggHeaders, aggRows);

  for (const tk of [...new Set(rows.map((r) => r.taskKey))].sort()) {
    env.write(`\n${tk}:`);
    renderTable(
      env,
      perTaskHeaders,
      rows
        .filter((r) => r.taskKey === tk)
        .sort(byQualityThenCost)
        .map(perTaskRow),
    );
  }
  env.write(
    "\nbest-first (highest pass%, then cheapest); cost is per graded case; " +
      "avg ms = full latency (flex queue included), tps = output tokens / full latency. " +
      "one task:  compound view compare <task_key>",
  );
  return { exitCode: 0 };
}

// --- shared helpers --------------------------------------------------------

/** Resolve an id or 8+ char prefix to a single full id; null if 0 or >1 match. */
function resolvePrefix(idOrPrefix: string, ids: readonly string[]): string | null {
  if (ids.includes(idOrPrefix)) return idOrPrefix;
  const matches = ids.filter((full) => full.startsWith(idOrPrefix));
  return matches.length === 1 ? (matches[0] as string) : null;
}

function ambiguous(env: CommandEnvironment, id: string, kind: string): CommandResult {
  env.write(`error: '${id}' matches no ${kind}, or more than one — use a longer id prefix`);
  return { exitCode: 1 };
}

export function runViewCommand(args: ParsedArgs, env: CommandEnvironment): CommandResult {
  const subject = args.positional[0];
  const id = args.positional[1];
  const full = args.flags.full === true;
  const db = env.openDatabase(stringFlag(args.flags, "db") ?? DEFAULT_DATABASE_PATH);
  try {
    switch (subject) {
      case undefined:
        return overview(db, env);
      case "gate":
        return id === undefined ? gateList(db, env) : gateDetail(db, env, id, full);
      case "case":
        if (id === undefined) {
          env.write("error: usage: compound view case <case_id>");
          return { exitCode: 2 };
        }
        return caseDetail(db, env, id, full);
      case "trace":
        if (id === undefined) {
          env.write("error: usage: compound view trace <trace_id>");
          return { exitCode: 2 };
        }
        return traceDetail(db, env, id, full);
      case "experiment":
      case "exp":
        return id === undefined ? experimentList(db, env) : experimentDetail(db, env, id);
      case "compare":
        return compareView(db, env, id);
      default:
        env.write(
          `error: unknown view '${subject}'; one of: gate, case, trace, experiment, compare (or no argument for the overview)`,
        );
        return { exitCode: 2 };
    }
  } finally {
    db.close();
  }
}
