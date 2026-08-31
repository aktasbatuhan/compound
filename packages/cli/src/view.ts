/**
 * `compound view [gate|case|trace|experiment] [id]` — a read-only browser for a
 * run's results and data. Zero side effects: every subview is a formatted read
 * over the storage layer, so a reviewer can drill from the overview into a gate's
 * disagreements, a case's competing outputs, or a raw trace without SQL.
 *
 * Ids may be given as an 8+ char prefix; a unique match resolves. `--full`
 * disables the output truncation that keeps the default views scannable.
 */
import { type CompoundConfig, loadConfig } from "@compound/config";
import { costFromUsage, type TokenPrice } from "@compound/execution";
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
import { DEFAULT_CONFIG_PATH, DEFAULT_DATABASE_PATH } from "./commands";
import {
  type AxisValues,
  formatPriority,
  normalizePriority,
  type ParetoPoint,
  type PriorityVector,
  paretoChartLines,
  parsePriority,
  rankRows,
} from "./rank";
import { parseMonthlyVolumeFlag, writeGateCostProjection } from "./savings";

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
  monthlyVolume: number | undefined,
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
  if (result.outcome === "meets_gate" && spec !== null) {
    writeGateCostProjection(db, env, {
      taskKey: spec.taskKey,
      candidateExperimentId: result.candidateExperimentId,
      referenceExperimentId: result.referenceExperimentId,
      ...(monthlyVolume !== undefined ? { monthlyVolume } : {}),
    });
  }

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
  /** The bare provider name (no upstream/tier folded in), for price lookup (#34). */
  baseProvider: string;
  /** Doubleword service tier when set — selects the flex price table (#34). */
  serviceTier: string | null;
  partition: string;
  graded: number;
  passed: number;
  scoreSum: number;
  totalCost: number;
  latencySum: number;
  latencyCount: number;
  tpsSum: number;
  tpsCount: number;
  inputTokens: number;
  outputTokens: number;
  /** Reported cached prompt tokens; null when the provider never reported (#34). */
  cachedTokens: number | null;
  /** Input tokens on the completions that reported cached — the hit-rate denominator. */
  reportedInputTokens: number;
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
/** Provider cache hit rate; "—" when the provider never reported cached tokens (#34). */
const hitRate = (r: CompareRow): string =>
  r.cachedTokens === null || r.reportedInputTokens === 0
    ? "—"
    : `${((r.cachedTokens / r.reportedInputTokens) * 100).toFixed(1)}%`;

/**
 * The price entry for a compare row, mirroring `resolveModel`'s fallback order:
 * the provider's own table first, then the matching global table. A compare row
 * no longer knows its transport, so a recorded service tier prefers the flex
 * table; a model priced in only one table resolves unambiguously either way.
 */
function priceEntryFor(config: CompoundConfig | undefined, r: CompareRow): TokenPrice | undefined {
  if (config === undefined) return undefined;
  const tables = [
    config.providers?.[r.baseProvider]?.pricing_usd_per_million_tokens,
    ...(r.serviceTier !== null
      ? [config.flex_pricing_usd_per_million_tokens, config.pricing_usd_per_million_tokens]
      : [config.pricing_usd_per_million_tokens, config.flex_pricing_usd_per_million_tokens]),
  ];
  for (const table of tables) {
    const entry = table?.[r.model];
    if (entry !== undefined) return entry;
  }
  return undefined;
}

/**
 * Effective $/case (#34): cached input priced at the config's `cached_input`
 * rate. Falls back to the stored (list-price) cost — marked with `*` — when no
 * cached rate is configured for the model OR the provider never reported cached
 * tokens; an unreported provider is never silently flattered with a discount.
 */
const effPerCase = (r: CompareRow, config: CompoundConfig | undefined): string => {
  const cost = effCostPerCase(r, config);
  if (cost === null) return "—";
  return cost.effective ? `$${cost.value.toFixed(5)}` : `${perCase(r.totalCost, r.graded)}*`;
};

/**
 * The numeric form of `effPerCase`, shared with the priority ranking (#29):
 * cached-rate effective $/case when computable, else the stored list-price
 * $/case (`effective: false`). Null only when nothing was graded.
 */
function effCostPerCase(
  r: CompareRow,
  config: CompoundConfig | undefined,
): { value: number; effective: boolean } | null {
  if (r.graded === 0) return null;
  const entry = priceEntryFor(config, r);
  if (entry?.cached_input !== undefined && r.cachedTokens !== null) {
    const eff = costFromUsage(
      {
        input_tokens: r.inputTokens,
        output_tokens: r.outputTokens,
        cached_input_tokens: r.cachedTokens,
      },
      entry,
    );
    return { value: eff / r.graded, effective: true };
  }
  return { value: r.totalCost / r.graded, effective: false };
}
/** Best-first: highest pass rate, then cheapest per case. */
const byQualityThenCost = (a: CompareRow, b: CompareRow): number =>
  b.passed / b.graded - a.passed / a.graded || a.totalCost / a.graded - b.totalCost / b.graded;

// --- priority ranking + Pareto frontier (#29) ------------------------------

/**
 * The raw per-axis values a route is scored on. Quality is the pass rate; cost
 * is the EFFECTIVE $/case (cached-rate when computable, list price otherwise);
 * latency/throughput are null when never measured, which excludes the row from
 * a ranking that weights them (reported, not silent).
 */
function axisValuesFor(r: CompareRow, config: CompoundConfig | undefined): AxisValues {
  return {
    quality: r.graded > 0 ? r.passed / r.graded : null,
    cost: effCostPerCase(r, config)?.value ?? null,
    latency: r.latencyCount > 0 ? r.latencySum / r.latencyCount : null,
    throughput: r.tpsCount > 0 ? r.tpsSum / r.tpsCount : null,
  };
}

const routeName = (r: CompareRow): string => `${r.model} @${r.provider}`;

/** Chart mark for a printed rank: 1–9, then a, b, c … */
const rankMark = (rank: number): string =>
  rank <= 9 ? String(rank) : String.fromCharCode(97 + ((rank - 10) % 26));

/**
 * The decision layer under a priority vector: the weighted ranked table, the
 * Pareto frontier over the weighted axes, a cost-vs-quality chart, and explicit
 * notes for every row or axis that could not take part in the scoring.
 */
function renderPriorityRanking(
  env: CommandEnvironment,
  rows: readonly CompareRow[],
  config: CompoundConfig | undefined,
  priority: PriorityVector,
  source: string,
  includePartition: boolean,
): void {
  env.write(`\npriority ranking — ${formatPriority(priority)}   (${source})`);
  const result = rankRows(rows, (r) => axisValuesFor(r, config), priority);

  if (result.ranked.length > 0) {
    const headers = [
      "rank",
      "score",
      "pareto",
      "model",
      "provider",
      ...(includePartition ? ["partition"] : []),
      "pass%",
      "eff $/case",
      "avg ms",
      "tps",
    ];
    renderTable(
      env,
      headers,
      result.ranked.map((row) => [
        String(row.rank),
        row.score.toFixed(3),
        row.onFrontier ? "*" : "",
        row.row.model,
        row.row.provider,
        ...(includePartition ? [row.row.partition] : []),
        rate(row.row.passed, row.row.graded),
        effPerCase(row.row, config),
        avgMs(row.row),
        avgTps(row.row),
      ]),
    );
  } else {
    env.write("  (no rows scorable under this priority)");
  }

  for (const axis of result.droppedAxes) {
    env.write(`  note: axis '${axis}' dropped from scoring — every scorable row is equal on it.`);
  }
  if (result.allDegenerate) {
    env.write(
      "  note: every weighted axis is equal across rows — no ranking signal; all rows tie.",
    );
  }
  for (const { row, missingAxes } of result.excluded) {
    env.write(
      `  note: ${routeName(row)} excluded from ranking — no measured ${missingAxes.join(", ")}.`,
    );
  }
  for (const row of result.ranked) {
    if (row.dominatedBy !== null) {
      env.write(
        `  dominated: ${routeName(row.row)} — at least as bad as ` +
          `${routeName(row.dominatedBy)} on every weighted axis, worse on one.`,
      );
    }
  }

  const points: ParetoPoint[] = result.ranked.flatMap((row) => {
    const v = axisValuesFor(row.row, config);
    if (v.cost === null || v.quality === null) return [];
    return [
      { mark: rankMark(row.rank), cost: v.cost, quality: v.quality, onFrontier: row.onFrontier },
    ];
  });
  const chart = paretoChartLines(points);
  if (chart.length > 0) {
    env.write("\n  pareto — pass% (up) vs eff $/case (right; cheaper is left)");
    env.write("  marks = rank of frontier rows, · = dominated, + = overlapping points");
    for (const line of chart) env.write(line);
  } else if (result.ranked.length >= 2) {
    env.write("  (pareto chart omitted — rows do not spread on both cost and quality)");
  }
  env.write(
    "  score = weighted sum of min-max normalized axes (cost, latency inverted); " +
      "pareto * = no row is at least as good on every weighted axis and better on one.",
  );
}

function compareView(
  db: ReturnType<CommandEnvironment["openDatabase"]>,
  env: CommandEnvironment,
  taskKey: string | undefined,
  config: CompoundConfig | undefined,
  priorityFlag: string | boolean | undefined,
  monthlyVolume: number | undefined,
): CommandResult {
  // Resolve the priority vector up front so a malformed flag fails before any
  // query output. Flag wins; a per-task default in compound.yaml
  // (task_keys.<task>.priority) applies to the per-task view.
  let priority: PriorityVector | undefined;
  let prioritySource = "--priority";
  if (priorityFlag !== undefined) {
    if (typeof priorityFlag !== "string") {
      env.write(
        "error: --priority needs a value, e.g. --priority quality=0.5,cost=0.3,latency=0.2",
      );
      return { exitCode: 2 };
    }
    try {
      priority = parsePriority(priorityFlag);
    } catch (error) {
      env.write(`error: ${error instanceof Error ? error.message : error}`);
      return { exitCode: 2 };
    }
  } else if (taskKey !== undefined) {
    const configured = config?.task_keys?.[taskKey]?.priority;
    if (configured !== undefined) {
      priority = normalizePriority(configured);
      prioritySource = `compound.yaml task_keys.${taskKey}.priority`;
    }
  }
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
      baseProvider: sc.provider ?? "?",
      serviceTier: sc.serviceTier,
      partition: e.partition,
      graded: sc.graded,
      passed: sc.passed,
      scoreSum: (sc.meanScore ?? 0) * sc.graded,
      totalCost: sc.totalCostUsd,
      latencySum: (sc.meanLatencyMs ?? 0) * sc.latencyCount,
      latencyCount: sc.latencyCount,
      tpsSum: (sc.meanTps ?? 0) * sc.tpsCount,
      tpsCount: sc.tpsCount,
      inputTokens: sc.totalInputTokens,
      outputTokens: sc.totalOutputTokens,
      cachedTokens: sc.cachedInputTokens,
      reportedInputTokens: sc.reportedInputTokens,
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
    "eff $/case",
    "hit%",
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
    effPerCase(r, config),
    hitRate(r),
    `$${r.totalCost.toFixed(5)}`,
  ];
  const cacheLegend =
    "eff $/case prices reported cached input at pricing's cached_input rate; " +
    "* = list price (no cached_input rate configured, or cached tokens unreported). " +
    "hit% = provider cache hit rate; — = provider did not report cached tokens.";

  const renderGateProjections = (onlyTask: string | undefined): void => {
    const latestByTask = new Map<string, ReturnType<typeof listGateResults>[number]>();
    for (const gate of listGateResults(db, 5000)) {
      if (gate.result.outcome !== "meets_gate") continue;
      if (onlyTask !== undefined && gate.spec.taskKey !== onlyTask) continue;
      if (!latestByTask.has(gate.spec.taskKey)) latestByTask.set(gate.spec.taskKey, gate);
    }

    const rendered: string[] = [];
    for (const gate of [...latestByTask.values()].sort((a, b) =>
      a.spec.taskKey.localeCompare(b.spec.taskKey),
    )) {
      const lines: string[] = [];
      const wrote = writeGateCostProjection(
        db,
        { write: (line) => lines.push(line) },
        {
          taskKey: gate.spec.taskKey,
          candidateExperimentId: gate.result.candidateExperimentId,
          referenceExperimentId: gate.result.referenceExperimentId,
          ...(monthlyVolume !== undefined ? { monthlyVolume } : {}),
        },
      );
      if (!wrote) continue;
      rendered.push(
        `  ${gate.spec.taskKey}: ${gate.spec.candidateModel} vs ${gate.spec.referenceModel} ` +
          `(latest MEETS GATE ${short(gate.result.id)})`,
        ...lines.map((line) => `  ${line}`),
      );
    }
    if (rendered.length === 0) return;
    env.write("\nprojected cost impact from latest MEETS GATE decisions:");
    for (const line of rendered) env.write(line);
  };

  if (taskKey !== undefined) {
    env.write(`cost / score — ${taskKey}\n`);
    renderTable(env, perTaskHeaders, [...rows].sort(byQualityThenCost).map(perTaskRow));
    renderGateProjections(taskKey);
    if (priority !== undefined) {
      renderPriorityRanking(env, rows, config, priority, prioritySource, true);
    }
    env.write(
      "\nlatest run per model+partition; cost is per graded case. " +
        "avg ms = full request latency (flex queue included); tps = output tokens / full latency. " +
        cacheLegend,
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
      acc.inputTokens += r.inputTokens;
      acc.outputTokens += r.outputTokens;
      // null stays null unless SOME row reported cached tokens (#34).
      if (r.cachedTokens !== null) {
        acc.cachedTokens = (acc.cachedTokens ?? 0) + r.cachedTokens;
      }
      acc.reportedInputTokens += r.reportedInputTokens;
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
    "eff $/case",
    "hit%",
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
      effPerCase(r, config),
      hitRate(r),
      `$${r.totalCost.toFixed(5)}`,
    ]);
  env.write("cost / score — aggregated (per model, across all tasks)\n");
  renderTable(env, aggHeaders, aggRows);
  renderGateProjections(undefined);
  if (priority !== undefined) {
    renderPriorityRanking(env, [...agg.values()], config, priority, prioritySource, false);
  }

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
      cacheLegend +
      " one task:  compound view compare <task_key>",
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
  let monthlyVolume: number | undefined;
  if (subject === "gate" || subject === "compare") {
    try {
      monthlyVolume = parseMonthlyVolumeFlag(args.flags["monthly-volume"]);
    } catch (error) {
      env.write(`error: ${error instanceof Error ? error.message : error}`);
      return { exitCode: 2 };
    }
  }
  if (subject === "compare" && id === undefined && monthlyVolume !== undefined) {
    env.write(
      "error: --monthly-volume with view compare requires one task_key; " +
        "an all-task view cannot apply one volume assumption to every task",
    );
    return { exitCode: 2 };
  }
  const db = env.openDatabase(stringFlag(args.flags, "db") ?? DEFAULT_DATABASE_PATH);
  try {
    switch (subject) {
      case undefined:
        return overview(db, env);
      case "gate":
        return id === undefined ? gateList(db, env) : gateDetail(db, env, id, full, monthlyVolume);
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
      case "compare": {
        // Config supplies the pricing tables for effective cost (#34). Best
        // effort: without one, eff $/case simply falls back to the stored
        // list-price figures (marked `*`) — a view never fails on a missing yaml.
        let config: CompoundConfig | undefined;
        try {
          config = loadConfig(stringFlag(args.flags, "config") ?? DEFAULT_CONFIG_PATH);
        } catch {
          config = undefined;
        }
        return compareView(db, env, id, config, args.flags.priority, monthlyVolume);
      }
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
