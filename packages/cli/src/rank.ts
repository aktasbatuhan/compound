/**
 * Priority-weighted ranking and Pareto-frontier math for `view compare` (#29).
 *
 * Pure functions over abstract axis values so the scoring is testable without a
 * database. The compare view supplies one `AxisValues` per route (model @
 * provider) and a priority vector; this module normalizes, scores, ranks, and
 * flags Pareto-optimal routes.
 *
 * Conventions:
 * - Axes are {quality, cost, latency, throughput}. Cost and latency are
 *   "lower is better" and get inverted during normalization.
 * - A row missing a WEIGHTED axis value is excluded from scoring (reported,
 *   never silent). Zero-weight axes never exclude anyone.
 * - A weighted axis on which every scorable row is equal carries no signal:
 *   it is dropped and the remaining weights renormalized (reported).
 */

export const PRIORITY_AXES = ["quality", "cost", "latency", "throughput"] as const;
export type PriorityAxis = (typeof PRIORITY_AXES)[number];

/** Normalized weights over all four axes; entries sum to 1. */
export type PriorityVector = Record<PriorityAxis, number>;

/** Axes where a numerically SMALLER raw value is better. */
const LOWER_IS_BETTER: ReadonlySet<PriorityAxis> = new Set(["cost", "latency"]);

/** Raw per-route axis values. `null` = not measured (excludes the row iff the axis is weighted). */
export type AxisValues = Record<PriorityAxis, number | null>;

/**
 * Normalize a partial weight map to a full vector summing to 1.
 * Throws when no axis has positive weight or any weight is negative/non-finite.
 */
export function normalizePriority(weights: Partial<Record<PriorityAxis, number>>): PriorityVector {
  let total = 0;
  for (const axis of PRIORITY_AXES) {
    const w = weights[axis] ?? 0;
    if (!Number.isFinite(w) || w < 0) {
      throw new Error(`priority weight for '${axis}' must be a non-negative number, got '${w}'`);
    }
    total += w;
  }
  if (total <= 0) {
    throw new Error("priority needs at least one positive weight");
  }
  const vector = {} as PriorityVector;
  for (const axis of PRIORITY_AXES) vector[axis] = (weights[axis] ?? 0) / total;
  return vector;
}

/**
 * Parse `--priority quality=0.5,cost=0.3,latency=0.2` into a normalized vector.
 * Unspecified axes get weight 0. Unknown axes and malformed numbers are errors
 * with the valid vocabulary spelled out.
 */
export function parsePriority(spec: string): PriorityVector {
  const weights: Partial<Record<PriorityAxis, number>> = {};
  for (const part of spec.split(",")) {
    const pair = part.trim();
    if (pair.length === 0) continue;
    const eq = pair.indexOf("=");
    if (eq === -1) {
      throw new Error(
        `priority '${pair}' is not axis=weight; expected e.g. quality=0.5,cost=0.3,latency=0.2`,
      );
    }
    const axis = pair.slice(0, eq).trim();
    const raw = pair.slice(eq + 1).trim();
    if (!(PRIORITY_AXES as readonly string[]).includes(axis)) {
      throw new Error(`unknown priority axis '${axis}'; one of: ${PRIORITY_AXES.join(", ")}`);
    }
    const weight = Number.parseFloat(raw);
    if (!Number.isFinite(weight) || weight < 0 || raw.length === 0) {
      throw new Error(`priority weight for '${axis}' must be a non-negative number, got '${raw}'`);
    }
    if (weights[axis as PriorityAxis] !== undefined) {
      throw new Error(`priority axis '${axis}' given twice`);
    }
    weights[axis as PriorityAxis] = weight;
  }
  if (Object.keys(weights).length === 0) {
    throw new Error("priority is empty; expected e.g. quality=0.5,cost=0.3,latency=0.2");
  }
  return normalizePriority(weights);
}

/** `quality=0.50 cost=0.30` — only the positively weighted axes, for headers. */
export function formatPriority(priority: PriorityVector): string {
  return PRIORITY_AXES.filter((a) => priority[a] > 0)
    .map((a) => `${a}=${priority[a].toFixed(2)}`)
    .join(" ");
}

export interface RankedRow<T> {
  row: T;
  values: AxisValues;
  /** Weighted sum of min-max-normalized axis values, in [0, 1]. */
  score: number;
  /** 1-based; ties share a rank and the next rank skips (competition ranking). */
  rank: number;
  onFrontier: boolean;
  /** One route that is at least as good on every weighted axis and better on one. */
  dominatedBy: T | null;
}

export interface RankingResult<T> {
  /** Best-first. */
  ranked: RankedRow<T>[];
  /** Rows left out of scoring: a weighted axis has no measured value. */
  excluded: { row: T; missingAxes: PriorityAxis[] }[];
  /** Weighted axes on which every scorable row is equal — no ranking signal. */
  droppedAxes: PriorityAxis[];
  /** Weights actually used after dropping degenerate axes (renormalized). */
  effectiveWeights: PriorityVector;
  /** True when every weighted axis was degenerate: scores are all 0 and meaningless. */
  allDegenerate: boolean;
}

const EPSILON = 1e-9;

/**
 * Min-max normalize each weighted axis across the scorable rows (cost/latency
 * inverted so 1 = best), score by the priority weights, rank best-first, and
 * flag Pareto-optimal rows across the weighted axes.
 */
export function rankRows<T>(
  rows: readonly T[],
  values: (row: T) => AxisValues,
  priority: PriorityVector,
): RankingResult<T> {
  const weightedAxes = PRIORITY_AXES.filter((a) => priority[a] > 0);

  const scorable: { row: T; values: AxisValues }[] = [];
  const excluded: { row: T; missingAxes: PriorityAxis[] }[] = [];
  for (const row of rows) {
    const v = values(row);
    const missing = weightedAxes.filter((a) => v[a] === null);
    if (missing.length > 0) excluded.push({ row, missingAxes: missing });
    else scorable.push({ row, values: v });
  }

  // An axis where all scorable rows are equal cannot rank anyone; drop it and
  // renormalize the remaining weights so they still sum to 1.
  const range = new Map<PriorityAxis, { min: number; max: number }>();
  for (const axis of weightedAxes) {
    const nums = scorable.map((s) => s.values[axis] as number);
    range.set(axis, { min: Math.min(...nums), max: Math.max(...nums) });
  }
  const liveAxes = weightedAxes.filter((a) => {
    const r = range.get(a) as { min: number; max: number };
    return scorable.length > 0 && r.max - r.min > EPSILON;
  });
  const droppedAxes = weightedAxes.filter((a) => !liveAxes.includes(a));
  const allDegenerate = scorable.length > 0 && liveAxes.length === 0;

  const liveTotal = liveAxes.reduce((sum, a) => sum + priority[a], 0);
  const effectiveWeights = {} as PriorityVector;
  for (const axis of PRIORITY_AXES) {
    effectiveWeights[axis] =
      liveAxes.includes(axis) && liveTotal > 0 ? priority[axis] / liveTotal : 0;
  }

  const scoreOf = (v: AxisValues): number => {
    let score = 0;
    for (const axis of liveAxes) {
      const { min, max } = range.get(axis) as { min: number; max: number };
      const raw = v[axis] as number;
      const normalized = LOWER_IS_BETTER.has(axis)
        ? (max - raw) / (max - min)
        : (raw - min) / (max - min);
      score += effectiveWeights[axis] * normalized;
    }
    return score;
  };

  // Pareto dominance over the weighted axes, in better-is-higher orientation.
  const oriented = (v: AxisValues, axis: PriorityAxis): number => {
    const raw = v[axis] as number;
    return LOWER_IS_BETTER.has(axis) ? -raw : raw;
  };
  const dominates = (a: AxisValues, b: AxisValues): boolean => {
    let strictlyBetter = false;
    for (const axis of weightedAxes) {
      const diff = oriented(a, axis) - oriented(b, axis);
      if (diff < -EPSILON) return false;
      if (diff > EPSILON) strictlyBetter = true;
    }
    return strictlyBetter;
  };

  const scored = scorable.map((s) => ({
    row: s.row,
    values: s.values,
    score: scoreOf(s.values),
    dominatedBy:
      scorable.find((other) => other.row !== s.row && dominates(other.values, s.values))?.row ??
      null,
  }));
  scored.sort((a, b) => b.score - a.score);

  const ranked: RankedRow<T>[] = scored.map((s) => ({
    row: s.row,
    values: s.values,
    score: s.score,
    rank: 0, // filled below (competition ranking needs the previous row's final rank)
    onFrontier: s.dominatedBy === null,
    dominatedBy: s.dominatedBy,
  }));
  for (let i = 0; i < ranked.length; i += 1) {
    const current = ranked[i] as RankedRow<T>;
    const prev = ranked[i - 1];
    current.rank =
      prev !== undefined && Math.abs(prev.score - current.score) <= EPSILON ? prev.rank : i + 1;
  }

  return { ranked, excluded, droppedAxes, effectiveWeights, allDegenerate };
}

// --- ASCII Pareto chart ----------------------------------------------------

export interface ParetoPoint {
  /** Single-character mark, e.g. the row's rank id. */
  mark: string;
  cost: number;
  quality: number;
  onFrontier: boolean;
}

/**
 * A plain-text cost-vs-quality scatter: quality up, cost right (cheap = left,
 * so the frontier hugs the upper-left). Frontier points draw as their rank
 * mark; dominated points draw as `·`. Returns [] when the spread on either
 * axis is too small to plot.
 */
export function paretoChartLines(
  points: readonly ParetoPoint[],
  width = 44,
  height = 12,
): string[] {
  if (points.length < 2) return [];
  const costs = points.map((p) => p.cost);
  const qualities = points.map((p) => p.quality);
  const costMin = Math.min(...costs);
  const costMax = Math.max(...costs);
  const qMin = Math.min(...qualities);
  const qMax = Math.max(...qualities);
  if (costMax - costMin <= EPSILON || qMax - qMin <= EPSILON) return [];

  const grid: string[][] = Array.from({ length: height }, () => Array(width).fill(" "));
  for (const p of points) {
    const col = Math.round(((p.cost - costMin) / (costMax - costMin)) * (width - 1));
    const rowIdx = height - 1 - Math.round(((p.quality - qMin) / (qMax - qMin)) * (height - 1));
    const cell = (grid[rowIdx] as string[])[col];
    (grid[rowIdx] as string[])[col] = cell === " " ? (p.onFrontier ? p.mark : "·") : "+";
  }

  const yLabel = (q: number): string => `${(q * 100).toFixed(0)}%`.padStart(5);
  const lines: string[] = [];
  for (let i = 0; i < height; i += 1) {
    const label = i === 0 ? yLabel(qMax) : i === height - 1 ? yLabel(qMin) : "     ";
    lines.push(`  ${label} |${(grid[i] as string[]).join("")}`);
  }
  lines.push(`        +${"-".repeat(width)}`);
  const xLeft = `$${costMin.toFixed(5)}`;
  const xRight = `$${costMax.toFixed(5)}`;
  const gap = Math.max(1, width - xLeft.length - xRight.length);
  lines.push(`         ${xLeft}${" ".repeat(gap)}${xRight}`);
  return lines;
}
