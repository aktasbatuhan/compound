/**
 * The statistic behind a gate: a paired bootstrap confidence interval on the
 * per-case difference (candidate minus reference).
 *
 * We use a bootstrap rather than a normal/McNemar approximation because case
 * counts are small, scores can be weighted (not just binary), and it assumes no
 * distribution (docs/gate-decision-v1.md, "The statistic"). The RNG is seeded
 * from the spec hash so a decision is reproducible and cannot be quietly
 * reseeded until it passes.
 */

/** mulberry32: a small, fast, deterministic PRNG. Returns a fn giving [0,1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Fold a string into a 32-bit seed (FNV-1a). */
export function seedFromString(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

export function mean(xs: readonly number[]): number {
  if (xs.length === 0) return 0;
  let sum = 0;
  for (const x of xs) sum += x;
  return sum / xs.length;
}

/** Linear-interpolated quantile of an already-sorted ascending array. */
export function quantileSorted(sorted: readonly number[], q: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0] as number;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  const frac = pos - lo;
  return (sorted[lo] as number) * (1 - frac) + (sorted[hi] as number) * frac;
}

/** Horner evaluation of a polynomial with `coeffs` in descending degree. */
function poly(coeffs: readonly number[], x: number): number {
  return coeffs.reduce((acc, coef) => acc * x + coef, 0);
}

// Acklam's coefficients. The central and tail rational approximations share the
// `c`/`d` tables; a trailing `1` appended to `b`/`d` supplies each denominator's
// unit term.
const ACKLAM_A = [
  -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2,
  -3.066479806614716e1, 2.506628277459239,
];
const ACKLAM_B = [
  -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1,
  -1.328068155288572e1, 1,
];
const ACKLAM_C = [
  -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734,
  4.374664141464968, 2.938163982698783,
];
const ACKLAM_D = [
  7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416, 1,
];

/**
 * Standard-normal quantile z such that Φ(z) = p, via Acklam's rational
 * approximation (|error| < 1.2e-9 across (0,1)). Used to turn a confidence level
 * into the two-sided critical value for the pre-run power estimate.
 */
export function invNormCdf(p: number): number {
  if (p <= 0) return Number.NEGATIVE_INFINITY;
  if (p >= 1) return Number.POSITIVE_INFINITY;
  const plow = 0.02425;
  const phigh = 1 - plow;
  if (p < plow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return poly(ACKLAM_C, q) / poly(ACKLAM_D, q);
  }
  if (p <= phigh) {
    const q = p - 0.5;
    return (poly(ACKLAM_A, q * q) * q) / poly(ACKLAM_B, q * q);
  }
  const q = Math.sqrt(-2 * Math.log(1 - p));
  return -(poly(ACKLAM_C, q) / poly(ACKLAM_D, q));
}

/** Two-sided critical z for a confidence level (e.g. 0.95 → 1.95996). */
export function zForConfidence(confidence: number): number {
  return invNormCdf(1 - (1 - confidence) / 2);
}

/** One-sided z quantile for a target statistical power (e.g. 0.80 → 0.8416). */
export function zForPower(power: number): number {
  return invNormCdf(power);
}

/** Below this many sealed cases, a paired gate rarely resolves a small regression. */
export const RECOMMENDED_MIN_DECISION_CASES = 20;

/** Default target power for the pre-run estimate: an 80% chance to detect a true regression. */
export const DEFAULT_TARGET_POWER = 0.8;

/**
 * Assumed per-case difference SD when there is no prior data. On a pass-rate
 * metric each paired difference lies in [-1, 1]; 0.5 is a moderate guess. The
 * WORST case is 1.0 (maximal disagreement), which makes the estimate ~2× wider —
 * so this is an assumption, not a conservative bound.
 */
export const ASSUMED_DIFF_SD = 0.5;

/**
 * A pre-run precision estimate for a paired gate: the minimum detectable effect
 * (smallest true regression detectable with probability `power`) for a two-sided
 * test at the given confidence, under the normal approximation:
 *
 *   MDE = (z_{1−α/2} + z_{power}) · sd / √n
 *
 * The `z_{power}` term is what separates an MDE from a bare CI half-width
 * (`z_{1−α/2}·sd/√n`) — without it the figure understates how large a regression
 * you can actually resolve. It is an estimate, not the bootstrap CI the decision
 * uses, but it lets a user see BEFORE spending that a tiny set resolves only a
 * huge regression. `sd` is assumed (see ASSUMED_DIFF_SD); the real SD scales it.
 */
export interface PowerEstimate {
  n: number;
  confidence: number;
  power: number;
  sd: number;
  /** Minimum detectable regression in metric units [0, 1]; Infinity when n < 2. */
  minDetectableEffect: number;
}

export function powerEstimate(
  n: number,
  confidence: number,
  power = DEFAULT_TARGET_POWER,
  sd = ASSUMED_DIFF_SD,
): PowerEstimate {
  const minDetectableEffect =
    n >= 2
      ? ((zForConfidence(confidence) + zForPower(power)) * sd) / Math.sqrt(n)
      : Number.POSITIVE_INFINITY;
  return { n, confidence, power, sd, minDetectableEffect };
}

/**
 * The sealed-set size needed to detect a target effect (e.g. the configured
 * margin) with probability `power` at the given confidence — the actionable
 * "curate ~N cases" nudge. Infinity when the target effect is not positive.
 */
export function casesForDetectableEffect(
  effect: number,
  confidence: number,
  power = DEFAULT_TARGET_POWER,
  sd = ASSUMED_DIFF_SD,
): number {
  if (!(effect > 0)) return Number.POSITIVE_INFINITY;
  return Math.ceil((((zForConfidence(confidence) + zForPower(power)) * sd) / effect) ** 2);
}

export interface BootstrapCi {
  point: number;
  lo: number;
  hi: number;
  n: number;
}

/**
 * Two-sided bootstrap CI on the mean of `diffs` at the given confidence. With
 * one observation the CI collapses to the point estimate (there is nothing to
 * resample); with zero it is [0,0]. `seed` makes the resampling deterministic.
 */
export function pairedBootstrapCi(
  diffs: readonly number[],
  confidence: number,
  seed: number,
  iterations = 10000,
): BootstrapCi {
  const n = diffs.length;
  const point = mean(diffs);
  if (n <= 1) return { point, lo: point, hi: point, n };

  const rng = mulberry32(seed);
  const means = new Array<number>(iterations);
  for (let b = 0; b < iterations; b++) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(rng() * n);
      sum += diffs[idx] as number;
    }
    means[b] = sum / n;
  }
  means.sort((x, y) => x - y);
  const alpha = 1 - confidence;
  return {
    point,
    lo: quantileSorted(means, alpha / 2),
    hi: quantileSorted(means, 1 - alpha / 2),
    n,
  };
}
