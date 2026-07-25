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
