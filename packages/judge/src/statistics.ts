/**
 * Agreement statistics for judge calibration: Cohen's kappa between the judge's
 * binary labels and the human labels, with a seeded bootstrap confidence
 * interval (docs/judges-v1.md — "report confidence intervals, never a bare
 * mean"). Self-contained so the judge package stays decoupled from the gate.
 */

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

export function seedFromString(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function quantileSorted(sorted: readonly number[], q: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0] as number;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  const frac = pos - lo;
  return (sorted[lo] as number) * (1 - frac) + (sorted[hi] as number) * frac;
}

/**
 * Cohen's kappa for two raters over binary (0/1) labels. Returns a value in
 * [-1, 1]: 1 is perfect agreement, 0 is chance-level. When both raters give a
 * single constant label (chance agreement is total), kappa is defined as 1 if
 * they always agree and 0 otherwise, rather than dividing by zero.
 */
export function cohenKappa(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  if (n === 0) return 0;
  let agree = 0;
  let a1 = 0;
  let b1 = 0;
  for (let i = 0; i < n; i++) {
    if (a[i] === b[i]) agree += 1;
    if (a[i] === 1) a1 += 1;
    if (b[i] === 1) b1 += 1;
  }
  const po = agree / n;
  const pa1 = a1 / n;
  const pb1 = b1 / n;
  const pe = pa1 * pb1 + (1 - pa1) * (1 - pb1);
  if (pe >= 1) return po >= 1 ? 1 : 0;
  return (po - pe) / (1 - pe);
}

export interface KappaCi {
  kappa: number;
  lo: number;
  hi: number;
  n: number;
}

/**
 * Bootstrap CI for Cohen's kappa over the paired label vectors. Resamples case
 * indices with replacement; the seed makes it reproducible. With one pair the CI
 * collapses to the point estimate.
 */
export function bootstrapKappaCi(
  a: readonly number[],
  b: readonly number[],
  confidence: number,
  seed: number,
  iterations = 5000,
): KappaCi {
  const n = Math.min(a.length, b.length);
  const kappa = cohenKappa(a, b);
  if (n <= 1) return { kappa, lo: kappa, hi: kappa, n };

  const rng = mulberry32(seed);
  const samples = new Array<number>(iterations);
  const ra = new Array<number>(n);
  const rb = new Array<number>(n);
  for (let k = 0; k < iterations; k++) {
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(rng() * n);
      ra[i] = a[idx] as number;
      rb[i] = b[idx] as number;
    }
    samples[k] = cohenKappa(ra, rb);
  }
  samples.sort((x, y) => x - y);
  const alpha = 1 - confidence;
  return {
    kappa,
    lo: quantileSorted(samples, alpha / 2),
    hi: quantileSorted(samples, 1 - alpha / 2),
    n,
  };
}
