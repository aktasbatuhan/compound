/**
 * Deterministic text-similarity metrics — the cheap middle grader between exact
 * assertions and a paid LLM judge (docs/reference/openai-evals-graders-reference-20260725.md,
 * "text_similarity"). Every metric is a pure function returning a score in
 * [0, 1]; none makes a network call or costs anything.
 *
 * Scope is a defensible subset of OpenAI's list — fuzzy, cosine, jaccard and
 * ROUGE (1/2/L). BLEU/METEOR/GLEU are deferred; they need more machinery and
 * their definitions vary by implementation.
 */

export const SIMILARITY_METRICS = [
  "fuzzy",
  "cosine",
  "jaccard",
  "rouge_1",
  "rouge_2",
  "rouge_l",
] as const;

export type SimilarityMetric = (typeof SIMILARITY_METRICS)[number];

export function tokenize(text: string): string[] {
  return text.split(/[^\p{L}\p{N}]+/u).filter((t) => t.length > 0);
}

/** Levenshtein edit distance between two strings. */
export function levenshtein(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  let curr = new Array<number>(b.length + 1);
  for (let i = 1; i <= a.length; i++) {
    curr[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(
        (curr[j - 1] as number) + 1,
        (prev[j] as number) + 1,
        (prev[j - 1] as number) + cost,
      );
    }
    [prev, curr] = [curr, prev];
  }
  return prev[b.length] as number;
}

/** 1 − normalized edit distance. Identical strings → 1; fully disjoint → ~0. */
export function fuzzyRatio(a: string, b: string): number {
  const maxLen = Math.max(a.length, b.length);
  if (maxLen === 0) return 1;
  return 1 - levenshtein(a, b) / maxLen;
}

function counts(tokens: string[]): Map<string, number> {
  const m = new Map<string, number>();
  for (const t of tokens) m.set(t, (m.get(t) ?? 0) + 1);
  return m;
}

export function cosineSimilarity(a: string[], b: string[]): number {
  if (a.length === 0 && b.length === 0) return 1;
  if (a.length === 0 || b.length === 0) return 0;
  const ca = counts(a);
  const cb = counts(b);
  let dot = 0;
  for (const [tok, n] of ca) dot += n * (cb.get(tok) ?? 0);
  const norm = (m: Map<string, number>) =>
    Math.sqrt([...m.values()].reduce((s, n) => s + n * n, 0));
  const denom = norm(ca) * norm(cb);
  return denom === 0 ? 0 : dot / denom;
}

export function jaccardSimilarity(a: string[], b: string[]): number {
  const sa = new Set(a);
  const sb = new Set(b);
  if (sa.size === 0 && sb.size === 0) return 1;
  let inter = 0;
  for (const t of sa) if (sb.has(t)) inter += 1;
  const union = sa.size + sb.size - inter;
  return union === 0 ? 0 : inter / union;
}

function ngrams(tokens: string[], n: number): string[] {
  if (tokens.length < n) return [];
  const out: string[] = [];
  for (let i = 0; i + n <= tokens.length; i++) out.push(tokens.slice(i, i + n).join(""));
  return out;
}

/** ROUGE-N F1: n-gram overlap between candidate and reference. */
export function rougeN(candidate: string[], reference: string[], n: number): number {
  const cand = ngrams(candidate, n);
  const ref = ngrams(reference, n);
  if (cand.length === 0 && ref.length === 0) return 1;
  if (cand.length === 0 || ref.length === 0) return 0;
  const refCounts = counts(ref);
  let overlap = 0;
  const seen = new Map<string, number>();
  for (const g of cand) {
    const used = seen.get(g) ?? 0;
    if (used < (refCounts.get(g) ?? 0)) {
      overlap += 1;
      seen.set(g, used + 1);
    }
  }
  const precision = overlap / cand.length;
  const recall = overlap / ref.length;
  return precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
}

/** Length of the longest common subsequence of two token lists. */
export function lcsLength(a: string[], b: string[]): number {
  if (a.length === 0 || b.length === 0) return 0;
  let prev = new Array<number>(b.length + 1).fill(0);
  let curr = new Array<number>(b.length + 1).fill(0);
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      curr[j] =
        a[i - 1] === b[j - 1]
          ? (prev[j - 1] as number) + 1
          : Math.max(prev[j] as number, curr[j - 1] as number);
    }
    [prev, curr] = [curr, prev];
    curr.fill(0);
  }
  return prev[b.length] as number;
}

/** ROUGE-L F1 based on the longest common subsequence. */
export function rougeL(candidate: string[], reference: string[]): number {
  if (candidate.length === 0 && reference.length === 0) return 1;
  if (candidate.length === 0 || reference.length === 0) return 0;
  const lcs = lcsLength(candidate, reference);
  const precision = lcs / candidate.length;
  const recall = lcs / reference.length;
  return precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
}

/** Compute one metric between two raw strings. */
export function similarity(metric: SimilarityMetric, candidate: string, reference: string): number {
  if (metric === "fuzzy") return fuzzyRatio(candidate, reference);
  const ct = tokenize(candidate);
  const rt = tokenize(reference);
  switch (metric) {
    case "cosine":
      return cosineSimilarity(ct, rt);
    case "jaccard":
      return jaccardSimilarity(ct, rt);
    case "rouge_1":
      return rougeN(ct, rt, 1);
    case "rouge_2":
      return rougeN(ct, rt, 2);
    case "rouge_l":
      return rougeL(ct, rt);
  }
}
