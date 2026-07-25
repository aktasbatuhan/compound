import { describe, expect, test } from "bun:test";
import type { Message } from "@compound/contract";
import {
  cosineSimilarity,
  evaluateAssertions,
  fuzzyRatio,
  jaccardSimilarity,
  rougeL,
  rougeN,
  similarity,
  tokenize,
} from "../src/index";

function output(text: string): { output: Message } {
  return { output: { role: "assistant", content: text } as Message };
}

describe("similarity primitives", () => {
  test("identical strings score 1 on every metric", () => {
    for (const m of ["fuzzy", "cosine", "jaccard", "rouge_1", "rouge_2", "rouge_l"] as const) {
      expect(similarity(m, "the quick brown fox", "the quick brown fox")).toBeCloseTo(1, 10);
    }
  });

  test("fuzzy degrades with edits", () => {
    expect(fuzzyRatio("kitten", "sitting")).toBeCloseTo(1 - 3 / 7, 10);
    expect(fuzzyRatio("abc", "xyz")).toBe(0);
  });

  test("cosine and jaccard on token overlap", () => {
    const a = tokenize("refund the order now");
    const b = tokenize("refund the order today");
    expect(cosineSimilarity(a, b)).toBeGreaterThan(0.5);
    expect(jaccardSimilarity(a, b)).toBeCloseTo(3 / 5, 10);
  });

  test("rouge rewards overlapping n-grams and subsequences", () => {
    const cand = tokenize("the cat sat on the mat");
    const ref = tokenize("the cat sat on a mat");
    expect(rougeN(cand, ref, 1)).toBeGreaterThan(0.7);
    expect(rougeN(cand, ref, 2)).toBeGreaterThan(0.4);
    expect(rougeL(cand, ref)).toBeGreaterThan(0.7);
  });

  test("disjoint token sets score 0 on token metrics", () => {
    expect(cosineSimilarity(tokenize("alpha beta"), tokenize("gamma delta"))).toBe(0);
    expect(jaccardSimilarity(tokenize("alpha"), tokenize("beta"))).toBe(0);
    expect(rougeN(tokenize("alpha beta"), tokenize("gamma delta"), 1)).toBe(0);
  });
});

describe("text_similarity assertion", () => {
  test("passes at or above the threshold", () => {
    const report = evaluateAssertions(
      [
        {
          type: "text_similarity",
          reference: "please issue a refund for the order",
          metric: "rouge_1",
          pass_threshold: 0.6,
        },
      ],
      output("please issue a refund for this order"),
    );
    expect(report.passed).toBe(true);
    expect(report.results[0]?.detail).toContain("rouge_1");
  });

  test("fails below the threshold and reports the score, not the output", () => {
    const report = evaluateAssertions(
      [
        {
          type: "text_similarity",
          reference: "the mitochondria is the powerhouse of the cell",
          metric: "cosine",
          pass_threshold: 0.5,
        },
      ],
      output("your refund has been processed"),
    );
    expect(report.passed).toBe(false);
    expect(report.results[0]?.detail).not.toContain("refund");
  });

  test("ignore_case normalizes both sides", () => {
    const report = evaluateAssertions(
      [
        {
          type: "text_similarity",
          reference: "REFUND APPROVED",
          metric: "fuzzy",
          pass_threshold: 0.99,
          ignore_case: true,
        },
      ],
      output("refund approved"),
    );
    expect(report.passed).toBe(true);
  });
});
