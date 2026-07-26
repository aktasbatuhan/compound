import { describe, expect, test } from "bun:test";
import { canonicalizeRule, type GateRule, hashRule } from "../src/spec";

const BASE: GateRule = {
  taskKey: "support",
  candidateModel: "cheap",
  referenceModel: "expensive",
  metric: "pass_rate",
  mode: "non_inferiority",
  margin: 0.05,
  confidence: 0.95,
  minCases: 20,
  judgeAbstainMax: 0,
};

describe("gate rule hashing with an optimized-prompt candidate", () => {
  test("absent/null prompt hash leaves the baseline hash unchanged (backward compat)", () => {
    expect(hashRule({ ...BASE, candidatePromptHash: null })).toBe(hashRule(BASE));
    expect(canonicalizeRule(BASE)).not.toContain("candidate_prompt_hash");
  });

  test("a prompt hash makes it a DIFFERENT declared rule than the baseline", () => {
    const adoption = { ...BASE, candidatePromptHash: "sha256:abc" };
    expect(hashRule(adoption)).not.toBe(hashRule(BASE));
    expect(canonicalizeRule(adoption)).toContain("candidate_prompt_hash");
  });

  test("two different optimized prompts are two different rules", () => {
    const a = hashRule({ ...BASE, candidatePromptHash: "sha256:abc" });
    const b = hashRule({ ...BASE, candidatePromptHash: "sha256:def" });
    expect(a).not.toBe(b);
  });
});
