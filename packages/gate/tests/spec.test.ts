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

describe("gate rule hashing with the provider axis", () => {
  test("absent/null providers leave the baseline hash unchanged", () => {
    expect(hashRule({ ...BASE, candidateProvider: null, referenceProvider: null })).toBe(
      hashRule(BASE),
    );
    expect(canonicalizeRule(BASE)).not.toContain("candidate_provider");
  });

  test("naming a provider makes it a DIFFERENT declared rule", () => {
    const onA = { ...BASE, candidateProvider: "together" };
    expect(hashRule(onA)).not.toBe(hashRule(BASE));
    expect(canonicalizeRule(onA)).toContain("candidate_provider");
  });

  test("the same model on two providers are two different rules", () => {
    const onA = hashRule({
      ...BASE,
      candidateProvider: "together",
      referenceProvider: "openrouter",
    });
    const onB = hashRule({
      ...BASE,
      candidateProvider: "fireworks",
      referenceProvider: "openrouter",
    });
    expect(onA).not.toBe(onB);
  });
});
