/**
 * The pre-declared gate rule and its content hash.
 *
 * The hash is taken over a canonical form of the rule so that declaring the
 * same rule twice yields the same hash (and reuses the stored row), while any
 * change to a decision-relevant field yields a different hash — the mechanism
 * that stops a rule being quietly edited after results exist
 * (docs/gate-decision-v1.md, "Pre-declared rule").
 */
import { createHash } from "node:crypto";
import type { GateMetric, GateMode } from "@compound/storage";

export interface GateRule {
  taskKey: string;
  candidateModel: string;
  referenceModel: string;
  metric: GateMetric;
  mode: GateMode;
  margin: number;
  confidence: number;
  minCases: number;
  judgeAbstainMax: number;
  /**
   * When the candidate runs with an optimized prompt, its content hash. Part of
   * the rule identity: "model M with prompt P" is a different declaration than
   * plain "model M", so an adoption gate never reuses a baseline gate's spec.
   */
  candidatePromptHash?: string | null;
}

/**
 * Canonical string over the decision-relevant fields only. The firewall reason
 * and experiment ids are deliberately excluded: the same rule decided over the
 * same models is one rule, whatever the stated reason or which run supplied the
 * evidence.
 */
export function canonicalizeRule(rule: GateRule): string {
  return JSON.stringify([
    ["task_key", rule.taskKey],
    ["candidate_model", rule.candidateModel],
    ["reference_model", rule.referenceModel],
    ["metric", rule.metric],
    ["mode", rule.mode],
    ["margin", rule.margin],
    ["confidence", rule.confidence],
    ["min_cases", rule.minCases],
    ["judge_abstain_max", rule.judgeAbstainMax],
    // Only present for an adoption gate, so every baseline hash is unchanged.
    ...(rule.candidatePromptHash != null
      ? [["candidate_prompt_hash", rule.candidatePromptHash]]
      : []),
  ]);
}

export function hashRule(rule: GateRule): string {
  return `sha256:${createHash("sha256").update(canonicalizeRule(rule)).digest("hex")}`;
}
