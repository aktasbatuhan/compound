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
  ]);
}

export function hashRule(rule: GateRule): string {
  return `sha256:${createHash("sha256").update(canonicalizeRule(rule)).digest("hex")}`;
}
