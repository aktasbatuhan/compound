/**
 * Decide a gate: pair the candidate and reference experiments case-by-case on
 * the sealed decision partition, compute the paired bootstrap CI on the chosen
 * metric, apply the pre-declared rule, and persist the verdict.
 *
 * The gate makes ZERO provider calls. It consumes two already-completed
 * experiments; running them (money-safe, capped, cached) is the caller's job.
 */
import {
  type CompoundDatabase,
  createGateSpec,
  type ExperimentResultRow,
  type ExperimentRow,
  type GateMetric,
  type GateResultRow,
  type GateSpecRow,
  getExperiment,
  getExperimentResults,
  type PriorDecisions,
  priorDecisions,
  recordGateResult,
  sealedPartitionVersion,
} from "@compound/storage";
import { decideOutcome } from "./rule";
import { type GateRule, hashRule } from "./spec";
import { pairedBootstrapCi, seedFromString } from "./statistics";

export interface DecideGateInput extends GateRule {
  candidateExperimentId: string;
  referenceExperimentId: string;
  /** Required: the stated reason for opening the sealed partition. */
  firewallReason: string;
  /** Provenance for an adoption gate: the optimization artifact under test. */
  optimizationRunId?: string;
  bootstrapIterations?: number;
  /**
   * Persist the pre-declared spec and the decided result (default true). A
   * dry-run preview passes `false`: it computes and returns the same verdict but
   * writes nothing and does not "open" the seal — recording a verdict is a
   * deliberate, paid, one-time act, not a side effect of previewing one.
   */
  persist?: boolean;
  /**
   * The peeking guard (#22): once the sealed set has been decided under an
   * adoption gate, block a further paid decision on the SAME held-out set —
   * repeated examination erodes its statistical guarantee. A re-curation (which
   * changes the partition version) resets the budget. Ignored on a preview.
   */
  blockRepeatAfterAdoption?: boolean;
  /** Override the peeking block for a deliberate, stated re-decision. */
  force?: boolean;
}

/** One case present in both runs, with each side's metric value. */
export interface PairedCase {
  caseId: string;
  candidateScore: number;
  referenceScore: number;
  candidatePassed: boolean | null;
  referencePassed: boolean | null;
  diff: number;
  abstained: boolean;
}

export interface DecideGateResult {
  spec: GateSpecRow;
  result: GateResultRow;
  pairs: PairedCase[];
  /** Fingerprint of the sealed set decided on; null if the task has no sealed cases. */
  partitionVersion: string | null;
  /** How often this sealed set was decided BEFORE this decision (the peeking budget, #22). */
  priorDecisions: PriorDecisions;
  /** Whether this gate puts an optimized prompt under test (an adoption decision). */
  isAdoption: boolean;
}

export class GateInputError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GateInputError";
  }
}

function metricValue(row: ExperimentResultRow, metric: GateMetric): number {
  if (metric === "pass_rate") return row.passed ? 1 : 0;
  return row.score ?? 0;
}

function byCase(rows: ExperimentResultRow[]): Map<string, ExperimentResultRow> {
  const m = new Map<string, ExperimentResultRow>();
  for (const r of rows) m.set(r.caseId, r);
  return m;
}

function assertComparable(
  candidate: ExperimentRow,
  reference: ExperimentRow,
  rule: GateRule,
): void {
  if (candidate.partition !== "decision_test" || reference.partition !== "decision_test") {
    throw new GateInputError(
      "a gate decides only on the sealed decision_test partition; " +
        `got candidate=${candidate.partition}, reference=${reference.partition}`,
    );
  }
  if (candidate.taskKey !== rule.taskKey || reference.taskKey !== rule.taskKey) {
    throw new GateInputError("candidate and reference experiments must match the rule's task_key");
  }
  if (candidate.status !== "completed" || reference.status !== "completed") {
    throw new GateInputError("both experiments must be completed before a gate can decide");
  }
}

/**
 * Build the paired sample. A case contributes only if BOTH runs graded it
 * (skipped/dry-run cases can't yield a difference). A case where either side's
 * judge abstained is counted toward the abstention fraction and excluded from
 * the diff sample.
 */
export function pairCases(
  candidateRows: ExperimentResultRow[],
  referenceRows: ExperimentResultRow[],
  metric: GateMetric,
): { pairs: PairedCase[]; abstainedCount: number; presentCount: number } {
  const ref = byCase(referenceRows);
  const pairs: PairedCase[] = [];
  let abstainedCount = 0;
  let presentCount = 0;

  for (const c of candidateRows) {
    const r = ref.get(c.caseId);
    if (r === undefined) continue;
    const bothGraded = c.status === "graded" && r.status === "graded";
    const abstained = c.judgeAbstained || r.judgeAbstained;
    if (!bothGraded && !abstained) continue; // not comparable (a skip on one side)
    presentCount += 1;
    if (abstained) {
      abstainedCount += 1;
      continue;
    }
    const candidateScore = metricValue(c, metric);
    const referenceScore = metricValue(r, metric);
    pairs.push({
      caseId: c.caseId,
      candidateScore,
      referenceScore,
      candidatePassed: c.passed,
      referencePassed: r.passed,
      diff: candidateScore - referenceScore,
      abstained: false,
    });
  }
  return { pairs, abstainedCount, presentCount };
}

export function decideGate(db: CompoundDatabase, input: DecideGateInput): DecideGateResult {
  if (input.firewallReason.trim().length === 0) {
    throw new GateInputError("a gate needs a firewall reason: state why the sealed set is opened");
  }
  if (input.mode === "non_inferiority" && input.margin <= 0) {
    throw new GateInputError("non-inferiority requires a positive margin");
  }

  const candidate = getExperiment(db, input.candidateExperimentId);
  const reference = getExperiment(db, input.referenceExperimentId);
  if (candidate === null || reference === null) {
    throw new GateInputError("candidate or reference experiment not found");
  }
  assertComparable(candidate, reference, input);

  const rule: GateRule = {
    taskKey: input.taskKey,
    candidateModel: input.candidateModel,
    referenceModel: input.referenceModel,
    metric: input.metric,
    mode: input.mode,
    margin: input.margin,
    confidence: input.confidence,
    minCases: input.minCases,
    judgeAbstainMax: input.judgeAbstainMax,
    candidatePromptHash: input.candidatePromptHash ?? null,
    candidateProvider: input.candidateProvider ?? null,
    referenceProvider: input.referenceProvider ?? null,
  };
  const specHash = hashRule(rule);

  // Pairing and the bootstrap CI are pure reads — safe on a preview.
  const { pairs, abstainedCount, presentCount } = pairCases(
    getExperimentResults(db, candidate.id),
    getExperimentResults(db, reference.id),
    rule.metric,
  );

  const diffs = pairs.map((p) => p.diff);
  const ci = pairedBootstrapCi(
    diffs,
    rule.confidence,
    seedFromString(specHash),
    input.bootstrapIterations,
  );
  const judgeAbstainedFraction = presentCount > 0 ? abstainedCount / presentCount : 0;

  const outcome = decideOutcome({
    mode: rule.mode,
    margin: rule.margin,
    ciLo: ci.lo,
    ciHi: ci.hi,
    n: pairs.length,
    minCases: rule.minCases,
    judgeAbstainedFraction,
    judgeAbstainMax: rule.judgeAbstainMax,
  });

  const candidateRate = pairs.length > 0 ? mean(pairs.map((p) => p.candidateScore)) : 0;
  const referenceRate = pairs.length > 0 ? mean(pairs.map((p) => p.referenceScore)) : 0;

  // The peeking budget (#22): how often this exact sealed set was already
  // decided. Computed on a preview too, so a dry run can warn before you pay.
  const partitionVersion = sealedPartitionVersion(db, rule.taskKey);
  const prior = priorDecisions(db, rule.taskKey, partitionVersion);
  const isAdoption = rule.candidatePromptHash != null || input.optimizationRunId != null;

  // A dry-run preview (`persist: false`) writes nothing: it neither declares the
  // rule nor records a verdict, so the sealed set is not "opened". Only a
  // deliberate (paid) decision persists the spec-before-result pair.
  if (input.persist === false) {
    const now = new Date();
    const spec: GateSpecRow = {
      id: "preview",
      specHash,
      taskKey: rule.taskKey,
      candidateModel: rule.candidateModel,
      referenceModel: rule.referenceModel,
      metric: rule.metric,
      mode: rule.mode,
      margin: rule.margin,
      confidence: rule.confidence,
      minCases: rule.minCases,
      judgeAbstainMax: rule.judgeAbstainMax,
      candidatePromptHash: rule.candidatePromptHash ?? null,
      optimizationRunId: input.optimizationRunId ?? null,
      candidateProvider: rule.candidateProvider ?? null,
      referenceProvider: rule.referenceProvider ?? null,
      firewallReason: input.firewallReason,
      createdAt: now,
    };
    const result: GateResultRow = {
      id: "preview",
      gateSpecId: "preview",
      candidateExperimentId: candidate.id,
      referenceExperimentId: reference.id,
      outcome,
      delta: ci.point,
      ciLo: ci.lo,
      ciHi: ci.hi,
      n: pairs.length,
      candidateRate,
      referenceRate,
      judgeAbstainedFraction,
      decisionPartitionVersion: partitionVersion,
      decidedAt: now,
    };
    return { spec, result, pairs, partitionVersion, priorDecisions: prior, isAdoption };
  }

  // The peeking guard: once this sealed set has been adopted against, a further
  // paid decision on the SAME held-out labels is blocked unless deliberately
  // forced. A re-curation changes the partition version and resets the budget.
  if (input.blockRepeatAfterAdoption && prior.adoptionCount >= 1 && input.force !== true) {
    const first = prior.firstDecidedAt?.toISOString() ?? "an earlier run";
    throw new GateInputError(
      `the sealed decision set for '${rule.taskKey}' has already been decided ${prior.count}× ` +
        `(${prior.adoptionCount} adoption decision(s), first ${first}); deciding it again ` +
        "re-examines the held-out labels and erodes the guarantee. Re-curate the decision set " +
        "for a fresh test, or pass --force with a stated escalation reason to override.",
    );
  }

  // Store the pre-declared rule (idempotent on hash) BEFORE recording the result.
  const spec = createGateSpec(db, {
    specHash,
    taskKey: rule.taskKey,
    candidateModel: rule.candidateModel,
    referenceModel: rule.referenceModel,
    metric: rule.metric,
    mode: rule.mode,
    margin: rule.margin,
    confidence: rule.confidence,
    minCases: rule.minCases,
    judgeAbstainMax: rule.judgeAbstainMax,
    ...(rule.candidatePromptHash != null ? { candidatePromptHash: rule.candidatePromptHash } : {}),
    ...(input.optimizationRunId !== undefined
      ? { optimizationRunId: input.optimizationRunId }
      : {}),
    ...(rule.candidateProvider != null ? { candidateProvider: rule.candidateProvider } : {}),
    ...(rule.referenceProvider != null ? { referenceProvider: rule.referenceProvider } : {}),
    firewallReason: input.firewallReason,
  });

  const result = recordGateResult(db, {
    gateSpecId: spec.id,
    candidateExperimentId: candidate.id,
    referenceExperimentId: reference.id,
    outcome,
    delta: ci.point,
    ciLo: ci.lo,
    ciHi: ci.hi,
    n: pairs.length,
    candidateRate,
    referenceRate,
    judgeAbstainedFraction,
    decisionPartitionVersion: partitionVersion,
  });

  return { spec, result, pairs, partitionVersion, priorDecisions: prior, isAdoption };
}

function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}
