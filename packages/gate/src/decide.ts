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
  decidedCohort,
  decisionTestCaseIds,
  type ExperimentResultRow,
  type ExperimentRow,
  type GateMetric,
  type GateResultRow,
  type GateSpecRow,
  getExperiment,
  getExperimentResults,
  type PriorDecisions,
  priorDecisions,
  recordDecisionCohort,
  recordGateResult,
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
   * Coverage gate (#5): the largest fraction of the sealed set that may be
   * omitted (skipped/abstained on either side) before the decision is void.
   * When set and exceeded — or when the two sides skip DIFFERENT cases — the
   * verdict is forced to `insufficient_data` rather than decided on a silent,
   * self-selected subset. Undefined = report coverage but don't enforce.
   */
  maxSkipFraction?: number;
  /**
   * Persist the pre-declared spec and the decided result (default true). A
   * dry-run preview passes `false`: it computes and returns the same verdict but
   * writes nothing and does not "open" the seal — recording a verdict is a
   * deliberate, paid, one-time act, not a side effect of previewing one.
   */
  persist?: boolean;
  /**
   * The peeking guard (#22, #3): once this task's held-out labels have been
   * decided, block a further paid decision that REUSES any of the same cases —
   * repeated examination erodes the statistical guarantee, and extending the
   * cohort by a case does not escape it. Curate a fresh, non-overlapping
   * decision set to decide again, or pass `force`. Ignored on a preview.
   */
  blockRepeatDecision?: boolean;
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

/** How much of the sealed set the decision actually covered (#5, #11). */
export interface DecisionCoverage {
  /** Cases in the task's sealed decision partition. */
  sealedTotal: number;
  /** Graded on BOTH sides and not abstained — the diff sample (= pairs.length). */
  paired: number;
  /** Sealed cases excluded because a judge abstained on a side (never also counted as skipped). */
  abstained: number;
  /**
   * Sealed cases NOT graded on the candidate side — skipped, dry-run, or absent
   * from the run entirely — excluding abstentions. Counted over the WHOLE sealed
   * cohort, so a case missing from both runs is not silently dropped (#11). This
   * is a PER-SIDE total: a case skipped on BOTH sides is counted here AND in
   * `skippedReference`, so the two do not sum — use `skippedBoth` to reconcile.
   */
  skippedCandidate: number;
  /** Sealed cases NOT graded on the reference side, excluding abstentions; per-side total, overlaps `skippedCandidate` (#11). */
  skippedReference: number;
  /**
   * Sealed cases skipped on BOTH sides (excluding abstentions) — the overlap of
   * `skippedCandidate` and `skippedReference`. Disjoint from `asymmetric`, so the
   * skipped cases partition exactly as `skippedBoth + asymmetric` (#11).
   */
  skippedBoth: number;
  /** Cases graded on exactly ONE side — the two runs disagree on what they could grade. */
  asymmetric: number;
  /** (sealedTotal − paired) / sealedTotal: the omitted fraction the guard checks. */
  skipFraction: number;
  /** True when coverage voided the verdict (fraction exceeded, or asymmetric skips). */
  shortfall: boolean;
}

export interface DecideGateResult {
  spec: GateSpecRow;
  result: GateResultRow;
  pairs: PairedCase[];
  /** How much of the sealed set was actually decided vs silently omitted (#5). */
  coverage: DecisionCoverage;
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

/**
 * Measure how much of the sealed set the decision actually covered (#5). The
 * gate decides on the paired intersection, which can silently shrink the sample
 * (e.g. agentic cases skipped for a blocked tool). This surfaces the omissions —
 * per side and, crucially, when the two runs skipped DIFFERENT cases — so a
 * verdict on a self-selected subset can be flagged instead of trusted.
 */
export function decisionCoverage(
  sealedCaseIds: readonly string[],
  candidateRows: ExperimentResultRow[],
  referenceRows: ExperimentResultRow[],
  pairedCount: number,
  maxSkipFraction?: number,
): DecisionCoverage {
  const cand = byCase(candidateRows);
  const ref = byCase(referenceRows);
  const graded = (row: ExperimentResultRow | undefined): boolean =>
    row?.status === "graded" && !row.judgeAbstained;
  const abstainedOn = (row: ExperimentResultRow | undefined): boolean =>
    row?.judgeAbstained === true;
  const sealedTotal = sealedCaseIds.length;
  let abstained = 0;
  let skippedCandidate = 0;
  let skippedReference = 0;
  let skippedBoth = 0;
  let asymmetric = 0;
  // Iterate the WHOLE sealed cohort so a case absent from BOTH runs is still
  // counted. Abstention is its own exclusive bucket; the remaining skipped cases
  // partition disjointly into skippedBoth vs asymmetric (one side only). The
  // per-side `skippedCandidate`/`skippedReference` are totals that overlap on
  // `skippedBoth`, so they are NOT meant to be summed (#11).
  for (const id of sealedCaseIds) {
    const c = cand.get(id);
    const r = ref.get(id);
    const cGraded = graded(c);
    const rGraded = graded(r);
    if (cGraded && rGraded) continue; // paired
    if (abstainedOn(c) || abstainedOn(r)) {
      abstained += 1; // excluded by abstention, not a skip
      continue;
    }
    if (!cGraded) skippedCandidate += 1;
    if (!rGraded) skippedReference += 1;
    if (!cGraded && !rGraded) skippedBoth += 1; // exclusive: skipped on both sides
    if (cGraded !== rGraded) asymmetric += 1; // exclusive: gradeable on exactly one side
  }
  const skipFraction = sealedTotal > 0 ? (sealedTotal - pairedCount) / sealedTotal : 0;
  const overFraction = maxSkipFraction !== undefined && skipFraction > maxSkipFraction;
  const shortfall = maxSkipFraction !== undefined && (overFraction || asymmetric > 0);
  return {
    sealedTotal,
    paired: pairedCount,
    abstained,
    skippedCandidate,
    skippedReference,
    skippedBoth,
    asymmetric,
    skipFraction,
    shortfall,
  };
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
    // The coverage threshold changes the verdict, so it is part of the rule
    // identity (#6): a gate at 0.6 and one at 0.4 must not share a spec hash.
    maxSkipFraction: input.maxSkipFraction ?? null,
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

  const decided = decideOutcome({
    mode: rule.mode,
    margin: rule.margin,
    ciLo: ci.lo,
    ciHi: ci.hi,
    n: pairs.length,
    minCases: rule.minCases,
    judgeAbstainedFraction,
    judgeAbstainMax: rule.judgeAbstainMax,
  });

  // Coverage gate (#5): the decision runs on the paired intersection, which can
  // silently omit sealed cases. If too much of the set was omitted, or the two
  // runs skipped different cases, the sample is self-selected — void the verdict
  // to `insufficient_data` rather than decide on it.
  const sealedCaseIds = decisionTestCaseIds(db, rule.taskKey);
  const coverage = decisionCoverage(
    sealedCaseIds,
    getExperimentResults(db, candidate.id),
    getExperimentResults(db, reference.id),
    pairs.length,
    input.maxSkipFraction,
  );
  const outcome = coverage.shortfall ? "insufficient_data" : decided;

  const candidateRate = pairs.length > 0 ? mean(pairs.map((p) => p.candidateScore)) : 0;
  const referenceRate = pairs.length > 0 ? mean(pairs.map((p) => p.referenceScore)) : 0;

  // The peeking budget (#22, #3): the EXACT held-out cohort these two runs
  // decided, and how often any of those same labels were already decided.
  // Computed on a preview too, so a dry run can warn before you pay.
  const cohort = decidedCohort(db, candidate.id, reference.id);
  const partitionVersion = cohort.version;
  const prior = priorDecisions(db, rule.taskKey, cohort.contentHashes);
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
      maxSkipFraction: rule.maxSkipFraction ?? null,
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
    return { spec, result, pairs, coverage, partitionVersion, priorDecisions: prior, isAdoption };
  }

  // The peeking guard: a paid decision that reuses ANY previously-decided label
  // for this task is blocked unless deliberately forced — reusing part of the
  // held-out set re-examines it just the same, so extending the cohort by a case
  // does not escape the guard (#3). Legacy verdicts with no recorded cohort
  // (pre-guard, or pruned experiments) also trip it: their overlap can't be
  // proven, so an override must be explicit (#9).
  // Fail closed (#5): a non-empty paired sample whose cohort can't be
  // reconstructed (cases pruned, no recorded content hash) can't be checked for
  // overlap at all — so the guard can't prove this is a fresh decision. Refuse
  // rather than silently treat an unverifiable cohort as "never decided".
  if (
    input.blockRepeatDecision &&
    pairs.length > 0 &&
    cohort.contentHashes.length === 0 &&
    input.force !== true
  ) {
    throw new GateInputError(
      `cannot verify the held-out cohort for '${rule.taskKey}': ${pairs.length} paired case(s) ` +
        "were decided but their content hashes are no longer reconstructable (cases pruned). " +
        "The peeking guard cannot prove this decision does not reuse already-decided labels. " +
        "Re-curate the decision set, or pass --force with a stated escalation reason.",
    );
  }

  if (
    input.blockRepeatDecision &&
    (prior.count >= 1 || prior.legacyCount >= 1) &&
    input.force !== true
  ) {
    const first = prior.firstDecidedAt?.toISOString() ?? "an earlier run";
    const legacyNote =
      prior.legacyCount >= 1
        ? ` and ${prior.legacyCount} earlier verdict(s) with no recorded cohort`
        : "";
    throw new GateInputError(
      `the held-out decision set for '${rule.taskKey}' has already been decided: ` +
        `${prior.count} prior verdict(s) reuse these same cases (first ${first})${legacyNote}. ` +
        "Deciding again re-examines the held-out labels and erodes the guarantee. Curate a " +
        "fresh, non-overlapping decision set, or pass --force with a stated escalation reason.",
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
    ...(rule.maxSkipFraction != null ? { maxSkipFraction: rule.maxSkipFraction } : {}),
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
  // Record the exact cohort this verdict decided, so a future decision that
  // reuses any of these labels is caught by the peeking guard (#3).
  recordDecisionCohort(db, result.id, cohort.contentHashes);

  return { spec, result, pairs, coverage, partitionVersion, priorDecisions: prior, isAdoption };
}

function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}
