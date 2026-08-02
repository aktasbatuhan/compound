import { describe, expect, test } from "bun:test";
import {
  type CaseResultInput,
  type CompoundDatabase,
  createDatabase,
  createExperiment,
  createGateSpec,
  finishExperiment,
  insertCases,
  listGateResults,
  recordCaseResults,
  recordGateResult,
} from "@compound/storage";
import { decideGate, GateInputError } from "../src/decide";

function db(): CompoundDatabase {
  return createDatabase({ path: ":memory:", migrate: true });
}

function completedExperiment(handle: CompoundDatabase, model: string, results: CaseResultInput[]) {
  const exp = createExperiment(handle, {
    taskKey: "support",
    candidateModel: model,
    provider: "openrouter",
    partition: "decision_test",
    paid: false,
  });
  recordCaseResults(handle, exp.id, results);
  finishExperiment(handle, exp.id, "completed", {});
  return exp;
}

/** n graded cases; the first `passes` pass. */
function passResults(n: number, passes: number): CaseResultInput[] {
  return Array.from({ length: n }, (_, i) => ({
    caseId: `c${i}`,
    status: "graded" as const,
    passed: i < passes,
    score: i < passes ? 1 : 0,
  }));
}

const rule = {
  taskKey: "support",
  candidateModel: "cand",
  referenceModel: "ref",
  metric: "pass_rate" as const,
  mode: "non_inferiority" as const,
  margin: 0.05,
  confidence: 0.95,
  minCases: 20,
  judgeAbstainMax: 0,
  firewallReason: "final support gate",
};

describe("decideGate", () => {
  test("candidate matching reference meets the gate", () => {
    const handle = db();
    const cand = completedExperiment(handle, "cand", passResults(30, 27));
    const ref = completedExperiment(handle, "ref", passResults(30, 27));
    const { result, spec, pairs } = decideGate(handle, {
      ...rule,
      candidateExperimentId: cand.id,
      referenceExperimentId: ref.id,
    });
    expect(pairs).toHaveLength(30);
    expect(result.outcome).toBe("meets_gate");
    expect(result.delta).toBeCloseTo(0, 10);
    expect(spec.firewallReason).toBe("final support gate");
    handle.close();
  });

  test("a candidate far below reference fails", () => {
    const handle = db();
    const cand = completedExperiment(handle, "cand", passResults(30, 12));
    const ref = completedExperiment(handle, "ref", passResults(30, 28));
    const { result } = decideGate(handle, {
      ...rule,
      candidateExperimentId: cand.id,
      referenceExperimentId: ref.id,
    });
    expect(result.outcome).toBe("fails_gate");
    expect(result.delta).toBeLessThan(0);
    handle.close();
  });

  test("too few paired cases is insufficient_data", () => {
    const handle = db();
    const cand = completedExperiment(handle, "cand", passResults(10, 9));
    const ref = completedExperiment(handle, "ref", passResults(10, 9));
    const { result } = decideGate(handle, {
      ...rule,
      candidateExperimentId: cand.id,
      referenceExperimentId: ref.id,
    });
    expect(result.outcome).toBe("insufficient_data");
    expect(result.n).toBe(10);
    handle.close();
  });

  test("only cases graded on both sides are paired", () => {
    const handle = db();
    const cand = completedExperiment(handle, "cand", [
      ...passResults(25, 24),
      { caseId: "skip1", status: "skipped" },
    ]);
    const ref = completedExperiment(handle, "ref", [
      ...passResults(25, 24),
      { caseId: "skip1", status: "graded", passed: true, score: 1 },
    ]);
    const { pairs } = decideGate(handle, {
      ...rule,
      candidateExperimentId: cand.id,
      referenceExperimentId: ref.id,
    });
    expect(pairs).toHaveLength(25);
    expect(pairs.find((p) => p.caseId === "skip1")).toBeUndefined();
    handle.close();
  });

  test("declaring the identical rule reuses the spec row", () => {
    const handle = db();
    const cand = completedExperiment(handle, "cand", passResults(30, 27));
    const ref = completedExperiment(handle, "ref", passResults(30, 27));
    const first = decideGate(handle, {
      ...rule,
      candidateExperimentId: cand.id,
      referenceExperimentId: ref.id,
    });
    const second = decideGate(handle, {
      ...rule,
      candidateExperimentId: cand.id,
      referenceExperimentId: ref.id,
    });
    expect(first.spec.id).toBe(second.spec.id);
    handle.close();
  });

  test("refuses a non-decision partition", () => {
    const handle = db();
    const cand = createExperiment(handle, {
      taskKey: "support",
      candidateModel: "cand",
      provider: "openrouter",
      partition: "optimizer_validation",
      paid: false,
    });
    finishExperiment(handle, cand.id, "completed", {});
    const ref = completedExperiment(handle, "ref", passResults(30, 27));
    expect(() =>
      decideGate(handle, {
        ...rule,
        candidateExperimentId: cand.id,
        referenceExperimentId: ref.id,
      }),
    ).toThrow(GateInputError);
    handle.close();
  });

  test("a preview (persist: false) computes the verdict but writes nothing", () => {
    const handle = db();
    const cand = completedExperiment(handle, "cand", passResults(30, 27));
    const ref = completedExperiment(handle, "ref", passResults(30, 27));
    const { result, spec } = decideGate(handle, {
      ...rule,
      persist: false,
      candidateExperimentId: cand.id,
      referenceExperimentId: ref.id,
    });
    // The verdict is still computed and returned...
    expect(result.outcome).toBe("meets_gate");
    // ...but nothing was recorded and the seal was not "opened": no spec, no result.
    expect(spec.id).toBe("preview");
    expect(result.id).toBe("preview");
    expect(listGateResults(handle)).toHaveLength(0);
    handle.close();
  });

  test("a decision (default persist) records the rule with its provider and prompt hash", () => {
    const handle = db();
    const cand = completedExperiment(handle, "cand", passResults(30, 27));
    const ref = completedExperiment(handle, "ref", passResults(30, 27));
    decideGate(handle, {
      ...rule,
      candidateProvider: "openrouter",
      candidatePromptHash: "sha256:deadbeef",
      candidateExperimentId: cand.id,
      referenceExperimentId: ref.id,
    });
    const [decided] = listGateResults(handle, 1);
    expect(decided?.spec.candidateProvider).toBe("openrouter");
    expect(decided?.spec.candidatePromptHash).toBe("sha256:deadbeef");
    handle.close();
  });

  // --- The peeking guard (#22) -------------------------------------------
  // Seed sealed decision_test cases so the partition has a stable version.
  function sealCases(handle: CompoundDatabase, hashes: string[]) {
    insertCases(
      handle,
      hashes.map((hash, i) => ({
        caseId: `seal-${hash}`,
        taskKey: "support",
        sourceTraceId: `trace-${i}`,
        contentHash: hash,
        provenance: "human_golden" as const,
        partition: "decision_test" as const,
        input: {},
        expected: {},
      })),
    );
  }

  /** Graded results keyed by the SEALED case ids, so the decided cohort a gate
   *  binds to is exactly these held-out cases (the #3 fix depends on this). */
  function decisionResults(hashes: string[]): CaseResultInput[] {
    return hashes.map((h) => ({
      caseId: `seal-${h}`,
      status: "graded" as const,
      passed: true,
      score: 1,
    }));
  }

  /** A candidate+reference pair that both decided exactly the given sealed cases. */
  function decidedOn(handle: CompoundDatabase, hashes: string[]) {
    return {
      candidateExperimentId: completedExperiment(handle, "cand", decisionResults(hashes)).id,
      referenceExperimentId: completedExperiment(handle, "ref", decisionResults(hashes)).id,
    };
  }

  test("the first decision on a sealed set reports no prior decisions", () => {
    const handle = db();
    sealCases(handle, ["h1", "h2", "h3"]);
    const { partitionVersion, priorDecisions } = decideGate(handle, {
      ...rule,
      ...decidedOn(handle, ["h1", "h2", "h3"]),
    });
    expect(partitionVersion).not.toBeNull();
    expect(priorDecisions.count).toBe(0);
    handle.close();
  });

  test("a repeat decision counts the prior ones and names the first timestamp", () => {
    const handle = db();
    sealCases(handle, ["h1", "h2", "h3"]);
    decideGate(handle, { ...rule, ...decidedOn(handle, ["h1", "h2", "h3"]) });
    const second = decideGate(handle, { ...rule, ...decidedOn(handle, ["h1", "h2", "h3"]) });
    expect(second.priorDecisions.count).toBe(1);
    expect(second.priorDecisions.firstDecidedAt).toBeInstanceOf(Date);
    handle.close();
  });

  test("a preview does not spend the budget: prior count stays put", () => {
    const handle = db();
    sealCases(handle, ["h1", "h2", "h3"]);
    decideGate(handle, { ...rule, ...decidedOn(handle, ["h1", "h2", "h3"]), persist: false });
    decideGate(handle, { ...rule, ...decidedOn(handle, ["h1", "h2", "h3"]), persist: false });
    // Neither preview recorded a verdict, so a real decision still sees zero prior.
    const real = decideGate(handle, { ...rule, ...decidedOn(handle, ["h1", "h2", "h3"]) });
    expect(real.priorDecisions.count).toBe(0);
    handle.close();
  });

  test("blockRepeatDecision blocks any repeat that reuses the held-out labels", () => {
    const handle = db();
    sealCases(handle, ["h1", "h2", "h3"]);
    decideGate(handle, { ...rule, ...decidedOn(handle, ["h1", "h2", "h3"]) });
    // A second decision on the same held-out set is blocked...
    expect(() =>
      decideGate(handle, {
        ...rule,
        ...decidedOn(handle, ["h1", "h2", "h3"]),
        blockRepeatDecision: true,
      }),
    ).toThrow(GateInputError);
    // ...unless deliberately forced.
    const forced = decideGate(handle, {
      ...rule,
      ...decidedOn(handle, ["h1", "h2", "h3"]),
      blockRepeatDecision: true,
      force: true,
    });
    expect(forced.result.id).not.toBe("preview");
    handle.close();
  });

  test("extending the cohort by one case does NOT reset the guard (#3)", () => {
    const handle = db();
    sealCases(handle, ["h1", "h2", "h3"]);
    decideGate(handle, { ...rule, ...decidedOn(handle, ["h1", "h2", "h3"]) });
    // Re-curate: add a fresh case, then decide over ALL FOUR. The prior 3 labels
    // are reused, so this overlaps the prior decision and is still blocked — the
    // "add one case to reset the budget" bypass is closed.
    sealCases(handle, ["h4"]);
    const reused = decideGate(handle, {
      ...rule,
      ...decidedOn(handle, ["h1", "h2", "h3", "h4"]),
    });
    expect(reused.priorDecisions.count).toBe(1);
    expect(() =>
      decideGate(handle, {
        ...rule,
        ...decidedOn(handle, ["h1", "h2", "h3", "h4"]),
        blockRepeatDecision: true,
      }),
    ).toThrow(GateInputError);
    handle.close();
  });

  test("a genuinely fresh, non-overlapping decision set is not blocked (#3)", () => {
    const handle = db();
    sealCases(handle, ["h1", "h2", "h3", "h9", "h10"]);
    decideGate(handle, {
      ...rule,
      ...decidedOn(handle, ["h1", "h2", "h3"]),
      blockRepeatDecision: true,
    });
    // A decision over disjoint held-out labels shares nothing with the prior one.
    const fresh = decideGate(handle, {
      ...rule,
      ...decidedOn(handle, ["h9", "h10"]),
      blockRepeatDecision: true,
    });
    expect(fresh.priorDecisions.count).toBe(0);
    expect(fresh.priorDecisions.legacyCount).toBe(0);
    expect(fresh.result.id).not.toBe("preview");
    handle.close();
  });

  test("a legacy verdict with no recorded cohort trips the guard until forced (#9)", () => {
    const handle = db();
    sealCases(handle, ["h1", "h2", "h3"]);
    // A pre-guard verdict: recorded WITHOUT cohort membership (as the migration
    // backfill would leave one whose experiments were pruned). Its overlap can't
    // be proven, so an enabled guard must not silently allow a fresh decision.
    const spec = createGateSpec(handle, {
      specHash: "sha256:legacy",
      taskKey: "support",
      candidateModel: "cand",
      referenceModel: "ref",
      metric: "pass_rate",
      mode: "non_inferiority",
      margin: 0.05,
      confidence: 0.95,
      minCases: 20,
      judgeAbstainMax: 0,
      firewallReason: "a pre-guard release gate",
    });
    const legacy = decidedOn(handle, ["h1", "h2", "h3"]);
    recordGateResult(handle, {
      gateSpecId: spec.id,
      candidateExperimentId: legacy.candidateExperimentId,
      referenceExperimentId: legacy.referenceExperimentId,
      outcome: "meets_gate",
      delta: 0,
      ciLo: 0,
      ciHi: 0,
      n: 3,
      candidateRate: 1,
      referenceRate: 1,
      judgeAbstainedFraction: 0,
      decisionPartitionVersion: null,
    });
    const next = { ...rule, ...decidedOn(handle, ["h1", "h2", "h3"]) };
    const preview = decideGate(handle, { ...next, persist: false });
    expect(preview.priorDecisions.legacyCount).toBe(1);
    expect(() => decideGate(handle, { ...next, blockRepeatDecision: true })).toThrow(
      GateInputError,
    );
    const forced = decideGate(handle, { ...next, blockRepeatDecision: true, force: true });
    expect(forced.result.id).not.toBe("preview");
    handle.close();
  });

  test("the decided version reflects the cohort, not the live partition (#3)", () => {
    const handle = db();
    sealCases(handle, ["h1", "h2", "h3"]);
    const first = decideGate(handle, { ...rule, ...decidedOn(handle, ["h1", "h2", "h3"]) });
    // Growing the partition must NOT change the version of a decision made over
    // the original three cases — the version is bound to what was decided.
    sealCases(handle, ["h4"]);
    const sameCohort = decideGate(handle, { ...rule, ...decidedOn(handle, ["h1", "h2", "h3"]) });
    expect(sameCohort.partitionVersion).toBe(first.partitionVersion);
    handle.close();
  });

  // --- Coverage gate (#5) -------------------------------------------------
  test("reports how much of the sealed set was actually decided", () => {
    const handle = db();
    sealCases(handle, ["h1", "h2", "h3", "h4"]);
    // Both runs grade only 3 of the 4 sealed cases (both skip h4).
    const withSkip = (): CaseResultInput[] => [
      ...decisionResults(["h1", "h2", "h3"]),
      { caseId: "seal-h4", status: "skipped" as const },
    ];
    const { coverage } = decideGate(handle, {
      ...rule,
      minCases: 3,
      candidateExperimentId: completedExperiment(handle, "cand", withSkip()).id,
      referenceExperimentId: completedExperiment(handle, "ref", withSkip()).id,
    });
    expect(coverage.sealedTotal).toBe(4);
    expect(coverage.paired).toBe(3);
    expect(coverage.skippedCandidate).toBe(1);
    expect(coverage.skippedReference).toBe(1);
    expect(coverage.asymmetric).toBe(0); // both skipped the SAME case
    expect(coverage.skipFraction).toBeCloseTo(0.25, 10);
  });

  test("asymmetric skips void the verdict when the coverage gate is on (#5)", () => {
    const handle = db();
    sealCases(handle, ["h1", "h2", "h3", "h4"]);
    // The candidate grades h1..h4; the reference skips h4 → they disagree on
    // what they could grade, so the paired sample is self-selected.
    const cand = completedExperiment(handle, "cand", decisionResults(["h1", "h2", "h3", "h4"]));
    const ref = completedExperiment(handle, "ref", [
      ...decisionResults(["h1", "h2", "h3"]),
      { caseId: "seal-h4", status: "skipped" as const },
    ]);
    const { result, coverage } = decideGate(handle, {
      ...rule,
      minCases: 3,
      maxSkipFraction: 0.5,
      candidateExperimentId: cand.id,
      referenceExperimentId: ref.id,
    });
    expect(coverage.asymmetric).toBe(1);
    expect(coverage.shortfall).toBe(true);
    expect(result.outcome).toBe("insufficient_data");
  });

  test("too much omission voids the verdict when a max skip fraction is set (#5)", () => {
    const handle = db();
    sealCases(handle, ["h1", "h2", "h3", "h4", "h5", "h6"]);
    // Only 2 of 6 graded on both → 67% omitted, over a 0.3 ceiling.
    const partial = (): CaseResultInput[] => [
      ...decisionResults(["h1", "h2"]),
      ...["h3", "h4", "h5", "h6"].map((h) => ({
        caseId: `seal-${h}`,
        status: "skipped" as const,
      })),
    ];
    const { result, coverage } = decideGate(handle, {
      ...rule,
      minCases: 2,
      maxSkipFraction: 0.3,
      candidateExperimentId: completedExperiment(handle, "cand", partial()).id,
      referenceExperimentId: completedExperiment(handle, "ref", partial()).id,
    });
    expect(coverage.skipFraction).toBeCloseTo(4 / 6, 10);
    expect(coverage.shortfall).toBe(true);
    expect(result.outcome).toBe("insufficient_data");
  });

  test("refuses an empty firewall reason", () => {
    const handle = db();
    const cand = completedExperiment(handle, "cand", passResults(30, 27));
    const ref = completedExperiment(handle, "ref", passResults(30, 27));
    expect(() =>
      decideGate(handle, {
        ...rule,
        firewallReason: "   ",
        candidateExperimentId: cand.id,
        referenceExperimentId: ref.id,
      }),
    ).toThrow(GateInputError);
    handle.close();
  });
});
