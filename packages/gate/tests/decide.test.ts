import { describe, expect, test } from "bun:test";
import {
  type CaseResultInput,
  type CompoundDatabase,
  createDatabase,
  createExperiment,
  finishExperiment,
  insertCases,
  listGateResults,
  recordCaseResults,
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

  test("the first decision on a sealed set reports no prior decisions", () => {
    const handle = db();
    sealCases(handle, ["h1", "h2", "h3"]);
    const cand = completedExperiment(handle, "cand", passResults(30, 27));
    const ref = completedExperiment(handle, "ref", passResults(30, 27));
    const { partitionVersion, priorDecisions } = decideGate(handle, {
      ...rule,
      candidateExperimentId: cand.id,
      referenceExperimentId: ref.id,
    });
    expect(partitionVersion).not.toBeNull();
    expect(priorDecisions.count).toBe(0);
    handle.close();
  });

  test("a repeat decision counts the prior ones and names the first timestamp", () => {
    const handle = db();
    sealCases(handle, ["h1", "h2", "h3"]);
    const cand = completedExperiment(handle, "cand", passResults(30, 27));
    const ref = completedExperiment(handle, "ref", passResults(30, 27));
    const opts = { ...rule, candidateExperimentId: cand.id, referenceExperimentId: ref.id };
    decideGate(handle, opts);
    const second = decideGate(handle, opts);
    expect(second.priorDecisions.count).toBe(1);
    expect(second.priorDecisions.firstDecidedAt).toBeInstanceOf(Date);
    handle.close();
  });

  test("a preview does not spend the budget: prior count stays put", () => {
    const handle = db();
    sealCases(handle, ["h1", "h2"]);
    const cand = completedExperiment(handle, "cand", passResults(30, 27));
    const ref = completedExperiment(handle, "ref", passResults(30, 27));
    const opts = { ...rule, candidateExperimentId: cand.id, referenceExperimentId: ref.id };
    decideGate(handle, { ...opts, persist: false });
    decideGate(handle, { ...opts, persist: false });
    // Neither preview recorded a verdict, so a real decision still sees zero prior.
    const real = decideGate(handle, opts);
    expect(real.priorDecisions.count).toBe(0);
    handle.close();
  });

  test("blockRepeatAfterAdoption blocks a repeat once the set was adopted against", () => {
    const handle = db();
    sealCases(handle, ["h1", "h2", "h3"]);
    const cand = completedExperiment(handle, "cand", passResults(30, 27));
    const ref = completedExperiment(handle, "ref", passResults(30, 27));
    const opts = { ...rule, candidateExperimentId: cand.id, referenceExperimentId: ref.id };
    // First decision is an ADOPTION (an optimized prompt under test).
    decideGate(handle, {
      ...opts,
      candidatePromptHash: "sha256:opt",
      blockRepeatAfterAdoption: true,
    });
    // A second decision on the same sealed set is now blocked...
    expect(() => decideGate(handle, { ...opts, blockRepeatAfterAdoption: true })).toThrow(
      GateInputError,
    );
    // ...unless deliberately forced.
    const forced = decideGate(handle, { ...opts, blockRepeatAfterAdoption: true, force: true });
    expect(forced.result.id).not.toBe("preview");
    handle.close();
  });

  test("a first non-adoption decision is never blocked", () => {
    const handle = db();
    sealCases(handle, ["h1", "h2", "h3"]);
    const cand = completedExperiment(handle, "cand", passResults(30, 27));
    const ref = completedExperiment(handle, "ref", passResults(30, 27));
    const opts = { ...rule, candidateExperimentId: cand.id, referenceExperimentId: ref.id };
    decideGate(handle, opts); // baseline (no prompt hash) — not an adoption
    // Still allowed to decide again: nothing was adopted yet, so warn-only.
    const second = decideGate(handle, { ...opts, blockRepeatAfterAdoption: true });
    expect(second.priorDecisions.count).toBe(1);
    expect(second.priorDecisions.adoptionCount).toBe(0);
    handle.close();
  });

  test("re-curating the decision set resets the budget (new partition version)", () => {
    const handle = db();
    sealCases(handle, ["h1", "h2", "h3"]);
    const cand = completedExperiment(handle, "cand", passResults(30, 27));
    const ref = completedExperiment(handle, "ref", passResults(30, 27));
    const opts = { ...rule, candidateExperimentId: cand.id, referenceExperimentId: ref.id };
    const first = decideGate(handle, { ...opts, candidatePromptHash: "sha256:opt" });
    // Re-curation adds a new sealed case → the partition version changes.
    sealCases(handle, ["h4"]);
    const afterRecurate = decideGate(handle, { ...opts, blockRepeatAfterAdoption: true });
    expect(afterRecurate.partitionVersion).not.toBe(first.partitionVersion);
    // A genuinely new held-out set is a fresh test: no prior decisions, not blocked.
    expect(afterRecurate.priorDecisions.count).toBe(0);
    handle.close();
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
