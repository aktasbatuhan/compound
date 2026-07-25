import { describe, expect, test } from "bun:test";
import {
  type CaseResultInput,
  type CompoundDatabase,
  createDatabase,
  createExperiment,
  finishExperiment,
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
