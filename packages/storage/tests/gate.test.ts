import { describe, expect, test } from "bun:test";
import {
  createExperiment,
  createGateSpec,
  decidedCohort,
  getExperimentResults,
  getGateSpecByHash,
  insertCases,
  listGateResults,
  recordCaseResults,
  recordGateResult,
} from "../src/index";
import { freshDatabase } from "./helpers";

function anExperiment(handle: ReturnType<typeof freshDatabase>, model: string) {
  return createExperiment(handle, {
    taskKey: "support",
    candidateModel: model,
    provider: "openrouter",
    partition: "decision_test",
    paid: false,
  });
}

describe("per-case experiment results", () => {
  test("persist and read back, keyed by (experiment, case)", () => {
    const handle = freshDatabase();
    const exp = anExperiment(handle, "cand");
    recordCaseResults(handle, exp.id, [
      { caseId: "c1", status: "graded", passed: true, score: 1 },
      { caseId: "c2", status: "graded", passed: false, score: 0 },
      { caseId: "c3", status: "skipped" },
    ]);

    const rows = getExperimentResults(handle, exp.id);
    expect(rows.map((r) => r.caseId)).toEqual(["c1", "c2", "c3"]);
    expect(rows[0]?.passed).toBe(true);
    expect(rows[2]?.status).toBe("skipped");
    expect(rows[2]?.passed).toBeNull();
    handle.close();
  });

  test("decidedCohort reconstructs from recorded content hashes even after cases are pruned (#5)", () => {
    const handle = freshDatabase();
    const cand = anExperiment(handle, "cand");
    const ref = anExperiment(handle, "ref");
    // Record the content hash ON each result row (as the runner now does). No
    // `cases` rows exist at all — the join fallback would find nothing.
    const results = [
      { caseId: "c1", status: "graded" as const, passed: true, score: 1, contentHash: "h1" },
      { caseId: "c2", status: "graded" as const, passed: true, score: 1, contentHash: "h2" },
    ];
    recordCaseResults(handle, cand.id, results);
    recordCaseResults(handle, ref.id, results);
    const cohort = decidedCohort(handle, cand.id, ref.id);
    expect(cohort.contentHashes).toEqual(["h1", "h2"]);
    expect(cohort.version).not.toBeNull();
    handle.close();
  });

  test("decidedCohort is unreconstructable (null version) when neither the row nor a case has the hash (#5)", () => {
    const handle = freshDatabase();
    const cand = anExperiment(handle, "cand");
    const ref = anExperiment(handle, "ref");
    // Pre-migration rows: no content hash recorded, and the cases were pruned.
    const results = [{ caseId: "c1", status: "graded" as const, passed: true, score: 1 }];
    recordCaseResults(handle, cand.id, results);
    recordCaseResults(handle, ref.id, results);
    const cohort = decidedCohort(handle, cand.id, ref.id);
    expect(cohort.contentHashes).toEqual([]);
    expect(cohort.version).toBeNull();
    handle.close();
  });

  test("decidedCohort still falls back to the cases table for legacy rows with no recorded hash (#5)", () => {
    const handle = freshDatabase();
    insertCases(handle, [
      {
        caseId: "c1",
        taskKey: "support",
        sourceTraceId: "t1",
        contentHash: "h1",
        provenance: "human_golden",
        partition: "decision_test",
        input: {},
        expected: {},
      },
    ]);
    const cand = anExperiment(handle, "cand");
    const ref = anExperiment(handle, "ref");
    const results = [{ caseId: "c1", status: "graded" as const, passed: true, score: 1 }];
    recordCaseResults(handle, cand.id, results);
    recordCaseResults(handle, ref.id, results);
    const cohort = decidedCohort(handle, cand.id, ref.id);
    expect(cohort.contentHashes).toEqual(["h1"]);
    handle.close();
  });

  test("re-recording the same case overwrites rather than duplicates", () => {
    const handle = freshDatabase();
    const exp = anExperiment(handle, "cand");
    recordCaseResults(handle, exp.id, [
      { caseId: "c1", status: "graded", passed: false, score: 0 },
    ]);
    recordCaseResults(handle, exp.id, [{ caseId: "c1", status: "graded", passed: true, score: 1 }]);

    const rows = getExperimentResults(handle, exp.id);
    expect(rows).toHaveLength(1);
    expect(rows[0]?.passed).toBe(true);
    handle.close();
  });
});

describe("gate spec and result", () => {
  const specInput = {
    specHash: "sha256:abc",
    taskKey: "support",
    candidateModel: "cand",
    referenceModel: "ref",
    metric: "pass_rate" as const,
    mode: "non_inferiority" as const,
    margin: 0.05,
    confidence: 0.95,
    minCases: 20,
    judgeAbstainMax: 0,
    firewallReason: "final gate for support",
  };

  test("declaring the same spec hash twice reuses the row", () => {
    const handle = freshDatabase();
    const a = createGateSpec(handle, specInput);
    const b = createGateSpec(handle, specInput);
    expect(a.id).toBe(b.id);
    expect(getGateSpecByHash(handle, "sha256:abc")?.id).toBe(a.id);
    handle.close();
  });

  test("record a result and list it joined to its spec", () => {
    const handle = freshDatabase();
    const spec = createGateSpec(handle, specInput);
    const cand = anExperiment(handle, "cand");
    const ref = anExperiment(handle, "ref");
    recordGateResult(handle, {
      gateSpecId: spec.id,
      candidateExperimentId: cand.id,
      referenceExperimentId: ref.id,
      outcome: "meets_gate",
      delta: -0.01,
      ciLo: -0.03,
      ciHi: 0.01,
      n: 30,
      candidateRate: 0.9,
      referenceRate: 0.91,
      judgeAbstainedFraction: 0,
    });

    const listed = listGateResults(handle);
    expect(listed).toHaveLength(1);
    expect(listed[0]?.result.outcome).toBe("meets_gate");
    expect(listed[0]?.spec.firewallReason).toBe("final gate for support");
    handle.close();
  });
});
