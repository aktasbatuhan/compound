import { describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createDatabase } from "../src/db";
import {
  BudgetExceededError,
  createExperiment,
  openReservationsUsd,
  releaseSpend,
  reserveSpend,
  settleSpend,
  totalSpendUsd,
} from "../src/execution";
import {
  claimDecisionCohort,
  cohortDigest,
  createGateSpec,
  recordDecisionCohort,
  recordGateResult,
  releaseDecisionClaim,
} from "../src/gate";
import { spendReservations } from "../src/schema";

const limits = { experimentCapUsd: 1, globalHardLimitUsd: 1 };

function memory() {
  return createDatabase({ path: ":memory:", migrate: true });
}

describe("spend reservations (#51)", () => {
  test("a reservation counts against the limit until it is settled", () => {
    const db = memory();
    const first = reserveSpend(db, {
      fingerprint: "a",
      estimatedCost: 0.6,
      experimentId: "e",
      ...limits,
    });
    expect(openReservationsUsd(db)).toBeCloseTo(0.6);
    expect(() =>
      reserveSpend(db, { fingerprint: "b", estimatedCost: 0.6, experimentId: "e", ...limits }),
    ).toThrow(BudgetExceededError);
    settleSpend(db, first, { fingerprint: "a", costUsd: 0.3, experimentId: "e" });
    expect(openReservationsUsd(db)).toBe(0);
    expect(totalSpendUsd(db)).toBeCloseTo(0.3);
    // Settling at the actual charge frees the difference.
    reserveSpend(db, { fingerprint: "b", estimatedCost: 0.6, experimentId: "e", ...limits });
    db.close();
  });

  test("two handles on one file cannot both reserve what only one can afford", () => {
    const dir = mkdtempSync(join(tmpdir(), "compound-reserve-"));
    const path = join(dir, "compound.db");
    const a = createDatabase({ path, migrate: true });
    const b = createDatabase({ path, migrate: false });
    reserveSpend(a, { fingerprint: "a", estimatedCost: 0.6, experimentId: "e", ...limits });
    expect(() =>
      reserveSpend(b, { fingerprint: "b", estimatedCost: 0.6, experimentId: "e", ...limits }),
    ).toThrow(BudgetExceededError);
    a.close();
    b.close();
  });

  test("release drops the reservation without recording spend", () => {
    const db = memory();
    const id = reserveSpend(db, {
      fingerprint: "a",
      estimatedCost: 0.6,
      experimentId: "e",
      ...limits,
    });
    releaseSpend(db, id);
    expect(openReservationsUsd(db)).toBe(0);
    expect(totalSpendUsd(db)).toBe(0);
    db.close();
  });

  test("a stale reservation from a dead process is ignored", () => {
    const db = memory();
    reserveSpend(db, { fingerprint: "a", estimatedCost: 0.9, experimentId: "e", ...limits });
    const stale = new Date(Date.now() - 20 * 60 * 1000);
    db.db.update(spendReservations).set({ createdAt: stale }).run();
    expect(openReservationsUsd(db)).toBe(0);
    reserveSpend(db, { fingerprint: "b", estimatedCost: 0.9, experimentId: "e", ...limits });
    db.close();
  });

  test("a re-executed fingerprint is ledgered again, not silently dropped", () => {
    const db = memory();
    const one = reserveSpend(db, {
      fingerprint: "same",
      estimatedCost: 0.1,
      experimentId: "e",
      ...limits,
    });
    settleSpend(db, one, { fingerprint: "same", costUsd: 0.1, experimentId: "e" });
    const two = reserveSpend(db, {
      fingerprint: "same",
      estimatedCost: 0.1,
      experimentId: "e",
      ...limits,
    });
    settleSpend(db, two, { fingerprint: "same", costUsd: 0.1, experimentId: "e" });
    expect(totalSpendUsd(db)).toBeCloseTo(0.2);
    db.close();
  });
});

describe("decision cohort claims (#54)", () => {
  const hashes = ["h1", "h2", "h3"];

  test("the second claimant on a cohort is refused until the first releases", () => {
    const db = memory();
    const first = claimDecisionCohort(db, "support", hashes);
    expect(first.claimed).toBe(true);
    const second = claimDecisionCohort(db, "support", [...hashes].reverse());
    expect(second.claimed).toBe(false);
    if (!second.claimed) expect(second.reason).toBe("held_by_another_gate");
    releaseDecisionClaim(db, first.digest);
    expect(claimDecisionCohort(db, "support", hashes).claimed).toBe(true);
    db.close();
  });

  test("a prior verdict on the cohort refuses the claim", () => {
    const db = memory();
    const spec = createGateSpec(db, {
      specHash: "sha256:claim-test",
      taskKey: "support",
      candidateModel: "cand",
      referenceModel: "ref",
      metric: "pass_rate",
      mode: "non_inferiority",
      margin: 0.05,
      confidence: 0.95,
      minCases: 3,
      judgeAbstainMax: 0,
      firewallReason: "claim test",
    });
    const experiment = (model: string) =>
      createExperiment(db, {
        taskKey: "support",
        candidateModel: model,
        provider: "openrouter",
        partition: "decision_test",
        paid: false,
      });
    const result = recordGateResult(db, {
      gateSpecId: spec.id,
      candidateExperimentId: experiment("cand").id,
      referenceExperimentId: experiment("ref").id,
      outcome: "meets_gate",
      delta: 0,
      ciLo: -0.01,
      ciHi: 0.01,
      n: 3,
      candidateRate: 1,
      referenceRate: 1,
      judgeAbstainedFraction: 0,
      decisionPartitionVersion: cohortDigest("support", hashes),
    });
    recordDecisionCohort(db, result.id, hashes);
    const claim = claimDecisionCohort(db, "support", hashes);
    expect(claim.claimed).toBe(false);
    if (!claim.claimed) expect(claim.reason).toBe("prior_decision");
    db.close();
  });

  test("the digest ignores hash order", () => {
    expect(cohortDigest("t", ["a", "b"])).toBe(cohortDigest("t", ["b", "a"]));
    expect(cohortDigest("t", ["a"])).not.toBe(cohortDigest("u", ["a"]));
  });
});
