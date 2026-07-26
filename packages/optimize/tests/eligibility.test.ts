import { describe, expect, test } from "bun:test";
import { assessEligibility } from "../src/eligibility";

describe("assessEligibility", () => {
  test("the GLM case — behind but within the band — is eligible", () => {
    const e = assessEligibility({ outcome: "insufficient_data", delta: -0.08 });
    expect(e.eligible).toBe(true);
    expect(e.reason).toBe("eligible");
    expect(e.gap).toBeCloseTo(0.08, 6);
  });

  test("a candidate that meets the gate is not eligible", () => {
    const e = assessEligibility({ outcome: "meets_gate", delta: 0 });
    expect(e.eligible).toBe(false);
    expect(e.reason).toBe("already_meets");
  });

  test("a candidate not behind the reference has no gap to close", () => {
    const e = assessEligibility({ outcome: "no_reliable_improvement", delta: 0.02 });
    expect(e.eligible).toBe(false);
    expect(e.reason).toBe("no_gap");
  });

  test("a hopeless gap says switch the model", () => {
    const e = assessEligibility({ outcome: "fails_gate", delta: -0.4 }, { ceiling: 0.25 });
    expect(e.eligible).toBe(false);
    expect(e.reason).toBe("hopeless");
  });

  test("no budget blocks eligibility regardless of the gap", () => {
    const e = assessEligibility({ outcome: "insufficient_data", delta: -0.08 }, { budgetUsd: 0 });
    expect(e.eligible).toBe(false);
    expect(e.reason).toBe("no_budget");
  });
});
