import { describe, expect, test } from "bun:test";
import { type DecideInput, decideOutcome } from "../src/rule";

const base: DecideInput = {
  mode: "non_inferiority",
  margin: 0.05,
  ciLo: 0,
  ciHi: 0,
  n: 30,
  minCases: 20,
  judgeAbstainedFraction: 0,
  judgeAbstainMax: 0,
};

describe("decideOutcome — non-inferiority", () => {
  test("meets when the lower bound clears -margin", () => {
    expect(decideOutcome({ ...base, ciLo: -0.04, ciHi: 0.02 })).toBe("meets_gate");
  });
  test("meets when candidate is dead even (CI [0,0])", () => {
    expect(decideOutcome({ ...base, ciLo: 0, ciHi: 0 })).toBe("meets_gate");
  });
  test("fails when even the best case is worse than -margin", () => {
    expect(decideOutcome({ ...base, ciLo: -0.2, ciHi: -0.08 })).toBe("fails_gate");
  });
  test("insufficient when the CI straddles -margin", () => {
    expect(decideOutcome({ ...base, ciLo: -0.09, ciHi: 0.01 })).toBe("insufficient_data");
  });
});

describe("decideOutcome — gates independent of the CI", () => {
  test("too few cases is insufficient_data", () => {
    expect(decideOutcome({ ...base, n: 5, ciLo: 0.1, ciHi: 0.2 })).toBe("insufficient_data");
  });
  test("excess judge abstention voids the decision", () => {
    expect(
      decideOutcome({ ...base, judgeAbstainedFraction: 0.5, judgeAbstainMax: 0.1, ciLo: 0.1 }),
    ).toBe("judge_abstained");
  });
  test("abstention check precedes the min-cases check", () => {
    expect(
      decideOutcome({ ...base, n: 2, judgeAbstainedFraction: 0.9, judgeAbstainMax: 0.1 }),
    ).toBe("judge_abstained");
  });
});

describe("decideOutcome — superiority", () => {
  const sup: DecideInput = { ...base, mode: "superiority", margin: 0 };
  test("meets only when the whole CI is above zero", () => {
    expect(decideOutcome({ ...sup, ciLo: 0.02, ciHi: 0.09 })).toBe("meets_gate");
  });
  test("no reliable improvement when the CI includes zero", () => {
    expect(decideOutcome({ ...sup, ciLo: -0.01, ciHi: 0.05 })).toBe("no_reliable_improvement");
  });
  test("fails when reliably worse", () => {
    expect(decideOutcome({ ...sup, ciLo: -0.1, ciHi: -0.02 })).toBe("fails_gate");
  });
});
