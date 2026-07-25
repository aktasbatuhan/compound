import { describe, expect, test } from "bun:test";
import { bootstrapKappaCi, cohenKappa, mulberry32, seedFromString } from "../src/statistics";

describe("cohenKappa", () => {
  test("perfect agreement is 1", () => {
    expect(cohenKappa([1, 0, 1, 0, 1], [1, 0, 1, 0, 1])).toBeCloseTo(1, 10);
  });

  test("perfect disagreement is negative", () => {
    expect(cohenKappa([1, 1, 0, 0], [0, 0, 1, 1])).toBeLessThan(0);
  });

  test("chance-level agreement is near 0", () => {
    // Independent-ish labels: agreement no better than chance.
    const a = [1, 0, 1, 0, 1, 0, 1, 0];
    const b = [1, 1, 0, 0, 1, 1, 0, 0];
    expect(Math.abs(cohenKappa(a, b))).toBeLessThan(0.5);
  });

  test("both raters constant and agreeing is defined as 1", () => {
    expect(cohenKappa([1, 1, 1], [1, 1, 1])).toBe(1);
  });

  test("both raters constant but disagreeing is 0", () => {
    expect(cohenKappa([1, 1, 1], [0, 0, 0])).toBe(0);
  });

  test("empty input is 0", () => {
    expect(cohenKappa([], [])).toBe(0);
  });
});

describe("bootstrapKappaCi", () => {
  test("collapses to the point estimate for one pair", () => {
    expect(bootstrapKappaCi([1], [1], 0.95, 1)).toEqual({ kappa: 1, lo: 1, hi: 1, n: 1 });
  });

  test("is reproducible for a seed and brackets the point", () => {
    const a = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0];
    const b = [1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0];
    const seed = seedFromString("judge:v1");
    const x = bootstrapKappaCi(a, b, 0.95, seed, 2000);
    const y = bootstrapKappaCi(a, b, 0.95, seed, 2000);
    expect(x).toEqual(y);
    expect(x.lo).toBeLessThanOrEqual(x.kappa);
    expect(x.hi).toBeGreaterThanOrEqual(x.kappa);
  });

  test("strong agreement keeps the lower bound well above zero", () => {
    const a = Array.from({ length: 30 }, (_, i) => i % 2);
    const ci = bootstrapKappaCi(a, a, 0.95, seedFromString("agree"));
    expect(ci.lo).toBeGreaterThan(0.5);
  });
});

describe("mulberry32", () => {
  test("is deterministic and in range", () => {
    const r = mulberry32(42);
    const s = mulberry32(42);
    for (let i = 0; i < 100; i++) {
      const x = r();
      expect(x).toBe(s());
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
    }
  });
});
