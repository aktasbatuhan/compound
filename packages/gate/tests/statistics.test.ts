import { describe, expect, test } from "bun:test";
import {
  mean,
  mulberry32,
  pairedBootstrapCi,
  quantileSorted,
  seedFromString,
} from "../src/statistics";

describe("mulberry32", () => {
  test("is deterministic for a seed", () => {
    const a = mulberry32(123);
    const b = mulberry32(123);
    expect([a(), a(), a()]).toEqual([b(), b(), b()]);
  });
  test("stays in [0,1)", () => {
    const r = mulberry32(seedFromString("compound"));
    for (let i = 0; i < 1000; i++) {
      const x = r();
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
    }
  });
});

describe("quantileSorted", () => {
  test("interpolates between points", () => {
    expect(quantileSorted([0, 10], 0.5)).toBe(5);
    expect(quantileSorted([0, 1, 2, 3, 4], 0)).toBe(0);
    expect(quantileSorted([0, 1, 2, 3, 4], 1)).toBe(4);
  });
});

describe("pairedBootstrapCi", () => {
  test("collapses to the point with 0 or 1 observations", () => {
    expect(pairedBootstrapCi([], 0.95, 1)).toEqual({ point: 0, lo: 0, hi: 0, n: 0 });
    expect(pairedBootstrapCi([0.2], 0.95, 1)).toEqual({ point: 0.2, lo: 0.2, hi: 0.2, n: 1 });
  });

  test("all-equal diffs give a zero-width CI at the constant", () => {
    const ci = pairedBootstrapCi([0, 0, 0, 0, 0], 0.95, seedFromString("x"));
    expect(ci.point).toBe(0);
    expect(ci.lo).toBe(0);
    expect(ci.hi).toBe(0);
  });

  test("brackets the true mean and is reproducible for a seed", () => {
    const diffs = [1, -1, 1, -1, 1, -1, 1, -1, 0, 0];
    const seed = seedFromString("sha256:fixed");
    const a = pairedBootstrapCi(diffs, 0.95, seed, 2000);
    const b = pairedBootstrapCi(diffs, 0.95, seed, 2000);
    expect(a).toEqual(b);
    expect(a.point).toBeCloseTo(mean(diffs), 10);
    expect(a.lo).toBeLessThanOrEqual(a.point);
    expect(a.hi).toBeGreaterThanOrEqual(a.point);
  });

  test("a strongly positive sample keeps its lower bound above zero", () => {
    const diffs = new Array(40).fill(0.3);
    const ci = pairedBootstrapCi(diffs, 0.95, seedFromString("pos"));
    expect(ci.lo).toBeGreaterThan(0);
  });
});
