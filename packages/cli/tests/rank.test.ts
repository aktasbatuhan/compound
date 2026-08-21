import { describe, expect, test } from "bun:test";
import {
  type AxisValues,
  formatPriority,
  normalizePriority,
  paretoChartLines,
  parsePriority,
  rankRows,
} from "../src/rank";

describe("parsePriority", () => {
  test("parses and normalizes an already-normalized vector", () => {
    const p = parsePriority("quality=0.5,cost=0.3,latency=0.2");
    expect(p.quality).toBeCloseTo(0.5, 9);
    expect(p.cost).toBeCloseTo(0.3, 9);
    expect(p.latency).toBeCloseTo(0.2, 9);
    expect(p.throughput).toBe(0); // unspecified axes weigh 0
  });

  test("normalizes weights on any positive scale", () => {
    const p = parsePriority("quality=2,cost=2");
    expect(p.quality).toBeCloseTo(0.5, 9);
    expect(p.cost).toBeCloseTo(0.5, 9);
    expect(p.latency).toBe(0);
  });

  test("rejects an unknown axis, naming the vocabulary", () => {
    expect(() => parsePriority("speed=1")).toThrow(
      "unknown priority axis 'speed'; one of: quality, cost, latency, throughput",
    );
  });

  test("rejects a malformed weight", () => {
    expect(() => parsePriority("quality=abc")).toThrow(
      "priority weight for 'quality' must be a non-negative number",
    );
  });

  test("rejects a negative weight", () => {
    expect(() => parsePriority("quality=-1")).toThrow("non-negative");
  });

  test("rejects an all-zero vector", () => {
    expect(() => parsePriority("quality=0,cost=0")).toThrow(
      "priority needs at least one positive weight",
    );
  });

  test("rejects a duplicated axis", () => {
    expect(() => parsePriority("quality=1,quality=2")).toThrow("given twice");
  });

  test("rejects a pair without '='", () => {
    expect(() => parsePriority("quality")).toThrow("is not axis=weight");
  });

  test("formatPriority prints only the positively weighted axes", () => {
    expect(formatPriority(parsePriority("quality=0.5,cost=0.5"))).toBe("quality=0.50 cost=0.50");
  });
});

describe("normalizePriority", () => {
  test("fills unspecified axes with 0 and sums to 1", () => {
    const p = normalizePriority({ quality: 3, cost: 1 });
    expect(p.quality).toBeCloseTo(0.75, 9);
    expect(p.cost).toBeCloseTo(0.25, 9);
    expect(p.latency).toBe(0);
    expect(p.throughput).toBe(0);
  });

  test("rejects a vector with no positive weight", () => {
    expect(() => normalizePriority({})).toThrow("at least one positive weight");
  });
});

// A hand-computed three-route fixture. With quality=0.5, cost=0.3, latency=0.2:
//   quality  min 0.5 max 1.0 → A 1,   B 0,   C 0.6
//   cost     min .005 max .01 (inverted) → A 0, B 0.8, C 1
//   latency  min 250 max 1000 (inverted) → A 2/3, B 0, C 1
//   score(A) = .5·1 + .3·0 + .2·(2/3) = 0.63333…
//   score(B) = .5·0 + .3·.8 + .2·0    = 0.24
//   score(C) = .5·.6 + .3·1 + .2·1    = 0.8
// C dominates B (better on all three); A trades quality for cost/latency, so
// the frontier is {C, A}.
const THREE_ROUTES: { name: string; values: AxisValues }[] = [
  { name: "A", values: { quality: 1.0, cost: 0.01, latency: 500, throughput: null } },
  { name: "B", values: { quality: 0.5, cost: 0.006, latency: 1000, throughput: null } },
  { name: "C", values: { quality: 0.8, cost: 0.005, latency: 250, throughput: null } },
];

describe("rankRows", () => {
  const priority = parsePriority("quality=0.5,cost=0.3,latency=0.2");

  test("scores and ranks the fixture exactly as hand-computed", () => {
    const result = rankRows(THREE_ROUTES, (r) => r.values, priority);
    expect(result.ranked.map((r) => r.row.name)).toEqual(["C", "A", "B"]);
    expect(result.ranked.map((r) => r.rank)).toEqual([1, 2, 3]);
    expect(result.ranked[0]?.score).toBeCloseTo(0.8, 9);
    expect(result.ranked[1]?.score).toBeCloseTo(0.5 + 0.2 * (2 / 3), 9);
    expect(result.ranked[2]?.score).toBeCloseTo(0.24, 9);
    expect(result.excluded).toEqual([]);
    expect(result.droppedAxes).toEqual([]);
    expect(result.allDegenerate).toBe(false);
  });

  test("flags the Pareto frontier and names the dominator of the dominated row", () => {
    const result = rankRows(THREE_ROUTES, (r) => r.values, priority);
    const byName = new Map(result.ranked.map((r) => [r.row.name, r]));
    expect(byName.get("C")?.onFrontier).toBe(true);
    expect(byName.get("A")?.onFrontier).toBe(true); // best quality — not dominated
    expect(byName.get("B")?.onFrontier).toBe(false);
    expect(byName.get("B")?.dominatedBy?.name).toBe("C");
  });

  test("an unweighted axis plays no part: zero throughput weight ignores throughput nulls", () => {
    const result = rankRows(THREE_ROUTES, (r) => r.values, priority);
    expect(result.excluded).toEqual([]); // throughput is null everywhere but unweighted
  });

  test("excludes a row missing a weighted axis, naming the axis", () => {
    const rows = [
      { name: "A", values: { quality: 1, cost: 0.01, latency: 500, throughput: null } },
      { name: "B", values: { quality: 0.5, cost: 0.005, latency: null, throughput: null } },
    ];
    const result = rankRows(rows, (r) => r.values, priority);
    expect(result.ranked.map((r) => r.row.name)).toEqual(["A"]);
    expect(result.excluded).toHaveLength(1);
    expect(result.excluded[0]?.row.name).toBe("B");
    expect(result.excluded[0]?.missingAxes).toEqual(["latency"]);
  });

  test("drops a degenerate axis and renormalizes the remaining weights", () => {
    // Quality identical everywhere → only cost can rank; its weight becomes 1.
    const rows = [
      { name: "cheap", values: { quality: 0.8, cost: 1, latency: null, throughput: null } },
      { name: "dear", values: { quality: 0.8, cost: 2, latency: null, throughput: null } },
    ];
    const result = rankRows(rows, (r) => r.values, parsePriority("quality=0.5,cost=0.5"));
    expect(result.droppedAxes).toEqual(["quality"]);
    expect(result.effectiveWeights.cost).toBeCloseTo(1, 9);
    expect(result.ranked.map((r) => r.row.name)).toEqual(["cheap", "dear"]);
    expect(result.ranked[0]?.score).toBeCloseTo(1, 9);
    expect(result.ranked[1]?.score).toBeCloseTo(0, 9);
    expect(result.allDegenerate).toBe(false);
  });

  test("all axes degenerate: no signal, every row ties at rank 1 on the frontier", () => {
    const rows = [
      { name: "X", values: { quality: 0.8, cost: 1, latency: null, throughput: null } },
      { name: "Y", values: { quality: 0.8, cost: 1, latency: null, throughput: null } },
    ];
    const result = rankRows(rows, (r) => r.values, parsePriority("quality=0.5,cost=0.5"));
    expect(result.allDegenerate).toBe(true);
    expect(result.droppedAxes).toEqual(["quality", "cost"]);
    expect(result.ranked.map((r) => r.rank)).toEqual([1, 1]);
    expect(result.ranked.every((r) => r.score === 0)).toBe(true);
    expect(result.ranked.every((r) => r.onFrontier)).toBe(true); // nothing strictly better
  });

  test("exact score ties share a rank; the next rank skips (competition ranking)", () => {
    const rows = [
      { name: "T1", values: { quality: 1, cost: 1, latency: null, throughput: null } },
      { name: "T2", values: { quality: 1, cost: 1, latency: null, throughput: null } },
      { name: "W", values: { quality: 0, cost: 2, latency: null, throughput: null } },
    ];
    const result = rankRows(rows, (r) => r.values, parsePriority("quality=0.5,cost=0.5"));
    expect(result.ranked.map((r) => r.rank)).toEqual([1, 1, 3]);
    expect(result.ranked[0]?.score).toBeCloseTo(1, 9);
    expect(result.ranked[1]?.score).toBeCloseTo(1, 9);
  });
});

describe("paretoChartLines", () => {
  const points = [
    { mark: "1", cost: 0.005, quality: 0.8, onFrontier: true },
    { mark: "2", cost: 0.01, quality: 1.0, onFrontier: true },
    { mark: "3", cost: 0.006, quality: 0.5, onFrontier: false },
  ];

  test("plots frontier points as their mark and dominated points as ·", () => {
    const lines = paretoChartLines(points);
    expect(lines.length).toBeGreaterThan(0);
    const body = lines.join("\n");
    expect(body).toContain("1");
    expect(body).toContain("2");
    expect(body).toContain("·");
    // Axis labels: quality extremes and cost extremes.
    expect(body).toContain("100%");
    expect(body).toContain("50%");
    expect(body).toContain("$0.00500");
    expect(body).toContain("$0.01000");
  });

  test("returns nothing when an axis has no spread", () => {
    const flat = points.map((p) => ({ ...p, quality: 0.8 }));
    expect(paretoChartLines(flat)).toEqual([]);
    expect(paretoChartLines(points.slice(0, 1))).toEqual([]);
  });
});
