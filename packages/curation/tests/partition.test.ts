import { describe, expect, test } from "bun:test";
import {
  assignPartition,
  DEFAULT_PARTITION_RATIOS,
  InvalidPartitionRatiosError,
  type PartitionRatios,
  partitionFraction,
} from "../src/partition";

function randomHash(): string {
  return crypto.randomUUID().replace(/-/g, "").padEnd(64, "0");
}

describe("assignPartition", () => {
  test("is deterministic: the same hash and salt always yield the same partition", () => {
    const hash = randomHash();
    const salt = { taskKey: "support" };
    const first = assignPartition(hash, salt);
    for (let i = 0; i < 20; i += 1) {
      expect(assignPartition(hash, salt)).toBe(first);
    }
  });

  test("the same work under different tasks can land in different partitions", () => {
    // Not guaranteed for one hash, but across many, the task salt must matter.
    let differed = false;
    for (let i = 0; i < 200 && !differed; i += 1) {
      const hash = randomHash();
      if (assignPartition(hash, { taskKey: "a" }) !== assignPartition(hash, { taskKey: "b" })) {
        differed = true;
      }
    }
    expect(differed).toBe(true);
  });

  test("bumping the version re-partitions (deliberately, never by accident)", () => {
    let differed = false;
    for (let i = 0; i < 200 && !differed; i += 1) {
      const hash = randomHash();
      const a = assignPartition(hash, { taskKey: "t", version: "v1" });
      const b = assignPartition(hash, { taskKey: "t", version: "v2" });
      if (a !== b) differed = true;
    }
    expect(differed).toBe(true);
  });

  test("respects the configured ratios within tolerance over many hashes", () => {
    const counts: Record<string, number> = {};
    const n = 20_000;
    for (let i = 0; i < n; i += 1) {
      const p = assignPartition(randomHash(), { taskKey: "t" });
      counts[p] = (counts[p] ?? 0) + 1;
    }
    // Default 70/15/10/5. Allow generous slack for a 20k sample.
    expect((counts.optimization_train ?? 0) / n).toBeCloseTo(0.7, 1);
    expect((counts.optimizer_validation ?? 0) / n).toBeCloseTo(0.15, 1);
    expect((counts.judge_calibration ?? 0) / n).toBeCloseTo(0.1, 1);
    expect((counts.decision_test ?? 0) / n).toBeCloseTo(0.05, 1);
  });

  test("a zero-decision ratio never assigns to the sealed set", () => {
    const ratios: PartitionRatios = {
      optimization_train: 80,
      optimizer_validation: 15,
      judge_calibration: 5,
      decision_test: 0,
    };
    for (let i = 0; i < 5_000; i += 1) {
      expect(assignPartition(randomHash(), { taskKey: "t" }, ratios)).not.toBe("decision_test");
    }
  });

  test("rejects invalid ratios rather than silently normalizing them", () => {
    expect(() =>
      assignPartition(
        "hash",
        { taskKey: "t" },
        {
          optimization_train: -1,
          optimizer_validation: 1,
          judge_calibration: 1,
          decision_test: 1,
        },
      ),
    ).toThrow(InvalidPartitionRatiosError);

    expect(() =>
      assignPartition(
        "hash",
        { taskKey: "t" },
        {
          optimization_train: 0,
          optimizer_validation: 0,
          judge_calibration: 0,
          decision_test: 0,
        },
      ),
    ).toThrow(InvalidPartitionRatiosError);
  });

  test("uses all default ratios keys", () => {
    // Guard against a ratios key being dropped from the default.
    expect(Object.keys(DEFAULT_PARTITION_RATIOS).sort()).toEqual([
      "decision_test",
      "judge_calibration",
      "optimization_train",
      "optimizer_validation",
    ]);
  });
});

describe("partitionFraction", () => {
  test("is in [0, 1)", () => {
    for (let i = 0; i < 1000; i += 1) {
      const f = partitionFraction(randomHash(), "salt");
      expect(f).toBeGreaterThanOrEqual(0);
      expect(f).toBeLessThan(1);
    }
  });

  test("is stable for the same inputs", () => {
    const hash = randomHash();
    expect(partitionFraction(hash, "s")).toBe(partitionFraction(hash, "s"));
  });
});
