import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  type CaseInsert,
  type CompoundDatabase,
  countCases,
  countCasesByPartition,
  countCasesByProvenance,
  createDatabase,
  getCase,
  InvalidPromotionError,
  insertCases,
  listCases,
  migrate,
  openDecisionFirewall,
  reviewCase,
  UnknownCaseError,
} from "../src/index";

let db: CompoundDatabase;

beforeEach(() => {
  db = createDatabase();
  migrate(db);
});

afterEach(() => {
  db.close();
});

let seq = 0;
function caseInsert(overrides: Partial<CaseInsert> = {}): CaseInsert {
  seq += 1;
  return {
    caseId: `case:${seq}`,
    taskKey: "support",
    sourceTraceId: `langfuse:tr-${seq}`,
    contentHash: `hash-${seq}`,
    provenance: "observed_output",
    partition: "optimization_train",
    input: { model: "gpt-4o", input: [{ role: "user", content: "hi" }] },
    expected: { role: "assistant", content: "hello" },
    ...overrides,
  };
}

describe("insertCases", () => {
  test("inserts new cases and returns their ids", () => {
    const result = insertCases(db, [caseInsert(), caseInsert()]);
    expect(result.inserted).toBe(2);
    expect(result.duplicates).toBe(0);
    expect(countCases(db)).toBe(2);
  });

  test("dedupes on (task_key, content_hash), counting duplicates on the survivor", () => {
    const first = caseInsert({ contentHash: "shared", caseId: "case:a" });
    insertCases(db, [first]);
    const dup = caseInsert({ contentHash: "shared", caseId: "case:b" });
    const result = insertCases(db, [dup]);

    expect(result.inserted).toBe(0);
    expect(result.duplicates).toBe(1);
    expect(countCases(db)).toBe(1);
    expect(getCase(db, "case:a")?.duplicateCount).toBe(1);
    expect(getCase(db, "case:b")).toBeNull();
  });

  test("the same content hash under a different task is a distinct case", () => {
    insertCases(db, [caseInsert({ contentHash: "shared", taskKey: "support", caseId: "s" })]);
    const result = insertCases(db, [
      caseInsert({ contentHash: "shared", taskKey: "billing", caseId: "b" }),
    ]);
    expect(result.inserted).toBe(1);
    expect(countCases(db)).toBe(2);
  });
});

describe("the decision-test firewall", () => {
  beforeEach(() => {
    insertCases(db, [
      caseInsert({ partition: "optimization_train" }),
      caseInsert({ partition: "optimizer_validation" }),
      caseInsert({ partition: "judge_calibration" }),
      caseInsert({ partition: "decision_test" }),
      caseInsert({ partition: "decision_test" }),
    ]);
  });

  test("listCases excludes decision_test by default", () => {
    const rows = listCases(db, {});
    expect(rows).toHaveLength(3);
    expect(rows.every((row) => row.partition !== "decision_test")).toBe(true);
  });

  test("even an explicit decision_test filter returns nothing without the firewall", () => {
    // The seal wins over the filter: you cannot opt in by asking nicely.
    expect(listCases(db, { partition: "decision_test" })).toHaveLength(0);
  });

  test("countCases never counts the sealed set without the firewall", () => {
    expect(countCases(db)).toBe(3);
    expect(countCases(db, { partition: "decision_test" })).toBe(0);
  });

  test("an opened firewall returns the sealed cases", () => {
    const token = openDecisionFirewall("final gate for task support, 2026-07-23");
    const rows = listCases(db, { partition: "decision_test", openDecisionFirewall: token });
    expect(rows).toHaveLength(2);
    expect(rows.every((row) => row.partition === "decision_test")).toBe(true);
  });

  test("opening the firewall requires a non-empty reason", () => {
    expect(() => openDecisionFirewall("")).toThrow();
    expect(() => openDecisionFirewall("   ")).toThrow();
  });

  test("countCasesByPartition reveals the sealed size without revealing the cases", () => {
    // Knowing 2 cases are sealed is fine; reading them is not.
    const byPartition = countCasesByPartition(db);
    const decision = byPartition.find((row) => row.partition === "decision_test");
    expect(decision?.count).toBe(2);
  });
});

describe("counts", () => {
  test("countCasesByProvenance groups by provenance", () => {
    insertCases(db, [
      caseInsert({ provenance: "observed_output" }),
      caseInsert({ provenance: "observed_output" }),
      caseInsert({ provenance: "deterministic_outcome" }),
    ]);
    const byProvenance = countCasesByProvenance(db);
    const observed = byProvenance.find((row) => row.provenance === "observed_output");
    expect(observed?.count).toBe(2);
  });
});

describe("reviewCase", () => {
  test("moves a case to approved", () => {
    insertCases(db, [caseInsert({ caseId: "c1" })]);
    const reviewed = reviewCase(db, "c1", { reviewState: "approved" });
    expect(reviewed.reviewState).toBe("approved");
  });

  test("approving with promotion is the ONLY path to human_golden", () => {
    insertCases(db, [caseInsert({ caseId: "c1", provenance: "observed_output" })]);
    const golden = reviewCase(db, "c1", {
      reviewState: "approved",
      expected: { role: "assistant", content: "human-verified answer" },
      promoteToGolden: true,
    });
    expect(golden.provenance).toBe("human_golden");
    expect((golden.expected as { content: string }).content).toBe("human-verified answer");
  });

  test("refuses to promote without approval", () => {
    insertCases(db, [caseInsert({ caseId: "c1" })]);
    expect(() =>
      reviewCase(db, "c1", { reviewState: "needs_edit", promoteToGolden: true }),
    ).toThrow(InvalidPromotionError);
    // Provenance is unchanged after a refused promotion.
    expect(getCase(db, "c1")?.provenance).toBe("observed_output");
  });

  test("throws for an unknown case", () => {
    expect(() => reviewCase(db, "nope", { reviewState: "approved" })).toThrow(UnknownCaseError);
  });
});
