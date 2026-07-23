import { describe, expect, test } from "bun:test";
import { type Assertion, gradeCaseObservedOutput } from "../src/index";

const assertions: Assertion[] = [
  { type: "valid_json" },
  { type: "not_contains", value: "as an AI" },
];

describe("gradeCaseObservedOutput", () => {
  test("grades an observed_output case's stored message", () => {
    const report = gradeCaseObservedOutput(
      {
        caseId: "case:1",
        taskKey: "support",
        provenance: "observed_output",
        expected: { role: "assistant", content: '{"answer":"30 days"}' },
      },
      assertions,
    );
    expect(report.graded).toBe(true);
    expect(report.passed).toBe(true);
    expect(report.caseId).toBe("case:1");
  });

  test("fails a case whose observed output breaks a required assertion", () => {
    const report = gradeCaseObservedOutput(
      {
        caseId: "case:2",
        taskKey: "support",
        provenance: "observed_output",
        expected: { role: "assistant", content: "as an AI I cannot" },
      },
      assertions,
    );
    expect(report.graded).toBe(true);
    expect(report.passed).toBe(false);
  });

  test("does not grade a non-observed_output case", () => {
    const report = gradeCaseObservedOutput(
      {
        caseId: "case:3",
        taskKey: "support",
        provenance: "deterministic_outcome",
        expected: { status: "success" },
      },
      assertions,
    );
    // Nothing to check yet — reported as ungraded, not failed.
    expect(report.graded).toBe(false);
    expect(report.results).toHaveLength(0);
  });

  test("does not grade a case whose expected is not a message", () => {
    const report = gradeCaseObservedOutput(
      {
        caseId: "case:4",
        taskKey: "support",
        provenance: "observed_output",
        expected: null,
      },
      assertions,
    );
    expect(report.graded).toBe(false);
  });
});
