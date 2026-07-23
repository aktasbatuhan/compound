import { describe, expect, test } from "bun:test";
import type { Message } from "@compound/contract";
import { type Assertion, type AssertionResult, evaluateAssertions } from "../src/index";

function textOutput(text: string): { output: Message } {
  return { output: { role: "assistant", content: text } };
}

function toolOutput(toolCalls: NonNullable<Message["tool_calls"]>): { output: Message } {
  return { output: { role: "assistant", content: null, tool_calls: toolCalls } };
}

function run(assertion: Assertion, subject: { output: Message | null }): AssertionResult {
  const result = evaluateAssertions([assertion], subject).results[0];
  if (result === undefined) throw new Error("expected one result");
  return result;
}

describe("valid_json", () => {
  test("passes on JSON output, fails on prose", () => {
    expect(run({ type: "valid_json" }, textOutput('{"a":1}')).passed).toBe(true);
    expect(run({ type: "valid_json" }, textOutput("hello")).passed).toBe(false);
  });

  test("fails cleanly (no throw) when there is no output", () => {
    const result = run({ type: "valid_json" }, { output: null });
    expect(result.passed).toBe(false);
    expect(result.detail).toContain("no text content");
  });
});

describe("json_schema", () => {
  const schema = {
    type: "object",
    properties: { name: { type: "string" } },
    required: ["name"],
  };
  test("passes matching JSON, fails on schema mismatch", () => {
    expect(run({ type: "json_schema", schema }, textOutput('{"name":"x"}')).passed).toBe(true);
    const bad = run({ type: "json_schema", schema }, textOutput('{"age":1}'));
    expect(bad.passed).toBe(false);
    expect(bad.detail).toContain("schema mismatch");
  });
});

describe("contains / not_contains", () => {
  test("substring presence", () => {
    expect(run({ type: "contains", value: "refund" }, textOutput("your refund")).passed).toBe(true);
    expect(run({ type: "not_contains", value: "as an AI" }, textOutput("Sure!")).passed).toBe(true);
    expect(run({ type: "not_contains", value: "as an AI" }, textOutput("as an AI I")).passed).toBe(
      false,
    );
  });

  test("ignore_case", () => {
    expect(
      run({ type: "contains", value: "REFUND", ignore_case: true }, textOutput("refund")).passed,
    ).toBe(true);
  });
});

describe("regex", () => {
  test("matches and reports an invalid pattern without throwing", () => {
    expect(run({ type: "regex", pattern: "^\\d{3}$" }, textOutput("123")).passed).toBe(true);
    const invalid = run({ type: "regex", pattern: "(" }, textOutput("x"));
    expect(invalid.passed).toBe(false);
    expect(invalid.detail).toContain("invalid regex");
  });
});

describe("equals and json_path_equals", () => {
  test("equals on the whole text", () => {
    expect(run({ type: "equals", value: "yes" }, textOutput("yes")).passed).toBe(true);
    expect(run({ type: "equals", value: "yes" }, textOutput("no")).passed).toBe(false);
  });

  test("json_path_equals into parsed output", () => {
    const out = textOutput('{"data":{"items":[{"id":7}]}}');
    expect(run({ type: "json_path_equals", path: "data.items.0.id", value: 7 }, out).passed).toBe(
      true,
    );
    expect(run({ type: "json_path_equals", path: "data.items.5.id", value: 7 }, out).passed).toBe(
      false,
    );
  });
});

describe("tool assertions", () => {
  const output = toolOutput([
    { id: "1", name: "lookup_order", arguments: { order_id: "A1" } },
    { id: "2", name: "lookup_order", arguments: { order_id: "A2" } },
  ]);

  test("tool_called with min_times", () => {
    expect(run({ type: "tool_called", name: "lookup_order" }, output).passed).toBe(true);
    expect(run({ type: "tool_called", name: "lookup_order", min_times: 2 }, output).passed).toBe(
      true,
    );
    expect(run({ type: "tool_called", name: "lookup_order", min_times: 3 }, output).passed).toBe(
      false,
    );
    expect(run({ type: "tool_called", name: "refund" }, output).passed).toBe(false);
  });

  test("tool_not_called", () => {
    expect(run({ type: "tool_not_called", name: "refund" }, output).passed).toBe(true);
    expect(run({ type: "tool_not_called", name: "lookup_order" }, output).passed).toBe(false);
  });

  test("tool_arg_equals across multiple calls", () => {
    expect(
      run({ type: "tool_arg_equals", name: "lookup_order", arg: "order_id", value: "A2" }, output)
        .passed,
    ).toBe(true);
    expect(
      run({ type: "tool_arg_equals", name: "lookup_order", arg: "order_id", value: "Z9" }, output)
        .passed,
    ).toBe(false);
  });

  test("tool assertions on a text-only output fail cleanly", () => {
    expect(run({ type: "tool_called", name: "x" }, textOutput("hi")).passed).toBe(false);
  });
});

describe("max_length", () => {
  test("bounds the output length", () => {
    expect(run({ type: "max_length", max: 5 }, textOutput("hi")).passed).toBe(true);
    expect(run({ type: "max_length", max: 5 }, textOutput("hello world")).passed).toBe(false);
  });
});

describe("aggregate report", () => {
  test("passed is true only when every required assertion passes", () => {
    const report = evaluateAssertions(
      [{ type: "valid_json" }, { type: "contains", value: "name" }],
      textOutput('{"name":"x"}'),
    );
    expect(report.passed).toBe(true);
    expect(report.score).toBe(1);
  });

  test("an optional failing assertion lowers score but does not fail the case", () => {
    const report = evaluateAssertions(
      [{ type: "valid_json" }, { type: "contains", value: "missing", required: false }],
      textOutput('{"a":1}'),
    );
    expect(report.passed).toBe(true);
    expect(report.score).toBe(0.5);
  });

  test("a required failing assertion fails the case", () => {
    const report = evaluateAssertions(
      [{ type: "valid_json" }, { type: "contains", value: "missing" }],
      textOutput('{"a":1}'),
    );
    expect(report.passed).toBe(false);
  });

  test("weights bias the score", () => {
    const report = evaluateAssertions(
      [
        { type: "contains", value: "yes", weight: 3, required: false },
        { type: "contains", value: "no", weight: 1, required: false },
      ],
      textOutput("yes"),
    );
    expect(report.score).toBe(0.75);
  });

  test("no assertions is a vacuous pass with score 1", () => {
    const report = evaluateAssertions([], textOutput("anything"));
    expect(report.passed).toBe(true);
    expect(report.score).toBe(1);
  });

  test("never echoes the raw output in a failure detail", () => {
    const secret = "SENSITIVE-abc123-token";
    const report = evaluateAssertions([{ type: "valid_json" }], textOutput(secret));
    expect(JSON.stringify(report)).not.toContain(secret);
  });
});
