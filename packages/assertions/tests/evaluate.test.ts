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

  test("equals is key-order insensitive on objects (#10)", () => {
    const out = textOutput('{"data":{"b":2,"a":1}}');
    // The value resolved at the path is an object whose keys differ in order from
    // the expected value; structural equality passes where JSON.stringify fails.
    expect(run({ type: "equals", path: "data", value: { a: 1, b: 2 } }, out).passed).toBe(true);
  });

  test("json_path_equals is key-order insensitive on objects (#10)", () => {
    const out = textOutput('{"data":{"b":2,"a":1}}');
    expect(run({ type: "json_path_equals", path: "data", value: { a: 1, b: 2 } }, out).passed).toBe(
      true,
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

  test("tool_arg_equals is key-order insensitive on object arguments (#10)", () => {
    const objArg = toolOutput([{ id: "1", name: "book", arguments: { seat: { row: 1, col: 2 } } }]);
    expect(
      run({ type: "tool_arg_equals", name: "book", arg: "seat", value: { col: 2, row: 1 } }, objArg)
        .passed,
    ).toBe(true);
  });

  test("tool assertions on a text-only output fail cleanly", () => {
    expect(run({ type: "tool_called", name: "x" }, textOutput("hi")).passed).toBe(false);
  });
});

describe("tool_call_arg", () => {
  const disputeRight = toolOutput([
    {
      id: "1",
      name: "dispute_charge",
      arguments: { amount: 23, account: "acct_9", reason: "fraud" },
    },
  ]);
  const disputeWrong = toolOutput([
    {
      id: "1",
      name: "dispute_charge",
      arguments: { amount: 99, account: "acct_9", reason: "fraud" },
    },
  ]);

  // The core correctness hole from #21: the right tool with the wrong amount
  // must FAIL, where the old `tool_called` check would have passed it.
  test("equals grades the argument value, not just that the tool ran", () => {
    const assertion: Assertion = {
      type: "tool_call_arg",
      name: "dispute_charge",
      arg: "amount",
      match: { equals: 23 },
    };
    expect(run(assertion, disputeRight).passed).toBe(true);
    const wrong = run(assertion, disputeWrong);
    expect(wrong.passed).toBe(false);
    expect(wrong.detail).toContain("never");
  });

  test("subset pins the arguments that matter and tolerates the rest", () => {
    const assertion: Assertion = {
      type: "tool_call_arg",
      name: "dispute_charge",
      match: { subset: { amount: 23, account: "acct_9" } },
    };
    expect(run(assertion, disputeRight).passed).toBe(true);
    expect(run(assertion, disputeWrong).passed).toBe(false);
  });

  test("a wrong argument's diagnostic names the mismatched argument (#21)", () => {
    const result = run(
      { type: "tool_call_arg", name: "dispute_charge", match: { subset: { amount: 23 } } },
      disputeWrong,
    );
    expect(result.passed).toBe(false);
    expect(result.detail).toContain("'amount' expected 23, got 99");
  });

  test("subset matches nested arguments recursively — extra nested keys allowed", () => {
    const output = toolOutput([
      {
        id: "1",
        name: "search",
        arguments: { query: "flights", filters: { min: 10, max: 50, sort: "asc" } },
      },
    ]);
    const assertion: Assertion = {
      type: "tool_call_arg",
      name: "search",
      match: { subset: { filters: { min: 10, max: 50 } } },
    };
    expect(run(assertion, output).passed).toBe(true);
  });

  test("a wrong nested argument fails and the diagnostic names its dot-path (#21)", () => {
    const output = toolOutput([
      { id: "1", name: "search", arguments: { filters: { min: 12, max: 50 } } },
    ]);
    const result = run(
      { type: "tool_call_arg", name: "search", match: { subset: { filters: { min: 10 } } } },
      output,
    );
    expect(result.passed).toBe(false);
    expect(result.detail).toContain("'filters.min' expected 10, got 12");
  });

  test("a pinned argument the call never sent fails as missing, by name (#21)", () => {
    const result = run(
      { type: "tool_call_arg", name: "dispute_charge", match: { subset: { memo: "wire" } } },
      disputeRight,
    );
    expect(result.passed).toBe(false);
    expect(result.detail).toContain("missing 'memo'");
  });

  test("regex matches a stringified argument", () => {
    const output = toolOutput([{ id: "1", name: "email", arguments: { to: "user@example.com" } }]);
    expect(
      run(
        { type: "tool_call_arg", name: "email", arg: "to", match: { regex: "@example\\.com$" } },
        output,
      ).passed,
    ).toBe(true);
    expect(
      run(
        { type: "tool_call_arg", name: "email", arg: "to", match: { regex: "@other\\.com$" } },
        output,
      ).passed,
    ).toBe(false);
  });

  test("schema validates the argument's shape", () => {
    const output = toolOutput([{ id: "1", name: "book", arguments: { seats: 2 } }]);
    const schema = { type: "integer", minimum: 1 };
    expect(
      run({ type: "tool_call_arg", name: "book", arg: "seats", match: { schema } }, output).passed,
    ).toBe(true);
    const bad = toolOutput([{ id: "1", name: "book", arguments: { seats: 0 } }]);
    expect(
      run({ type: "tool_call_arg", name: "book", arg: "seats", match: { schema } }, bad).passed,
    ).toBe(false);
  });

  test("a dot-path reaches a nested argument", () => {
    const output = toolOutput([
      { id: "1", name: "search", arguments: { filters: { min: 10, max: 50 } } },
    ]);
    expect(
      run(
        { type: "tool_call_arg", name: "search", arg: "filters.min", match: { equals: 10 } },
        output,
      ).passed,
    ).toBe(true);
  });

  test("passes when ANY call satisfies the matcher", () => {
    const output = toolOutput([
      { id: "1", name: "dispute_charge", arguments: { amount: 5 } },
      { id: "2", name: "dispute_charge", arguments: { amount: 23 } },
    ]);
    expect(
      run(
        { type: "tool_call_arg", name: "dispute_charge", arg: "amount", match: { equals: 23 } },
        output,
      ).passed,
    ).toBe(true);
  });

  test("fails cleanly when the tool was never called", () => {
    const result = run(
      { type: "tool_call_arg", name: "dispute_charge", arg: "amount", match: { equals: 23 } },
      textOutput("no tools here"),
    );
    expect(result.passed).toBe(false);
    expect(result.detail).toContain("was not called");
  });

  test("an invalid regex fails with a reason rather than throwing", () => {
    const output = toolOutput([{ id: "1", name: "email", arguments: { to: "x" } }]);
    const result = run(
      { type: "tool_call_arg", name: "email", arg: "to", match: { regex: "(" } },
      output,
    );
    expect(result.passed).toBe(false);
    expect(result.detail).toContain("invalid regex");
  });

  test("a missing matcher fails SAFE — fails the assertion, never throws (#8)", () => {
    // Config validation should catch this, but if a malformed assertion reaches
    // the engine it must fail cleanly rather than throw mid-experiment.
    const assertion = { type: "tool_call_arg", name: "dispute_charge" } as unknown as Assertion;
    const result = run(assertion, disputeRight);
    expect(result.passed).toBe(false);
    expect(result.detail).toContain("matcher");
  });

  test("a malformed subset matcher fails SAFE, never passes vacuously (#9)", () => {
    // `{subset: 23}` used to reach Object.entries(23) → [] → `.every()` true for
    // ANY argument. It must fail the assertion with a reason instead.
    const assertion = {
      type: "tool_call_arg",
      name: "dispute_charge",
      match: { subset: 23 },
    } as unknown as Assertion;
    const result = run(assertion, disputeRight);
    expect(result.passed).toBe(false);
    expect(result.detail).toContain("subset");
  });

  test("an EMPTY subset fails SAFE — a matcher that constrains nothing never passes (#9)", () => {
    // `{subset: {}}` is a plain object, so it slipped the type guard, but
    // `[].every()` is vacuously true — it would pass against ANY call. It must
    // fail with a reason instead of rubber-stamping every argument.
    const assertion = {
      type: "tool_call_arg",
      name: "dispute_charge",
      match: { subset: {} },
    } as unknown as Assertion;
    const result = run(assertion, disputeRight);
    expect(result.passed).toBe(false);
    expect(result.detail).toContain("subset");
  });

  test("two matchers at once fail SAFE — the runtime enforces exactly one (#9)", () => {
    // `{equals: 23, regex: "nope"}` used to silently pick whichever key was
    // checked first (equals) and ignore the rest — a false pass on the amount.
    // Config rejects this, but the grader must fail safe if it ever reaches here.
    const assertion = {
      type: "tool_call_arg",
      name: "dispute_charge",
      arg: "amount",
      match: { equals: 23, regex: "nope" },
    } as unknown as Assertion;
    const result = run(assertion, disputeRight);
    expect(result.passed).toBe(false);
    expect(result.detail).toContain("exactly one");
  });

  test("equals matches regardless of argument key order (#10)", () => {
    // The recorded value and the call's arguments carry the same keys in a
    // different order — a structural match, not a JSON.stringify match.
    const output = toolOutput([{ id: "1", name: "book", arguments: { seat: { row: 1, col: 2 } } }]);
    const assertion: Assertion = {
      type: "tool_call_arg",
      name: "book",
      arg: "seat",
      match: { equals: { col: 2, row: 1 } },
    };
    expect(run(assertion, output).passed).toBe(true);
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
