import { describe, expect, test } from "bun:test";
import type { Message } from "@compound/contract";
import { type Assertion, suggestAssertions } from "../src/index";

function toolMsg(...names: string[]): Message {
  return {
    role: "assistant",
    content: null,
    tool_calls: names.map((name, i) => ({ id: `c${i}`, name, arguments: {} })),
  };
}
function textMsg(text: string): Message {
  return { role: "assistant", content: text };
}

describe("suggestAssertions — high-precision, explainable", () => {
  test("proposes tool_called for a tool used in almost every accepted output", () => {
    const outputs = [
      toolMsg("dispute_charge"),
      toolMsg("dispute_charge"),
      toolMsg("dispute_charge"),
      toolMsg("dispute_charge"),
      textMsg("no tool here"),
    ];
    const [top] = suggestAssertions(outputs);
    expect(top?.assertion).toEqual({ type: "tool_called", name: "dispute_charge" });
    expect(top?.support).toBe(4);
    expect(top?.total).toBe(5);
    expect(top?.rationale).toContain("4/5");
  });

  test("does not propose a tool used in only a minority of outputs", () => {
    const outputs = [toolMsg("rare"), textMsg("a"), textMsg("b"), textMsg("c"), textMsg("d")];
    const tool = suggestAssertions(outputs).find((s) => s.assertion.type === "tool_called");
    expect(tool).toBeUndefined();
  });

  test("proposes valid_json when the output shape always parses", () => {
    const outputs = [
      textMsg('{"ok":1}'),
      textMsg('{"ok":2}'),
      textMsg('{"ok":3}'),
      textMsg('{"ok":4}'),
    ];
    const json = suggestAssertions(outputs).find((s) => s.assertion.type === "valid_json");
    expect(json?.assertion).toEqual({ type: "valid_json" });
    expect(json?.support).toBe(4);
  });

  test("does not propose valid_json when a third of outputs are prose", () => {
    const outputs = [textMsg('{"ok":1}'), textMsg('{"ok":2}'), textMsg("hello there")];
    const json = suggestAssertions(outputs).find((s) => s.assertion.type === "valid_json");
    expect(json).toBeUndefined();
  });

  test("proposes a max_length guardrail above the longest accepted text output", () => {
    const outputs = [textMsg("a".repeat(120)), textMsg("b".repeat(80)), textMsg("c".repeat(40))];
    const max = suggestAssertions(outputs).find((s) => s.assertion.type === "max_length");
    // ceil(120 * 1.5 / 50) * 50 = 200; strictly above the observed 120.
    expect(max?.assertion).toEqual({ type: "max_length", max: 200 });
    expect(max?.rationale).toContain("120");
  });

  test("never re-suggests an assertion already declared", () => {
    const outputs = [
      toolMsg("dispute_charge"),
      toolMsg("dispute_charge"),
      toolMsg("dispute_charge"),
    ];
    const existing: Assertion[] = [{ type: "tool_called", name: "dispute_charge" }];
    const tool = suggestAssertions(outputs, { existing }).find(
      (s) => s.assertion.type === "tool_called",
    );
    expect(tool).toBeUndefined();
  });

  test("returns nothing when there is too little evidence", () => {
    expect(suggestAssertions([toolMsg("x"), toolMsg("x")])).toEqual([]);
    expect(suggestAssertions([])).toEqual([]);
  });

  test("orders multiple tool suggestions by support, most-used first", () => {
    const outputs = [
      toolMsg("search", "book"),
      toolMsg("search", "book"),
      toolMsg("search", "book"),
      toolMsg("search", "book"),
      toolMsg("search"),
    ];
    const tools = suggestAssertions(outputs).filter((s) => s.assertion.type === "tool_called");
    expect(tools.map((s) => (s.assertion as { name: string }).name)).toEqual(["search", "book"]);
  });
});
