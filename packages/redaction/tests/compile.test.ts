import { describe, expect, test } from "bun:test";
import { compileRedaction, RedactionConfigError, redactTrace } from "../src/index";
import { baseTrace } from "./helpers";

describe("compileRedaction", () => {
  test("returns undefined for an absent config or an empty rule list", () => {
    expect(compileRedaction(undefined)).toBeUndefined();
    expect(compileRedaction({ rules: [] })).toBeUndefined();
  });

  test("resolves the contract rule id and the default marker per detector", () => {
    const compiled = compileRedaction({
      rules: [
        { name: "a", applies_to: ["metadata"], detector: "secret" },
        { name: "b", applies_to: ["metadata"], detector: "pii" },
        { name: "c", applies_to: ["metadata"], detector: "regex", pattern: "ORD-[0-9]{6}" },
      ],
    });
    expect(compiled?.rules.map((rule) => [rule.rule, rule.marker])).toEqual([
      ["secret", "⟦redacted:secret⟧"],
      ["pii", "⟦redacted:pii⟧"],
      ["custom:c", "⟦redacted:custom:c⟧"],
    ]);
  });

  test("an explicit marker overrides the default", () => {
    const compiled = compileRedaction({
      rules: [{ name: "a", applies_to: ["metadata"], detector: "pii", marker: "[gone]" }],
    });
    expect(compiled?.rules[0]?.marker).toBe("[gone]");
    expect(compiled?.rules[0]?.rule).toBe("pii");
  });

  test("an invalid regex throws at config time, naming the rule", () => {
    const config = {
      rules: [
        { name: "broken", applies_to: ["metadata"], detector: "regex" as const, pattern: "([" },
      ],
    };
    expect(() => compileRedaction(config)).toThrow(RedactionConfigError);
    expect(() => compileRedaction(config)).toThrow(/redaction rule "broken"/);
  });

  test("an invalid applies_to path throws, naming the rule and the entry", () => {
    expect(() =>
      compileRedaction({
        rules: [{ name: "bad_path", applies_to: ["steps[x]"], detector: "pii" }],
      }),
    ).toThrow(/redaction rule "bad_path" applies_to "steps\[x\]"/);
  });

  test("an invalid field_allowlist path throws, naming the entry", () => {
    expect(() =>
      compileRedaction({
        rules: [{ name: "ok", applies_to: ["metadata"], detector: "pii" }],
        field_allowlist: ["metadata..environment"],
      }),
    ).toThrow(/field_allowlist "metadata\.\.environment"/);
  });

  test("redactTrace surfaces the same error", () => {
    expect(() =>
      redactTrace(baseTrace(), {
        rules: [{ name: "broken", applies_to: ["metadata"], detector: "regex", pattern: "(" }],
      }),
    ).toThrow(RedactionConfigError);
  });
});
