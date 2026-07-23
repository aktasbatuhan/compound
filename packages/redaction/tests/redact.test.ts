import { describe, expect, test } from "bun:test";
import type { ModelCallStep, Trace } from "@compound/contract";
import { TraceSchema, validate } from "@compound/contract";
import type { RedactionConfig } from "../src/index";
import { redactTrace } from "../src/index";
import { baseTrace, contractFixture, piiRule, secretRule, serialize } from "./helpers";

const KEY = "sk-live-abcdefghij0123456789";
const EMAIL = "ada@example.com";

function modelCall(trace: Trace, i = 0): ModelCallStep {
  const step = trace.steps[i];
  if (step?.type !== "model_call") throw new Error("expected a model_call step");
  return step;
}

describe("no config", () => {
  test("undefined config returns the trace unchanged with no redactions", () => {
    const input = baseTrace((trace) => {
      trace.metadata = { note: `key ${KEY}` };
    });
    const result = redactTrace(input, undefined);
    expect(result.redactions).toEqual([]);
    expect(result.trace).toEqual(input);
  });

  test("a config with no rules is the same as no config", () => {
    const input = baseTrace((trace) => {
      trace.metadata = { note: `key ${KEY}` };
    });
    const result = redactTrace(input, { rules: [] });
    expect(result.redactions).toEqual([]);
    expect(result.trace).toEqual(input);
  });

  test("the returned trace is a new object either way", () => {
    const input = baseTrace();
    expect(redactTrace(input, undefined).trace).not.toBe(input);
  });
});

describe("string values", () => {
  test("only the matched substring is replaced; context survives", () => {
    const input = baseTrace((trace) => {
      modelCall(trace).input[0] = { role: "user", content: `use ${KEY} to call the API` };
    });
    const result = redactTrace(input, secretRule(["steps[*].input"]));
    expect(modelCall(result.trace).input[0]?.content).toBe("use ⟦redacted:secret⟧ to call the API");
    expect(result.redactions).toEqual([
      { path: "steps[0].input[0].content", rule: "secret", marker: "⟦redacted:secret⟧" },
    ]);
  });

  test("several matches in one string produce one record for that path", () => {
    const input = baseTrace((trace) => {
      modelCall(trace).input[0] = {
        role: "user",
        content: `mail ${EMAIL} or grace@example.com`,
      };
    });
    const result = redactTrace(input, piiRule(["steps[*].input"]));
    expect(modelCall(result.trace).input[0]?.content).toBe("mail ⟦redacted:pii⟧ or ⟦redacted:pii⟧");
    expect(result.redactions).toHaveLength(1);
  });

  test("a string with no match is untouched and produces no record", () => {
    const input = baseTrace((trace) => {
      modelCall(trace).input[0] = { role: "user", content: "max_tokens: 4096, invoice INV-2291" };
    });
    const result = redactTrace(input, secretRule(["steps[*].**"]));
    expect(result.redactions).toEqual([]);
    expect(result.trace).toEqual(input);
  });

  test("paths not covered by applies_to are not touched", () => {
    const input = baseTrace((trace) => {
      modelCall(trace).input[0] = { role: "user", content: `key ${KEY}` };
      trace.metadata = { note: `key ${KEY}` };
    });
    const result = redactTrace(input, secretRule(["metadata.**"]));
    expect(modelCall(result.trace).input[0]?.content).toBe(`key ${KEY}`);
    expect(result.redactions.map((r) => r.path)).toEqual(["metadata.note"]);
  });
});

describe("whole-value replacement", () => {
  test("a tool_execution input object targeted whole becomes the marker string", () => {
    const input = baseTrace((trace) => {
      trace.steps.push({
        type: "tool_execution",
        step_id: "tool-1",
        name: "call_api",
        input: { endpoint: "/v1/invoices", api_key: KEY },
      });
    });
    const result = redactTrace(input, secretRule(["steps[*].input"]));
    const step = result.trace.steps[1];
    expect(step?.type).toBe("tool_execution");
    expect(step?.type === "tool_execution" ? step.input : undefined).toBe("⟦redacted:secret⟧");
    expect(result.redactions).toEqual([
      { path: "steps[1].input", rule: "secret", marker: "⟦redacted:secret⟧" },
    ]);
  });

  test("a targeted object with nothing sensitive in it is left as an object", () => {
    const input = baseTrace((trace) => {
      trace.steps.push({
        type: "tool_execution",
        step_id: "tool-1",
        name: "call_api",
        input: { invoice_id: "INV-2291" },
      });
    });
    const result = redactTrace(input, secretRule(["steps[*].input"]));
    expect(result.trace).toEqual(input);
    expect(result.redactions).toEqual([]);
  });

  test("a metadata container is never replaced whole; its values are", () => {
    const input = baseTrace((trace) => {
      trace.metadata = { environment: "prod", customer: { email: EMAIL } };
    });
    const result = redactTrace(input, piiRule(["metadata.**"]));
    expect(result.trace.metadata).toEqual({ environment: "prod", customer: "⟦redacted:pii⟧" });
    expect(result.redactions).toEqual([
      { path: "metadata.customer", rule: "pii", marker: "⟦redacted:pii⟧" },
    ]);
  });

  test("tool_call arguments stay an object; their values are redacted", () => {
    const input = baseTrace((trace) => {
      modelCall(trace).output = {
        role: "assistant",
        content: null,
        tool_calls: [{ id: "call_1", name: "send", arguments: { to: EMAIL, retries: 2 } }],
      };
    });
    const result = redactTrace(input, piiRule(["steps[*].output"]));
    const call = modelCall(result.trace).output?.tool_calls?.[0];
    expect(call?.id).toBe("call_1");
    expect(call?.arguments).toEqual({ to: "⟦redacted:pii⟧", retries: 2 });
    expect(result.redactions.map((r) => r.path)).toEqual([
      "steps[0].output.tool_calls[0].arguments.to",
    ]);
  });
});

describe("content parts", () => {
  test("a fully redacted text part becomes a redacted part", () => {
    const input = baseTrace((trace) => {
      modelCall(trace).input[0] = {
        role: "user",
        content: [
          { type: "text", text: "look at this:" },
          { type: "text", text: KEY },
        ],
      };
    });
    const result = redactTrace(input, secretRule(["steps[*].input[*].content"]));
    expect(modelCall(result.trace).input[0]?.content).toEqual([
      { type: "text", text: "look at this:" },
      { type: "redacted", marker: "⟦redacted:secret⟧" },
    ]);
    expect(result.redactions).toEqual([
      { path: "steps[0].input[0].content[1]", rule: "secret", marker: "⟦redacted:secret⟧" },
    ]);
  });

  test("a partially redacted text part stays a text part with the marker inline", () => {
    const input = baseTrace((trace) => {
      modelCall(trace).input[0] = {
        role: "user",
        content: [{ type: "text", text: `use ${KEY} here` }],
      };
    });
    const result = redactTrace(input, secretRule(["steps[*].input[*].content"]));
    expect(modelCall(result.trace).input[0]?.content).toEqual([
      { type: "text", text: "use ⟦redacted:secret⟧ here" },
    ]);
    expect(result.redactions).toEqual([
      { path: "steps[0].input[0].content[0].text", rule: "secret", marker: "⟦redacted:secret⟧" },
    ]);
  });

  test("existing redacted and unsupported parts are left alone", () => {
    const input = baseTrace((trace) => {
      modelCall(trace).input[0] = {
        role: "user",
        content: [
          { type: "redacted", marker: "⟦redacted:pii⟧" },
          { type: "unsupported", media_type: "image/png" },
        ],
      };
    });
    const result = redactTrace(input, secretRule(["steps.**"]));
    expect(result.trace).toEqual(input);
    expect(result.redactions).toEqual([]);
  });
});

describe("markers", () => {
  test("defaults per detector, and custom:<name> for regex rules", () => {
    const config: RedactionConfig = {
      rules: [
        {
          name: "orders",
          applies_to: ["metadata.order"],
          detector: "regex",
          pattern: "ORD-\\d{6}",
        },
      ],
    };
    const input = baseTrace((trace) => {
      trace.metadata = { order: "shipped ORD-123456 today" };
    });
    const result = redactTrace(input, config);
    expect(result.trace.metadata?.order).toBe("shipped ⟦redacted:custom:orders⟧ today");
    expect(result.redactions).toEqual([
      { path: "metadata.order", rule: "custom:orders", marker: "⟦redacted:custom:orders⟧" },
    ]);
    // `custom:<name>` is the contract's rule namespace, so the record validates.
    expect(TraceSchema.safeParse(result.trace).success).toBe(true);
  });

  test("an explicit marker overrides the default", () => {
    const input = baseTrace((trace) => {
      trace.metadata = { note: `key ${KEY}` };
    });
    const result = redactTrace(input, secretRule(["metadata.**"], "[REDACTED]"));
    expect(result.trace.metadata?.note).toBe("key [REDACTED]");
    expect(result.redactions[0]?.marker).toBe("[REDACTED]");
    expect(result.redactions[0]?.rule).toBe("secret");
  });
});

describe("field_allowlist", () => {
  test("an allowlisted path survives a rule that matches it", () => {
    const input = baseTrace((trace) => {
      trace.metadata = { support_contact: EMAIL, customer_email: EMAIL };
    });
    const result = redactTrace(input, {
      ...piiRule(["metadata.**"]),
      field_allowlist: ["metadata.support_contact"],
    });
    expect(result.trace.metadata).toEqual({
      support_contact: EMAIL,
      customer_email: "⟦redacted:pii⟧",
    });
    expect(result.redactions.map((r) => r.path)).toEqual(["metadata.customer_email"]);
  });

  test("allowlisting a node allowlists its whole subtree", () => {
    const input = baseTrace((trace) => {
      trace.metadata = { support: { escalation: { email: EMAIL } } };
    });
    const result = redactTrace(input, {
      ...piiRule(["metadata.**"]),
      field_allowlist: ["metadata.support"],
    });
    expect(result.trace.metadata).toEqual({ support: { escalation: { email: EMAIL } } });
    expect(result.redactions).toEqual([]);
  });

  test("an allowlisted leaf inside a targeted object prevents whole-value replacement", () => {
    const input = baseTrace((trace) => {
      trace.steps.push({
        type: "tool_execution",
        step_id: "tool-1",
        name: "call_api",
        input: { reply_to: EMAIL, api_key: KEY },
      });
    });
    const result = redactTrace(input, {
      ...secretRule(["steps[*].input"]),
      field_allowlist: ["steps[*].input.reply_to"],
    });
    const step = result.trace.steps[1];
    expect(step?.type === "tool_execution" ? step.input : undefined).toEqual({
      reply_to: EMAIL,
      api_key: "⟦redacted:secret⟧",
    });
    expect(result.redactions.map((r) => r.path)).toEqual(["steps[1].input.api_key"]);
  });

  test("allowlist globs work like applies_to globs", () => {
    const input = baseTrace((trace) => {
      modelCall(trace).input[0] = { role: "system", content: `key ${KEY}` };
      modelCall(trace).input[1] = { role: "user", content: `key ${KEY}` };
    });
    const result = redactTrace(input, {
      ...secretRule(["steps[*].input"]),
      field_allowlist: ["steps[*].input[0].**"],
    });
    expect(modelCall(result.trace).input[0]?.content).toBe(`key ${KEY}`);
    expect(modelCall(result.trace).input[1]?.content).toBe("key ⟦redacted:secret⟧");
  });
});

describe("multiple rules", () => {
  test("rules apply in config order and each records its own hit", () => {
    const config: RedactionConfig = {
      rules: [
        { name: "api_keys", applies_to: ["steps[*].input"], detector: "secret" },
        { name: "customer_pii", applies_to: ["steps[*].input"], detector: "pii" },
        {
          name: "orders",
          applies_to: ["metadata.**"],
          detector: "regex",
          pattern: "ORD-\\d{6}",
          marker: "<order>",
        },
      ],
    };
    const input = baseTrace((trace) => {
      modelCall(trace).input[0] = { role: "user", content: `${KEY} for ${EMAIL}` };
      trace.metadata = { order: "ORD-123456" };
    });
    const result = redactTrace(input, config);
    expect(modelCall(result.trace).input[0]?.content).toBe("⟦redacted:secret⟧ for ⟦redacted:pii⟧");
    // Records come out in traversal order: metadata is walked before steps.
    expect(result.redactions).toEqual([
      { path: "metadata.order", rule: "custom:orders", marker: "<order>" },
      { path: "steps[0].input[0].content", rule: "secret", marker: "⟦redacted:secret⟧" },
      { path: "steps[0].input[0].content", rule: "pii", marker: "⟦redacted:pii⟧" },
    ]);
  });
});

describe("purity and provenance", () => {
  test("the input trace is not mutated", () => {
    const input = baseTrace((trace) => {
      trace.metadata = { customer: { email: EMAIL } };
      modelCall(trace).input[0] = { role: "user", content: `key ${KEY}` };
    });
    const before = serialize(input);
    redactTrace(input, {
      rules: [...secretRule(["steps.**"]).rules, ...piiRule(["metadata.**"]).rules],
    });
    expect(serialize(input)).toBe(before);
  });

  test("records already on the trace are preserved and the new ones appended", () => {
    const input = baseTrace((trace) => {
      trace.redactions = [{ path: "metadata.old", rule: "pii", marker: "⟦redacted:pii⟧" }];
      trace.metadata = { note: `key ${KEY}` };
    });
    const result = redactTrace(input, secretRule(["metadata.**"]));
    expect(result.trace.redactions).toHaveLength(2);
    expect(result.trace.redactions[0]?.path).toBe("metadata.old");
    expect(result.redactions).toHaveLength(1);
  });

  test("no original value survives anywhere in the result, records included", () => {
    const input = baseTrace((trace) => {
      trace.session_id = `session for ${EMAIL}`;
      trace.metadata = { customer: { email: EMAIL, card: "4111111111111111" } };
      modelCall(trace).input[0] = { role: "user", content: `key ${KEY}` };
      modelCall(trace).output = { role: "assistant", content: `sent to ${EMAIL}` };
      trace.steps.push({
        type: "tool_execution",
        step_id: "tool-1",
        name: "call_api",
        input: { api_key: KEY },
        output: { ok: true, contact: EMAIL },
      });
      trace.steps.push({ type: "other", step_id: "span-1", name: "trace", data: { token: KEY } });
      trace.outcome = { feedback: [{ kind: "comment", value: `reply to ${EMAIL}` }] };
    });
    const config: RedactionConfig = {
      rules: [
        { name: "api_keys", applies_to: ["**"], detector: "secret" },
        { name: "customer_pii", applies_to: ["**"], detector: "pii" },
      ],
    };
    const result = redactTrace(input, config);
    const dumped = serialize(result);
    expect(dumped).not.toContain(KEY);
    expect(dumped).not.toContain(EMAIL);
    expect(dumped).not.toContain("4111111111111111");
    expect(result.redactions.length).toBeGreaterThan(0);
  });
});

describe("contract conformance", () => {
  test("the redacted trace still validates against TraceSchema", () => {
    const input = baseTrace((trace) => {
      trace.metadata = { customer: { email: EMAIL } };
      modelCall(trace).input = [
        { role: "system", content: `key ${KEY}` },
        { role: "user", content: [{ type: "text", text: EMAIL }] },
      ];
      trace.steps.push({
        type: "tool_execution",
        step_id: "tool-1",
        name: "call_api",
        input: { api_key: KEY },
      });
    });
    const config: RedactionConfig = {
      rules: [
        { name: "api_keys", applies_to: ["**"], detector: "secret" },
        { name: "customer_pii", applies_to: ["**"], detector: "pii" },
      ],
    };
    const result = redactTrace(input, config);
    expect(TraceSchema.safeParse(result.trace).success).toBe(true);
  });

  test("an eval_ready contract fixture stays eval_ready after redaction", () => {
    const input = contractFixture("eval-ready-full.json");
    const config: RedactionConfig = {
      rules: [
        { name: "api_keys", applies_to: ["**"], detector: "secret" },
        { name: "customer_pii", applies_to: ["**"], detector: "pii" },
      ],
    };
    const before = validate(input);
    expect(before.class).toBe("eval_ready");
    const result = redactTrace(input, config);
    expect(validate(result.trace).class).toBe("eval_ready");
  });

  test("every contract fixture that validates keeps its class through redaction", () => {
    for (const name of [
      "eval-ready-minimal.json",
      "diagnostic-missing-focal.json",
      "diagnostic-no-model-calls.json",
      "diagnostic-unsupported-content.json",
    ]) {
      const input = contractFixture(name);
      const result = redactTrace(input, {
        rules: [{ name: "everything", applies_to: ["**"], detector: "pii" }],
      });
      expect(validate(result.trace).class).toBe(validate(input).class);
    }
  });
});
