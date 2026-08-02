import { describe, expect, test } from "bun:test";
import {
  chargeableCost,
  completionFingerprint,
  costFromUsage,
  estimateCost,
  type FingerprintInput,
} from "../src/index";

const base: FingerprintInput = {
  provider: "openrouter",
  request: {
    model: "gpt-4o",
    messages: [{ role: "user", content: "hi" }],
    params: { temperature: 0 },
  },
};

describe("completionFingerprint", () => {
  test("is stable for identical inputs", () => {
    expect(completionFingerprint(base)).toBe(completionFingerprint(base));
  });

  test("a transport override changes the hash, but its absence keeps the legacy identity (#8)", () => {
    const original = completionFingerprint(base);
    // Backward compat: no override tag → byte-identical to a pre-#8 fingerprint.
    expect(completionFingerprint({ ...base, transportOverride: undefined })).toBe(original);
    // A forced non-native transport → distinct, so chat and flex never collide.
    const overridden = completionFingerprint({ ...base, transportOverride: "chat_completions" });
    expect(overridden).not.toBe(original);
    expect(completionFingerprint({ ...base, transportOverride: "flex" })).not.toBe(overridden);
  });

  test("changes with model, params, provider, or revision", () => {
    const original = completionFingerprint(base);
    expect(completionFingerprint({ ...base, provider: "doubleword" })).not.toBe(original);
    expect(completionFingerprint({ ...base, providerRevision: "2026-07" })).not.toBe(original);
    expect(
      completionFingerprint({ ...base, request: { ...base.request, model: "other" } }),
    ).not.toBe(original);
    expect(
      completionFingerprint({
        ...base,
        request: { ...base.request, params: { temperature: 1 } },
      }),
    ).not.toBe(original);
  });

  test("different OpenRouter upstreams are distinct paid calls / cache identities (#9)", () => {
    // The routing block rides in params, so pinning fireworks vs together yields
    // different fingerprints — the cache never serves one host's answer for another.
    const fireworks = completionFingerprint({
      ...base,
      request: {
        ...base.request,
        params: { ...base.request.params, provider: { only: ["fireworks"] } },
      },
    });
    const together = completionFingerprint({
      ...base,
      request: {
        ...base.request,
        params: { ...base.request.params, provider: { only: ["together"] } },
      },
    });
    expect(fireworks).not.toBe(together);
    // And both differ from an unpinned run, so pinning never reuses a generic hit.
    expect(fireworks).not.toBe(completionFingerprint(base));
  });

  test("is insensitive to param key order (canonical JSON)", () => {
    const a = completionFingerprint({
      ...base,
      request: { ...base.request, params: { a: 1, b: 2 } },
    });
    const b = completionFingerprint({
      ...base,
      request: { ...base.request, params: { b: 2, a: 1 } },
    });
    expect(a).toBe(b);
  });

  test("trial 0 keeps the base identity; later trials differ", () => {
    expect(completionFingerprint(base, 0)).toBe(completionFingerprint(base));
    expect(completionFingerprint(base, 1)).not.toBe(completionFingerprint(base, 0));
    expect(completionFingerprint(base, 1)).not.toBe(completionFingerprint(base, 2));
  });
});

describe("costFromUsage", () => {
  test("bills input and output at their per-million rates", () => {
    const cost = costFromUsage(
      { input_tokens: 1_000_000, output_tokens: 500_000 },
      { input: 2, output: 6 },
    );
    expect(cost).toBeCloseTo(2 + 3, 9);
  });

  test("is zero when usage is null", () => {
    expect(costFromUsage(null, { input: 5, output: 5 })).toBe(0);
  });
});

describe("estimateCost", () => {
  test("is conservative: assumes the full output budget", () => {
    const estimate = estimateCost(
      { model: "m", messages: [{ role: "user", content: "hi" }], params: { max_tokens: 1000 } },
      { input: 1, output: 10 },
    );
    // ~ output-dominated: 1000 output tokens at $10/M ≈ $0.01.
    expect(estimate).toBeGreaterThanOrEqual(0.01);
  });

  test("counts tool schemas, so an agentic request estimates higher than its bare prompt (#4)", () => {
    const messages = [{ role: "user" as const, content: "hi" }];
    const tools = [
      {
        type: "function",
        function: {
          name: "dispute_charge",
          description: "Open a dispute for a charge on the customer's account",
          parameters: {
            type: "object",
            properties: { charge_id: { type: "string" }, amount: { type: "number" } },
            required: ["charge_id"],
          },
        },
      },
    ];
    const price = { input: 100, output: 1 };
    const withoutTools = estimateCost({ model: "m", messages }, price);
    const withTools = estimateCost({ model: "m", messages, tools }, price);
    // Tool JSON is billed as prompt tokens, so it must raise the input estimate.
    expect(withTools).toBeGreaterThan(withoutTools);
  });
});

describe("chargeableCost", () => {
  const req = { model: "m", messages: [{ role: "user" as const, content: "hi" }] };

  test("uses the measured cost when usage is present", () => {
    const { costUsd, usageKnown } = chargeableCost(
      { input_tokens: 1_000_000, output_tokens: 0 },
      req,
      { input: 3, output: 6 },
    );
    expect(costUsd).toBeCloseTo(3, 9);
    expect(usageKnown).toBe(true);
  });

  test("falls back to the estimate — never $0 — when the provider omits usage (#3)", () => {
    const { costUsd, usageKnown } = chargeableCost(null, req, { input: 1, output: 10 });
    expect(costUsd).toBeGreaterThan(0);
    expect(usageKnown).toBe(false);
  });
});
