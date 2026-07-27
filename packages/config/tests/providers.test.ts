import { describe, expect, test } from "bun:test";
import { KNOWN_PROVIDERS, knownProvider, providerBlockYaml, validateConfig } from "../src/index";

describe("the known-provider registry", () => {
  test("is non-empty and every entry is well-formed", () => {
    expect(KNOWN_PROVIDERS.length).toBeGreaterThan(0);
    for (const p of KNOWN_PROVIDERS) {
      expect(p.name).toMatch(/^[a-z0-9_]+$/);
      expect(p.apiKeyEnv).toMatch(/^[A-Z0-9_]+$/);
      expect(["openai_compatible", "flex"]).toContain(p.type);
      expect(["yes", "per_model"]).toContain(p.tools);
    }
  });

  test("names are unique", () => {
    const names = KNOWN_PROVIDERS.map((p) => p.name);
    expect(new Set(names).size).toBe(names.length);
  });

  test("covers the openai_compatible hosts batched in #13–#18", () => {
    const names = new Set(KNOWN_PROVIDERS.map((p) => p.name));
    for (const expected of [
      "fireworks",
      "together",
      "groq",
      "cerebras",
      "baseten",
      "self_hosted",
    ]) {
      expect(names.has(expected)).toBe(true);
    }
  });

  test("omits anthropic/google (they need native adapters, #11/#12)", () => {
    expect(knownProvider("anthropic")).toBeUndefined();
    expect(knownProvider("google")).toBeUndefined();
  });

  test("lookup finds a provider by name, misses are undefined", () => {
    expect(knownProvider("groq")?.baseUrl).toContain("groq.com");
    expect(knownProvider("nope")).toBeUndefined();
  });

  test("every known provider yields a block that validates as a real config", () => {
    // This is the guarantee that matters: a base_url that fails URL validation
    // or a bad type would make the paste block a broken suggestion.
    for (const p of KNOWN_PROVIDERS) {
      const result = validateConfig({
        version: 1,
        artifacts_dir: "a",
        manifests_dir: "m",
        benchmarks: {},
        providers: { [p.name]: { base_url: p.baseUrl, api_key_env: p.apiKeyEnv, type: p.type } },
      });
      expect(result.ok).toBe(true);
    }
  });

  test("providerBlockYaml renders the four fields, with a price stub for baseten", () => {
    const groqProvider = knownProvider("groq");
    const basetenProvider = knownProvider("baseten");
    expect(groqProvider).toBeDefined();
    expect(basetenProvider).toBeDefined();
    if (groqProvider === undefined || basetenProvider === undefined) return;

    const groq = providerBlockYaml(groqProvider);
    expect(groq).toContain("  groq:");
    expect(groq).toContain("base_url: https://api.groq.com/openai/v1");
    expect(groq).toContain("api_key_env: GROQ_API_KEY");
    expect(groq).toContain("type: openai_compatible");

    // Baseten pricing is per-deployment, so the block carries a commented stub.
    expect(providerBlockYaml(basetenProvider)).toContain("pricing_usd_per_million_tokens");
  });
});
