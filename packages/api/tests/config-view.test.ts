import { describe, expect, test } from "bun:test";
import { isSecretKey, stripSecrets } from "../src/config-view";

describe("isSecretKey", () => {
  test("treats credential-bearing names as secret", () => {
    for (const key of [
      "api_key",
      "apiKey",
      "api_keys",
      "secret",
      "client_secret",
      "password",
      "credentials",
      "private_key",
      "access_token",
      "auth_token",
      "bearer_token",
      "refresh_token",
    ]) {
      expect(isSecretKey(key)).toBe(true);
    }
  });

  /**
   * Regression: a bare `token` in the pattern matched `max_tokens` and
   * `pricing_usd_per_million_tokens`, silently hiding pricing and sampling
   * config from the dashboard. LLM token counts are not credentials.
   */
  test("does not treat LLM token counts or prices as secrets", () => {
    for (const key of [
      "max_tokens",
      "agent_max_tokens",
      "user_max_tokens",
      "output_tokens",
      "input_tokens",
      "reasoning_tokens",
      "cached_input_tokens",
      "total_tokens",
      "pricing_usd_per_million_tokens",
      "flex_pricing_usd_per_million_tokens",
    ]) {
      expect(isSecretKey(key)).toBe(false);
    }
  });

  test("keeps env-var references, which are pointers rather than credentials", () => {
    expect(isSecretKey("api_key_env")).toBe(false);
    expect(isSecretKey("secret_env")).toBe(false);
  });
});

describe("stripSecrets", () => {
  test("omits secret keys entirely instead of masking them", () => {
    const { value, omitted } = stripSecrets({
      providers: { openrouter: { api_key_env: "OPENROUTER_API_KEY", api_key: "sk-live-abc" } },
    });
    const serialized = JSON.stringify(value);
    expect(serialized).not.toContain("sk-live-abc");
    // Not masked in place: no placeholder betraying length or shape.
    expect(serialized).not.toContain("*");
    expect(serialized).toContain("OPENROUTER_API_KEY");
    expect(omitted).toEqual(["providers.openrouter.api_key"]);
  });

  test("recurses through arrays and reports indexed paths", () => {
    const { value, omitted } = stripSecrets({
      models: [{ id: "a", api_key: "secret-1" }, { id: "b" }],
    });
    expect(omitted).toEqual(["models[0].api_key"]);
    expect(value).toEqual({ models: [{ id: "a" }, { id: "b" }] });
  });

  test("leaves a config with no secrets untouched", () => {
    const config = {
      version: 1,
      budget: { hard_limit_usd: 25 },
      pricing_usd_per_million_tokens: { "some/model": { input: 1, output: 2 } },
      benchmarks: { tau_bench: { agent_max_tokens: 4096 } },
    };
    const { value, omitted } = stripSecrets(config);
    expect(value).toEqual(config);
    expect(omitted).toEqual([]);
  });

  test("preserves null and primitive values", () => {
    const { value } = stripSecrets({ a: null, b: 0, c: false, d: "x" });
    expect(value).toEqual({ a: null, b: 0, c: false, d: "x" });
  });
});
