import { describe, expect, test } from "bun:test";
import type { CompoundConfig } from "@compound/config";
import { ExecutionConfigError, FlexProvider, HttpProvider, resolveModel } from "../src/index";

/** A config with one model resolvable on two providers, each priced differently. */
const CONFIG = {
  version: 1,
  artifacts_dir: "artifacts",
  manifests_dir: "manifests",
  benchmarks: {},
  providers: {
    openrouter: {
      base_url: "https://openrouter.ai/api/v1",
      api_key_env: "OPENROUTER_API_KEY",
      type: "openai_compatible",
    },
    doubleword: {
      base_url: "https://api.doubleword.ai/v1",
      api_key_env: "DOUBLEWORD_API_KEY",
      type: "flex",
      pricing_usd_per_million_tokens: { "glm-5.2": { input: 0.7, output: 2.25 } },
    },
    together: {
      base_url: "https://api.together.xyz/v1",
      api_key_env: "TOGETHER_API_KEY",
      type: "openai_compatible",
      pricing_usd_per_million_tokens: { "glm-5.2": { input: 0.9, output: 3.0 } },
    },
    vertex: {
      base_url: "https://vertex.example.com/v1",
      api_key_env: "GOOGLE_API_KEY",
      type: "google",
    },
  },
  models: {
    candidates: [
      // Default provider is doubleword (flex); backend records that default.
      { id: "glm-5.2", provider: "doubleword", role: "candidate", backend: "flex" },
    ],
  },
  pricing_usd_per_million_tokens: { "glm-5.2": { input: 1.4, output: 4.4 } },
  flex_pricing_usd_per_million_tokens: { "glm-5.2": { input: 0.7, output: 2.25 } },
} as unknown as CompoundConfig;

const ENV = {
  OPENROUTER_API_KEY: "or-key",
  DOUBLEWORD_API_KEY: "dw-key",
  TOGETHER_API_KEY: "tg-key",
  GOOGLE_API_KEY: "g-key",
};

describe("resolveModel — the provider axis", () => {
  test("default provider comes from the model entry, flex type → FlexProvider", () => {
    const r = resolveModel(CONFIG, "glm-5.2", { env: ENV });
    expect(r.providerName).toBe("doubleword");
    expect(r.provider).toBeInstanceOf(FlexProvider);
    expect(r.price).toEqual({ input: 0.7, output: 2.25 });
  });

  test("--provider override runs the SAME model on another provider", () => {
    const r = resolveModel(CONFIG, "glm-5.2", { provider: "together", env: ENV });
    expect(r.providerName).toBe("together");
    // openai_compatible → the chat HttpProvider, not flex, even though the model
    // entry's backend is flex (the endpoint defines the protocol).
    expect(r.provider).toBeInstanceOf(HttpProvider);
    // The provider's own price table wins over the global one.
    expect(r.price).toEqual({ input: 0.9, output: 3.0 });
  });

  test("a flex-backend model on an openai_compatible provider uses CHAT, priced from the chat table", () => {
    // openrouter has no per-provider price → falls back to the global chat table
    // (not the flex table), because the transport is chat.
    const r = resolveModel(CONFIG, "glm-5.2", { provider: "openrouter", env: ENV });
    expect(r.provider).toBeInstanceOf(HttpProvider);
    expect(r.price).toEqual({ input: 1.4, output: 4.4 });
  });

  test("an unsupported provider type fails honestly rather than sending wrong-shaped requests", () => {
    expect(() => resolveModel(CONFIG, "glm-5.2", { provider: "vertex", env: ENV })).toThrow(
      ExecutionConfigError,
    );
  });

  test("an unknown provider name is a precise error", () => {
    expect(() => resolveModel(CONFIG, "glm-5.2", { provider: "nope", env: ENV })).toThrow(
      /provider 'nope' is not configured/,
    );
  });

  test("a missing api key names the env var and the provider", () => {
    expect(() => resolveModel(CONFIG, "glm-5.2", { provider: "together", env: {} })).toThrow(
      /TOGETHER_API_KEY is not set/,
    );
  });

  test("no price anywhere refuses to run", () => {
    const noPrice = {
      ...CONFIG,
      pricing_usd_per_million_tokens: {},
      flex_pricing_usd_per_million_tokens: {},
      providers: {
        ...CONFIG.providers,
        together: {
          base_url: "https://api.together.xyz/v1",
          api_key_env: "TOGETHER_API_KEY",
          type: "openai_compatible",
        },
      },
    } as unknown as CompoundConfig;
    expect(() => resolveModel(noPrice, "glm-5.2", { provider: "together", env: ENV })).toThrow(
      /refusing to run without a price/,
    );
  });
});
