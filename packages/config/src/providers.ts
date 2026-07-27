/**
 * A registry of known inference providers (issues #8–#18).
 *
 * After the provider axis (#4), every OpenAI-compatible host is a plain config
 * block — no adapter code. The remaining friction is discovery: which base_url,
 * which API-key env var, does it do tool calling. This table answers that so a
 * user adds a provider by name instead of hunting docs, and `compound providers`
 * emits a paste-ready block.
 *
 * These are NOT auto-injected into anyone's config — a provider only exists once
 * the user declares it. The registry is reference data plus the paste helper.
 *
 * `anthropic` and `google` are deliberately absent: they speak their own wire
 * protocols (not OpenAI-compatible) and need real adapters (#11, #12), which
 * resolveModel currently refuses honestly rather than sending wrong-shaped
 * requests.
 */

export interface KnownProvider {
  /** The suggested config key and the name `compound providers <name>` takes. */
  name: string;
  /** OpenAI-compatible base URL (…/v1), or the flex Responses endpoint. */
  baseUrl: string;
  /** Env var the API key is read from; never the key itself. */
  apiKeyEnv: string;
  /** Wire protocol — matches ProviderSchema.type. */
  type: "openai_compatible" | "flex";
  /**
   * Whether the host does function/tool calling. `"per_model"` means some
   * models do and some do not — gate it per candidate, don't assume.
   */
  tools: "yes" | "per_model";
  /** One line of anything a user must know before relying on it. */
  note?: string;
}

/**
 * Known providers, most-general first within each group. Base URLs and env
 * conventions are the vendor defaults as of the knowledge cutoff; a user can
 * always override either in their own config.
 */
export const KNOWN_PROVIDERS: readonly KnownProvider[] = [
  {
    name: "openrouter",
    baseUrl: "https://openrouter.ai/api/v1",
    apiKeyEnv: "OPENROUTER_API_KEY",
    type: "openai_compatible",
    tools: "per_model",
    note: "Aggregator: one key, many upstreams. Model ids are namespaced (openai/…, anthropic/…).",
  },
  {
    name: "openai",
    baseUrl: "https://api.openai.com/v1",
    apiKeyEnv: "OPENAI_API_KEY",
    type: "openai_compatible",
    tools: "yes",
  },
  {
    name: "fireworks",
    baseUrl: "https://api.fireworks.ai/inference/v1",
    apiKeyEnv: "FIREWORKS_API_KEY",
    type: "openai_compatible",
    tools: "per_model",
    note: "OSS-model host (Llama, DeepSeek, Qwen). Model ids look like accounts/fireworks/models/…",
  },
  {
    name: "together",
    baseUrl: "https://api.together.xyz/v1",
    apiKeyEnv: "TOGETHER_API_KEY",
    type: "openai_compatible",
    tools: "per_model",
    note: "Broad OSS catalog; HF-style model ids (meta-llama/…, deepseek-ai/…).",
  },
  {
    name: "groq",
    baseUrl: "https://api.groq.com/openai/v1",
    apiKeyEnv: "GROQ_API_KEY",
    type: "openai_compatible",
    tools: "yes",
    note: "Very high TPS — the showcase for the latency/TPS telemetry axis.",
  },
  {
    name: "cerebras",
    baseUrl: "https://api.cerebras.ai/v1",
    apiKeyEnv: "CEREBRAS_API_KEY",
    type: "openai_compatible",
    tools: "per_model",
    note: "Speed-focused host; good provider-axis speed comparison against a serverless API.",
  },
  {
    name: "baseten",
    baseUrl: "https://inference.baseten.co/v1",
    apiKeyEnv: "BASETEN_API_KEY",
    type: "openai_compatible",
    tools: "per_model",
    note: "Pricing is deployment-specific — set a per-provider price table, not the global one. Dedicated deployments have their own base_url.",
  },
  {
    name: "doubleword",
    baseUrl: "https://api.doubleword.ai/v1",
    apiKeyEnv: "DOUBLEWORD_API_KEY",
    type: "flex",
    tools: "yes",
    note: "Async Responses API (service_tier=flex): submit-then-poll, flex pricing. Each flex request reserves ~$0.02 extra cap headroom for reasoning-token overrun; also serves plain chat (compound experiment --transport chat_completions).",
  },
  {
    name: "self_hosted",
    baseUrl: "http://localhost:8000/v1",
    apiKeyEnv: "SELF_HOSTED_API_KEY",
    type: "openai_compatible",
    tools: "per_model",
    note: "Template for vLLM/Ollama/LM Studio/TGI/SGLang. Point base_url at your server (Ollama: :11434/v1). Any OpenAI-compatible endpoint works via this block.",
  },
];

/** Look up a known provider by name. */
export function knownProvider(name: string): KnownProvider | undefined {
  return KNOWN_PROVIDERS.find((provider) => provider.name === name);
}

/**
 * Render a known provider as a compound.yaml `providers:` block (the entry
 * lines under `providers:`, indented two spaces). Baseten carries a commented
 * price-table stub because its pricing is per-deployment.
 */
export function providerBlockYaml(provider: KnownProvider): string {
  const lines = [
    `  ${provider.name}:`,
    `    base_url: ${provider.baseUrl}`,
    `    api_key_env: ${provider.apiKeyEnv}`,
    `    type: ${provider.type}`,
  ];
  if (provider.name === "baseten") {
    lines.push(
      "    # Baseten pricing is per-deployment — declare it here, not globally:",
      "    # pricing_usd_per_million_tokens:",
      "    #   <your-model-id>: { input: 0.0, output: 0.0 }",
    );
  }
  return lines.join("\n");
}
