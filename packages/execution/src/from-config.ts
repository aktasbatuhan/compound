/**
 * Build a provider, price, and money controls from `compound.yaml`.
 *
 * This is where config becomes an executable experiment. It reads the API key
 * from the provider's `api_key_env` at call time; the key never lives in config
 * (docs/execution-v1.md, and the api's secret-omission rule).
 */
import type { CompoundConfig } from "@compound/config";
import type { TokenPrice } from "./fingerprint";
import { FlexProvider } from "./flex-provider";
import { HttpProvider, type Provider } from "./provider";

export class ExecutionConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ExecutionConfigError";
  }
}

interface ResolvedModel {
  provider: Provider;
  providerName: string;
  price: TokenPrice;
}

export interface ResolveModelOptions {
  /**
   * Run the model on THIS provider instead of the one named on its model entry
   * — the provider axis (`experiment <task> <model> --provider X`). The named
   * provider must exist in `providers`.
   */
  provider?: string;
  env?: NodeJS.ProcessEnv;
}

/** chat_completions | flex, derived from the provider's `type` or the model's `backend`. */
type Transport = "chat_completions" | "flex";

function transportFor(
  providerType: string | undefined,
  modelBackend: string | undefined,
): Transport {
  // The provider endpoint defines the protocol, so its `type` wins when set.
  if (providerType !== undefined) {
    if (providerType === "flex") return "flex";
    if (providerType === "openai_compatible") return "chat_completions";
    // A native adapter that isn't built yet: fail honestly rather than send
    // OpenAI-shaped requests to an incompatible API.
    throw new ExecutionConfigError(
      `provider type '${providerType}' is not supported yet; use an openai_compatible endpoint`,
    );
  }
  return modelBackend === "flex" ? "flex" : "chat_completions";
}

/**
 * Resolve a model id (optionally on an overriding provider) to a provider
 * client and its price.
 *
 * Looks the model up in `models.candidates`/`models.frontier`, picks the
 * provider (the override if given, else the model's own), reads the key from
 * `api_key_env`, chooses the transport from the provider's `type` (or the
 * model's `backend`), and takes the price from the provider's own price table
 * if it has one, else the matching global table. Throws with a precise reason if
 * anything required is missing, rather than guessing.
 */
export function resolveModel(
  config: CompoundConfig,
  modelId: string,
  options: ResolveModelOptions = {},
): ResolvedModel {
  const env = options.env ?? process.env;
  const entries = [...(config.models?.candidates ?? []), ...(config.models?.frontier ?? [])];
  const entry = entries.find((candidate) => candidate.id === modelId);
  if (entry === undefined) {
    throw new ExecutionConfigError(
      `model '${modelId}' is not in models.candidates or models.frontier`,
    );
  }

  const providerName = options.provider ?? entry.provider;
  const providerConfig = config.providers?.[providerName];
  if (providerConfig === undefined) {
    throw new ExecutionConfigError(`provider '${providerName}' is not configured`);
  }

  const apiKey = env[providerConfig.api_key_env];
  if (apiKey === undefined || apiKey.length === 0) {
    throw new ExecutionConfigError(
      `${providerConfig.api_key_env} is not set; cannot call provider '${providerName}'`,
    );
  }

  const transport = transportFor(
    (providerConfig as { type?: string }).type,
    (entry as { backend?: string }).backend,
  );

  // Price: the provider's own table first (identical weights cost differently
  // per host), then the matching global table. Flex and chat prices are kept
  // separate because the async route is billed differently.
  const providerPricing = (
    providerConfig as { pricing_usd_per_million_tokens?: Record<string, TokenPrice> }
  ).pricing_usd_per_million_tokens;
  const globalTable =
    transport === "flex"
      ? config.flex_pricing_usd_per_million_tokens
      : config.pricing_usd_per_million_tokens;
  const priceEntry = providerPricing?.[modelId] ?? globalTable?.[modelId];
  if (priceEntry === undefined) {
    const which =
      transport === "flex"
        ? "flex_pricing_usd_per_million_tokens"
        : "pricing_usd_per_million_tokens";
    throw new ExecutionConfigError(
      `no price for '${modelId}' on provider '${providerName}' ` +
        `(looked in providers.${providerName}.pricing_usd_per_million_tokens and ${which}); ` +
        "refusing to run without a price",
    );
  }

  const provider =
    transport === "flex"
      ? new FlexProvider({ name: providerName, baseUrl: providerConfig.base_url, apiKey })
      : new HttpProvider({ name: providerName, baseUrl: providerConfig.base_url, apiKey });

  return {
    provider,
    providerName,
    price: { input: priceEntry.input, output: priceEntry.output },
  };
}

export interface MoneyControls {
  paidRunsEnabled: boolean;
  globalHardLimitUsd: number;
}

/** The global money controls from `budget`. */
export function moneyControls(config: CompoundConfig): MoneyControls {
  return {
    paidRunsEnabled: config.budget?.paid_runs_enabled === true,
    globalHardLimitUsd: config.budget?.hard_limit_usd ?? 0,
  };
}
