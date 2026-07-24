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

/**
 * Resolve a candidate model id to a provider client and its price.
 *
 * Looks the model up in `models.candidates`/`models.frontier`, finds its
 * provider, reads the key from `api_key_env`, and takes the price from
 * `pricing_usd_per_million_tokens`. Throws with a precise reason if anything
 * required is missing, rather than guessing.
 */
export function resolveModel(
  config: CompoundConfig,
  modelId: string,
  env: NodeJS.ProcessEnv = process.env,
): ResolvedModel {
  const entries = [...(config.models?.candidates ?? []), ...(config.models?.frontier ?? [])];
  const entry = entries.find((candidate) => candidate.id === modelId);
  if (entry === undefined) {
    throw new ExecutionConfigError(
      `model '${modelId}' is not in models.candidates or models.frontier`,
    );
  }

  const providerName = entry.provider;
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

  // A model entry may declare its backend; default is chat completions.
  const backend =
    (entry as { backend?: "chat_completions" | "flex" }).backend ?? "chat_completions";

  // Flex models bill at the async flex rates; the two tables are kept separate
  // because Doubleword prices them differently.
  const priceTable =
    backend === "flex"
      ? config.flex_pricing_usd_per_million_tokens
      : config.pricing_usd_per_million_tokens;
  const priceEntry = priceTable?.[modelId];
  if (priceEntry === undefined) {
    const which =
      backend === "flex" ? "flex_pricing_usd_per_million_tokens" : "pricing_usd_per_million_tokens";
    throw new ExecutionConfigError(
      `no ${which} entry for '${modelId}'; refusing to run without a price`,
    );
  }

  const provider =
    backend === "flex"
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
