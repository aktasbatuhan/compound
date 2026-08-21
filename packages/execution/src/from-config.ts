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
  /**
   * The id to SEND to this provider. Equals the logical id unless the model
   * entry declares a `provider_ids` alias for the chosen provider (issue #19).
   * The logical id (the caller's `modelId`) remains the storage/fingerprint/
   * telemetry identity — only this wire id changes per provider.
   */
  wireModel: string;
  /** The resolved transport actually used (for display). */
  transport: Transport;
  /**
   * Set only when the transport was OVERRIDDEN away from the provider's native
   * one (issue #8) — e.g. running a flex host over plain chat. It joins the
   * completion fingerprint so a chat run and a flex run on the same host never
   * collide in the cache; the native transport contributes nothing, so every
   * existing (native) cache entry keeps hitting.
   */
  transportOverride?: string;
  /**
   * OpenRouter upstream routing to merge into the request params (issue #9),
   * e.g. `{ provider: { only: ["fireworks"], allow_fallbacks: false } }`. Set
   * only when an upstream was pinned; it flows into `request.params`, so it joins
   * the completion fingerprint automatically — pinning `fireworks` vs `together`
   * yields distinct cache identities and distinct paid calls.
   */
  routingParams?: Record<string, unknown>;
}

export interface ResolveModelOptions {
  /**
   * Run the model on THIS provider instead of the one named on its model entry
   * — the provider axis (`experiment <task> <model> --provider X`). The named
   * provider must exist in `providers`.
   */
  provider?: string;
  /**
   * Force the transport for this run (issue #8). A flex host also speaks plain
   * chat, so `--transport chat_completions` runs a Doubleword model over the
   * synchronous route for a queue-free comparison. A chat-only host cannot be
   * pushed to flex — that is refused. Omitted → the provider's native transport.
   */
  transport?: Transport;
  /**
   * Pin a specific OpenRouter UPSTREAM host for this run (issue #9), by
   * OpenRouter provider slug (e.g. "fireworks", "together", "baseten"). Sends
   * `provider: { only: [slug], allow_fallbacks: false }` so the single kimi-k3
   * model id can be benchmarked host-by-host. Only valid on an OpenRouter
   * provider; refused otherwise.
   */
  openrouterProvider?: string;
  /** Optional OpenRouter quantization filter for the pinned upstream (e.g. "fp8"). */
  openrouterQuant?: string;
  env?: NodeJS.ProcessEnv;
}

/** chat_completions | flex, derived from the provider's `type` or the model's `backend`. */
type Transport = "chat_completions" | "flex";

/**
 * Which transports a provider can actually speak. A flex host (Doubleword's
 * Responses API) ALSO serves plain chat completions, so it supports both; an
 * openai_compatible host serves only chat. This is what makes chat-vs-flex
 * selectable on the same Doubleword registry entry (issue #8).
 */
function supportedTransports(
  providerType: string | undefined,
  nativeTransport: Transport,
): Set<Transport> {
  if (providerType === "flex") return new Set<Transport>(["flex", "chat_completions"]);
  return new Set<Transport>([nativeTransport]);
}

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

  const providerType = (providerConfig as { type?: string }).type;
  const nativeTransport = transportFor(providerType, (entry as { backend?: string }).backend);
  // A per-run transport override (issue #8): allowed only to a transport the
  // endpoint can actually speak, so a chat-only host is never pushed to flex.
  let transport = nativeTransport;
  if (options.transport !== undefined && options.transport !== nativeTransport) {
    if (!supportedTransports(providerType, nativeTransport).has(options.transport)) {
      throw new ExecutionConfigError(
        `provider '${providerName}' (type ${providerType ?? "unset"}) cannot serve transport ` +
          `'${options.transport}'; it speaks ${nativeTransport}`,
      );
    }
    transport = options.transport;
  }
  // Only a non-native transport joins the fingerprint; the native one keeps the
  // existing identity so warm caches still hit (see ResolvedModel.transportOverride).
  const transportOverride = transport === nativeTransport ? undefined : transport;

  // Price: the provider's own table first (identical weights cost differently
  // per host), then the matching global table. Flex and chat prices are kept
  // separate because the async route is billed differently. A provider's own
  // table describes its NATIVE transport, so when the transport is overridden we
  // ignore it and bill on the global transport-specific table instead (#8).
  const providerPricing =
    transportOverride === undefined
      ? (providerConfig as { pricing_usd_per_million_tokens?: Record<string, TokenPrice> })
          .pricing_usd_per_million_tokens
      : undefined;
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

  // An optional per-provider request timeout override (long-output workloads).
  const timeoutOverride =
    typeof (providerConfig as { request_timeout_ms?: unknown }).request_timeout_ms === "number"
      ? { timeoutMs: (providerConfig as { request_timeout_ms: number }).request_timeout_ms }
      : {};
  const provider =
    transport === "flex"
      ? new FlexProvider({
          name: providerName,
          baseUrl: providerConfig.base_url,
          apiKey,
          ...timeoutOverride,
        })
      : new HttpProvider({
          name: providerName,
          baseUrl: providerConfig.base_url,
          apiKey,
          ...timeoutOverride,
        });

  // The wire id sent to this provider: a declared per-provider alias, else the
  // logical id. Price stays keyed by the logical id (what the user declares).
  const providerIds = (entry as { provider_ids?: Record<string, string> }).provider_ids;
  const wireModel = providerIds?.[providerName] ?? modelId;

  // OpenRouter upstream pinning (issue #9). Only OpenRouter accepts a `provider`
  // routing block; sending it to a direct host would be silently ignored (a
  // false comparison) or rejected, so refuse rather than pretend it pinned.
  let routingParams: Record<string, unknown> | undefined;
  if (options.openrouterProvider !== undefined) {
    if (!providerConfig.base_url.includes("openrouter.ai")) {
      throw new ExecutionConfigError(
        `--openrouter-provider only applies to an OpenRouter provider; ` +
          `'${providerName}' (${providerConfig.base_url}) is not one`,
      );
    }
    routingParams = {
      provider: {
        only: [options.openrouterProvider],
        allow_fallbacks: false,
        ...(options.openrouterQuant !== undefined
          ? { quantizations: [options.openrouterQuant] }
          : {}),
      },
    };
  }

  return {
    provider,
    providerName,
    price: {
      input: priceEntry.input,
      output: priceEntry.output,
      // The cached-input rate rides along when declared (#34), so a run's
      // ledgered cost is the effective bill, not the list price.
      ...(priceEntry.cached_input !== undefined ? { cached_input: priceEntry.cached_input } : {}),
    },
    wireModel,
    transport,
    ...(transportOverride !== undefined ? { transportOverride } : {}),
    ...(routingParams !== undefined ? { routingParams } : {}),
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
