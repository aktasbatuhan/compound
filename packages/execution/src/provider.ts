/**
 * The provider client: an OpenAI-compatible chat-completions call.
 *
 * It is an INTERFACE so the runner is tested without a network. The real HTTP
 * implementation and a mock both satisfy `Provider`, and tests always use the
 * mock — no test ever makes a paid call (docs/execution-v1.md).
 */
import type { Message } from "@compound/contract";

export interface CompletionRequest {
  /** Provider-native model id. */
  model: string;
  messages: Message[];
  /** Tool definitions the model may call (OpenAI function-tool shape). */
  tools?: unknown[];
  /** Sampling and control params: temperature, max_tokens, reasoning_effort… */
  params?: Record<string, unknown>;
}

export interface CompletionUsage {
  input_tokens: number;
  output_tokens: number;
  reasoning_tokens?: number;
  total_tokens?: number;
}

export interface CompletionResponse {
  output: Message;
  usage: CompletionUsage | null;
  finishReason: string | null;
  /** Model id the provider actually served, when it echoes one. */
  resolvedModel: string | null;
  /**
   * The UPSTREAM provider that actually served the call, when a routing broker
   * echoes one (issue #9). OpenRouter returns a top-level `provider` naming the
   * host it dispatched to (e.g. "Fireworks", "Together"); a direct provider does
   * not, so this is `null`/absent for them. Recorded so a single OpenRouter model
   * id can be compared host-by-host in telemetry.
   */
  upstreamProvider?: string | null;
  latencyMs: number;
  /**
   * For an ASYNC route (flex submit-then-poll): the portion of `latencyMs` spent
   * waiting in the provider's queue before compute began — decode time is
   * `latencyMs - queueMs`. `null`/absent for synchronous providers, which have no
   * queue. Recorded so the telemetry axis compares queueing and decode honestly
   * rather than lumping them into one wall-clock number (issue #8).
   */
  queueMs?: number | null;
}

export interface Provider {
  readonly name: string;
  complete(request: CompletionRequest): Promise<CompletionResponse>;
}

export class ProviderError extends Error {
  constructor(
    readonly provider: string,
    message: string,
  ) {
    super(`provider ${provider}: ${message}`);
    this.name = "ProviderError";
  }
}

interface HttpProviderConfig {
  name: string;
  baseUrl: string;
  apiKey: string;
  timeoutMs?: number;
}

/**
 * OpenAI-compatible chat-completions over HTTP.
 *
 * Deliberately thin: it makes the call and normalizes the response. Budget
 * enforcement, caching, and fingerprinting live in the runner, so this class
 * never decides whether a call is allowed — it only performs one it was told to.
 */
export class HttpProvider implements Provider {
  readonly name: string;
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly timeoutMs: number;

  constructor(config: HttpProviderConfig) {
    this.name = config.name;
    this.baseUrl = config.baseUrl.replace(/\/$/, "");
    this.apiKey = config.apiKey;
    this.timeoutMs = config.timeoutMs ?? 180_000;
  }

  async complete(request: CompletionRequest): Promise<CompletionResponse> {
    const body: Record<string, unknown> = {
      model: request.model,
      messages: request.messages,
      ...(request.tools && request.tools.length > 0 ? { tools: request.tools } : {}),
      ...(request.params ?? {}),
    };

    const started = performance.now();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (error) {
      throw new ProviderError(
        this.name,
        `request failed: ${error instanceof Error ? error.message : "unknown error"}`,
      );
    } finally {
      clearTimeout(timeout);
    }
    const latencyMs = Math.round(performance.now() - started);

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new ProviderError(this.name, `HTTP ${response.status}: ${text.slice(0, 200)}`);
    }

    const payload = (await response.json()) as ChatCompletionPayload;
    return normalizeChatCompletion(this.name, payload, latencyMs);
  }
}

interface ChatCompletionPayload {
  model?: string;
  /** OpenRouter names the upstream host it routed to here (issue #9). */
  provider?: string;
  choices?: Array<{
    message?: {
      role?: string;
      content?: string | null;
      tool_calls?: Array<{
        id?: string;
        function?: { name?: string; arguments?: string };
      }>;
    };
    finish_reason?: string | null;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
    completion_tokens_details?: { reasoning_tokens?: number };
  };
}

/** Normalize an OpenAI-shaped response into a contract `Message` + usage. */
export function normalizeChatCompletion(
  provider: string,
  payload: ChatCompletionPayload,
  latencyMs: number,
): CompletionResponse {
  const choice = payload.choices?.[0];
  if (choice?.message === undefined) {
    throw new ProviderError(provider, "response had no message");
  }

  type ToolCall = NonNullable<Message["tool_calls"]>[number];
  const toolCalls: ToolCall[] = (choice.message.tool_calls ?? [])
    .map((call): ToolCall => {
      let args: ToolCall["arguments"] = {};
      try {
        args = call.function?.arguments ? JSON.parse(call.function.arguments) : {};
      } catch {
        args = { _raw: call.function?.arguments ?? "" };
      }
      return { id: call.id ?? "", name: call.function?.name ?? "", arguments: args };
    })
    .filter((call) => call.name !== "");

  const output: Message =
    toolCalls.length > 0
      ? { role: "assistant", content: choice.message.content ?? null, tool_calls: toolCalls }
      : { role: "assistant", content: choice.message.content ?? null };

  const usage: CompletionUsage | null = payload.usage
    ? {
        input_tokens: payload.usage.prompt_tokens ?? 0,
        output_tokens: payload.usage.completion_tokens ?? 0,
        ...(payload.usage.completion_tokens_details?.reasoning_tokens !== undefined
          ? { reasoning_tokens: payload.usage.completion_tokens_details.reasoning_tokens }
          : {}),
        ...(payload.usage.total_tokens !== undefined
          ? { total_tokens: payload.usage.total_tokens }
          : {}),
      }
    : null;

  return {
    output,
    usage,
    finishReason: choice.finish_reason ?? null,
    resolvedModel: payload.model ?? null,
    upstreamProvider: payload.provider ?? null,
    latencyMs,
  };
}
