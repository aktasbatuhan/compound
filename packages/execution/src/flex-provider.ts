/**
 * Flex Responses provider for Doubleword's cheap candidates.
 *
 * These models (GLM, DeepSeek, Nemotron) are served through the OpenAI
 * Responses API with `background=true, service_tier=flex`, not chat completions
 * — established in the benchmark phase (src/compound/flex.py). The call is
 * submit-then-poll: create a background response, then GET it until terminal.
 *
 * It satisfies the same `Provider` interface as `HttpProvider`, so the runner
 * is unchanged: from the runner's view a completion is still one awaited call.
 */
import type { Message } from "@compound/contract";
import type { CompletionRequest, CompletionResponse, CompletionUsage, Provider } from "./provider";
import { ProviderError } from "./provider";

const PENDING_STATUSES = new Set(["queued", "in_progress"]);

export interface FlexProviderConfig {
  name: string;
  baseUrl: string;
  apiKey: string;
  serviceTier?: string;
  pollIntervalMs?: number;
  timeoutMs?: number;
  /** Injected for tests so polling is deterministic and instant. */
  sleep?: (ms: number) => Promise<void>;
  fetchImpl?: typeof fetch;
}

type ToolCall = NonNullable<Message["tool_calls"]>[number];

interface ResponsePayload {
  id?: string;
  status?: string;
  output_text?: string;
  output?: Array<{
    type?: string;
    content?: Array<{ type?: string; text?: string }>;
    // Responses-API function-call item fields.
    id?: string;
    call_id?: string;
    name?: string;
    arguments?: string;
  }>;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    output_tokens_details?: { reasoning_tokens?: number };
    total_tokens?: number;
  };
  model?: string;
  error?: { message?: string } | null;
}

interface ChatToolShape {
  type?: string;
  function?: { name?: string; description?: string; parameters?: unknown };
  name?: string;
  description?: string;
  parameters?: unknown;
}

/**
 * The runner produces tools in chat-completions shape
 * (`{type:"function", function:{name,description,parameters}}`), but the
 * Responses API flattens the function fields to the top level. Convert here.
 */
export function toResponsesTools(tools: readonly unknown[]): unknown[] {
  return tools.map((raw) => {
    const t = raw as ChatToolShape;
    const fn = t.function ?? t;
    return {
      type: "function",
      name: fn.name,
      description: fn.description,
      parameters: fn.parameters,
    };
  });
}

function parseArgs(raw: string | undefined): ToolCall["arguments"] {
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed !== null && typeof parsed === "object" ? parsed : { _raw: raw };
  } catch {
    return { _raw: raw };
  }
}

/**
 * Extract tool calls from a Responses payload. Prefers structured
 * `function_call` output items; falls back to parsing `<tool_call>{…}` blocks
 * that some served models (e.g. GLM) emit as plain text when they don't produce
 * a structured item.
 */
export function responsesToolCalls(payload: ResponsePayload): ToolCall[] {
  const calls: ToolCall[] = [];
  for (const item of payload.output ?? []) {
    if (
      (item.type === "function_call" || item.type === "tool_call") &&
      typeof item.name === "string"
    ) {
      calls.push({
        id: item.call_id ?? item.id ?? "",
        name: item.name,
        arguments: parseArgs(item.arguments),
      });
    }
  }
  if (calls.length > 0) return calls;

  // Text fallback: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
  const text = responsesOutputText(payload);
  const re = /<tool_call>\s*(\{[\s\S]*?\})\s*(?:<\/tool_call>|$)/g;
  let match: RegExpExecArray | null;
  // biome-ignore lint/suspicious/noAssignInExpressions: standard regex-exec loop
  while ((match = re.exec(text)) !== null) {
    try {
      const obj = JSON.parse(match[1] as string) as { name?: string; arguments?: unknown };
      if (typeof obj.name === "string") {
        const args =
          obj.arguments !== null && typeof obj.arguments === "object"
            ? (obj.arguments as ToolCall["arguments"])
            : {};
        calls.push({ id: "", name: obj.name, arguments: args });
      }
    } catch {
      // A malformed block yields no tool call rather than a guess.
    }
  }
  return calls;
}

/** Concatenate Responses-API output text (direct field or output items). */
export function responsesOutputText(payload: ResponsePayload): string {
  if (typeof payload.output_text === "string") return payload.output_text;
  const parts: string[] = [];
  for (const item of payload.output ?? []) {
    for (const content of item.content ?? []) {
      if (
        (content.type === "output_text" || content.type === "text") &&
        typeof content.text === "string"
      ) {
        parts.push(content.text);
      }
    }
  }
  return parts.join("");
}

function responsesUsage(payload: ResponsePayload): CompletionUsage | null {
  const raw = payload.usage;
  if (raw === undefined) return null;
  return {
    input_tokens: raw.input_tokens ?? 0,
    output_tokens: raw.output_tokens ?? 0,
    ...(raw.output_tokens_details?.reasoning_tokens !== undefined
      ? { reasoning_tokens: raw.output_tokens_details.reasoning_tokens }
      : {}),
    ...(raw.total_tokens !== undefined ? { total_tokens: raw.total_tokens } : {}),
  };
}

const defaultSleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

export class FlexProvider implements Provider {
  readonly name: string;
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly serviceTier: string;
  private readonly pollIntervalMs: number;
  private readonly timeoutMs: number;
  private readonly sleep: (ms: number) => Promise<void>;
  private readonly fetchImpl: typeof fetch;

  constructor(config: FlexProviderConfig) {
    this.name = config.name;
    this.baseUrl = config.baseUrl.replace(/\/$/, "");
    this.apiKey = config.apiKey;
    this.serviceTier = config.serviceTier ?? "flex";
    this.pollIntervalMs = config.pollIntervalMs ?? 2000;
    this.timeoutMs = config.timeoutMs ?? 3_600_000;
    this.sleep = config.sleep ?? defaultSleep;
    this.fetchImpl = config.fetchImpl ?? fetch;
  }

  private headers(): Record<string, string> {
    return { "content-type": "application/json", authorization: `Bearer ${this.apiKey}` };
  }

  async complete(request: CompletionRequest): Promise<CompletionResponse> {
    const started = performance.now();
    const responseId = await this.submit(request);
    const { payload, leftQueueAt } = await this.poll(responseId);
    const latencyMs = Math.round(performance.now() - started);
    // Queue time = submit-start until the response left `queued` (compute began).
    // Decode time is the remainder; telemetry derives it as latencyMs - queueMs.
    const queueMs = Math.round(leftQueueAt - started);

    if (payload.status !== "completed") {
      throw new ProviderError(
        this.name,
        `background response ${responseId} ended ${payload.status}: ${payload.error?.message ?? ""}`.trim(),
      );
    }

    const toolCalls = responsesToolCalls(payload);
    const text = responsesOutputText(payload);
    const output: Message =
      toolCalls.length > 0
        ? { role: "assistant", content: text.length > 0 ? text : null, tool_calls: toolCalls }
        : { role: "assistant", content: text };
    return {
      output,
      usage: responsesUsage(payload),
      finishReason: payload.status ?? null,
      resolvedModel: payload.model ?? null,
      latencyMs,
      queueMs,
    };
  }

  /** Submit the background response and return its id. */
  private async submit(request: CompletionRequest): Promise<string> {
    const body: Record<string, unknown> = {
      model: request.model,
      // Responses API takes `input`; render messages as its input array.
      input: request.messages,
      background: true,
      service_tier: this.serviceTier,
      ...(request.tools && request.tools.length > 0
        ? { tools: toResponsesTools(request.tools), tool_choice: "auto" }
        : {}),
      ...(typeof request.params?.max_tokens === "number"
        ? { max_output_tokens: request.params.max_tokens }
        : {}),
      ...(typeof request.params?.reasoning_effort === "string"
        ? { reasoning: { effort: request.params.reasoning_effort } }
        : {}),
    };

    const response = await this.call("POST", "/responses", body);
    if (typeof response.id !== "string" || response.id.length === 0) {
      throw new ProviderError(this.name, "background submission returned no id");
    }
    return response.id;
  }

  /**
   * Poll until the response is terminal or the timeout elapses. Also reports
   * `leftQueueAt`: the clock reading at which the response was first observed to
   * have left `queued` (compute started), so the caller can split queue time
   * from decode time.
   */
  private async poll(
    responseId: string,
  ): Promise<{ payload: ResponsePayload; leftQueueAt: number }> {
    const deadline = performance.now() + this.timeoutMs;
    // First read immediately; a fast model may already be running or done.
    let payload = await this.call("GET", `/responses/${responseId}`);
    let leftQueueAt: number | undefined =
      payload.status === "queued" ? undefined : performance.now();
    while (payload.status !== undefined && PENDING_STATUSES.has(payload.status)) {
      if (performance.now() > deadline) {
        throw new ProviderError(this.name, `background response ${responseId} timed out`);
      }
      await this.sleep(this.pollIntervalMs);
      payload = await this.call("GET", `/responses/${responseId}`);
      if (leftQueueAt === undefined && payload.status !== "queued") {
        leftQueueAt = performance.now();
      }
    }
    // If it never left `queued` before going terminal, queue spanned the run.
    return { payload, leftQueueAt: leftQueueAt ?? performance.now() };
  }

  private async call(
    method: "GET" | "POST",
    path: string,
    body?: unknown,
  ): Promise<ResponsePayload> {
    let response: Response;
    try {
      response = await this.fetchImpl(`${this.baseUrl}${path}`, {
        method,
        headers: this.headers(),
        ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
      });
    } catch (error) {
      throw new ProviderError(
        this.name,
        `request failed: ${error instanceof Error ? error.message : "unknown error"}`,
      );
    }
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new ProviderError(this.name, `HTTP ${response.status}: ${text.slice(0, 200)}`);
    }
    return (await response.json()) as ResponsePayload;
  }
}
