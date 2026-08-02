/**
 * Multi-turn / agentic trajectory execution (issue #23).
 *
 * v1 gates a single model call. An agentic task instead drives a candidate
 * across turns: it emits tool calls, we feed back tool results under a per-tool
 * REPLAY POLICY, and it continues until it answers or a turn budget is hit.
 *
 * The whole trajectory then collapses into ONE aggregate assistant message —
 * the final answer text plus every tool call made across turns — so the single
 * TS grader (assertions / judge) scores it unchanged. No trajectory-specific
 * grader, and the sealed-partition + gate machinery apply as they do for a
 * single call.
 *
 * The replay policy is what keeps replay side-effect-free: `recorded` and
 * `mocked` never touch a live system, and `blocked` refuses a side-effecting
 * tool outright. `live_read_only` (actually calling a read tool) is a documented
 * follow-up, not part of v1.
 */
import type { Message, ToolCall } from "@compound/contract";
import type { CompletionRequest, CompletionUsage, Provider } from "./provider";

export const TOOL_REPLAY_POLICIES = ["recorded", "mocked", "live_read_only", "blocked"] as const;
export type ToolReplayPolicy = (typeof TOOL_REPLAY_POLICIES)[number];

/** A tool result to replay during a `recorded` trajectory. */
export interface RecordedToolResult {
  /** Tool name this result answers for. */
  tool: string;
  /**
   * If present, this result only answers a call whose arguments deep-equal it,
   * so a tool invoked with different arguments across turns can be scripted
   * distinctly. Omit to answer any call to `tool`.
   */
  arguments?: unknown;
  /** The tool message content replayed back to the model. */
  result: string;
}

/** How each tool call is answered during replay (config `task_keys.<t>.replay`). */
export interface TrajectoryPolicy {
  default: ToolReplayPolicy;
  perTool?: Record<string, ToolReplayPolicy>;
}

export interface RunTrajectoryOptions {
  /** The initial request: system/user messages, tools, params. */
  request: CompletionRequest;
  /** Scripted results for the `recorded` policy. */
  recordedToolResults?: readonly RecordedToolResult[];
  policy: TrajectoryPolicy;
  /** Result returned for a `mocked` tool call (default "{}"). */
  mockResult?: string;
  /** Turn budget: model calls before the trajectory is cut off (default 8). */
  maxTurns?: number;
}

export interface TrajectoryResult {
  /**
   * The aggregate output the single TS grader scores: the final answer's text
   * plus EVERY tool call made across the trajectory, as one assistant message.
   */
  gradedOutput: Message;
  /** The last assistant message produced. */
  finalMessage: Message;
  /** Every tool call across all turns, in order. */
  toolCalls: ToolCall[];
  /** Model calls made (1 for a single-shot answer). */
  turns: number;
  /** Whether the run stopped because it hit the turn budget rather than answering. */
  truncated: boolean;
  /** Summed token usage across turns. */
  usage: CompletionUsage;
  /** Summed latency across turns (ms). */
  latencyMs: number;
  /** The full message history, including replayed tool results. */
  transcript: Message[];
}

export const DEFAULT_MAX_TURNS = 8;

export class ToolBlockedError extends Error {
  constructor(readonly tool: string) {
    super(`tool '${tool}' is blocked by the replay policy and must not run during replay`);
    this.name = "ToolBlockedError";
  }
}

export class MissingRecordedResultError extends Error {
  constructor(readonly tool: string) {
    super(`no recorded result for tool '${tool}': a 'recorded' trajectory cannot proceed`);
    this.name = "MissingRecordedResultError";
  }
}

export class UnsupportedReplayPolicyError extends Error {
  constructor(
    readonly tool: string,
    readonly policy: ToolReplayPolicy,
  ) {
    super(
      `replay policy '${policy}' for tool '${tool}' is not supported by the v1 trajectory runner`,
    );
    this.name = "UnsupportedReplayPolicyError";
  }
}

function policyFor(tool: string, policy: TrajectoryPolicy): ToolReplayPolicy {
  return policy.perTool?.[tool] ?? policy.default;
}

function deepEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

/** Answer one tool call per the replay policy; never touches a live system in v1. */
function resolveToolResult(
  call: ToolCall,
  policy: TrajectoryPolicy,
  recorded: readonly RecordedToolResult[],
  mockResult: string,
): string {
  const p = policyFor(call.name, policy);
  switch (p) {
    case "blocked":
      throw new ToolBlockedError(call.name);
    case "mocked":
      return mockResult;
    case "live_read_only":
      // Deliberately unsupported in v1: executing a live tool is a follow-up.
      throw new UnsupportedReplayPolicyError(call.name, p);
    case "recorded": {
      const match = recorded.find(
        (r) =>
          r.tool === call.name &&
          (r.arguments === undefined || deepEqual(r.arguments, call.arguments)),
      );
      if (match === undefined) throw new MissingRecordedResultError(call.name);
      return match.result;
    }
  }
}

function addUsage(a: CompletionUsage, b: CompletionUsage): CompletionUsage {
  const sum: CompletionUsage = {
    input_tokens: a.input_tokens + b.input_tokens,
    output_tokens: a.output_tokens + b.output_tokens,
  };
  if (a.reasoning_tokens !== undefined || b.reasoning_tokens !== undefined) {
    sum.reasoning_tokens = (a.reasoning_tokens ?? 0) + (b.reasoning_tokens ?? 0);
  }
  if (a.total_tokens !== undefined || b.total_tokens !== undefined) {
    sum.total_tokens = (a.total_tokens ?? 0) + (b.total_tokens ?? 0);
  }
  return sum;
}

function finalText(message: Message): string | null {
  const content = message.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map((part) => (part.type === "text" ? part.text : "")).join("");
  }
  return null;
}

/**
 * Drive a candidate across turns under the replay policy and return the
 * aggregate result. Makes one provider call per turn; tool results are replayed
 * (never executed live in v1). The caller owns money-safety and caching — this
 * is the pure execution loop.
 */
export async function runTrajectory(
  provider: Provider,
  options: RunTrajectoryOptions,
): Promise<TrajectoryResult> {
  const maxTurns = options.maxTurns ?? DEFAULT_MAX_TURNS;
  const recorded = options.recordedToolResults ?? [];
  const mockResult = options.mockResult ?? "{}";
  const messages: Message[] = [...options.request.messages];
  const toolCalls: ToolCall[] = [];
  let usage: CompletionUsage = { input_tokens: 0, output_tokens: 0 };
  let latencyMs = 0;
  let finalMessage: Message = { role: "assistant", content: null };
  let turns = 0;
  let truncated = true;

  while (turns < maxTurns) {
    const response = await provider.complete({ ...options.request, messages: [...messages] });
    turns += 1;
    latencyMs += response.latencyMs;
    if (response.usage !== null) usage = addUsage(usage, response.usage);
    finalMessage = response.output;
    messages.push(response.output);

    const calls = response.output.tool_calls ?? [];
    if (calls.length === 0) {
      truncated = false; // answered within budget
      break;
    }
    for (const call of calls) {
      toolCalls.push(call);
      const result = resolveToolResult(call, options.policy, recorded, mockResult);
      messages.push({ role: "tool", content: result, tool_call_id: call.id });
    }
  }

  const gradedOutput: Message = {
    role: "assistant",
    content: finalText(finalMessage),
    ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
  };

  return {
    gradedOutput,
    finalMessage,
    toolCalls,
    turns,
    truncated,
    usage,
    latencyMs,
    transcript: messages,
  };
}
