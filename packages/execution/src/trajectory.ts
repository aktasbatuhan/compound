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
import type { CompletionRequest, CompletionResponse, CompletionUsage, Provider } from "./provider";

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
  /**
   * Called BEFORE each provider call with that turn's exact request (1-indexed).
   * The caller uses it to enforce per-turn budget headroom against the real
   * request — money-safety must hold on every turn, not just the first. Throwing
   * (e.g. a budget error) aborts the trajectory; every turn already made was
   * reported via `onTurnComplete`, so nothing billed is silently lost (#23).
   */
  beforeTurn?: (turn: number, request: CompletionRequest) => void;
  /**
   * Called AFTER each provider response with that turn's request and response,
   * so the caller can persist spend and cache per turn. Critically, this fires
   * even for the turn that triggers a policy stop or truncation — the model call
   * was real and must be ledgered (#23, money-safety).
   */
  onTurnComplete?: (turn: number, request: CompletionRequest, response: CompletionResponse) => void;
}

/** Why a trajectory stopped. Only `answered` yields a gradeable outcome. */
export type TrajectoryStopReason =
  | "answered"
  | "truncated"
  | "tool_blocked"
  | "missing_recorded_result"
  | "unsupported_policy";

export interface TrajectoryStop {
  reason: TrajectoryStopReason;
  /** The tool that caused a policy stop (blocked / missing / unsupported). */
  tool?: string;
  /** The offending policy for an `unsupported_policy` stop. */
  policy?: ToolReplayPolicy;
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
  /** Why the trajectory stopped; only `answered` is a gradeable outcome. */
  stop: TrajectoryStop;
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

/** Human-readable one-liner for a non-`answered` stop, for skip detail/telemetry. */
export function describeStop(stop: TrajectoryStop): string {
  switch (stop.reason) {
    case "answered":
      return "answered within the turn budget";
    case "truncated":
      return "hit the turn budget without answering (truncated)";
    case "tool_blocked":
      return `tool '${stop.tool}' is blocked by the replay policy and must not run during replay`;
    case "missing_recorded_result":
      return `no recorded result for tool '${stop.tool}': a 'recorded' trajectory cannot proceed`;
    case "unsupported_policy":
      return `replay policy '${stop.policy}' for tool '${stop.tool}' is not supported by the v1 trajectory runner`;
  }
}

/** The skip-reason key a non-`answered` stop is counted under in the report. */
export function stopSkipKey(reason: Exclude<TrajectoryStopReason, "answered">): string {
  switch (reason) {
    case "truncated":
      return "turn_budget_exhausted";
    case "tool_blocked":
      return "tool_blocked";
    case "missing_recorded_result":
      return "missing_recorded_result";
    case "unsupported_policy":
      return "unsupported_replay_policy";
  }
}

function policyFor(tool: string, policy: TrajectoryPolicy): ToolReplayPolicy {
  return policy.perTool?.[tool] ?? policy.default;
}

function deepEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

/** A resolved tool result, or the reason replay cannot continue. */
type ResolveOutcome =
  | { ok: true; result: string }
  | {
      ok: false;
      reason: "tool_blocked" | "missing_recorded_result" | "unsupported_policy";
      tool: string;
      policy?: ToolReplayPolicy;
    };

/**
 * Answer one tool call per the replay policy; never touches a live system in v1.
 * Returns a stop reason instead of throwing, so the caller keeps the billing
 * state of every turn already made before the stop (#23, money-safety).
 */
function resolveToolResult(
  call: ToolCall,
  policy: TrajectoryPolicy,
  recorded: readonly RecordedToolResult[],
  mockResult: string,
): ResolveOutcome {
  const p = policyFor(call.name, policy);
  switch (p) {
    case "blocked":
      return { ok: false, reason: "tool_blocked", tool: call.name };
    case "mocked":
      return { ok: true, result: mockResult };
    case "live_read_only":
      // Deliberately unsupported in v1: executing a live tool is a follow-up.
      return { ok: false, reason: "unsupported_policy", tool: call.name, policy: p };
    case "recorded": {
      const match = recorded.find(
        (r) =>
          r.tool === call.name &&
          (r.arguments === undefined || deepEqual(r.arguments, call.arguments)),
      );
      if (match === undefined)
        return { ok: false, reason: "missing_recorded_result", tool: call.name };
      return { ok: true, result: match.result };
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
  // Default to `truncated`: if the loop exits by exhausting the turn budget
  // (last turn made a tool call), that is exactly what happened.
  let stop: TrajectoryStop = { reason: "truncated" };

  while (turns < maxTurns) {
    const turnRequest: CompletionRequest = { ...options.request, messages: [...messages] };
    // Budget headroom is enforced per turn against THIS turn's real request, not
    // once up front — a trajectory can call the provider many times (#23).
    options.beforeTurn?.(turns + 1, turnRequest);
    const response = await provider.complete(turnRequest);
    turns += 1;
    latencyMs += response.latencyMs;
    if (response.usage !== null) usage = addUsage(usage, response.usage);
    // Ledger this turn immediately — even if the tool calls below force a policy
    // stop, this model call was real and billable.
    options.onTurnComplete?.(turns, turnRequest, response);
    finalMessage = response.output;
    messages.push(response.output);

    const calls = response.output.tool_calls ?? [];
    if (calls.length === 0) {
      stop = { reason: "answered" }; // answered within budget
      break;
    }
    let policyStop: TrajectoryStop | undefined;
    for (const call of calls) {
      toolCalls.push(call);
      const outcome = resolveToolResult(call, options.policy, recorded, mockResult);
      if (!outcome.ok) {
        policyStop = {
          reason: outcome.reason,
          tool: outcome.tool,
          ...(outcome.policy !== undefined ? { policy: outcome.policy } : {}),
        };
        break;
      }
      messages.push({ role: "tool", content: outcome.result, tool_call_id: call.id });
    }
    if (policyStop !== undefined) {
      stop = policyStop;
      break;
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
    stop,
    truncated: stop.reason === "truncated",
    usage,
    latencyMs,
    transcript: messages,
  };
}
