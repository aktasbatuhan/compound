/**
 * Stable machine-readable reasons and dialect labels.
 *
 * Diagnostic strings are part of this package's API: they land in
 * `NormalizedTrace.diagnostics`, in the import report histogram, and
 * (eventually) in the diagnostic queue the user reads. They never change
 * meaning; new situations get new strings.
 */

export const DIAGNOSTICS = {
  /** Generation `input` was neither a message array nor a `{messages}` wrapper. */
  unparseableGenerationInput: "unparseable_generation_input",
  /** Generation `output` was not message-shaped, so no output message was built. */
  unparseableGenerationOutput: "unparseable_generation_output",
  /** An OpenAI-dialect `function.arguments` string did not parse to an object. */
  toolArgumentsUnparseable: "tool_arguments_unparseable",
  /** A tool_call carried no `id`; a synthetic one was minted for linkage. */
  toolCallMissingId: "tool_call_missing_id",
  /** A tool_call carried no name. */
  toolCallMissingName: "tool_call_missing_name",
  /** A legacy `function`-role message was mapped to `tool` with a null tool_call_id. */
  legacyFunctionRole: "legacy_function_role",
  /** Legacy `functions`/`function_call` were converted to ToolDefs. */
  legacyFunctions: "legacy_functions",
  /** A message role was outside the contract's four roles; the message was dropped. */
  unknownMessageRole: "unknown_message_role",
  /** Non-text content (image/audio/JSON blob) recorded as an `unsupported` part. */
  unsupportedContentPart: "unsupported_content_part",
  /** Observation `type` was outside the documented set; mapped to an `other` step. */
  unknownObservationType: "unknown_observation_type",
  /** A tool_execution's `tool_call_id` matched no generation tool_call. */
  unresolvedToolCallRef: "unresolved_tool_call_ref",
  /** A tool_execution had no name to carry into the contract. */
  toolExecutionMissingName: "tool_execution_missing_name",
  /** Focal step could not be determined by the v1 rule. */
  noReplayableFocalCall: "no_replayable_focal_call",
  /** A timestamp field was present but unparseable. */
  unparseableTimestamp: "unparseable_timestamp",
  /** A score's `source` was outside ANNOTATION/API/EVAL. */
  unknownScoreSource: "unknown_score_source",
  /** A non-CORRECTION score had no numeric value, so the contract cannot hold it. */
  nonNumericScoreSkipped: "non_numeric_score_skipped",
} as const;

export const DIALECTS = {
  /** Input was a bare message array. */
  messagesArray: "messages_array",
  /** Input was the OpenAI wrapper `{messages, tools}`. */
  openaiToolsWrapper: "openai_tools_wrapper",
  /** Input carried legacy `functions` / `function_call`. */
  legacyFunctions: "legacy_functions",
  /** Tool calls in OpenAI format (`function.arguments` as a JSON string). */
  openaiToolCalls: "openai_tool_calls",
  /** Tool calls in LangChain format (`args` already an object). */
  langchainToolCalls: "langchain_tool_calls",
  /** Tool calls recovered from `additional_kwargs.tool_calls`. */
  additionalKwargsToolCalls: "additional_kwargs_tool_calls",
  /** Usage read from `usageDetails`/`usage_details`. */
  usageDetails: "usage_details",
  /** Usage read from the deprecated `usage` object. */
  deprecatedUsage: "deprecated_usage",
} as const;

/** Collector threaded through normalization; one instance per import run. */
export class Collector {
  private readonly traceReasons = new Set<string>();
  readonly dialects = new Set<string>();
  readonly reasonHistogram = new Map<string, number>();

  diagnostic(reason: string): void {
    this.traceReasons.add(reason);
  }

  dialect(name: string): void {
    this.dialects.add(name);
  }

  /** Drain the current trace's reasons, folding them into the run histogram. */
  takeTraceDiagnostics(): string[] {
    const reasons = [...this.traceReasons].sort();
    this.traceReasons.clear();
    for (const reason of reasons) {
      this.reasonHistogram.set(reason, (this.reasonHistogram.get(reason) ?? 0) + 1);
    }
    return reasons;
  }
}
