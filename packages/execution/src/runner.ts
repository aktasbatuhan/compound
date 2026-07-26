/**
 * The eval runner: run a candidate model against a task's cases and grade.
 *
 * Money-safety is the spine (docs/execution-v1.md):
 * - paid calls are OFF unless explicitly enabled with a positive cap;
 * - the cap is enforced BEFORE every call;
 * - every call is cache-first, so a re-run is $0;
 * - dry-run makes zero calls and reports what a paid run would do.
 */
import { type Assertion, evaluateAssertions } from "@compound/assertions";
import type { Message } from "@compound/contract";
import {
  type CasePartition,
  type CompoundDatabase,
  cacheCompletion,
  createExperiment,
  type ExperimentReport,
  finishExperiment,
  getCachedCompletion,
  listCases,
  recordCaseResults,
  recordSpend,
  requireBudgetHeadroom,
} from "@compound/storage";
import { completionFingerprint, costFromUsage, estimateCost, type TokenPrice } from "./fingerprint";
import type { CompletionRequest, CompletionResponse, Provider } from "./provider";

export interface RunExperimentOptions {
  taskKey: string;
  candidateModel: string;
  provider: Provider;
  providerName: string;
  price: TokenPrice;
  assertions: readonly Assertion[];
  /** Defaults to `optimizer_validation`. `decision_test` requires the flag below. */
  partition?: CasePartition;
  /** Guard: running against the sealed set is refused without this. */
  allowDecisionTest?: boolean;
  params?: Record<string, unknown>;
  /**
   * Replace each case's system message with this prompt (adoption re-gates:
   * run the candidate WITH an optimized prompt). Applied before fingerprinting,
   * so overridden completions never collide with the baseline prompt's cache.
   */
  systemPromptOverride?: string;
  providerRevision?: string;
  /**
   * Money controls. Paid calls happen only when `paidRunsEnabled` is true,
   * `globalHardLimitUsd > 0`, and `experimentCapUsd > 0`. Otherwise the run is
   * a dry run: cache hits are served, but no provider call is ever made.
   */
  paidRunsEnabled?: boolean;
  experimentCapUsd?: number;
  globalHardLimitUsd?: number;
  /** Force a dry run even when paid calls are enabled. */
  dryRun?: boolean;
  maxCases?: number;
}

export interface CaseRunResult {
  caseId: string;
  status: "graded" | "skipped" | "cache_miss_dry_run";
  passed?: boolean;
  score?: number;
  costUsd?: number;
  cached?: boolean;
  detail?: string;
  /** Fingerprint of the completion produced, so a judge can grade it later. */
  completionFingerprint?: string;
}

export interface RunExperimentResult {
  experimentId: string;
  report: ExperimentReport;
  results: CaseRunResult[];
}

export class DecisionPartitionRefusedError extends Error {
  constructor() {
    super("refusing to run an experiment against the sealed decision_test partition");
    this.name = "DecisionPartitionRefusedError";
  }
}

/** Paid calls happen only when every money precondition is met. */
export function isPaidEnabled(options: RunExperimentOptions): boolean {
  return (
    options.paidRunsEnabled === true &&
    (options.globalHardLimitUsd ?? 0) > 0 &&
    (options.experimentCapUsd ?? 0) > 0 &&
    options.dryRun !== true
  );
}

/**
 * The trace contract stores tools UNWRAPPED (`{name, description, parameters}`),
 * but the OpenAI/OpenRouter chat API needs the function envelope
 * (`{type:"function", function:{…}}`). Re-wrap here so a candidate actually
 * receives the tools; without this the model gets malformed tools, ignores them,
 * and emits a text tool call the assertions can't see.
 */
function toOpenAiTool(tool: unknown): unknown {
  if (tool !== null && typeof tool === "object" && "function" in tool) return tool;
  return { type: "function", function: tool };
}

function requestForCase(
  candidateModel: string,
  input: unknown,
  params: Record<string, unknown> | undefined,
  systemPromptOverride?: string,
): CompletionRequest {
  const record = (input ?? {}) as { input?: Message[]; tools_available?: unknown[] | null };
  const tools = record.tools_available ?? null;
  let messages = record.input ?? [];
  if (systemPromptOverride !== undefined && messages.length > 0) {
    messages = [
      { role: "system", content: systemPromptOverride },
      ...messages.filter((m) => m.role !== "system"),
    ];
  }
  return {
    model: candidateModel,
    messages,
    ...(tools && tools.length > 0 ? { tools: tools.map(toOpenAiTool) } : {}),
    ...(params ? { params } : {}),
  };
}

/**
 * Run the experiment. Always creates an experiment row and always closes it, so
 * a crash never leaves one `running`.
 */
export async function runExperiment(
  db: CompoundDatabase,
  options: RunExperimentOptions,
): Promise<RunExperimentResult> {
  const partition: CasePartition = options.partition ?? "optimizer_validation";
  if (partition === "decision_test" && options.allowDecisionTest !== true) {
    throw new DecisionPartitionRefusedError();
  }

  const paid = isPaidEnabled(options);
  const experiment = createExperiment(db, {
    taskKey: options.taskKey,
    candidateModel: options.candidateModel,
    provider: options.providerName,
    partition,
    paid,
  });

  try {
    const cases = listCases(db, {
      taskKey: options.taskKey,
      partition,
      limit: options.maxCases ?? 1000,
      ...(partition === "decision_test" && options.allowDecisionTest
        ? { openDecisionFirewall: { reason: "experiment gate", openedAt: new Date() } }
        : {}),
    });

    const results: CaseRunResult[] = [];
    const skipReasons: Record<string, number> = {};
    let graded = 0;
    let skipped = 0;
    let passed = 0;
    let scoreSum = 0;
    let cacheHits = 0;
    let providerCalls = 0;
    let estimatedCost = 0;
    let actualCost = 0;

    for (const stored of cases) {
      const request = requestForCase(
        options.candidateModel,
        stored.input,
        options.params,
        options.systemPromptOverride,
      );
      if (request.messages.length === 0) {
        skipped += 1;
        skipReasons.empty_input = (skipReasons.empty_input ?? 0) + 1;
        results.push({ caseId: stored.caseId, status: "skipped", detail: "no input messages" });
        continue;
      }

      const fingerprint = completionFingerprint({
        provider: options.providerName,
        request,
        providerRevision: options.providerRevision,
      });

      let response: CompletionResponse;
      let costUsd = 0;
      let cached = false;

      const hit = getCachedCompletion(db, fingerprint);
      if (hit !== null) {
        cached = true;
        cacheHits += 1;
        response = {
          output: hit.outputJson as Message,
          usage: (hit.usageJson as CompletionResponse["usage"]) ?? null,
          finishReason: hit.finishReason,
          resolvedModel: hit.resolvedModel,
          latencyMs: hit.latencyMs ?? 0,
        };
        costUsd = hit.costUsd;
      } else if (!paid) {
        estimatedCost += estimateCost(request, options.price);
        results.push({
          caseId: stored.caseId,
          status: "cache_miss_dry_run",
          detail: "would call the provider (dry run)",
        });
        continue;
      } else {
        const estimate = estimateCost(request, options.price);
        // Enforce BOTH caps before spending a cent.
        requireBudgetHeadroom(db, {
          fingerprint,
          estimatedCost: estimate,
          experimentId: experiment.id,
          experimentCapUsd: options.experimentCapUsd as number,
          globalHardLimitUsd: options.globalHardLimitUsd as number,
        });
        estimatedCost += estimate;

        response = await options.provider.complete(request);
        costUsd = costFromUsage(response.usage, options.price);
        providerCalls += 1;
        actualCost += costUsd;

        // Persist before grading, so a crash after the paid call never loses it.
        recordSpend(db, { fingerprint, costUsd, experimentId: experiment.id });
        cacheCompletion(db, {
          fingerprint,
          provider: options.providerName,
          model: options.candidateModel,
          resolvedModel: response.resolvedModel,
          params: options.params ?? null,
          output: response.output,
          usage: response.usage,
          finishReason: response.finishReason,
          latencyMs: response.latencyMs,
          costUsd,
        });
      }

      const report = evaluateAssertions(options.assertions, { output: response.output });
      graded += 1;
      if (report.passed) passed += 1;
      scoreSum += report.score;
      results.push({
        caseId: stored.caseId,
        status: "graded",
        passed: report.passed,
        score: report.score,
        costUsd,
        cached,
        completionFingerprint: fingerprint,
      });
    }

    const report: ExperimentReport = {
      cases_total: cases.length,
      cases_graded: graded,
      cases_skipped: skipped,
      passed,
      pass_rate: graded > 0 ? passed / graded : 0,
      mean_score: graded > 0 ? scoreSum / graded : 0,
      cache_hits: cacheHits,
      provider_calls: providerCalls,
      estimated_cost_usd: estimatedCost,
      actual_cost_usd: actualCost,
      skip_reasons: skipReasons,
    };
    // Persist per-case outcomes so a gate can pair candidate vs reference by
    // case, and the dashboard can show run diffs (docs/gate-decision-v1.md).
    recordCaseResults(
      db,
      experiment.id,
      results.map((r) => ({
        caseId: r.caseId,
        status: r.status,
        ...(r.passed !== undefined ? { passed: r.passed } : {}),
        ...(r.score !== undefined ? { score: r.score } : {}),
        ...(r.completionFingerprint !== undefined
          ? { completionFingerprint: r.completionFingerprint }
          : {}),
      })),
    );

    finishExperiment(db, experiment.id, "completed", report);
    return { experimentId: experiment.id, report, results };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    finishExperiment(db, experiment.id, "failed", { error: message });
    throw error;
  }
}
