/**
 * The single typed client over the Compound HTTP API.
 *
 * The dashboard is a VIEW, not a second source of truth (docs/dashboard-v1.md):
 * every read and every action goes through these methods, which wrap `fetch`
 * against the running API (`compound serve`, default http://127.0.0.1:4319).
 * The dashboard holds no database and re-implements no grading or partition
 * logic — if it isn't an API route, it isn't here.
 *
 * All calls run server-side (server components and server actions), so there is
 * no CORS surface and no secret ever reaches the browser.
 */

import type {
  CasePartition,
  CaseProvenance,
  CaseReviewState,
  ExperimentReport,
  ExperimentStatus,
  GateMetric,
  GateMode,
  GateOutcome,
} from "@compound/storage";

export const DEFAULT_API_URL = "http://127.0.0.1:4319";

/** The base URL of the API, overridable for a non-default `compound serve`. */
export function apiBaseUrl(): string {
  return process.env.COMPOUND_API_URL?.replace(/\/$/, "") ?? DEFAULT_API_URL;
}

// --- error envelope --------------------------------------------------------

/** The API's standard error shape: `{ error: { code, message, details? } }`. */
export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    details?: unknown;
  };
}

/**
 * Thrown for any non-2xx response or unreachable API. Pages catch this and show
 * `message` rather than crashing — an offline API is a first-class state.
 */
export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly details: unknown;

  constructor(status: number, code: string, message: string, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

function isErrorBody(value: unknown): value is ApiErrorBody {
  return (
    typeof value === "object" &&
    value !== null &&
    "error" in value &&
    typeof (value as ApiErrorBody).error === "object" &&
    (value as ApiErrorBody).error !== null &&
    typeof (value as ApiErrorBody).error.message === "string"
  );
}

// --- serialized response types (mirror the API's snake_case wire shapes) ----

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  trace_schema_version: number;
  config_schema_version: number;
}

export interface ConfigResponse {
  config: unknown;
  omitted_secret_paths: string[];
}

/** One message in a case's replayable input conversation. */
export interface ConversationMessage {
  role?: string;
  content?: unknown;
  [key: string]: unknown;
}

/** The focal call's replayable request stored on a case. */
export interface CaseInput {
  model?: string;
  input?: ConversationMessage[];
  tools_available?: unknown;
  [key: string]: unknown;
}

export interface CaseResponse {
  case_id: string;
  task_key: string;
  source_trace_id: string;
  content_hash: string;
  provenance: CaseProvenance;
  partition: CasePartition;
  review_state: CaseReviewState;
  input: CaseInput;
  expected: unknown;
  duplicate_count: number;
  created_at: string;
  updated_at: string;
}

export interface CasesStatsResponse {
  by_partition: Array<{ partition: CasePartition; count: number }>;
  by_provenance: Array<{ provenance: CaseProvenance; count: number }>;
}

export interface AssertionResult {
  type: string;
  passed: boolean;
  detail: string;
  required: boolean;
  weight: number;
}

export interface AssertionReportResponse {
  caseId: string;
  graded: boolean;
  passed: boolean;
  score: number;
  results: AssertionResult[];
}

export interface TraceResponse {
  trace: Record<string, unknown>;
  validation_class: "eval_ready" | "diagnostic";
  diagnostic_reasons: string[];
  import_batch_id: string;
  content_hash: string;
  imported_at: string;
}

export interface TracesStatsResponse {
  total: number;
  by_validation_class: { eval_ready: number; diagnostic: number };
  by_task_key: Array<{ task_key: string | null; count: number }>;
  by_diagnostic_reason: Array<{ reason: string; count: number }>;
}

export interface ImportBatchResponse {
  id: string;
  importer: string;
  importer_version: string;
  source_fingerprint: string;
  status: "running" | "completed" | "failed";
  started_at: string;
  completed_at: string | null;
  report: unknown;
  created_at: string;
}

export interface ExperimentResponse {
  id: string;
  task_key: string;
  candidate_model: string;
  provider: string;
  partition: CasePartition;
  status: ExperimentStatus;
  paid: boolean;
  report: ExperimentReport | null;
  started_at: string;
  completed_at: string | null;
  created_at: string;
}

/** A decided gate: the verdict, the point estimate, and its confidence interval. */
export interface GateResponse {
  id: string;
  task_key: string;
  candidate_model: string;
  reference_model: string;
  metric: GateMetric;
  mode: GateMode;
  margin: number;
  confidence: number;
  min_cases: number;
  firewall_reason: string;
  /** Set on an adoption gate: the optimized prompt under test. */
  candidate_prompt_hash: string | null;
  optimization_run_id: string | null;
  outcome: GateOutcome;
  delta: number;
  ci: [number, number];
  n: number;
  candidate_rate: number;
  reference_rate: number;
  judge_abstained_fraction: number;
  candidate_experiment_id: string;
  reference_experiment_id: string;
  decided_at: string;
}

/** The latest judge calibration for a task and whether it may feed a gate. */
export interface JudgeCalibrationResponse {
  id: string;
  task_key: string;
  judge_model: string;
  prompt_version: string;
  rubric_hash: string;
  mode: "pointwise" | "pairwise";
  agreement_kappa: number;
  kappa_ci: [number, number];
  n: number;
  position_bias_rate: number;
  threshold: number;
  calibrated: boolean;
  measured_at: string;
}

/** Operational rollup for one task x model x provider group. */
export interface TelemetryResponse {
  task_key: string;
  model: string;
  provider: string;
  completions: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  mean_cost_usd: number;
  total_cost_usd: number;
  mean_input_tokens: number;
  mean_output_tokens: number;
  total_input_tokens: number;
  total_output_tokens: number;
  output_tps: number;
}

/** A stored GEPA optimization run (a proposed prompt improvement). */
export interface OptimizationResponse {
  id: string;
  task_key: string;
  candidate_model: string;
  seed_prompt: string;
  optimized_prompt: string;
  before_val_score: number;
  after_val_score: number;
  val_cases: number;
  reflection_calls: number;
  eligibility_reason: string | null;
  cost_usd: number;
  created_at: string;
}

// --- request filter shapes -------------------------------------------------

/**
 * The list filters accepted by `GET /api/cases`. There is deliberately NO
 * `decision_test` partition option — the sealed set is never requestable, and
 * the API would refuse it regardless.
 */
export interface CaseListFilters {
  task_key?: string;
  partition?: "optimization_train" | "optimizer_validation" | "judge_calibration";
  provenance?: CaseProvenance;
  review_state?: CaseReviewState;
  limit?: number;
  offset?: number;
}

export interface TraceListFilters {
  task_key?: string;
  validation_class?: "eval_ready" | "diagnostic";
  import_batch_id?: string;
  limit?: number;
  offset?: number;
}

export interface ExperimentListFilters {
  task_key?: string;
  candidate_model?: string;
  limit?: number;
  offset?: number;
}

export interface ReviewRequest {
  review_state: CaseReviewState;
  /** When present, replaces the expected output (may be null to clear it). */
  expected?: unknown;
  /** Only honored with `review_state: "approved"`; otherwise the API refuses. */
  promote_to_golden?: boolean;
}

// --- the client ------------------------------------------------------------

function buildQuery(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) search.set(key, String(value));
  }
  const query = search.toString();
  return query ? `?${query}` : "";
}

export interface ApiClient {
  baseUrl: string;
  getHealth(): Promise<HealthResponse>;
  getConfig(): Promise<ConfigResponse>;
  listCases(filters?: CaseListFilters): Promise<Page<CaseResponse>>;
  getCasesStats(taskKey?: string): Promise<CasesStatsResponse>;
  getCase(caseId: string): Promise<CaseResponse>;
  getCaseAssertions(caseId: string): Promise<AssertionReportResponse>;
  reviewCase(caseId: string, review: ReviewRequest): Promise<CaseResponse>;
  listTraces(filters?: TraceListFilters): Promise<Page<TraceResponse>>;
  getTracesStats(validationClass?: "eval_ready" | "diagnostic"): Promise<TracesStatsResponse>;
  listImports(limit?: number, offset?: number): Promise<Page<ImportBatchResponse>>;
  listExperiments(filters?: ExperimentListFilters): Promise<Page<ExperimentResponse>>;
  listGates(taskKey?: string): Promise<{ items: GateResponse[] }>;
  listJudges(): Promise<{ items: JudgeCalibrationResponse[] }>;
  listOptimizations(taskKey?: string): Promise<{ items: OptimizationResponse[] }>;
  listTelemetry(taskKey?: string): Promise<{ items: TelemetryResponse[] }>;
}

/**
 * Build a client bound to `baseUrl` (defaults to `apiBaseUrl()`). Every method
 * returns parsed JSON or throws {@link ApiError}; there is no silent failure.
 */
export function createApiClient(baseUrl: string = apiBaseUrl()): ApiClient {
  const base = baseUrl.replace(/\/$/, "");

  async function request<T>(path: string, init?: { method?: string; body?: unknown }): Promise<T> {
    const url = `${base}${path}`;
    let response: Response;
    try {
      response = await fetch(url, {
        method: init?.method ?? "GET",
        headers: init?.body === undefined ? undefined : { "content-type": "application/json" },
        body: init?.body === undefined ? undefined : JSON.stringify(init.body),
        // Always hit the API fresh: it is the source of truth, not a CDN cache.
        cache: "no-store",
      });
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : "unknown error";
      throw new ApiError(
        0,
        "api_unreachable",
        `could not reach the Compound API at ${base} — is \`compound serve\` running? (${message})`,
      );
    }

    let payload: unknown = null;
    const text = await response.text();
    if (text.length > 0) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = null;
      }
    }

    if (!response.ok) {
      if (isErrorBody(payload)) {
        throw new ApiError(
          response.status,
          payload.error.code,
          payload.error.message,
          payload.error.details,
        );
      }
      throw new ApiError(
        response.status,
        "http_error",
        `API responded ${response.status} ${response.statusText}`,
      );
    }

    return payload as T;
  }

  return {
    baseUrl: base,
    getHealth: () => request<HealthResponse>("/health"),
    getConfig: () => request<ConfigResponse>("/api/config"),
    listCases: (filters = {}) =>
      request<Page<CaseResponse>>(`/api/cases${buildQuery({ ...filters })}`),
    getCasesStats: (taskKey) =>
      request<CasesStatsResponse>(`/api/cases/stats${buildQuery({ task_key: taskKey })}`),
    getCase: (caseId) => request<CaseResponse>(`/api/cases/${encodeURIComponent(caseId)}`),
    getCaseAssertions: (caseId) =>
      request<AssertionReportResponse>(`/api/cases/${encodeURIComponent(caseId)}/assertions`),
    reviewCase: (caseId, review) =>
      request<CaseResponse>(`/api/cases/${encodeURIComponent(caseId)}/review`, {
        method: "POST",
        body: review,
      }),
    listTraces: (filters = {}) =>
      request<Page<TraceResponse>>(`/api/traces${buildQuery({ ...filters })}`),
    getTracesStats: (validationClass) =>
      request<TracesStatsResponse>(
        `/api/traces/stats${buildQuery({ validation_class: validationClass })}`,
      ),
    listImports: (limit, offset) =>
      request<Page<ImportBatchResponse>>(`/api/imports${buildQuery({ limit, offset })}`),
    listExperiments: (filters = {}) =>
      request<Page<ExperimentResponse>>(`/api/experiments${buildQuery({ ...filters })}`),
    listGates: (taskKey) =>
      request<{ items: GateResponse[] }>(`/api/gates${buildQuery({ task_key: taskKey })}`),
    listJudges: () => request<{ items: JudgeCalibrationResponse[] }>("/api/judges"),
    listOptimizations: (taskKey) =>
      request<{ items: OptimizationResponse[] }>(
        `/api/optimizations${buildQuery({ task_key: taskKey })}`,
      ),
    listTelemetry: (taskKey) =>
      request<{ items: TelemetryResponse[] }>(`/api/telemetry${buildQuery({ task_key: taskKey })}`),
  };
}
