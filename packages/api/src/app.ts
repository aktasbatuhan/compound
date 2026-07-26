/**
 * The Compound local API.
 *
 * Spec: docs/api-design-v1.md. Two rules from it shape this file:
 *
 * 1. No endpoint that lies. Routes exist only for features that are built.
 *    `POST /api/imports` exists as of Step 2 because ingest is real; it ran a
 *    genuine import before the route was added.
 * 2. The app is a function, not a server. `createApp` returns a Hono app with
 *    its dependencies injected, so every route is testable without a port.
 */

import { readFileSync } from "node:fs";
import { type Assertion, gradeCaseObservedOutput } from "@compound/assertions";
import { type CompoundConfig, validateConfig } from "@compound/config";
import { TRACE_SCHEMA_VERSION } from "@compound/contract";
import { curateTask } from "@compound/curation";
import { runImport, UnsupportedImporterError } from "@compound/pipeline";
import {
  CASE_PROVENANCES,
  CASE_REVIEW_STATES,
  type CaseRow,
  type CompoundDatabase,
  countCases,
  countCasesByPartition,
  countCasesByProvenance,
  countImportBatches,
  countTraces,
  countTracesByDiagnosticReason,
  countTracesByTaskKey,
  countTracesByValidationClass,
  type ExperimentRow,
  type GateResultRow,
  type GateSpecRow,
  getCase,
  getGateResult,
  getImportBatch,
  getTraceByTraceId,
  type ImportBatchRow,
  InvalidPromotionError,
  type JudgeCalibrationRow,
  listCases,
  listExperiments,
  listGateResults,
  listImportBatches,
  listLatestCalibrations,
  listOptimizationRuns,
  listTraces,
  type OptimizationRunRow,
  reviewCase,
  type StoredTrace,
  UnknownCaseError,
} from "@compound/storage";
import { Hono } from "hono";
import { stripSecrets } from "./config-view";
import { errorResponse, invalidRequest, notFound } from "./errors";
import { parseImportRequest } from "./import-request";
import { parseDateParam, parseEnumParam, parsePageParams, parseTaskKeyParam } from "./query";
import { parseReviewRequest } from "./review-request";

export const APP_VERSION = "0.1.0";
export const CONFIG_SCHEMA_VERSION = 1;

export interface AppDependencies {
  db: CompoundDatabase;
  config: CompoundConfig;
}

/**
 * Storage-derived fields travel BESIDE the contract payload, never merged into
 * it, so `trace` stays exactly what the importer produced.
 */
function serializeTrace(stored: StoredTrace) {
  return {
    trace: stored.trace,
    validation_class: stored.validationClass,
    diagnostic_reasons: stored.diagnosticReasons,
    import_batch_id: stored.importBatchId,
    content_hash: stored.contentHash,
    imported_at: stored.createdAt.toISOString(),
  };
}

function serializeCase(row: CaseRow) {
  return {
    case_id: row.caseId,
    task_key: row.taskKey,
    source_trace_id: row.sourceTraceId,
    content_hash: row.contentHash,
    provenance: row.provenance,
    partition: row.partition,
    review_state: row.reviewState,
    input: row.input,
    expected: row.expected,
    duplicate_count: row.duplicateCount,
    created_at: row.createdAt.toISOString(),
    updated_at: row.updatedAt.toISOString(),
  };
}

function serializeBatch(batch: ImportBatchRow) {
  return {
    id: batch.id,
    importer: batch.importer,
    importer_version: batch.importerVersion,
    source_fingerprint: batch.sourceFingerprint,
    status: batch.status,
    started_at: batch.startedAt.toISOString(),
    completed_at: batch.completedAt?.toISOString() ?? null,
    // Drizzle stores `report` as a JSON column and hands it back parsed.
    report: batch.report,
    created_at: batch.createdAt.toISOString(),
  };
}

function serializeExperiment(row: ExperimentRow) {
  return {
    id: row.id,
    task_key: row.taskKey,
    candidate_model: row.candidateModel,
    provider: row.provider,
    partition: row.partition,
    status: row.status,
    paid: row.paid,
    // Drizzle stores `report` as a JSON column and hands it back parsed; it is
    // null until the experiment finishes.
    report: row.report ?? null,
    started_at: row.startedAt.toISOString(),
    completed_at: row.completedAt?.toISOString() ?? null,
    created_at: row.createdAt.toISOString(),
  };
}

function serializeGate(result: GateResultRow, spec: GateSpecRow) {
  return {
    id: result.id,
    task_key: spec.taskKey,
    candidate_model: spec.candidateModel,
    reference_model: spec.referenceModel,
    metric: spec.metric,
    mode: spec.mode,
    margin: spec.margin,
    confidence: spec.confidence,
    min_cases: spec.minCases,
    // The stated reason the sealed set was opened — part of the lineage.
    firewall_reason: spec.firewallReason,
    // Set on an adoption gate: the optimized prompt under test and its artifact.
    candidate_prompt_hash: spec.candidatePromptHash,
    optimization_run_id: spec.optimizationRunId,
    outcome: result.outcome,
    delta: result.delta,
    ci: [result.ciLo, result.ciHi],
    n: result.n,
    candidate_rate: result.candidateRate,
    reference_rate: result.referenceRate,
    judge_abstained_fraction: result.judgeAbstainedFraction,
    candidate_experiment_id: result.candidateExperimentId,
    reference_experiment_id: result.referenceExperimentId,
    decided_at: result.decidedAt.toISOString(),
  };
}

function serializeCalibration(row: JudgeCalibrationRow) {
  return {
    id: row.id,
    task_key: row.taskKey,
    judge_model: row.judgeModel,
    prompt_version: row.promptVersion,
    rubric_hash: row.rubricHash,
    mode: row.mode,
    agreement_kappa: row.agreementKappa,
    kappa_ci: [row.kappaCiLo, row.kappaCiHi],
    n: row.n,
    position_bias_rate: row.positionBiasRate,
    threshold: row.threshold,
    calibrated: row.calibrated,
    measured_at: row.measuredAt.toISOString(),
  };
}

function serializeOptimization(row: OptimizationRunRow) {
  return {
    id: row.id,
    task_key: row.taskKey,
    candidate_model: row.candidateModel,
    seed_prompt: row.seedPrompt,
    optimized_prompt: row.optimizedPrompt,
    before_val_score: row.beforeValScore,
    after_val_score: row.afterValScore,
    val_cases: row.valCases,
    reflection_calls: row.reflectionCalls,
    eligibility_reason: row.eligibilityReason,
    cost_usd: row.costUsd,
    created_at: row.createdAt.toISOString(),
  };
}

export function createApp({ db, config }: AppDependencies): Hono {
  const app = new Hono();

  app.onError((error, c) => errorResponse(c, error));
  app.notFound((c) => errorResponse(c, notFound(`no route for ${c.req.method} ${c.req.path}`)));

  app.get("/health", (c) =>
    c.json({
      status: "ok",
      version: APP_VERSION,
      trace_schema_version: TRACE_SCHEMA_VERSION,
      config_schema_version: CONFIG_SCHEMA_VERSION,
    }),
  );

  // --- config: readable and validatable, never writable over HTTP ----------

  app.get("/api/config", (c) => {
    const { value, omitted } = stripSecrets(config);
    return c.json({ config: value, omitted_secret_paths: omitted });
  });

  app.post("/api/config/validate", async (c) => {
    let body: unknown;
    try {
      body = await c.req.json();
    } catch {
      throw invalidRequest("request body must be a JSON document");
    }
    const result = validateConfig(body);
    // A config that fails validation is a successful *answer* to "is this
    // valid?", not a failed request — the dashboard renders the issues.
    return result.ok
      ? c.json({ valid: true, issues: [] })
      : c.json({ valid: false, issues: result.issues });
  });

  // --- traces --------------------------------------------------------------

  // Registered before `/api/traces/:traceId` so "stats" is never read as an id.
  app.get("/api/traces/stats", (c) => {
    const query = new URL(c.req.url).searchParams;
    const validationClass = parseEnumParam(query, "validation_class", [
      "eval_ready",
      "diagnostic",
    ] as const);
    const filter = validationClass ? { validationClass } : {};
    const byClass = countTracesByValidationClass(db);
    const byTaskKey = countTracesByTaskKey(db, filter);
    // A trace contributes one count per reason it carries, so these do not sum
    // to the diagnostic total. The queue wants "which failure dominates?".
    const byReason = countTracesByDiagnosticReason(db, filter);
    return c.json({
      total: byClass.eval_ready + byClass.diagnostic,
      by_validation_class: byClass,
      by_task_key: byTaskKey.map((row) => ({
        task_key: row.taskKey,
        count: row.count,
      })),
      by_diagnostic_reason: byReason,
    });
  });

  app.get("/api/traces", (c) => {
    const query = new URL(c.req.url).searchParams;
    const { limit, offset } = parsePageParams(query);
    const filter = {
      taskKey: parseTaskKeyParam(query),
      validationClass: parseEnumParam(query, "validation_class", [
        "eval_ready",
        "diagnostic",
      ] as const),
      importBatchId: query.get("import_batch_id") ?? undefined,
      startedAtFrom: parseDateParam(query, "from"),
      startedAtTo: parseDateParam(query, "to"),
    };
    const items = listTraces(db, { ...filter, limit, offset });
    return c.json({
      items: items.map(serializeTrace),
      total: countTraces(db, filter),
      limit,
      offset,
    });
  });

  app.get("/api/traces/:traceId", (c) => {
    const traceId = c.req.param("traceId");
    const stored = getTraceByTraceId(db, traceId);
    if (stored === null) {
      throw notFound(`no trace with trace_id ${traceId}`, { trace_id: traceId });
    }
    return c.json(serializeTrace(stored));
  });

  // --- import batches ------------------------------------------------------

  app.get("/api/imports", (c) => {
    const query = new URL(c.req.url).searchParams;
    const { limit, offset } = parsePageParams(query);
    const status = parseEnumParam(query, "status", ["running", "completed", "failed"] as const);
    const filter = status ? { status } : {};
    const items = listImportBatches(db, { ...filter, limit, offset });
    return c.json({
      items: items.map(serializeBatch),
      total: countImportBatches(db, filter),
      limit,
      offset,
    });
  });

  app.post("/api/imports", async (c) => {
    let body: unknown;
    try {
      body = await c.req.json();
    } catch {
      throw invalidRequest("request body must be a JSON document");
    }
    const request = parseImportRequest(body);

    let content: string;
    if (request.content !== undefined) {
      content = request.content;
    } else {
      try {
        content = readFileSync(request.path as string, "utf8");
      } catch (error) {
        throw invalidRequest(
          `could not read ${request.path}: ${error instanceof Error ? error.message : "unknown error"}`,
          { parameter: "path" },
        );
      }
    }

    try {
      const { batch, report } = runImport(db, {
        importer: request.importer,
        content,
        config,
        projectId: request.project_id,
      });
      return c.json({ batch: serializeBatch(batch), report }, 201);
    } catch (error) {
      if (error instanceof UnsupportedImporterError) {
        throw invalidRequest(error.message, { parameter: "importer" });
      }
      throw error;
    }
  });

  app.get("/api/imports/:id", (c) => {
    const id = c.req.param("id");
    const batch = getImportBatch(db, id);
    if (batch === null) {
      throw notFound(`no import batch with id ${id}`, { id });
    }
    return c.json(serializeBatch(batch));
  });

  // --- cases ---------------------------------------------------------------

  app.post("/api/tasks/:taskKey/curate", (c) => {
    const taskKey = c.req.param("taskKey");
    const report = curateTask(db, { taskKey });
    return c.json(report, 201);
  });

  app.get("/api/cases", (c) => {
    const query = new URL(c.req.url).searchParams;
    const { limit, offset } = parsePageParams(query);
    // The decision partition is sealed: this route never opens the firewall,
    // so it can never surface a decision_test case, whatever is asked for.
    const items = listCases(db, {
      taskKey: query.get("task_key") ?? undefined,
      partition: parseEnumParam(query, "partition", [
        "optimization_train",
        "optimizer_validation",
        "judge_calibration",
      ] as const),
      provenance: parseEnumParam(query, "provenance", CASE_PROVENANCES),
      reviewState: parseEnumParam(query, "review_state", CASE_REVIEW_STATES),
      limit,
      offset,
    });
    return c.json({
      items: items.map(serializeCase),
      total: countCases(db, { taskKey: query.get("task_key") ?? undefined }),
      limit,
      offset,
    });
  });

  app.get("/api/cases/stats", (c) => {
    const query = new URL(c.req.url).searchParams;
    const taskKey = query.get("task_key") ?? undefined;
    return c.json({
      by_partition: countCasesByPartition(db, taskKey).map((row) => ({
        partition: row.partition,
        count: row.count,
      })),
      by_provenance: countCasesByProvenance(db, taskKey).map((row) => ({
        provenance: row.provenance,
        count: row.count,
      })),
    });
  });

  app.get("/api/cases/:caseId", (c) => {
    const caseId = c.req.param("caseId");
    const found = getCase(db, caseId);
    if (found === null) throw notFound(`no case with id ${caseId}`, { case_id: caseId });
    // A decision_test case is fetchable by its exact id — that is not a scan of
    // the sealed set, and hiding a case someone already holds the id for would
    // be confusing. Bulk listing remains sealed.
    return c.json(serializeCase(found));
  });

  app.get("/api/cases/:caseId/assertions", (c) => {
    const caseId = c.req.param("caseId");
    const found = getCase(db, caseId);
    if (found === null) throw notFound(`no case with id ${caseId}`, { case_id: caseId });

    // Free: grades the case's own observed output against the task's assertions.
    // No model call. Assertions come from config; an absent list grades nothing.
    const assertions = config.assertions?.[found.taskKey] ?? [];
    const report = gradeCaseObservedOutput(
      {
        caseId: found.caseId,
        taskKey: found.taskKey,
        provenance: found.provenance,
        expected: found.expected,
      },
      assertions as Assertion[],
    );
    return c.json(report);
  });

  app.post("/api/cases/:caseId/review", async (c) => {
    const caseId = c.req.param("caseId");
    let body: unknown;
    try {
      body = await c.req.json();
    } catch {
      throw invalidRequest("request body must be a JSON document");
    }
    const review = parseReviewRequest(body);
    try {
      const updated = reviewCase(db, caseId, review);
      return c.json(serializeCase(updated));
    } catch (error) {
      if (error instanceof UnknownCaseError) {
        throw notFound(`no case with id ${caseId}`, { case_id: caseId });
      }
      if (error instanceof InvalidPromotionError) {
        throw invalidRequest(error.message);
      }
      throw error;
    }
  });

  // --- experiments ---------------------------------------------------------

  // Read-only: the dashboard's model matrix shows what has been run. Launching a
  // paid run stays a deliberate CLI action (docs/dashboard-v1.md), so there is
  // no POST here.
  app.get("/api/experiments", (c) => {
    const query = new URL(c.req.url).searchParams;
    const { limit, offset } = parsePageParams(query);
    const filter = {
      taskKey: query.get("task_key") ?? undefined,
      candidateModel: query.get("candidate_model") ?? undefined,
    };
    const items = listExperiments(db, { ...filter, limit, offset });
    // No dedicated count helper exists in storage (which this task must not
    // change); the full filtered list gives an honest total. Experiment volume
    // is small, so this is cheap.
    const total = listExperiments(db, filter).length;
    return c.json({
      items: items.map(serializeExperiment),
      total,
      limit,
      offset,
    });
  });

  // --- gates ---------------------------------------------------------------

  // Read-only: gate decisions are shown here, but DECIDING a gate opens the
  // sealed decision set, which stays a deliberate CLI action (`compound gate`)
  // requiring a stated reason. The sealed cases themselves are never returned —
  // only the verdict, the delta, and its confidence interval.
  app.get("/api/gates", (c) => {
    const query = new URL(c.req.url).searchParams;
    const { limit } = parsePageParams(query);
    const taskKey = query.get("task_key") ?? undefined;
    const rows = listGateResults(db, limit).filter(
      (r) => taskKey === undefined || r.spec.taskKey === taskKey,
    );
    return c.json({ items: rows.map((r) => serializeGate(r.result, r.spec)) });
  });

  app.get("/api/gates/:id", (c) => {
    const id = c.req.param("id");
    const result = getGateResult(db, id);
    if (result === null) throw notFound(`no gate result with id ${id}`, { id });
    const match = listGateResults(db, 1000).find((r) => r.result.id === id);
    if (match === undefined) throw notFound(`no gate result with id ${id}`, { id });
    return c.json(serializeGate(match.result, match.spec));
  });

  // --- judges --------------------------------------------------------------

  // Read-only calibration status per task. Measuring a calibration is a paid
  // CLI action (`compound judge calibrate`); this only reports the latest
  // result and whether the judge may currently feed a gate.
  app.get("/api/judges", (c) => {
    const rows = listLatestCalibrations(db);
    return c.json({ items: rows.map(serializeCalibration) });
  });

  // --- optimizations -------------------------------------------------------

  // Read-only: GEPA runs are launched from the CLI (`compound optimize`); this
  // surfaces the stored proposals with their before/after validation scores.
  app.get("/api/optimizations", (c) => {
    const taskKey = new URL(c.req.url).searchParams.get("task_key") ?? undefined;
    return c.json({ items: listOptimizationRuns(db, taskKey).map(serializeOptimization) });
  });

  return app;
}
