/**
 * The Compound local API.
 *
 * Spec: docs/api-design-v1.md. Two rules from it shape this file:
 *
 * 1. No endpoint that lies. Routes exist only for features that are built.
 *    `POST /api/imports` is deliberately absent until ingest lands — a missing
 *    route is honest, a stub returning a fabricated shape is not.
 * 2. The app is a function, not a server. `createApp` returns a Hono app with
 *    its dependencies injected, so every route is testable without a port.
 */

import { type CompoundConfig, validateConfig } from "@compound/config";
import { TRACE_SCHEMA_VERSION } from "@compound/contract";
import {
  type CompoundDatabase,
  countImportBatches,
  countTraces,
  countTracesByDiagnosticReason,
  countTracesByTaskKey,
  countTracesByValidationClass,
  getImportBatch,
  getTraceByTraceId,
  type ImportBatchRow,
  listImportBatches,
  listTraces,
  type StoredTrace,
} from "@compound/storage";
import { Hono } from "hono";
import { stripSecrets } from "./config-view";
import { errorResponse, invalidRequest, notFound } from "./errors";
import { parseDateParam, parseEnumParam, parsePageParams, parseTaskKeyParam } from "./query";

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

  app.get("/api/imports/:id", (c) => {
    const id = c.req.param("id");
    const batch = getImportBatch(db, id);
    if (batch === null) {
      throw notFound(`no import batch with id ${id}`, { id });
    }
    return c.json(serializeBatch(batch));
  });

  return app;
}
