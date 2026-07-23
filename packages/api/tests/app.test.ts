import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import type { CompoundDatabase } from "@compound/storage";
import { completeImportBatch, failImportBatch } from "@compound/storage";
import { APP_VERSION, CONFIG_SCHEMA_VERSION } from "../src/app";
import { freshDatabase, getJson, postJson, seedTraces, testApp } from "./helpers";

let db: CompoundDatabase;

beforeEach(() => {
  db = freshDatabase();
});

afterEach(() => {
  db.close();
});

describe("GET /health", () => {
  test("reports status and the schema versions clients must agree on", async () => {
    const { status, body } = await getJson(testApp(db), "/health");
    expect(status).toBe(200);
    expect(body).toEqual({
      status: "ok",
      version: APP_VERSION,
      trace_schema_version: 1,
      config_schema_version: CONFIG_SCHEMA_VERSION,
    });
  });
});

describe("unknown routes", () => {
  test("return the standard error envelope, not an HTML page", async () => {
    const { status, body } = await getJson(testApp(db), "/api/nope");
    expect(status).toBe(404);
    expect(body.error.code).toBe("not_found");
    expect(body.error.message).toContain("/api/nope");
  });

  test("POST /api/imports does not exist until ingest is built", async () => {
    const { status, body } = await postJson(testApp(db), "/api/imports", {});
    expect(status).toBe(404);
    expect(body.error.code).toBe("not_found");
  });
});

describe("GET /api/config", () => {
  test("returns the loaded config", async () => {
    const { status, body } = await getJson(testApp(db), "/api/config");
    expect(status).toBe(200);
    expect(body.config.version).toBe(1);
    expect(body.config.providers.openrouter.api_key_env).toBe("OPENROUTER_API_KEY");
  });

  test("omits secret values entirely rather than masking them", async () => {
    const config = {
      ...(await getJson(testApp(db), "/api/config")).body.config,
      providers: {
        leaky: { base_url: "https://example.com", api_key_env: "X", api_key: "sk-real-secret" },
      },
    };
    const { body } = await getJson(testApp(db, config), "/api/config");
    const serialized = JSON.stringify(body);
    expect(serialized).not.toContain("sk-real-secret");
    expect(body.config.providers.leaky).not.toHaveProperty("api_key");
    // The env-var *name* survives: it is a pointer, not a credential.
    expect(body.config.providers.leaky.api_key_env).toBe("X");
    expect(body.omitted_secret_paths).toContain("providers.leaky.api_key");
  });
});

describe("POST /api/config/validate", () => {
  test("accepts a valid config", async () => {
    const app = testApp(db);
    const { body: current } = await getJson(app, "/api/config");
    const { status, body } = await postJson(app, "/api/config/validate", current.config);
    expect(status).toBe(200);
    expect(body).toEqual({ valid: true, issues: [] });
  });

  test("reports path-qualified issues for an invalid config", async () => {
    const app = testApp(db);
    const { body: current } = await getJson(app, "/api/config");
    const broken = {
      ...current.config,
      task_keys: { support: { replay: { default_tool_policy: "teleport" } } },
    };
    const { status, body } = await postJson(app, "/api/config/validate", broken);
    // Answering "is this valid?" with "no" is a successful request.
    expect(status).toBe(200);
    expect(body.valid).toBe(false);
    expect(body.issues.length).toBeGreaterThan(0);
    expect(body.issues[0].path).toContain("task_keys");
  });

  test("rejects a non-JSON body with invalid_request", async () => {
    const { status, body } = await postJson(
      testApp(db),
      "/api/config/validate",
      undefined,
      "not json",
    );
    expect(status).toBe(400);
    expect(body.error.code).toBe("invalid_request");
  });

  test("never writes the config file", async () => {
    const app = testApp(db);
    const before = (await getJson(app, "/api/config")).body.config;
    await postJson(app, "/api/config/validate", { version: 99 });
    const after = (await getJson(app, "/api/config")).body.config;
    expect(after).toEqual(before);
  });
});

describe("GET /api/traces", () => {
  test("returns an empty page against an empty store", async () => {
    const { status, body } = await getJson(testApp(db), "/api/traces");
    expect(status).toBe(200);
    expect(body).toEqual({ items: [], total: 0, limit: 50, offset: 0 });
  });

  test("returns the contract payload beside storage-derived fields", async () => {
    seedTraces(db, { fixtures: [{ name: "eval-ready-full", taskKey: "support" }] });
    const { body } = await getJson(testApp(db), "/api/traces");
    expect(body.total).toBe(1);
    const item = body.items[0];
    // The trace is exactly what the importer produced: no injected keys.
    expect(item.trace.schema).toBe("compound.trace");
    expect(item.trace).not.toHaveProperty("validation_class");
    expect(item.validation_class).toBe("eval_ready");
    expect(item.diagnostic_reasons).toEqual([]);
    expect(item.import_batch_id).toBeString();
  });

  test("filters by validation_class", async () => {
    seedTraces(db, {
      fixtures: [
        { name: "eval-ready-full", count: 2 },
        { name: "diagnostic-missing-focal", count: 1 },
      ],
    });
    const app = testApp(db);
    expect((await getJson(app, "/api/traces?validation_class=eval_ready")).body.total).toBe(2);
    const diagnostic = await getJson(app, "/api/traces?validation_class=diagnostic");
    expect(diagnostic.body.total).toBe(1);
    expect(diagnostic.body.items[0].diagnostic_reasons).toContain("missing_focal_step_id");
  });

  test("task_key=unassigned selects the null bucket, distinct from no filter", async () => {
    seedTraces(db, {
      fixtures: [
        { name: "eval-ready-full", count: 2, taskKey: "support" },
        { name: "eval-ready-minimal", count: 1, taskKey: null },
      ],
    });
    const app = testApp(db);
    expect((await getJson(app, "/api/traces")).body.total).toBe(3);
    expect((await getJson(app, "/api/traces?task_key=unassigned")).body.total).toBe(1);
    expect((await getJson(app, "/api/traces?task_key=support")).body.total).toBe(2);
  });

  test("paginates with a stable total", async () => {
    seedTraces(db, { fixtures: [{ name: "eval-ready-full", count: 5 }] });
    const { body } = await getJson(testApp(db), "/api/traces?limit=2&offset=2");
    expect(body.items).toHaveLength(2);
    expect(body.total).toBe(5);
    expect(body.limit).toBe(2);
    expect(body.offset).toBe(2);
  });

  test("rejects malformed pagination instead of coercing it", async () => {
    const app = testApp(db);
    for (const query of ["limit=banana", "limit=0", "limit=100000", "offset=-1"]) {
      const { status, body } = await getJson(app, `/api/traces?${query}`);
      expect(status).toBe(400);
      expect(body.error.code).toBe("invalid_request");
    }
  });

  test("rejects an unknown validation_class", async () => {
    const { status, body } = await getJson(testApp(db), "/api/traces?validation_class=rejected");
    expect(status).toBe(400);
    expect(body.error.details.parameter).toBe("validation_class");
  });

  test("rejects a malformed date filter", async () => {
    const { status } = await getJson(testApp(db), "/api/traces?from=yesterday");
    expect(status).toBe(400);
  });
});

describe("GET /api/traces/:traceId", () => {
  test("returns one trace", async () => {
    seedTraces(db, { fixtures: [{ name: "eval-ready-full" }] });
    const { status, body } = await getJson(testApp(db), "/api/traces/eval-ready-full-0");
    expect(status).toBe(200);
    expect(body.trace.trace_id).toBe("eval-ready-full-0");
  });

  test("404s for an unknown trace_id", async () => {
    const { status, body } = await getJson(testApp(db), "/api/traces/nope");
    expect(status).toBe(404);
    expect(body.error.code).toBe("not_found");
    expect(body.error.details.trace_id).toBe("nope");
  });

  test("'stats' is routed as the stats endpoint, not as a trace id", async () => {
    const { status, body } = await getJson(testApp(db), "/api/traces/stats");
    expect(status).toBe(200);
    expect(body).toHaveProperty("by_validation_class");
  });
});

describe("GET /api/traces/stats", () => {
  test("counts by class and by task key including the unassigned bucket", async () => {
    seedTraces(db, {
      fixtures: [
        { name: "eval-ready-full", count: 2, taskKey: "support" },
        { name: "eval-ready-minimal", count: 1, taskKey: null },
        { name: "diagnostic-missing-focal", count: 1, taskKey: "support" },
      ],
    });
    const { body } = await getJson(testApp(db), "/api/traces/stats");
    expect(body.total).toBe(4);
    expect(body.by_validation_class).toEqual({ eval_ready: 3, diagnostic: 1 });
    const support = body.by_task_key.find((row: any) => row.task_key === "support");
    const unassigned = body.by_task_key.find((row: any) => row.task_key === null);
    expect(support.count).toBe(3);
    expect(unassigned.count).toBe(1);
  });

  test("reports both classes as zero on an empty store", async () => {
    const { body } = await getJson(testApp(db), "/api/traces/stats");
    expect(body.by_validation_class).toEqual({ eval_ready: 0, diagnostic: 0 });
    expect(body.by_task_key).toEqual([]);
    expect(body.by_diagnostic_reason).toEqual([]);
  });

  test("groups diagnostic reasons so the queue can show what dominates", async () => {
    seedTraces(db, {
      fixtures: [
        { name: "diagnostic-missing-focal", count: 2 },
        { name: "eval-ready-full", count: 1 },
      ],
    });
    const { body } = await getJson(testApp(db), "/api/traces/stats");
    const missingFocal = body.by_diagnostic_reason.find(
      (row: any) => row.reason === "missing_focal_step_id",
    );
    expect(missingFocal.count).toBe(2);
  });
});

describe("import batches", () => {
  test("lists batches with their reports", async () => {
    const batch = seedTraces(db, { fixtures: [{ name: "eval-ready-full" }] });
    const report = {
      counts: { eval_ready: 1, diagnostic: 0, rejected: 2, duplicate: 0 },
      diagnostic_reasons: {},
      dialects: ["openai_tool_calls"],
    };
    completeImportBatch(db, batch.id, report);
    const { body } = await getJson(testApp(db), "/api/imports");
    expect(body.total).toBe(1);
    expect(body.items[0].status).toBe("completed");
    // Rejected records live only here as counts — never as stored traces.
    expect(body.items[0].report).toEqual(report);
  });

  test("returns one batch by id", async () => {
    const batch = seedTraces(db, { fixtures: [{ name: "eval-ready-full" }] });
    const { status, body } = await getJson(testApp(db), `/api/imports/${batch.id}`);
    expect(status).toBe(200);
    expect(body.id).toBe(batch.id);
    expect(body.importer).toBe("test");
    expect(body.report).toBeNull();
  });

  test("filters by status", async () => {
    const running = seedTraces(db, {});
    const failed = seedTraces(db, {});
    failImportBatch(db, failed.id);
    const app = testApp(db);
    const runningList = await getJson(app, "/api/imports?status=running");
    expect(runningList.body.total).toBe(1);
    expect(runningList.body.items[0].id).toBe(running.id);
    expect((await getJson(app, "/api/imports?status=failed")).body.total).toBe(1);
  });

  test("404s for an unknown batch id", async () => {
    const { status, body } = await getJson(testApp(db), "/api/imports/nope");
    expect(status).toBe(404);
    expect(body.error.code).toBe("not_found");
  });
});
