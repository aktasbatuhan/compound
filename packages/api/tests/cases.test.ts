import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { type CompoundDatabase, listCases, openDecisionFirewall } from "@compound/storage";
import { freshDatabase, getJson, postJson, testApp } from "./helpers";

/** A Langfuse export producing N distinct eval-ready traces under one task. */
function exportOf(count: number, taskKey = "support"): string {
  return JSON.stringify(
    Array.from({ length: count }, (_, i) => ({
      id: `tr-${i}`,
      timestamp: "2026-07-23T10:00:00Z",
      tags: [],
      public: false,
      environment: "production",
      metadata: { task_key: taskKey },
      observations: [
        {
          id: `g-${i}`,
          traceId: `tr-${i}`,
          type: "GENERATION",
          startTime: "2026-07-23T10:00:00Z",
          endTime: "2026-07-23T10:00:01Z",
          level: "DEFAULT",
          environment: "production",
          model: "gpt-4o",
          input: [{ role: "user", content: `question ${i}` }],
          output: { role: "assistant", content: `answer ${i}` },
          usageDetails: { input: 5, output: 2 },
        },
      ],
      scores: [],
    })),
  );
}

let db: CompoundDatabase;

beforeEach(() => {
  db = freshDatabase();
});

afterEach(() => {
  db.close();
});

async function importAndCurate(app: ReturnType<typeof testApp>, count: number) {
  await postJson(app, "/api/imports", { importer: "langfuse", content: exportOf(count) });
  return postJson(app, "/api/tasks/support/curate", {});
}

describe("POST /api/tasks/:taskKey/curate", () => {
  test("curates imported traces into cases and reports the split", async () => {
    const app = testApp(db);
    const { status, body } = await importAndCurate(app, 30);
    expect(status).toBe(201);
    expect(body.casesCreated).toBe(30);
    expect(Object.values(body.byPartition).reduce((a: number, b) => a + (b as number), 0)).toBe(30);
  });

  test("is idempotent over HTTP", async () => {
    const app = testApp(db);
    await importAndCurate(app, 10);
    const second = await postJson(app, "/api/tasks/support/curate", {});
    expect(second.body.casesCreated).toBe(0);
    expect(second.body.duplicates).toBe(10);
  });
});

describe("GET /api/cases", () => {
  test("lists cases but never the sealed decision set", async () => {
    const app = testApp(db);
    await importAndCurate(app, 100);

    const listed = await getJson(app, "/api/cases?limit=500");
    expect(listed.body.items.every((c: any) => c.partition !== "decision_test")).toBe(true);

    // Prove some cases really are sealed, so the exclusion is meaningful.
    const sealed = listCases(db, {
      partition: "decision_test",
      openDecisionFirewall: openDecisionFirewall("test assertion"),
    });
    expect(sealed.length).toBeGreaterThan(0);
    // The listed total also excludes the sealed set is NOT asserted here: total
    // counts all non-sealed; the point is the sealed ones never appear.
    expect(listed.body.items.length).toBe(listed.body.total);
  });

  test("an explicit decision_test partition filter is rejected as an unknown enum", async () => {
    // The route does not even offer decision_test as a choice.
    const { status, body } = await getJson(db && testApp(db), "/api/cases?partition=decision_test");
    expect(status).toBe(400);
    expect(body.error.details.parameter).toBe("partition");
  });

  test("filters by provenance", async () => {
    const app = testApp(db);
    await importAndCurate(app, 20);
    const observed = await getJson(app, "/api/cases?provenance=observed_output&limit=500");
    expect(observed.body.items.every((c: any) => c.provenance === "observed_output")).toBe(true);
  });
});

describe("GET /api/cases/stats", () => {
  test("reports the partition and provenance split", async () => {
    const app = testApp(db);
    await importAndCurate(app, 40);
    const { body } = await getJson(app, "/api/cases/stats?task_key=support");
    const partitions = Object.fromEntries(
      body.by_partition.map((r: any) => [r.partition, r.count]),
    );
    // The sealed size is visible here (knowing 5% is sealed != reading it).
    expect(partitions.optimization_train).toBeGreaterThan(0);
    const total = body.by_partition.reduce((a: number, r: any) => a + r.count, 0);
    expect(total).toBe(40);
  });
});

describe("case review", () => {
  async function firstCaseId(app: ReturnType<typeof testApp>): Promise<string> {
    const listed = await getJson(app, "/api/cases?limit=1");
    return listed.body.items[0].case_id;
  }

  test("fetches a single case by id", async () => {
    const app = testApp(db);
    await importAndCurate(app, 5);
    const id = await firstCaseId(app);
    const { status, body } = await getJson(app, `/api/cases/${encodeURIComponent(id)}`);
    expect(status).toBe(200);
    expect(body.case_id).toBe(id);
    expect(body.review_state).toBe("unreviewed");
  });

  test("approving with promotion produces a human_golden case", async () => {
    const app = testApp(db);
    await importAndCurate(app, 5);
    const id = await firstCaseId(app);

    const { status, body } = await postJson(app, `/api/cases/${encodeURIComponent(id)}/review`, {
      review_state: "approved",
      expected: { role: "assistant", content: "human-verified" },
      promote_to_golden: true,
    });
    expect(status).toBe(200);
    expect(body.provenance).toBe("human_golden");
    expect(body.review_state).toBe("approved");
  });

  test("refuses promotion without approval", async () => {
    const app = testApp(db);
    await importAndCurate(app, 5);
    const id = await firstCaseId(app);
    const { status } = await postJson(app, `/api/cases/${encodeURIComponent(id)}/review`, {
      review_state: "needs_edit",
      promote_to_golden: true,
    });
    expect(status).toBe(400);
  });

  test("404s reviewing an unknown case", async () => {
    const { status, body } = await postJson(testApp(db), "/api/cases/case:nope/review", {
      review_state: "approved",
    });
    expect(status).toBe(404);
    expect(body.error.code).toBe("not_found");
  });

  test("rejects an unknown review_state", async () => {
    const app = testApp(db);
    await importAndCurate(app, 2);
    const id = await firstCaseId(app);
    const { status, body } = await postJson(app, `/api/cases/${encodeURIComponent(id)}/review`, {
      review_state: "loved_it",
    });
    expect(status).toBe(400);
    expect(body.error.details.parameter).toBe("review_state");
  });
});
