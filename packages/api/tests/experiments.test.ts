import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { type CompoundDatabase, createExperiment, finishExperiment } from "@compound/storage";
import { freshDatabase, getJson, testApp } from "./helpers";

let db: CompoundDatabase;

beforeEach(() => {
  db = freshDatabase();
});

afterEach(() => {
  db.close();
});

describe("GET /api/experiments", () => {
  test("returns an empty page when nothing has been run", async () => {
    const { status, body } = await getJson(testApp(db), "/api/experiments");
    expect(status).toBe(200);
    expect(body.items).toEqual([]);
    expect(body.total).toBe(0);
    expect(body.limit).toBe(50);
    expect(body.offset).toBe(0);
  });

  test("serializes a finished experiment with its report", async () => {
    const created = createExperiment(db, {
      taskKey: "support",
      candidateModel: "zai-org/GLM-5.2-FP8",
      provider: "doubleword",
      partition: "optimization_train",
      paid: true,
    });
    finishExperiment(db, created.id, "completed", {
      cases_total: 10,
      passed: 8,
      pass_rate: 0.8,
      actual_cost_usd: 0.12,
    });

    const { status, body } = await getJson(testApp(db), "/api/experiments");
    expect(status).toBe(200);
    expect(body.total).toBe(1);
    const [item] = body.items;
    expect(item.task_key).toBe("support");
    expect(item.candidate_model).toBe("zai-org/GLM-5.2-FP8");
    expect(item.provider).toBe("doubleword");
    expect(item.partition).toBe("optimization_train");
    expect(item.status).toBe("completed");
    expect(item.paid).toBe(true);
    expect(item.report.pass_rate).toBe(0.8);
    expect(item.report.actual_cost_usd).toBe(0.12);
    expect(typeof item.started_at).toBe("string");
    expect(typeof item.completed_at).toBe("string");
  });

  test("a running experiment has a null report and completed_at", async () => {
    createExperiment(db, {
      taskKey: "support",
      candidateModel: "deepseek-ai/DeepSeek-V4-Flash",
      provider: "doubleword",
      partition: "optimization_train",
      paid: false,
    });
    const { body } = await getJson(testApp(db), "/api/experiments");
    const [item] = body.items;
    expect(item.status).toBe("running");
    expect(item.report).toBeNull();
    expect(item.completed_at).toBeNull();
  });

  test("filters by task_key and candidate_model", async () => {
    createExperiment(db, {
      taskKey: "support",
      candidateModel: "model-a",
      provider: "doubleword",
      partition: "optimization_train",
      paid: false,
    });
    createExperiment(db, {
      taskKey: "billing",
      candidateModel: "model-b",
      provider: "doubleword",
      partition: "optimization_train",
      paid: false,
    });

    const byTask = await getJson(testApp(db), "/api/experiments?task_key=support");
    expect(byTask.body.total).toBe(1);
    expect(byTask.body.items[0].task_key).toBe("support");

    const byModel = await getJson(testApp(db), "/api/experiments?candidate_model=model-b");
    expect(byModel.body.total).toBe(1);
    expect(byModel.body.items[0].candidate_model).toBe("model-b");
  });

  test("rejects an invalid limit with the standard error envelope", async () => {
    const { status, body } = await getJson(testApp(db), "/api/experiments?limit=banana");
    expect(status).toBe(400);
    expect(body.error.code).toBe("invalid_request");
    expect(body.error.details.parameter).toBe("limit");
  });
});
