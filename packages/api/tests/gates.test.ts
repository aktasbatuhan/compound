import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  type CompoundDatabase,
  createExperiment,
  createGateSpec,
  finishExperiment,
  recordGateResult,
} from "@compound/storage";
import { freshDatabase, getJson, testApp } from "./helpers";

let db: CompoundDatabase;

beforeEach(() => {
  db = freshDatabase();
});

afterEach(() => {
  db.close();
});

function decidedGate(handle: CompoundDatabase, outcome = "meets_gate") {
  const cand = createExperiment(handle, {
    taskKey: "support",
    candidateModel: "glm",
    provider: "doubleword",
    partition: "decision_test",
    paid: false,
  });
  finishExperiment(handle, cand.id, "completed", {});
  const ref = createExperiment(handle, {
    taskKey: "support",
    candidateModel: "opus",
    provider: "openrouter",
    partition: "decision_test",
    paid: false,
  });
  finishExperiment(handle, ref.id, "completed", {});
  const spec = createGateSpec(handle, {
    specHash: `sha256:${outcome}`,
    taskKey: "support",
    candidateModel: "glm",
    referenceModel: "opus",
    metric: "pass_rate",
    mode: "non_inferiority",
    margin: 0.05,
    confidence: 0.95,
    minCases: 20,
    judgeAbstainMax: 0,
    firewallReason: "final support gate",
  });
  return recordGateResult(handle, {
    gateSpecId: spec.id,
    candidateExperimentId: cand.id,
    referenceExperimentId: ref.id,
    outcome: outcome as "meets_gate",
    delta: -0.01,
    ciLo: -0.03,
    ciHi: 0.02,
    n: 30,
    candidateRate: 0.9,
    referenceRate: 0.91,
    judgeAbstainedFraction: 0,
  });
}

describe("GET /api/gates", () => {
  test("is empty before any gate is decided", async () => {
    const { status, body } = await getJson(testApp(db), "/api/gates");
    expect(status).toBe(200);
    expect(body.items).toEqual([]);
  });

  test("returns a decided gate with its verdict, delta, and CI — not the cases", async () => {
    decidedGate(db);
    const { status, body } = await getJson(testApp(db), "/api/gates");
    expect(status).toBe(200);
    expect(body.items).toHaveLength(1);
    const gate = body.items[0];
    expect(gate.outcome).toBe("meets_gate");
    expect(gate.task_key).toBe("support");
    expect(gate.candidate_model).toBe("glm");
    expect(gate.reference_model).toBe("opus");
    expect(gate.ci).toEqual([-0.03, 0.02]);
    expect(gate.n).toBe(30);
    expect(gate.firewall_reason).toBe("final support gate");
    // The sealed cases must never be exposed by the gate view.
    expect(gate).not.toHaveProperty("cases");
    expect(gate).not.toHaveProperty("pairs");
  });

  test("filters by task_key", async () => {
    decidedGate(db);
    const other = await getJson(testApp(db), "/api/gates?task_key=billing");
    expect(other.body.items).toEqual([]);
    const same = await getJson(testApp(db), "/api/gates?task_key=support");
    expect(same.body.items).toHaveLength(1);
  });
});

describe("GET /api/gates/:id", () => {
  test("returns a single gate by id", async () => {
    const created = decidedGate(db);
    const { status, body } = await getJson(testApp(db), `/api/gates/${created.id}`);
    expect(status).toBe(200);
    expect(body.id).toBe(created.id);
    expect(body.outcome).toBe("meets_gate");
  });

  test("404 for an unknown id", async () => {
    const { status } = await getJson(testApp(db), "/api/gates/nope");
    expect(status).toBe(404);
  });
});
