import { describe, expect, test } from "bun:test";
import { type Trace, TraceSchema, validate } from "../src/index";
import { cloneFixture, loadFixture } from "./fixtures";

function expectClass(result: ReturnType<typeof validate>, expected: string) {
  expect(result.class).toBe(expected as never);
}

describe("eval_ready", () => {
  test("fully-populated trace with tool loop is eval_ready", () => {
    const result = validate(loadFixture("eval-ready-full.json"));
    expectClass(result, "eval_ready");
    if (result.class !== "eval_ready") throw new Error("unreachable");
    expect(result.trace.focal_step_id).toBe("gen-2");
    expect(result.trace.steps).toHaveLength(4);
    const focal = result.trace.steps.find((s) => s.step_id === "gen-2");
    if (focal?.type !== "model_call") throw new Error("focal must be a model_call");
    expect(focal.usage?.input_tokens).toBe(583);
    expect(focal.usage?.cached_input_tokens).toBe(412);
    expect(focal.output?.content).toContain("INV-2291");
  });

  test("minimal trace with timestamps but no usage is eval_ready", () => {
    const result = validate(loadFixture("eval-ready-minimal.json"));
    expectClass(result, "eval_ready");
  });

  test("focal call with error but no output is eval_ready (explicit error being studied)", () => {
    const trace = cloneFixture(loadFixture("eval-ready-minimal.json")) as Record<string, unknown>;
    const steps = trace.steps as Array<Record<string, unknown>>;
    const step = steps[0];
    if (step === undefined) throw new Error("fixture must have a step");
    step.output = null;
    step.error = "provider timeout after 30s";
    const result = validate(trace);
    expectClass(result, "eval_ready");
  });
});

describe("diagnostic", () => {
  test("missing focal_step_id", () => {
    const result = validate(loadFixture("diagnostic-missing-focal.json"));
    expectClass(result, "diagnostic");
    if (result.class !== "diagnostic") throw new Error("unreachable");
    expect(result.diagnostic_reasons).toEqual(["missing_focal_step_id"]);
  });

  test("focal pointing at a tool_execution step", () => {
    const result = validate(loadFixture("diagnostic-focal-tool-execution.json"));
    expectClass(result, "diagnostic");
    if (result.class !== "diagnostic") throw new Error("unreachable");
    expect(result.diagnostic_reasons).toEqual(["focal_step_not_model_call"]);
  });

  test("unsupported content part forces diagnostic", () => {
    const result = validate(loadFixture("diagnostic-unsupported-content.json"));
    expectClass(result, "diagnostic");
    if (result.class !== "diagnostic") throw new Error("unreachable");
    expect(result.diagnostic_reasons).toEqual(["unsupported_content_part"]);
  });

  test("no model_call steps", () => {
    const result = validate(loadFixture("diagnostic-no-model-calls.json"));
    expectClass(result, "diagnostic");
    if (result.class !== "diagnostic") throw new Error("unreachable");
    expect(result.diagnostic_reasons).toContain("no_model_call_steps");
    expect(result.diagnostic_reasons).toContain("missing_focal_step_id");
  });

  test("focal_step_id pointing at a nonexistent step", () => {
    const trace = cloneFixture(loadFixture("eval-ready-minimal.json")) as Record<string, unknown>;
    trace.focal_step_id = "gen-does-not-exist";
    const result = validate(trace);
    expectClass(result, "diagnostic");
    if (result.class !== "diagnostic") throw new Error("unreachable");
    expect(result.diagnostic_reasons).toEqual(["focal_step_not_found"]);
  });

  test("focal call with neither output nor error", () => {
    const trace = cloneFixture(loadFixture("eval-ready-minimal.json")) as Record<string, unknown>;
    const steps = trace.steps as Array<Record<string, unknown>>;
    const step = steps[0];
    if (step === undefined) throw new Error("fixture must have a step");
    step.output = null;
    const result = validate(trace);
    expectClass(result, "diagnostic");
    if (result.class !== "diagnostic") throw new Error("unreachable");
    expect(result.diagnostic_reasons).toEqual(["focal_step_missing_output_and_error"]);
  });

  test("focal call with empty input", () => {
    const trace = cloneFixture(loadFixture("eval-ready-minimal.json")) as Record<string, unknown>;
    const steps = trace.steps as Array<Record<string, unknown>>;
    const step = steps[0];
    if (step === undefined) throw new Error("fixture must have a step");
    step.input = [];
    const result = validate(trace);
    expectClass(result, "diagnostic");
    if (result.class !== "diagnostic") throw new Error("unreachable");
    expect(result.diagnostic_reasons).toEqual(["focal_step_missing_input"]);
  });

  test("focal call with neither usage nor complete timestamps", () => {
    const trace = cloneFixture(loadFixture("eval-ready-minimal.json")) as Record<string, unknown>;
    const steps = trace.steps as Array<Record<string, unknown>>;
    const step = steps[0];
    if (step === undefined) throw new Error("fixture must have a step");
    step.ended_at = null;
    const result = validate(trace);
    expectClass(result, "diagnostic");
    if (result.class !== "diagnostic") throw new Error("unreachable");
    expect(result.diagnostic_reasons).toEqual(["focal_step_missing_usage_and_timestamps"]);
  });

  test("duplicate step ids", () => {
    const trace = cloneFixture(loadFixture("diagnostic-missing-focal.json")) as Record<
      string,
      unknown
    >;
    const steps = trace.steps as Array<Record<string, unknown>>;
    const step = steps[0];
    if (step === undefined) throw new Error("fixture must have a step");
    steps.push(cloneFixture(step));
    const result = validate(trace);
    expectClass(result, "diagnostic");
    if (result.class !== "diagnostic") throw new Error("unreachable");
    expect(result.diagnostic_reasons).toContain("duplicate_step_ids");
  });
});

describe("rejected", () => {
  test("bad envelope (wrong schema name and version)", () => {
    const result = validate(loadFixture("rejected-bad-envelope.json"));
    expectClass(result, "rejected");
    if (result.class !== "rejected") throw new Error("unreachable");
    const paths = result.issues.map((issue) => issue.path.join("."));
    expect(paths).toContain("schema");
    expect(paths).toContain("schema_version");
  });

  test("wrong field types", () => {
    const result = validate(loadFixture("rejected-wrong-types.json"));
    expectClass(result, "rejected");
    if (result.class !== "rejected") throw new Error("unreachable");
    expect(result.issues.length).toBeGreaterThan(0);
    const paths = result.issues.map((issue) => issue.path.join("."));
    expect(paths).toContain("steps");
    expect(paths).toContain("trace_id");
  });

  test("non-object input", () => {
    expectClass(validate("not a trace"), "rejected");
    expectClass(validate(null), "rejected");
    expectClass(validate([]), "rejected");
  });

  test("unknown extra keys are rejected (strict contract)", () => {
    const trace = cloneFixture(loadFixture("eval-ready-minimal.json")) as Record<string, unknown>;
    trace.surprise_field = true;
    expectClass(validate(trace), "rejected");
  });

  test("non-UTC timestamp offset is rejected", () => {
    const trace = cloneFixture(loadFixture("eval-ready-minimal.json")) as Record<string, unknown>;
    trace.started_at = "2026-07-18T09:00:00+02:00";
    expectClass(validate(trace), "rejected");
  });
});

describe("schema types", () => {
  test("parsed fixture satisfies the inferred Trace type", () => {
    const trace: Trace = TraceSchema.parse(loadFixture("eval-ready-full.json"));
    expect(trace.schema).toBe("compound.trace");
    expect(trace.schema_version).toBe(1);
    expect(trace.permissions.fine_tuning).toBe(false);
    expect(trace.redactions[0]?.rule).toBe("pii");
  });

  test("custom redaction rule accepts custom:<name> and rejects bare custom", () => {
    const trace = cloneFixture(loadFixture("eval-ready-full.json")) as Record<string, unknown>;
    const redactions = trace.redactions as Array<Record<string, unknown>>;
    const redaction = redactions[0];
    if (redaction === undefined) throw new Error("fixture must have a redaction");
    redaction.rule = "custom:internal_ticket_ids";
    expectClass(validate(trace), "eval_ready");
    redaction.rule = "custom:";
    expectClass(validate(trace), "rejected");
  });
});
