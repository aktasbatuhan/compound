import { describe, expect, test } from "bun:test";
import { type Trace, validate } from "@compound/contract";
import { computeContentHash } from "../src/content-hash";

function trace(overrides: Record<string, unknown> = {}): Trace {
  const raw = {
    schema: "compound.trace",
    schema_version: 1,
    trace_id: "t-1",
    task_key: "support",
    started_at: "2026-07-23T10:00:00Z",
    source: { importer: "test", importer_version: "1", source_ids: {} },
    steps: [
      {
        type: "model_call",
        step_id: "s1",
        model: "gpt-4o",
        input: [{ role: "user", content: "dispute my $23 charge" }],
        output: { role: "assistant", content: "done" },
      },
    ],
    focal_step_id: "s1",
    permissions: { judging: true, optimization: true, fine_tuning: false },
    redactions: [],
    ...overrides,
  };
  const result = validate(raw);
  if (result.class === "rejected") throw new Error("bad trace");
  return result.trace;
}

const toolStep = (name: string, output: unknown) => ({
  type: "tool_execution",
  step_id: `x-${name}`,
  name,
  input: { q: 1 },
  output,
});

describe("computeContentHash", () => {
  test("collapses two traces with the same focal request (dedup)", () => {
    expect(computeContentHash(trace({ trace_id: "a" }))).toBe(
      computeContentHash(trace({ trace_id: "b" })),
    );
  });

  test("folds the replay script into the hash — different tool outputs, different case (#6)", () => {
    const focalSteps = trace().steps;
    const withA = trace({ steps: [focalSteps[0], toolStep("get_charge", { amount: 23 })] });
    const withB = trace({ steps: [focalSteps[0], toolStep("get_charge", { amount: 99 })] });
    // Same focal request, different recorded tool outcome → distinct identities.
    expect(computeContentHash(withA)).not.toBe(computeContentHash(withB));
  });

  test("adding tool steps changes the hash; a plain trace is stable", () => {
    // The fold fires only when tool_execution steps exist, so a single-call case
    // keeps its identity, while adding a recorded tool step is a new case.
    const focalSteps = trace().steps;
    const plainHash = computeContentHash(trace());
    expect(computeContentHash(trace())).toBe(plainHash); // stable
    const withTool = trace({ steps: [focalSteps[0], toolStep("get_charge", { amount: 23 })] });
    expect(computeContentHash(withTool)).not.toBe(plainHash);
  });

  test("the single-call (v1) identity is byte-stable — a durable-identity lock (#7)", () => {
    // content_hash is a durable identity; every hash in existing databases is a
    // single-call one. The agentic identity carries an explicit version, but the
    // NON-agentic subject must never change — pin its value so any accidental
    // change to the hashed subject is caught here instead of silently
    // re-identifying every stored case.
    expect(computeContentHash(trace())).toBe(
      "72a0fbefdaf00694834398679e89eba7693d3385ce28d5b3a79c936fc72c0e54",
    );
  });

  test("the agentic (versioned) identity is distinct from the single-call one (#7)", () => {
    // Same first request, but the presence of a replay script makes this the
    // versioned agentic identity — it must never collide with the v1 single-call
    // hash, so a future algorithm change is a clean break, not a re-identification.
    const focalSteps = trace().steps;
    const agentic = trace({ steps: [focalSteps[0], toolStep("get_charge", { amount: 23 })] });
    expect(computeContentHash(agentic)).not.toBe(computeContentHash(trace()));
  });
});
