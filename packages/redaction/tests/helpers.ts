import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { Trace } from "@compound/contract";
import type { RedactionConfig } from "../src/index";

const CONTRACT_FIXTURES = join(import.meta.dir, "..", "..", "contract", "fixtures");

/** Load a fixture from `packages/contract/fixtures`, already deep-cloned. */
export function contractFixture(name: string): Trace {
  return JSON.parse(readFileSync(join(CONTRACT_FIXTURES, name), "utf8")) as Trace;
}

/** A minimal schema-valid trace; tests patch in the shapes they exercise. */
export function baseTrace(patch: (trace: Trace) => void = () => {}): Trace {
  const trace: Trace = {
    schema: "compound.trace",
    schema_version: 1,
    trace_id: "test:trace-1",
    task_key: "support.triage",
    started_at: "2026-07-20T14:03:11.000Z",
    source: {
      importer: "json",
      importer_version: "0.1.0",
      source_ids: { trace_id: "trace-1" },
    },
    steps: [
      {
        type: "model_call",
        step_id: "gen-1",
        input: [{ role: "user", content: "hello" }],
        output: { role: "assistant", content: "hi" },
      },
    ],
    focal_step_id: "gen-1",
    permissions: { judging: true, optimization: true, fine_tuning: false },
    redactions: [],
  };
  patch(trace);
  return trace;
}

export function secretRule(appliesTo: string[], marker?: string): RedactionConfig {
  return {
    rules: [
      {
        name: "api_keys",
        applies_to: appliesTo,
        detector: "secret",
        ...(marker ? { marker } : {}),
      },
    ],
  };
}

export function piiRule(appliesTo: string[]): RedactionConfig {
  return { rules: [{ name: "customer_pii", applies_to: appliesTo, detector: "pii" }] };
}

/** Every string that appears anywhere in a value, including object keys. */
export function serialize(value: unknown): string {
  return JSON.stringify(value);
}
