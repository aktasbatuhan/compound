/**
 * Shared config cases. Used by both the zod tests and the JSON Schema
 * round-trip test, so the two artifacts are proven to agree on every case.
 */

/** Minimal benchmark-only config: what a workspace looks like before ingest. */
export function baseConfig(): Record<string, unknown> {
  return {
    version: 1,
    artifacts_dir: "artifacts",
    manifests_dir: "benchmarks/manifests",
    benchmarks: {
      ds1000: {
        task_key: "data_processing",
        sample_count: 30,
        partitions: { optimizer_train: 15, optimizer_validation: 5, decision_test: 10 },
      },
    },
  };
}

export function withSection(section: string, value: unknown): Record<string, unknown> {
  return { ...baseConfig(), [section]: value };
}

export const validCases: Record<string, Record<string, unknown>> = {
  "benchmark-only": baseConfig(),
  "task_keys minimal": withSection("task_keys", {
    support_agent: { replay: { default_tool_policy: "recorded" } },
  }),
  "task_keys full": withSection("task_keys", {
    support_agent: {
      description: "Customer support chat agent",
      replay: {
        default_tool_policy: "recorded",
        per_tool: { search_orders: "live_read_only", issue_refund: "blocked" },
      },
    },
  }),
  "redaction all detectors": withSection("redaction", {
    rules: [
      { name: "api_keys", applies_to: ["steps[*].input", "steps[*].output"], detector: "secret" },
      { name: "customer_pii", applies_to: ["metadata.**"], detector: "pii", marker: "[pii]" },
      {
        name: "order_ids",
        applies_to: ["steps[*].input"],
        detector: "regex",
        pattern: "ORD-[0-9]{6}",
      },
    ],
    field_allowlist: ["metadata.environment"],
  }),
  "redaction empty rules": withSection("redaction", { rules: [] }),
  "ingest minimal": withSection("ingest", {
    default_permissions: { judging: true, optimization: true, fine_tuning: false },
  }),
  "ingest with sources": withSection("ingest", {
    default_permissions: { judging: true, optimization: false, fine_tuning: false },
    sources: [
      { name: "langfuse-prod", importer: "langfuse", path: "exports/langfuse.jsonl" },
      { name: "manual", importer: "json" },
    ],
  }),
  "assertions with tool_call_arg": withSection("assertions", {
    "finance.dispute_charge": [
      { type: "tool_called", name: "dispute_charge" },
      { type: "tool_call_arg", name: "dispute_charge", arg: "amount", match: { equals: 23 } },
      {
        type: "tool_call_arg",
        name: "dispute_charge",
        match: { subset: { amount: 23, account: "acct_9" } },
        weight: 2,
      },
    ],
  }),
};

export const invalidCases: Record<string, unknown> = {
  "not an object": "version: 1",
  "wrong version": { ...baseConfig(), version: 2 },
  "missing benchmarks": (() => {
    const config = baseConfig();
    delete config.benchmarks;
    return config;
  })(),
  "benchmark missing task_key": withSection("benchmarks", {
    ds1000: { sample_count: 30, partitions: { decision_test: 30 } },
  }),
  "unknown replay policy": withSection("task_keys", {
    support_agent: { replay: { default_tool_policy: "replayed" } },
  }),
  "unknown per_tool policy": withSection("task_keys", {
    support_agent: {
      replay: { default_tool_policy: "recorded", per_tool: { search: "cached" } },
    },
  }),
  "task key without replay": withSection("task_keys", {
    support_agent: { description: "no policy declared" },
  }),
  "task key unknown field": withSection("task_keys", {
    support_agent: { replay: { default_tool_policy: "recorded" }, retention_days: 30 },
  }),
  "regex detector without pattern": withSection("redaction", {
    rules: [{ name: "order_ids", applies_to: ["steps[*].input"], detector: "regex" }],
  }),
  "pattern on non-regex detector": withSection("redaction", {
    rules: [
      { name: "api_keys", applies_to: ["steps[*].input"], detector: "secret", pattern: "sk-.*" },
    ],
  }),
  "unknown detector": withSection("redaction", {
    rules: [{ name: "api_keys", applies_to: ["steps[*].input"], detector: "entropy" }],
  }),
  "rule without applies_to": withSection("redaction", {
    rules: [{ name: "api_keys", detector: "secret" }],
  }),
  "redaction unknown field": withSection("redaction", { rules: [], drop_unmatched: true }),
  "ingest without default_permissions": withSection("ingest", { sources: [] }),
  "ingest partial permissions": withSection("ingest", {
    default_permissions: { judging: true, optimization: true },
  }),
  "ingest unknown importer": withSection("ingest", {
    default_permissions: { judging: true, optimization: true, fine_tuning: false },
    sources: [{ name: "otel", importer: "otel" }],
  }),
  "ingest source unknown field": withSection("ingest", {
    default_permissions: { judging: true, optimization: true, fine_tuning: false },
    sources: [{ name: "langfuse-prod", importer: "langfuse", project: "prod" }],
  }),
  "assertion unknown type": withSection("assertions", {
    "finance.dispute_charge": [{ type: "tool_call_argument", name: "dispute_charge" }],
  }),
};
