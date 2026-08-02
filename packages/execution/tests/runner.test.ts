import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import type { Assertion } from "@compound/assertions";
import { validate } from "@compound/contract";
import { curateTask } from "@compound/curation";
import {
  BudgetExceededError,
  type CompoundDatabase,
  createDatabase,
  createImportBatch,
  insertTraces,
  migrate,
  telemetryRollup,
  totalSpendUsd,
  traceRecordFromValidation,
} from "@compound/storage";
import {
  type CompletionRequest,
  type CompletionResponse,
  DecisionPartitionRefusedError,
  type Provider,
  runExperiment,
} from "../src/index";

/** A mock provider: records calls, returns a scripted output. No network. */
class MockProvider implements Provider {
  readonly name = "mock";
  calls: CompletionRequest[] = [];
  constructor(
    private readonly content: string,
    private readonly usage = { input_tokens: 10, output_tokens: 5 },
  ) {}
  async complete(request: CompletionRequest): Promise<CompletionResponse> {
    this.calls.push(request);
    return {
      output: { role: "assistant", content: this.content },
      usage: this.usage,
      finishReason: "stop",
      resolvedModel: request.model,
      latencyMs: 1,
    };
  }
}

const PRICE = { input: 1, output: 2 };
const ASSERTIONS: Assertion[] = [{ type: "valid_json" }];

let db: CompoundDatabase;

beforeEach(() => {
  db = createDatabase();
  migrate(db);
});

afterEach(() => {
  db.close();
});

/** Seed N eval-ready traces for a task and curate them into cases. */
function seedCases(taskKey: string, n: number): void {
  const batch = createImportBatch(db, {
    importer: "test",
    importerVersion: "1",
    sourceFingerprint: `${taskKey}-${n}`,
  });
  const records = [];
  for (let i = 0; i < n; i += 1) {
    const raw = {
      schema: "compound.trace",
      schema_version: 1,
      trace_id: `t-${taskKey}-${i}`,
      task_key: taskKey,
      started_at: "2026-07-24T10:00:00Z",
      source: { importer: "test", importer_version: "1", source_ids: {} },
      steps: [
        {
          type: "model_call",
          step_id: "s1",
          model: "gpt-4o",
          input: [{ role: "user", content: `question ${i}` }],
          output: { role: "assistant", content: "answer" },
          usage: { input_tokens: 5, output_tokens: 2 },
          started_at: "2026-07-24T10:00:00Z",
          ended_at: "2026-07-24T10:00:01Z",
        },
      ],
      focal_step_id: "s1",
      permissions: { judging: true, optimization: true, fine_tuning: false },
      redactions: [],
    };
    const record = traceRecordFromValidation(validate(raw), `hash-${taskKey}-${i}`);
    if (record !== null) records.push(record);
  }
  insertTraces(db, batch.id, records);
  curateTask(db, { taskKey });
}

describe("runExperiment money-safety", () => {
  test("a dry run makes ZERO provider calls and spends nothing", async () => {
    seedCases("support", 20);
    const provider = new MockProvider('{"ok":true}');
    const { report } = await runExperiment(db, {
      taskKey: "support",
      candidateModel: "cheap-model",
      provider,
      providerName: "mock",
      price: PRICE,
      assertions: ASSERTIONS,
      partition: "optimization_train",
      // paidRunsEnabled omitted → dry run.
    });

    expect(provider.calls).toHaveLength(0);
    expect(report.provider_calls).toBe(0);
    expect(report.actual_cost_usd).toBe(0);
    expect(totalSpendUsd(db)).toBe(0);
    expect(report.estimated_cost_usd ?? 0).toBeGreaterThan(0);
  });

  test("re-wraps contract tools into the OpenAI function envelope for the provider", async () => {
    // Seed one trace whose focal step exposes a tool in the CONTRACT shape
    // ({name, description, parameters}) — the shape ingest normalizes to.
    const batch = createImportBatch(db, {
      importer: "test",
      importerVersion: "1",
      sourceFingerprint: "tools",
    });
    const raw = {
      schema: "compound.trace",
      schema_version: 1,
      trace_id: "t-tools-1",
      task_key: "toolt",
      started_at: "2026-07-24T10:00:00Z",
      source: { importer: "test", importer_version: "1", source_ids: {} },
      steps: [
        {
          type: "model_call",
          step_id: "s1",
          model: "gpt-4o",
          input: [{ role: "user", content: "dispute it" }],
          tools_available: [
            {
              name: "dispute_charge",
              description: "d",
              parameters: { type: "object", properties: {} },
            },
          ],
          output: { role: "assistant", content: "answer" },
          usage: { input_tokens: 5, output_tokens: 2 },
          started_at: "2026-07-24T10:00:00Z",
          ended_at: "2026-07-24T10:00:01Z",
        },
      ],
      focal_step_id: "s1",
      permissions: { judging: true, optimization: true, fine_tuning: false },
      redactions: [],
    };
    const record = traceRecordFromValidation(validate(raw), "hash-tools-1");
    if (record !== null) insertTraces(db, batch.id, [record]);
    curateTask(db, { taskKey: "toolt" });

    const provider = new MockProvider('{"ok":true}');
    await runExperiment(db, {
      taskKey: "toolt",
      candidateModel: "cheap",
      provider,
      providerName: "mock",
      price: PRICE,
      assertions: ASSERTIONS,
      partition: "optimization_train",
      paidRunsEnabled: true,
      experimentCapUsd: 1,
      globalHardLimitUsd: 1,
    });

    // The provider must receive the function envelope, not the bare contract tool.
    const tools = provider.calls[0]?.tools as Array<{
      type?: string;
      function?: { name?: string };
    }>;
    expect(tools?.[0]?.type).toBe("function");
    expect(tools?.[0]?.function?.name).toBe("dispute_charge");
  });

  test("paid calls stay off unless enabled with a positive cap", async () => {
    seedCases("support", 5);
    const provider = new MockProvider('{"ok":true}');
    // Enabled but zero cap → still a dry run, no calls.
    await runExperiment(db, {
      taskKey: "support",
      candidateModel: "cheap",
      provider,
      providerName: "mock",
      price: PRICE,
      assertions: ASSERTIONS,
      partition: "optimization_train",
      paidRunsEnabled: true,
      experimentCapUsd: 0,
      globalHardLimitUsd: 0,
    });
    expect(provider.calls).toHaveLength(0);
  });

  test("an enabled paid run makes calls, grades, and records spend", async () => {
    seedCases("support", 10);
    const provider = new MockProvider('{"ok":true}');
    const { report } = await runExperiment(db, {
      taskKey: "support",
      candidateModel: "cheap",
      provider,
      providerName: "mock",
      price: PRICE,
      assertions: ASSERTIONS,
      partition: "optimization_train",
      paidRunsEnabled: true,
      experimentCapUsd: 5,
      globalHardLimitUsd: 25,
    });
    expect(provider.calls.length).toBeGreaterThan(0);
    expect(report.provider_calls).toBe(provider.calls.length);
    expect(report.pass_rate).toBe(1); // valid_json passes on '{"ok":true}'
    expect(report.actual_cost_usd).toBeGreaterThan(0);
    expect(totalSpendUsd(db)).toBeCloseTo(report.actual_cost_usd as number, 9);
  });

  test("a re-run is served entirely from cache at $0", async () => {
    seedCases("support", 8);
    const first = new MockProvider('{"ok":true}');
    await runExperiment(db, {
      taskKey: "support",
      candidateModel: "cheap",
      provider: first,
      providerName: "mock",
      price: PRICE,
      assertions: ASSERTIONS,
      partition: "optimization_train",
      paidRunsEnabled: true,
      experimentCapUsd: 5,
      globalHardLimitUsd: 25,
    });
    const spentAfterFirst = totalSpendUsd(db);

    const second = new MockProvider('{"ok":true}');
    const { report } = await runExperiment(db, {
      taskKey: "support",
      candidateModel: "cheap",
      provider: second,
      providerName: "mock",
      price: PRICE,
      assertions: ASSERTIONS,
      partition: "optimization_train",
      paidRunsEnabled: true,
      experimentCapUsd: 5,
      globalHardLimitUsd: 25,
    });

    expect(second.calls).toHaveLength(0); // everything cached
    expect(report.cache_hits).toBe(report.cases_graded);
    expect(report.actual_cost_usd).toBe(0);
    expect(totalSpendUsd(db)).toBe(spentAfterFirst); // no new spend
  });

  test("a new trial re-runs fresh (own paid calls), not from the trial-0 cache", async () => {
    seedCases("support", 8);
    const base = { input_tokens: 10, output_tokens: 5 };
    const shared = {
      taskKey: "support",
      candidateModel: "cheap",
      providerName: "mock",
      price: PRICE,
      assertions: ASSERTIONS,
      partition: "optimization_train" as const,
      paidRunsEnabled: true,
      experimentCapUsd: 5,
      globalHardLimitUsd: 25,
    };
    const t0 = new MockProvider('{"ok":true}', base);
    await runExperiment(db, { ...shared, provider: t0 }); // trial 0
    const cases = t0.calls.length; // however many landed in this partition
    expect(cases).toBeGreaterThan(0);
    const spentT0 = totalSpendUsd(db);

    // Trial 1: a distinct fingerprint per case → fresh calls, new spend, and a
    // second latency/TPS sample per case rather than a $0 replay of trial 0.
    const t1 = new MockProvider('{"ok":true}', base);
    const { report } = await runExperiment(db, { ...shared, provider: t1, trial: 1 });
    expect(t1.calls).toHaveLength(cases); // nothing cached across trials
    expect(report.cache_hits).toBe(0);
    expect(report.actual_cost_usd).toBeGreaterThan(0);
    expect(totalSpendUsd(db)).toBeGreaterThan(spentT0);
  });

  test("the hard cap stops a run before exceeding the global limit", async () => {
    seedCases("support", 200);
    const provider = new MockProvider('{"ok":true}', { input_tokens: 1000, output_tokens: 1000 });
    // A tiny global limit will be exhausted after a few calls.
    await expect(
      runExperiment(db, {
        taskKey: "support",
        candidateModel: "cheap",
        provider,
        providerName: "mock",
        price: { input: 100, output: 100 },
        assertions: ASSERTIONS,
        partition: "optimization_train",
        paidRunsEnabled: true,
        experimentCapUsd: 1000,
        globalHardLimitUsd: 0.01,
      }),
    ).rejects.toBeInstanceOf(BudgetExceededError);
    // Whatever was spent stayed under the limit at the moment of the throw.
    expect(totalSpendUsd(db)).toBeLessThanOrEqual(0.01);
  });

  test("refuses the sealed decision_test partition without the guard", async () => {
    seedCases("support", 5);
    const provider = new MockProvider("{}");
    await expect(
      runExperiment(db, {
        taskKey: "support",
        candidateModel: "cheap",
        provider,
        providerName: "mock",
        price: PRICE,
        assertions: ASSERTIONS,
        partition: "decision_test",
      }),
    ).rejects.toBeInstanceOf(DecisionPartitionRefusedError);
  });
});

describe("runExperiment grading", () => {
  test("caches the completion so the output can be inspected later", async () => {
    seedCases("support", 3);
    const provider = new MockProvider("not json");
    const { results } = await runExperiment(db, {
      taskKey: "support",
      candidateModel: "cheap",
      provider,
      providerName: "mock",
      price: PRICE,
      assertions: ASSERTIONS,
      partition: "optimization_train",
      paidRunsEnabled: true,
      experimentCapUsd: 5,
      globalHardLimitUsd: 25,
    });
    // valid_json fails on "not json".
    expect(results.every((r) => r.status !== "graded" || r.passed === false)).toBe(true);
  });

  test("with no assertions, runs but grades vacuously (nothing to check)", async () => {
    seedCases("support", 4);
    const provider = new MockProvider("anything");
    const { report } = await runExperiment(db, {
      taskKey: "support",
      candidateModel: "cheap",
      provider,
      providerName: "mock",
      price: PRICE,
      assertions: [],
      partition: "optimization_train",
      paidRunsEnabled: true,
      experimentCapUsd: 5,
      globalHardLimitUsd: 25,
    });
    expect(report.pass_rate).toBe(1);
  });
});

describe("runExperiment wireModel (per-provider wire ids, #19)", () => {
  const paid = { paidRunsEnabled: true, experimentCapUsd: 5, globalHardLimitUsd: 25 } as const;

  test("sends the wire id to the provider but records the LOGICAL id for grouping", async () => {
    seedCases("support", 3);
    const provider = new MockProvider('{"ok":true}');
    await runExperiment(db, {
      taskKey: "support",
      candidateModel: "gpt-4o-mini", // logical identity
      wireModel: "openai/gpt-4o-mini", // what the provider is actually sent
      provider,
      providerName: "openrouter",
      price: PRICE,
      assertions: ASSERTIONS,
      partition: "optimization_train",
      ...paid,
    });

    // The provider received the wire id.
    expect(provider.calls.length).toBeGreaterThan(0);
    expect(provider.calls.every((c) => c.model === "openai/gpt-4o-mini")).toBe(true);

    // Telemetry groups by the LOGICAL id, so the same model on another provider
    // would land in a sibling row under the same model name — not a new model.
    const rollup = telemetryRollup(db, "support");
    expect(rollup.map((g) => g.model)).toContain("gpt-4o-mini");
    expect(rollup.some((g) => g.model === "openai/gpt-4o-mini")).toBe(false);
  });

  test("the wire id feeds the fingerprint: a different wire id does not reuse the cache", async () => {
    seedCases("support", 4);
    // Same provider, same logical model, DIFFERENT wire id — isolates the wire
    // id's effect on the cache key from the provider's.
    const first = new MockProvider('{"ok":true}');
    await runExperiment(db, {
      taskKey: "support",
      candidateModel: "gpt-4o-mini",
      wireModel: "gpt-4o-mini",
      provider: first,
      providerName: "mock",
      price: PRICE,
      assertions: ASSERTIONS,
      partition: "optimization_train",
      ...paid,
    });
    expect(first.calls.length).toBeGreaterThan(0);

    const second = new MockProvider('{"ok":true}');
    await runExperiment(db, {
      taskKey: "support",
      candidateModel: "gpt-4o-mini",
      wireModel: "openai/gpt-4o-mini",
      provider: second,
      providerName: "mock",
      price: PRICE,
      assertions: ASSERTIONS,
      partition: "optimization_train",
      ...paid,
    });
    // A new wire id → new fingerprint → real calls, not cache hits.
    expect(second.calls.length).toBe(first.calls.length);
  });

  test("a transport override does not reuse a native run's cache (#8)", async () => {
    seedCases("support", 6);
    // Native run (no override) warms the cache.
    const native = new MockProvider('{"ok":true}');
    await runExperiment(db, {
      taskKey: "support",
      candidateModel: "glm",
      provider: native,
      providerName: "doubleword",
      price: PRICE,
      assertions: ASSERTIONS,
      partition: "optimization_train",
      ...paid,
    });
    expect(native.calls.length).toBeGreaterThan(0);

    // Same model/provider/messages but a forced non-native transport → distinct
    // fingerprint → real calls, not cache hits.
    const overridden = new MockProvider('{"ok":true}');
    await runExperiment(db, {
      taskKey: "support",
      candidateModel: "glm",
      provider: overridden,
      providerName: "doubleword",
      price: PRICE,
      assertions: ASSERTIONS,
      partition: "optimization_train",
      transportOverride: "chat_completions",
      ...paid,
    });
    expect(overridden.calls.length).toBe(native.calls.length);
  });

  test("wireModel defaults to the logical id (unchanged behavior)", async () => {
    seedCases("support", 3);
    const provider = new MockProvider('{"ok":true}');
    await runExperiment(db, {
      taskKey: "support",
      candidateModel: "cheap-model",
      // wireModel omitted
      provider,
      providerName: "mock",
      price: PRICE,
      assertions: ASSERTIONS,
      partition: "optimization_train",
      ...paid,
    });
    expect(provider.calls.every((c) => c.model === "cheap-model")).toBe(true);
  });
});

describe("runExperiment flex cap reservation (#8)", () => {
  test("a flex run reserves extra headroom that the same chat run does not", async () => {
    seedCases("support", 10);
    // One call's token estimate is ~$0.008 (4096 output tokens @ $2/M). A $0.01
    // cap admits it as chat, but the +$0.02 flex reserve pushes it over.
    const chat = new MockProvider('{"ok":true}');
    await runExperiment(db, {
      taskKey: "support",
      candidateModel: "m-chat",
      provider: chat,
      providerName: "mock",
      price: PRICE,
      assertions: ASSERTIONS,
      partition: "optimization_train",
      paidRunsEnabled: true,
      experimentCapUsd: 0.01,
      globalHardLimitUsd: 25,
      maxCases: 1,
    });
    expect(chat.calls).toHaveLength(1);

    // Same estimate, but transport: flex adds the reserve → over the same cap.
    const flex = new MockProvider('{"ok":true}');
    await expect(
      runExperiment(db, {
        taskKey: "support",
        candidateModel: "m-flex", // distinct id so it can't reuse the chat cache
        provider: flex,
        providerName: "mock",
        price: PRICE,
        assertions: ASSERTIONS,
        partition: "optimization_train",
        transport: "flex",
        paidRunsEnabled: true,
        experimentCapUsd: 0.01,
        globalHardLimitUsd: 25,
        maxCases: 1,
      }),
    ).rejects.toBeInstanceOf(BudgetExceededError);
    expect(flex.calls).toHaveLength(0);
  });
});

describe("runExperiment systemPromptOverride (adoption re-gates)", () => {
  /** Seed traces whose input already has a system message, then curate. */
  function seedCasesWithSystem(taskKey: string, n: number): void {
    const batch = createImportBatch(db, {
      importer: "test",
      importerVersion: "1",
      sourceFingerprint: `${taskKey}-sys-${n}`,
    });
    const records = [];
    for (let i = 0; i < n; i += 1) {
      const raw = {
        schema: "compound.trace",
        schema_version: 1,
        trace_id: `t-sys-${taskKey}-${i}`,
        task_key: taskKey,
        started_at: "2026-07-24T10:00:00Z",
        source: { importer: "test", importer_version: "1", source_ids: {} },
        steps: [
          {
            type: "model_call",
            step_id: "s1",
            model: "gpt-4o",
            input: [
              { role: "system", content: "original baseline prompt" },
              { role: "user", content: `question ${i}` },
            ],
            output: { role: "assistant", content: "answer" },
            usage: { input_tokens: 5, output_tokens: 2 },
            started_at: "2026-07-24T10:00:00Z",
            ended_at: "2026-07-24T10:00:01Z",
          },
        ],
        focal_step_id: "s1",
        permissions: { judging: true, optimization: true, fine_tuning: false },
        redactions: [],
      };
      const record = traceRecordFromValidation(validate(raw), `hash-sys-${taskKey}-${i}`);
      if (record !== null) records.push(record);
    }
    insertTraces(db, batch.id, records);
    curateTask(db, { taskKey });
  }

  const OVERRIDE = "You are the optimized prompt.";

  test("replaces the case's system message with the override, keeping the rest", async () => {
    seedCasesWithSystem("support", 12);
    const provider = new MockProvider('{"ok":true}');
    await runExperiment(db, {
      taskKey: "support",
      candidateModel: "cheap-model",
      provider,
      providerName: "mock",
      price: PRICE,
      assertions: ASSERTIONS,
      partition: "optimization_train",
      systemPromptOverride: OVERRIDE,
      paidRunsEnabled: true,
      experimentCapUsd: 5,
      globalHardLimitUsd: 5,
    });

    expect(provider.calls.length).toBeGreaterThan(0);
    for (const call of provider.calls) {
      const systems = call.messages.filter((m) => m.role === "system");
      expect(systems).toHaveLength(1);
      expect(systems[0]?.content).toBe(OVERRIDE);
      // The user turn survives untouched.
      expect(call.messages.some((m) => m.role === "user")).toBe(true);
    }
  });

  test("an overridden run never reuses the baseline prompt's cache (and vice versa)", async () => {
    seedCasesWithSystem("support", 12);
    const run = (provider: MockProvider, override?: string) =>
      runExperiment(db, {
        taskKey: "support",
        candidateModel: "cheap-model",
        provider,
        providerName: "mock",
        price: PRICE,
        assertions: ASSERTIONS,
        partition: "optimization_train",
        ...(override !== undefined ? { systemPromptOverride: override } : {}),
        paidRunsEnabled: true,
        experimentCapUsd: 5,
        globalHardLimitUsd: 5,
      });

    const baseline = new MockProvider('{"ok":true}');
    await run(baseline);
    expect(baseline.calls.length).toBeGreaterThan(0);

    // Different prompt → different fingerprint → real calls, not cache hits.
    const overridden = new MockProvider('{"ok":true}');
    await run(overridden, OVERRIDE);
    expect(overridden.calls.length).toBe(baseline.calls.length);

    // Same override again → served fully from cache, $0.
    const cached = new MockProvider('{"ok":true}');
    const { report } = await run(cached, OVERRIDE);
    expect(cached.calls).toHaveLength(0);
    expect(report.cache_hits).toBe(baseline.calls.length);
  });
});
