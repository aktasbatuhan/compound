import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import type { Assertion } from "@compound/assertions";
import { type Message, validate } from "@compound/contract";
import { curateTask } from "@compound/curation";
import {
  BudgetExceededError,
  type CompoundDatabase,
  createDatabase,
  createImportBatch,
  getCachedCompletion,
  insertCases,
  insertTraces,
  migrate,
  telemetryRollup,
  totalSpendUsd,
  traceRecordFromValidation,
} from "@compound/storage";
import {
  CACHE_BUST_MARKER,
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

  test("a paid call with no usage ledgers the estimate, never $0 (#3)", async () => {
    seedCases("support", 1);
    // A provider that completes but reports no usage (e.g. Doubleword Flex on
    // some long outputs). The money was spent; costFromUsage would read $0.
    class NullUsageProvider implements Provider {
      readonly name = "mock";
      calls: CompletionRequest[] = [];
      async complete(request: CompletionRequest): Promise<CompletionResponse> {
        this.calls.push(request);
        return {
          output: { role: "assistant", content: '{"ok":true}' },
          usage: null,
          finishReason: "stop",
          resolvedModel: request.model,
          latencyMs: 1,
        };
      }
    }
    const provider = new NullUsageProvider();
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
    expect(provider.calls).toHaveLength(1);
    // Ledger and reported cost are the estimate, not $0.
    expect(report.actual_cost_usd as number).toBeGreaterThan(0);
    expect(report.cost_unknown_calls).toBe(1);
    expect(totalSpendUsd(db)).toBeCloseTo(report.actual_cost_usd as number, 9);
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

  const paidRun = { paidRunsEnabled: true, experimentCapUsd: 5, globalHardLimitUsd: 25 } as const;
  const systemContent = (req: CompletionRequest | undefined): string => {
    const sys = req?.messages.find((m) => m.role === "system");
    return typeof sys?.content === "string" ? sys.content : "";
  };

  const freshRun = {
    taskKey: "support",
    candidateModel: "cheap",
    providerName: "mock",
    price: PRICE,
    assertions: ASSERTIONS,
    partition: "optimization_train" as const,
    ...paidRun,
  };

  test("a fresh run injects a distinct cache-bust nonce into each call (#25)", async () => {
    seedCases("support", 12);
    const provider = new MockProvider('{"ok":true}');
    await runExperiment(db, { ...freshRun, provider, fresh: true });
    expect(provider.calls.length).toBeGreaterThan(0);
    // Every call carries the nonce marker, and the nonces are all distinct.
    const markers = provider.calls.map(systemContent);
    for (const m of markers) expect(m).toContain(CACHE_BUST_MARKER);
    expect(new Set(markers).size).toBe(provider.calls.length);
  });

  test("a fresh run defeats the client cache: a repeat still makes real calls (#25)", async () => {
    seedCases("support", 12);
    const first = new MockProvider('{"ok":true}');
    await runExperiment(db, { ...freshRun, provider: first, fresh: true });
    const second = new MockProvider('{"ok":true}');
    const { report } = await runExperiment(db, { ...freshRun, provider: second, fresh: true });
    // No cache hits — each fresh nonce is a new fingerprint, so it really runs.
    expect(second.calls.length).toBe(first.calls.length);
    expect(first.calls.length).toBeGreaterThan(0);
    expect(report.cache_hits).toBe(0);
  });

  test("fresh completions never populate the correctness cache a gate reads (#25)", async () => {
    seedCases("support", 12);
    const fresh = new MockProvider('{"ok":true}');
    await runExperiment(db, { ...freshRun, provider: fresh, fresh: true });
    // A normal (non-fresh) run afterwards still makes real calls: the baseline
    // fingerprint differs from every nonce-bearing fresh one, so a gate's
    // correctness cache is untouched by the measurement run.
    const normal = new MockProvider('{"ok":true}');
    const { report } = await runExperiment(db, { ...freshRun, provider: normal });
    expect(normal.calls.length).toBe(fresh.calls.length);
    expect(report.cache_hits).toBe(0);
    // The baseline request carries no nonce.
    expect(systemContent(normal.calls[0])).not.toContain(CACHE_BUST_MARKER);
  });

  test("a fresh completion persists its salt; a normal one stays unsalted (#25)", async () => {
    seedCases("support", 6);
    const provider = new MockProvider('{"ok":true}');
    const { results } = await runExperiment(db, { ...freshRun, provider, fresh: true });
    const graded = results.filter((r) => r.status === "graded");
    expect(graded.length).toBeGreaterThan(0);
    // Every fresh completion row carries the exact salt its request was sent
    // with, so salted measurements are identifiable in storage/telemetry and
    // never silently mixed with unsalted quality data.
    for (const r of graded) {
      const row = getCachedCompletion(db, r.completionFingerprint as string);
      expect(row?.measurementNonce).toBeTruthy();
      const salted = provider.calls.filter((c) =>
        systemContent(c).includes(row?.measurementNonce as string),
      );
      expect(salted).toHaveLength(1);
    }
    // A normal run's completion rows carry no salt.
    const normal = new MockProvider('{"ok":true}');
    const { results: baseline } = await runExperiment(db, { ...freshRun, provider: normal });
    for (const r of baseline.filter((x) => x.status === "graded")) {
      const row = getCachedCompletion(db, r.completionFingerprint as string);
      expect(row?.measurementNonce).toBeNull();
    }
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

describe("runExperiment agentic multi-turn (#23)", () => {
  /** A provider that walks a scripted sequence of assistant messages. */
  class ScriptedProvider implements Provider {
    readonly name = "mock";
    calls: CompletionRequest[] = [];
    private i = 0;
    constructor(
      private readonly outputs: Message[],
      private readonly usage = { input_tokens: 10, output_tokens: 5 },
    ) {}
    async complete(request: CompletionRequest): Promise<CompletionResponse> {
      this.calls.push(request);
      const output = this.outputs[Math.min(this.i, this.outputs.length - 1)] as Message;
      this.i += 1;
      return {
        output,
        usage: this.usage,
        finishReason: "stop",
        resolvedModel: request.model,
        latencyMs: 1,
      };
    }
  }

  /** A case whose model loops forever (always calls a tool, never answers). */
  function seedLoopingCase(): void {
    insertCases(db, [
      {
        caseId: "loop-1",
        taskKey: "agent",
        sourceTraceId: "t-loop-1",
        contentHash: "loop-hash-1",
        provenance: "human_golden",
        partition: "optimization_train",
        input: {
          input: [{ role: "user", content: "loop" }],
          tools_available: [{ name: "spin", description: "", parameters: {} }],
          recorded_tool_results: [{ tool: "spin", result: "{}" }],
        },
        expected: {},
      },
    ]);
  }

  const spin: Message = {
    role: "assistant",
    content: null,
    tool_calls: [{ id: "c1", name: "spin", arguments: {} }],
  };

  function seedAgenticCase(): void {
    insertCases(db, [
      {
        caseId: "agentic-1",
        taskKey: "agent",
        sourceTraceId: "t-agent-1",
        contentHash: "agent-hash-1",
        provenance: "human_golden",
        partition: "optimization_train",
        input: {
          input: [{ role: "user", content: "dispute my $23 charge" }],
          tools_available: [{ name: "dispute_charge", description: "", parameters: {} }],
          recorded_tool_results: [{ tool: "dispute_charge", result: '{"ok":true}' }],
        },
        expected: {},
      },
    ]);
  }

  const dispute: Message = {
    role: "assistant",
    content: null,
    tool_calls: [{ id: "c1", name: "dispute_charge", arguments: { amount: 23 } }],
  };

  test("grades a whole trajectory with the one TS grader (tool_call_arg over turns)", async () => {
    seedAgenticCase();
    const provider = new ScriptedProvider([dispute, { role: "assistant", content: "done" }]);
    const { report } = await runExperiment(db, {
      taskKey: "agent",
      candidateModel: "cand",
      provider,
      providerName: "mock",
      price: PRICE,
      // A #21 argument assertion grades the tool call MADE DURING the trajectory.
      assertions: [
        { type: "tool_call_arg", name: "dispute_charge", arg: "amount", match: { equals: 23 } },
      ],
      partition: "optimization_train",
      agentic: true,
      replayPolicy: { default: "recorded" },
      paidRunsEnabled: true,
      experimentCapUsd: 5,
      globalHardLimitUsd: 25,
    });
    expect(report.cases_graded).toBe(1);
    expect(report.pass_rate).toBe(1);
    // Two turns → two provider calls counted (not one).
    expect(report.provider_calls).toBe(2);
    expect(provider.calls.length).toBe(2);
  });

  test("a blocked tool skips the case with a reason (no side effect)", async () => {
    seedAgenticCase();
    const provider = new ScriptedProvider([dispute, { role: "assistant", content: "done" }]);
    const { report } = await runExperiment(db, {
      taskKey: "agent",
      candidateModel: "cand",
      provider,
      providerName: "mock",
      price: PRICE,
      assertions: [{ type: "tool_called", name: "dispute_charge" }],
      partition: "optimization_train",
      agentic: true,
      replayPolicy: { default: "recorded", perTool: { dispute_charge: "blocked" } },
      paidRunsEnabled: true,
      experimentCapUsd: 5,
      globalHardLimitUsd: 25,
    });
    expect(report.cases_graded).toBe(0);
    expect(report.cases_skipped).toBe(1);
    expect(report.skip_reasons?.tool_blocked).toBe(1);
    // #1 money-safety: the model call that surfaced the blocked tool was REAL.
    // It must be counted and ledgered, never a phantom paid call reporting $0.
    expect(provider.calls).toHaveLength(1);
    expect(report.provider_calls).toBe(1);
    expect(report.actual_cost_usd as number).toBeGreaterThan(0);
    expect(totalSpendUsd(db)).toBeCloseTo(report.actual_cost_usd as number, 9);
  });

  test("re-running an agentic case is served from cache at $0", async () => {
    seedAgenticCase();
    const shared = {
      taskKey: "agent",
      candidateModel: "cand",
      providerName: "mock",
      price: PRICE,
      assertions: [{ type: "tool_called", name: "dispute_charge" }] as Assertion[],
      partition: "optimization_train" as const,
      agentic: true,
      replayPolicy: { default: "recorded" as const },
      paidRunsEnabled: true,
      experimentCapUsd: 5,
      globalHardLimitUsd: 25,
    };
    await runExperiment(db, {
      ...shared,
      provider: new ScriptedProvider([dispute, { role: "assistant", content: "done" }]),
    });
    const second = new ScriptedProvider([dispute, { role: "assistant", content: "done" }]);
    const { report } = await runExperiment(db, { ...shared, provider: second });
    expect(second.calls).toHaveLength(0); // whole trajectory cached
    expect(report.cache_hits).toBe(1);
    expect(report.actual_cost_usd).toBe(0);
  });

  test("re-running a NON-answered trajectory is served from cache — no re-charge (#1)", async () => {
    seedAgenticCase();
    const shared = {
      taskKey: "agent",
      candidateModel: "cand",
      providerName: "mock",
      price: PRICE,
      assertions: [{ type: "tool_called", name: "dispute_charge" }] as Assertion[],
      partition: "optimization_train" as const,
      agentic: true,
      // dispute_charge is blocked, so the first turn stops non-answered (skipped).
      replayPolicy: {
        default: "recorded" as const,
        perTool: { dispute_charge: "blocked" as const },
      },
      paidRunsEnabled: true,
      experimentCapUsd: 5,
      globalHardLimitUsd: 25,
    };
    const first = new ScriptedProvider([dispute, { role: "assistant", content: "done" }]);
    const firstRun = await runExperiment(db, { ...shared, provider: first });
    expect(firstRun.report.cases_skipped).toBe(1);
    expect(first.calls).toHaveLength(1); // one real, ledgered call
    const spendAfterFirst = totalSpendUsd(db);
    expect(spendAfterFirst).toBeGreaterThan(0);

    // The bug (#1): a re-run finds no aggregate cache, re-runs the turn, and the
    // per-turn ledger idempotency lets the fresh paid call through un-charged.
    // With the skip cached, the re-run makes ZERO provider calls and the ledger
    // does not move.
    const second = new ScriptedProvider([dispute, { role: "assistant", content: "done" }]);
    const secondRun = await runExperiment(db, { ...shared, provider: second });
    expect(second.calls).toHaveLength(0);
    expect(secondRun.report.cache_hits).toBe(1);
    expect(secondRun.report.cases_skipped).toBe(1);
    expect(secondRun.report.skip_reasons?.tool_blocked).toBe(1);
    expect(secondRun.report.actual_cost_usd).toBe(0);
    expect(totalSpendUsd(db)).toBeCloseTo(spendAfterFirst, 9);
  });

  test("a re-run RESUMES a budget-interrupted trajectory from per-turn cache — no re-charge (#1)", async () => {
    seedLoopingCase();
    const shared = {
      taskKey: "agent",
      candidateModel: "cand",
      providerName: "mock",
      price: PRICE,
      assertions: [{ type: "tool_called", name: "spin" }] as Assertion[],
      partition: "optimization_train" as const,
      agentic: true,
      // Endless loop under `mocked`, so the per-turn budget (not a consumed
      // recorded result) is what stops it. maxTurns is part of the trajectory
      // identity, so it MUST match across attempts for the resume to line up.
      replayPolicy: { default: "mocked" as const },
      maxTurns: 5,
      globalHardLimitUsd: 1000,
    };

    // Attempt 1: each turn bills ~$1 (measured). A $1.5 cap lets exactly two turns
    // complete; the third's headroom check throws. The trajectory NEVER reaches a
    // terminal stop, so NO aggregate is cached — only the two completed turns are
    // cached individually under their per-turn fingerprints.
    const first = new ScriptedProvider([spin], { input_tokens: 1_000_000, output_tokens: 0 });
    await expect(
      runExperiment(db, {
        ...shared,
        provider: first,
        paidRunsEnabled: true,
        experimentCapUsd: 1.5,
      }),
    ).rejects.toBeInstanceOf(BudgetExceededError);
    expect(first.calls).toHaveLength(2); // two real, ledgered calls
    const spendAfterFirst = totalSpendUsd(db);
    expect(spendAfterFirst).toBeCloseTo(2, 9);

    // Attempt 2 (the retry, cap raised): the bug (#1) was that a re-run re-executed
    // the two already-completed turns as fresh provider calls whose per-turn
    // fingerprint is already charged, so the ledger idempotency let the new BILLED
    // calls through un-ledgered. With per-turn resume, those two turns replay at $0
    // (no provider call) and only the remaining turns hit the provider.
    const second = new ScriptedProvider([spin], { input_tokens: 1_000_000, output_tokens: 0 });
    const { report } = await runExperiment(db, {
      ...shared,
      provider: second,
      paidRunsEnabled: true,
      experimentCapUsd: 10,
    });
    // Turns 1-2 resumed from cache → the provider only sees turns 3,4,5.
    expect(second.calls).toHaveLength(3);
    expect(report.provider_calls).toBe(3); // only real calls, not the 2 replayed
    expect(report.cases_skipped).toBe(1); // truncated at maxTurns
    expect(report.skip_reasons?.turn_budget_exhausted).toBe(1);
    // The ledger moved by exactly the 3 new calls, and the grand total equals every
    // real provider call ever made (2 + 3) — no billed call went un-ledgered.
    expect(totalSpendUsd(db)).toBeCloseTo(spendAfterFirst + 3, 9);
    expect(totalSpendUsd(db)).toBeCloseTo(first.calls.length + second.calls.length, 9);
  });

  test("a truncated trajectory is skipped, never graded as a pass (#4)", async () => {
    seedLoopingCase();
    // The model never answers; with maxTurns=3 the run hits the budget. A
    // `tool_called` assertion WOULD pass on the aggregate — but a truncated run
    // has no gradeable outcome and must be skipped, not counted as a pass.
    const provider = new ScriptedProvider([spin]);
    const { report } = await runExperiment(db, {
      taskKey: "agent",
      candidateModel: "cand",
      provider,
      providerName: "mock",
      price: PRICE,
      assertions: [{ type: "tool_called", name: "spin" }],
      partition: "optimization_train",
      agentic: true,
      // `mocked`: an endless-loop fixture needs a policy with nothing to exhaust,
      // so it truncates on the turn budget rather than running out of recorded
      // results (which a `recorded` policy now consumes once each, #8).
      replayPolicy: { default: "mocked" },
      maxTurns: 3,
      paidRunsEnabled: true,
      experimentCapUsd: 5,
      globalHardLimitUsd: 25,
    });
    expect(report.cases_graded).toBe(0);
    expect(report.passed).toBe(0);
    expect(report.cases_skipped).toBe(1);
    expect(report.skip_reasons?.turn_budget_exhausted).toBe(1);
    expect(report.provider_calls).toBe(3); // all three real calls counted
    expect(totalSpendUsd(db)).toBeCloseTo(report.actual_cost_usd as number, 9);
  });

  test("budget is enforced BEFORE every turn, bounding a runaway trajectory (#2)", async () => {
    seedLoopingCase();
    // Each turn costs ~$1 (1M input tokens); the cap allows only a few. The run
    // must stop mid-trajectory with a budget error — not silently make all 8
    // calls — and every call actually made must be ledgered (no loss, no overrun
    // of the pre-per-turn-check kind the old single up-front check allowed).
    const provider = new ScriptedProvider([spin], { input_tokens: 1_000_000, output_tokens: 0 });
    await expect(
      runExperiment(db, {
        taskKey: "agent",
        candidateModel: "cand",
        provider,
        providerName: "mock",
        price: PRICE,
        assertions: [{ type: "tool_called", name: "spin" }],
        partition: "optimization_train",
        agentic: true,
        // Endless loop → `mocked` so the budget (not a consumed recorded result,
        // #8) is what stops it.
        replayPolicy: { default: "mocked" },
        maxTurns: 8,
        paidRunsEnabled: true,
        experimentCapUsd: 3.5,
        globalHardLimitUsd: 1000,
      }),
    ).rejects.toBeInstanceOf(BudgetExceededError);
    // It stopped well before the 8-turn budget, and only the calls it made are
    // in the ledger — the per-turn check bounded the spend.
    expect(provider.calls.length).toBeGreaterThan(0);
    expect(provider.calls.length).toBeLessThan(8);
    expect(totalSpendUsd(db)).toBeCloseTo(provider.calls.length, 9);
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
