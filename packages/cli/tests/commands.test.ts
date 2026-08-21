import { beforeAll, describe, expect, test } from "bun:test";
import { readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  cacheCompletion,
  createDatabase,
  createExperiment,
  createGateSpec,
  insertCases,
  listGateResults,
  migrate,
  recordCaseResults,
  recordDecisionCohort,
  recordGateResult,
  recordOptimizationRun,
  totalSpendUsd,
} from "@compound/storage";
import { type CommandEnvironment, parseArgs, runCommand } from "../src/commands";
import { configGateMetric, verdictExitCode } from "../src/gate";

const EXPORT_JSON = JSON.stringify([
  {
    id: "tr-cli-1",
    timestamp: "2026-07-23T10:00:00Z",
    tags: [],
    public: false,
    environment: "production",
    metadata: { task_key: "support" },
    observations: [
      {
        id: "gen-1",
        traceId: "tr-cli-1",
        type: "GENERATION",
        startTime: "2026-07-23T10:00:00Z",
        endTime: "2026-07-23T10:00:02Z",
        level: "DEFAULT",
        environment: "production",
        model: "gpt-4o",
        input: [{ role: "user", content: "hello" }],
        output: { role: "assistant", content: "hi" },
        usageDetails: { input: 5, output: 2 },
      },
    ],
    scores: [],
  },
]);

/** One in-memory database shared across a single test's commands. */
function testEnvironment() {
  const lines: string[] = [];
  const db = createDatabase();
  migrate(db);
  const env: CommandEnvironment = {
    write: (line) => lines.push(line),
    // Reused so `import` then `status` see the same store; close is a no-op
    // here because the test owns the lifetime.
    openDatabase: () => ({ ...db, close: () => {} }),
    cwd: process.cwd(),
  };
  return { env, lines, output: () => lines.join("\n"), db };
}

async function withTempFile(contents: string, run: (path: string) => Promise<void>): Promise<void> {
  const path = join(tmpdir(), `compound-cli-${crypto.randomUUID()}.json`);
  writeFileSync(path, contents);
  try {
    await run(path);
  } finally {
    rmSync(path, { force: true });
  }
}

describe("parseArgs", () => {
  test("separates command, positionals and flags", () => {
    const parsed = parseArgs(["import", "export.json", "--db", "x.db", "--verbose"]);
    expect(parsed.command).toBe("import");
    expect(parsed.positional).toEqual(["export.json"]);
    expect(parsed.flags).toEqual({ db: "x.db", verbose: true });
  });
});

describe("help", () => {
  test("no command prints help and exits non-zero", async () => {
    const { env, output } = testEnvironment();
    expect((await runCommand([], env)).exitCode).toBe(2);
    expect(output()).toContain("Usage:");
  });

  test("explicit help exits zero", async () => {
    const { env } = testEnvironment();
    expect((await runCommand(["help"], env)).exitCode).toBe(0);
  });

  test("an unknown command is an error, not a silent no-op", async () => {
    const { env, output } = testEnvironment();
    expect((await runCommand(["frobnicate"], env)).exitCode).toBe(2);
    expect(output()).toContain("unknown command 'frobnicate'");
  });
});

describe("init", () => {
  test("scaffolds a compound.yaml that is itself valid, and refuses to clobber it", async () => {
    const { env, output } = testEnvironment();
    const path = join(tmpdir(), `compound-init-${crypto.randomUUID()}.yaml`);
    try {
      const written = await runCommand(["init", "--config", path], env);
      expect(written.exitCode).toBe(0);
      expect(output()).toContain(`wrote ${path}`);

      // The scaffold must pass our own validator — a starter file that fails
      // validation would be a broken first impression.
      const { env: v, output: vOut } = testEnvironment();
      const validated = await runCommand(["validate", "--config", path], v);
      expect(validated.exitCode).toBe(0);
      expect(vOut()).toContain("valid:");

      // Refuses to overwrite without --force; --force overwrites.
      const clobber = await runCommand(["init", "--config", path], env);
      expect(clobber.exitCode).toBe(1);
      expect(output()).toContain("already exists");
      const forced = await runCommand(["init", "--config", path, "--force"], env);
      expect(forced.exitCode).toBe(0);
      expect(output()).toContain("overwritten");
    } finally {
      rmSync(path, { force: true });
    }
  });
});

describe("validate", () => {
  test("the repo's real compound.yaml is valid", async () => {
    const { env, output } = testEnvironment();
    const result = await runCommand(["validate", "--config", "compound.yaml"], env);
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("valid: compound.yaml");
  });

  test("a schema-invalid config exits 1 with a path-qualified issue", async () => {
    const { env, output } = testEnvironment();
    const path = join(tmpdir(), `compound-bad-${crypto.randomUUID()}.yaml`);
    writeFileSync(path, "version: 1\nartifacts_dir: a\nmanifests_dir: m\n");
    try {
      const result = await runCommand(["validate", "--config", path], env);
      expect(result.exitCode).toBe(1);
      expect(output()).toContain("invalid:");
      expect(output()).toContain("benchmarks");
    } finally {
      rmSync(path, { force: true });
    }
  });

  test("warns (but stays valid) when a model names an undeclared provider with no price", async () => {
    const { env, output } = testEnvironment();
    const path = join(tmpdir(), `compound-warn-${crypto.randomUUID()}.yaml`);
    writeFileSync(
      path,
      [
        "version: 1",
        "artifacts_dir: a",
        "manifests_dir: m",
        "benchmarks: {}",
        "providers:",
        "  openrouter:",
        "    base_url: https://openrouter.ai/api/v1",
        "    api_key_env: OPENROUTER_API_KEY",
        "models:",
        "  candidates:",
        "    - id: foo/bar",
        "      provider: nonesuch",
        "      role: candidate",
        "",
      ].join("\n"),
    );
    try {
      const result = await runCommand(["validate", "--config", path], env);
      expect(result.exitCode).toBe(0);
      expect(output()).toContain("not under providers:");
      expect(output()).toContain("has no price");
    } finally {
      rmSync(path, { force: true });
    }
  });

  test("points at `compound providers <name>` when the undeclared provider is a known one", async () => {
    const { env, output } = testEnvironment();
    const path = join(tmpdir(), `compound-known-${crypto.randomUUID()}.yaml`);
    writeFileSync(
      path,
      [
        "version: 1",
        "artifacts_dir: a",
        "manifests_dir: m",
        "benchmarks: {}",
        "providers:",
        "  openrouter:",
        "    base_url: https://openrouter.ai/api/v1",
        "    api_key_env: OPENROUTER_API_KEY",
        "models:",
        "  candidates:",
        "    - id: llama-3.3-70b",
        "      provider: groq",
        "      role: candidate",
        "pricing_usd_per_million_tokens:",
        "  llama-3.3-70b: { input: 0.59, output: 0.79 }",
        "",
      ].join("\n"),
    );
    try {
      const result = await runCommand(["validate", "--config", path], env);
      expect(result.exitCode).toBe(0);
      expect(output()).toContain("run: compound providers groq");
    } finally {
      rmSync(path, { force: true });
    }
  });
});

describe("providers", () => {
  test("lists the known providers with base_url, env, and tool support", async () => {
    const { env, output } = testEnvironment();
    const result = await runCommand(["providers"], env);
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("groq");
    expect(output()).toContain("https://api.groq.com/openai/v1");
    expect(output()).toContain("GROQ_API_KEY");
    // Anthropic/Google are called out as not-yet-supported, not silently absent.
    expect(output()).toContain("anthropic and google");
  });

  test("a name emits a paste-ready providers block", async () => {
    const { env, output } = testEnvironment();
    const result = await runCommand(["providers", "fireworks"], env);
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("providers:");
    expect(output()).toContain("  fireworks:");
    expect(output()).toContain("api_key_env: FIREWORKS_API_KEY");
  });

  test("an unknown provider name is a helpful error", async () => {
    const { env, output } = testEnvironment();
    const result = await runCommand(["providers", "nope"], env);
    expect(result.exitCode).toBe(1);
    expect(output()).toContain("no known provider 'nope'");
    expect(output()).toContain("self_hosted");
  });
});

describe("import", () => {
  test("imports a file and reports concrete counts", async () => {
    const { env, output } = testEnvironment();
    await withTempFile(EXPORT_JSON, async (path) => {
      const result = await runCommand(["import", path, "--config", "compound.yaml"], env);
      expect(result.exitCode).toBe(0);
    });
    expect(output()).toContain("eval_ready:  1");
    expect(output()).toContain("diagnostic:  0");
    expect(output()).toContain("rejected:    0");
  });

  test("requires a file argument", async () => {
    const { env, output } = testEnvironment();
    expect((await runCommand(["import"], env)).exitCode).toBe(2);
    expect(output()).toContain("a file to import is required");
  });

  test("reports an unreadable file without a stack trace", async () => {
    const { env, output } = testEnvironment();
    expect((await runCommand(["import", "/nonexistent/nope.json"], env)).exitCode).toBe(1);
    expect(output()).toContain("could not read");
  });

  test("warns loudly when config is missing, since redaction is then not applied", async () => {
    const { env, output } = testEnvironment();
    await withTempFile(EXPORT_JSON, async (path) => {
      await runCommand(["import", path, "--config", "/nonexistent/compound.yaml"], env);
    });
    expect(output()).toContain("importing WITHOUT redaction rules");
  });

  test("reports duplicates on a second import rather than failing", async () => {
    const { env, output } = testEnvironment();
    await withTempFile(EXPORT_JSON, async (path) => {
      await runCommand(["import", path, "--config", "compound.yaml"], env);
      const second = await runCommand(["import", path, "--config", "compound.yaml"], env);
      expect(second.exitCode).toBe(0);
    });
    expect(output()).toContain("duplicate:   1");
  });

  test("lists diagnostic reasons when traces are not replayable", async () => {
    const broken = JSON.parse(EXPORT_JSON);
    broken[0].observations[0].input = 42;
    const { env, output } = testEnvironment();
    await withTempFile(JSON.stringify(broken), async (path) => {
      await runCommand(["import", path, "--config", "compound.yaml"], env);
    });
    expect(output()).toContain("diagnostic reasons:");
    expect(output()).toContain("unparseable_generation_input");
  });

  test("rejects an unsupported importer", async () => {
    const { env, output } = testEnvironment();
    await withTempFile(EXPORT_JSON, async (path) => {
      expect((await runCommand(["import", path, "--importer", "braintrust"], env)).exitCode).toBe(
        1,
      );
    });
    expect(output()).toContain("unsupported importer");
  });
});

describe("curate", () => {
  test("curates imported traces into cases and names the seal", async () => {
    const { env, output } = testEnvironment();
    await withTempFile(EXPORT_JSON, async (path) => {
      await runCommand(["import", path, "--config", "compound.yaml"], env);
    });
    const result = await runCommand(["curate", "support"], env);
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("cases created:   1");
    expect(output()).toContain("decision_test");
  });

  test("requires a task key", async () => {
    const { env, output } = testEnvironment();
    expect((await runCommand(["curate"], env)).exitCode).toBe(2);
    expect(output()).toContain("a task key to curate is required");
  });

  test("re-running curate reports duplicates, not new cases", async () => {
    const { env, output } = testEnvironment();
    await withTempFile(EXPORT_JSON, async (path) => {
      await runCommand(["import", path, "--config", "compound.yaml"], env);
    });
    await runCommand(["curate", "support"], env);
    const again = await runCommand(["curate", "support"], env);
    expect(again.exitCode).toBe(0);
    expect(output()).toContain("duplicates:      1");
  });

  test("reports decision-set power and nudges when under the floor (issue #24)", async () => {
    const { env, output, db } = testEnvironment();
    // Seed 14 sealed cases directly — under the recommended 20.
    insertCases(
      db,
      Array.from({ length: 14 }, (_, i) => ({
        caseId: `pow-${i}`,
        taskKey: "powertask",
        sourceTraceId: `pt-${i}`,
        contentHash: `pow-hash-${i}`,
        provenance: "human_golden" as const,
        partition: "decision_test" as const,
        input: {},
        expected: {},
      })),
    );
    const result = await runCommand(["curate", "powertask"], env);
    expect(result.exitCode).toBe(0);
    // A precision read on the sealed set, plus a curate-more nudge (14 < 20).
    expect(output()).toContain("decision power:");
    expect(output()).toContain("below the recommended 20+");
    // Honest about what a small set means, with a concrete target: at the
    // default 5pp margin the suggested n is ~785 (95% gate, 80% power, SD 0.5).
    expect(output()).toContain("INSUFFICIENT DATA");
    expect(output()).toContain("5.0pp margin needs ~785 sealed cases");
  });
});

describe("experiment", () => {
  // Provider resolution reads the key from the provider's api_key_env even for
  // a dry run (the provider object is built but never called). Set dummy keys
  // so these tests are hermetic and never depend on a developer's .env.
  beforeAll(() => {
    process.env.OPENROUTER_API_KEY ??= "test-openrouter-key";
    process.env.DOUBLEWORD_API_KEY ??= "test-doubleword-key";
  });

  async function importAndCurate(env: CommandEnvironment): Promise<void> {
    await withTempFile(EXPORT_JSON, async (path) => {
      await runCommand(["import", path, "--config", "compound.yaml"], env);
    });
    await runCommand(["curate", "support"], env);
  }

  test("a dry run makes no provider calls and reports estimated cost", async () => {
    const { env, output } = testEnvironment();
    await importAndCurate(env);
    // GLM is a configured candidate with pricing in the repo compound.yaml.
    const result = await runCommand(
      ["experiment", "support", "zai-org/GLM-5.2-FP8", "--partition", "optimization_train"],
      env,
    );
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("dry run (no provider calls)");
    expect(output()).toContain("provider calls: 0");
  });

  test("requires a task key and a model", async () => {
    const { env } = testEnvironment();
    expect((await runCommand(["experiment", "support"], env)).exitCode).toBe(2);
  });

  test("--paid without an enabled budget is refused", async () => {
    const { env, output } = testEnvironment();
    await importAndCurate(env);
    // The repo compound.yaml has paid_runs_enabled: true, so the block that
    // trips first here is the missing --cap.
    const result = await runCommand(
      ["experiment", "support", "zai-org/GLM-5.2-FP8", "--paid"],
      env,
    );
    expect(result.exitCode).toBe(1);
    expect(output()).toContain("--cap");
  });

  test("--fresh is announced as measurement-only (issue #25)", async () => {
    const { env, output } = testEnvironment();
    await importAndCurate(env);
    const result = await runCommand(
      ["experiment", "support", "zai-org/GLM-5.2-FP8", "--fresh"],
      env,
    );
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("fresh:");
    expect(output()).toContain("measurement-only");
  });

  test("--agentic announces the multi-turn replay policy (issue #23)", async () => {
    const { env, output } = testEnvironment();
    await importAndCurate(env);
    const result = await runCommand(
      ["experiment", "support", "zai-org/GLM-5.2-FP8", "--agentic"],
      env,
    );
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("agentic:");
    expect(output()).toContain("replay: recorded");
  });

  test("an unknown model is a clear config error", async () => {
    const { env, output } = testEnvironment();
    await importAndCurate(env);
    const result = await runCommand(["experiment", "support", "no-such-model"], env);
    expect(result.exitCode).toBe(1);
    expect(output()).toContain("not in models");
  });

  test("--max-tokens must be a positive integer", async () => {
    const { env, output } = testEnvironment();
    await importAndCurate(env);
    const result = await runCommand(
      ["experiment", "support", "zai-org/GLM-5.2-FP8", "--max-tokens", "0"],
      env,
    );
    expect(result.exitCode).toBe(2);
    expect(output()).toContain("--max-tokens must be a positive integer");
  });

  test("--max-tokens is echoed and bounds the output budget", async () => {
    const { env, output } = testEnvironment();
    await importAndCurate(env);
    const result = await runCommand(
      [
        "experiment",
        "support",
        "zai-org/GLM-5.2-FP8",
        "--partition",
        "optimization_train",
        "--max-tokens",
        "1200",
      ],
      env,
    );
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("max_tokens:   1200");
  });
});

describe("gate", () => {
  beforeAll(() => {
    process.env.OPENROUTER_API_KEY ??= "test-openrouter-key";
    process.env.DOUBLEWORD_API_KEY ??= "test-doubleword-key";
  });

  async function importAndCurate(env: CommandEnvironment): Promise<void> {
    await withTempFile(EXPORT_JSON, async (path) => {
      await runCommand(["import", path, "--config", "compound.yaml"], env);
    });
    await runCommand(["curate", "support"], env);
  }

  test("requires candidate, reference, and reason", async () => {
    const { env } = testEnvironment();
    expect((await runCommand(["gate", "support"], env)).exitCode).toBe(2);
  });

  test("refuses without a firewall reason", async () => {
    const { env, output } = testEnvironment();
    const result = await runCommand(
      [
        "gate",
        "support",
        "--candidate",
        "zai-org/GLM-5.2-FP8",
        "--reference",
        "anthropic/claude-opus-4.8",
      ],
      env,
    );
    expect(result.exitCode).toBe(2);
    expect(output()).toContain("--reason is required");
  });

  test("refuses --max on a paid gate — a decision must cover the whole sealed set", async () => {
    const { env, output } = testEnvironment();
    await importAndCurate(env);
    const result = await runCommand(
      [
        "gate",
        "support",
        "--candidate",
        "zai-org/GLM-5.2-FP8",
        "--reference",
        "anthropic/claude-opus-4.8",
        "--reason",
        "trying to gate a truncated slice",
        "--paid",
        "--cap",
        "5",
        "--max",
        "3",
      ],
      env,
    );
    expect(result.exitCode).toBe(2);
    expect(output()).toContain("--max is not allowed on a paid gate");
  });

  test("a dry run previews without opening the seal or recording a verdict (issue #20)", async () => {
    const { env, output, db } = testEnvironment();
    await importAndCurate(env);
    const result = await runCommand(
      [
        "gate",
        "support",
        "--candidate",
        "zai-org/GLM-5.2-FP8",
        "--reference",
        "anthropic/claude-opus-4.8",
        "--reason",
        "smoke test of the gate path",
      ],
      env,
    );
    expect(result.exitCode).toBe(0);
    // A dry run is a side-effect-free PREVIEW: it does not claim to open the seal
    // and it labels the verdict as a preview, not a recorded decision.
    expect(output()).not.toContain("opening the sealed decision set");
    expect(output()).toContain("preview (dry run)");
    expect(output()).toContain("GATE (preview):");
    // No cached completions on the sealed set in a dry run → an honest verdict,
    // not a fabricated pass.
    expect(output()).toContain("INSUFFICIENT DATA");
    // Crucially, nothing is persisted — a preview must not pollute the audit trail.
    expect(listGateResults(db)).toHaveLength(0);
  });

  test("a dry run reports the sealed set's decision power before any spend (issue #24)", async () => {
    const { env, output, db } = testEnvironment();
    await importAndCurate(env);
    // Seed extra sealed cases so the power branch (n ≥ 2) is exercised.
    insertCases(
      db,
      Array.from({ length: 13 }, (_, i) => ({
        caseId: `gate-pow-${i}`,
        taskKey: "support",
        sourceTraceId: `gpt-${i}`,
        contentHash: `gate-pow-hash-${i}`,
        provenance: "human_golden" as const,
        partition: "decision_test" as const,
        input: {},
        expected: {},
      })),
    );
    const result = await runCommand(
      [
        "gate",
        "support",
        "--candidate",
        "zai-org/GLM-5.2-FP8",
        "--reference",
        "anthropic/claude-opus-4.8",
        "--reason",
        "previewing the gate's precision before paying",
      ],
      env,
    );
    expect(result.exitCode).toBe(0);
    // The preview sizes what a paid decision could detect on this sealed set,
    // at the declared margin — so underpowered sets are caught before spend.
    expect(output()).toContain("decision power:");
    expect(output()).toContain("min. detectable regression");
    expect(output()).toContain("5.0pp margin needs ~785 sealed cases");
  });

  test("warns when the sealed set has already been decided (issue #22)", async () => {
    const { env, output, db } = testEnvironment();
    await importAndCurate(env);
    // Ensure the task has a sealed set (a tiny corpus may curate none into it).
    insertCases(
      db,
      ["s1", "s2", "s3"].map((h, i) => ({
        caseId: `sealed-${h}`,
        taskKey: "support",
        sourceTraceId: `seal-trace-${i}`,
        contentHash: h,
        provenance: "human_golden" as const,
        partition: "decision_test" as const,
        input: {},
        expected: {},
      })),
    );
    // Seed a prior decision whose recorded cohort reuses these sealed cases.
    const mkExp = (model: string) =>
      createExperiment(db, {
        taskKey: "support",
        candidateModel: model,
        provider: "openrouter",
        partition: "decision_test",
        paid: false,
      });
    const prior = createGateSpec(db, {
      specHash: "sha256:prior-rule",
      taskKey: "support",
      candidateModel: "zai-org/GLM-5.2-FP8",
      referenceModel: "anthropic/claude-opus-4.8",
      metric: "pass_rate",
      mode: "non_inferiority",
      margin: 0.05,
      confidence: 0.95,
      minCases: 20,
      judgeAbstainMax: 0,
      firewallReason: "an earlier release gate",
    });
    const priorResult = recordGateResult(db, {
      gateSpecId: prior.id,
      candidateExperimentId: mkExp("zai-org/GLM-5.2-FP8").id,
      referenceExperimentId: mkExp("anthropic/claude-opus-4.8").id,
      outcome: "meets_gate",
      delta: 0,
      ciLo: 0,
      ciHi: 0,
      n: 25,
      candidateRate: 1,
      referenceRate: 1,
      judgeAbstainedFraction: 0,
      decisionPartitionVersion: "sha256:prior-cohort",
    });
    // The prior decided exactly these held-out labels, so the next gate over the
    // same set overlaps it and is flagged.
    recordDecisionCohort(db, priorResult.id, ["s1", "s2", "s3"]);

    await runCommand(
      [
        "gate",
        "support",
        "--candidate",
        "zai-org/GLM-5.2-FP8",
        "--reference",
        "anthropic/claude-opus-4.8",
        "--reason",
        "re-checking the same sealed set",
      ],
      env,
    );
    // Even a preview flags that the held-out set has been examined before.
    expect(output()).toContain("already been decided 1×");
  });

  test("a paid re-decision is refused BEFORE running when the guard is on (#2)", async () => {
    const { env, output, db } = testEnvironment();
    await importAndCurate(env);
    insertCases(
      db,
      ["s1", "s2", "s3"].map((h, i) => ({
        caseId: `sealed-${h}`,
        taskKey: "support",
        sourceTraceId: `seal-trace-${i}`,
        contentHash: h,
        provenance: "human_golden" as const,
        partition: "decision_test" as const,
        input: {},
        expected: {},
      })),
    );
    const mkExp = (model: string) =>
      createExperiment(db, {
        taskKey: "support",
        candidateModel: model,
        provider: "openrouter",
        partition: "decision_test",
        paid: false,
      });
    const prior = createGateSpec(db, {
      specHash: "sha256:prior-rule-2",
      taskKey: "support",
      candidateModel: "zai-org/GLM-5.2-FP8",
      referenceModel: "anthropic/claude-opus-4.8",
      metric: "pass_rate",
      mode: "non_inferiority",
      margin: 0.05,
      confidence: 0.95,
      minCases: 20,
      judgeAbstainMax: 0,
      firewallReason: "an earlier release gate",
    });
    const priorResult = recordGateResult(db, {
      gateSpecId: prior.id,
      candidateExperimentId: mkExp("zai-org/GLM-5.2-FP8").id,
      referenceExperimentId: mkExp("anthropic/claude-opus-4.8").id,
      outcome: "meets_gate",
      delta: 0,
      ciLo: 0,
      ciHi: 0,
      n: 25,
      candidateRate: 1,
      referenceRate: 1,
      judgeAbstainedFraction: 0,
      decisionPartitionVersion: "sha256:prior-cohort",
    });
    recordDecisionCohort(db, priorResult.id, ["s1", "s2", "s3"]);

    // A config that opts into the peeking block.
    const cfg = readFileSync("compound.yaml", "utf8").replace(
      "gate:\n",
      "gate:\n  block_repeat_decision: true\n",
    );
    const path = join(tmpdir(), `compound-gate-${crypto.randomUUID()}.yaml`);
    writeFileSync(path, cfg);
    try {
      const result = await runCommand(
        [
          "gate",
          "support",
          "--candidate",
          "zai-org/GLM-5.2-FP8",
          "--reference",
          "anthropic/claude-opus-4.8",
          "--reason",
          "trying to re-decide the same sealed set",
          "--paid",
          "--cap",
          "5",
          "--config",
          path,
        ],
        env,
      );
      // The guard fires in the preflight — exit 1, and the message says no
      // provider calls were made (the whole point of #2).
      expect(result.exitCode).toBe(1);
      expect(output()).toContain("Refused before running");
      // Assert the PROPERTY, not just the message: the paid path was never
      // entered. The seal is never announced ("opening the sealed decision set"
      // prints immediately before the experiments run), and nothing was spent —
      // so no provider call could have happened before the refusal.
      expect(output()).not.toContain("opening the sealed decision set");
      expect(totalSpendUsd(db)).toBe(0);
    } finally {
      rmSync(path, { force: true });
    }
  });

  test("--prompt-artifact must name a stored optimization run", async () => {
    const { env, output } = testEnvironment();
    await importAndCurate(env);
    const result = await runCommand(
      [
        "gate",
        "support",
        "--candidate",
        "zai-org/GLM-5.2-FP8",
        "--reference",
        "anthropic/claude-opus-4.8",
        "--reason",
        "adoption attempt",
        "--prompt-artifact",
        "nope-no-such-artifact",
      ],
      env,
    );
    expect(result.exitCode).toBe(1);
    expect(output()).toContain("not found");
  });

  test("--prompt-artifact refuses an artifact from a different task", async () => {
    const { env, output, db } = testEnvironment();
    await importAndCurate(env);
    const artifact = recordOptimizationRun(db, {
      taskKey: "some_other_task",
      candidateModel: "zai-org/GLM-5.2-FP8",
      seedPrompt: "seed",
      optimizedPrompt: "optimized",
      beforeValScore: 0.5,
      afterValScore: 0.9,
      valCases: 4,
      reflectionCalls: 1,
    });
    const result = await runCommand(
      [
        "gate",
        "support",
        "--candidate",
        "zai-org/GLM-5.2-FP8",
        "--reference",
        "anthropic/claude-opus-4.8",
        "--reason",
        "adoption attempt",
        "--prompt-artifact",
        artifact.id,
      ],
      env,
    );
    expect(result.exitCode).toBe(1);
    expect(output()).toContain("belongs to task");
  });

  // A dry-run preview announces the artifact under test but records nothing;
  // that the persisted rule captures the artifact + prompt hash is covered at the
  // gate-decision level (packages/gate/tests/decide.test.ts).
  test("an adoption re-gate previews the optimized prompt under test", async () => {
    const { env, output, db } = testEnvironment();
    await importAndCurate(env);
    const artifact = recordOptimizationRun(db, {
      taskKey: "support",
      candidateModel: "zai-org/GLM-5.2-FP8",
      seedPrompt: "seed prompt",
      optimizedPrompt: "You are the optimized support assistant.",
      beforeValScore: 0.5,
      afterValScore: 0.9,
      valCases: 4,
      reflectionCalls: 1,
    });
    const result = await runCommand(
      [
        "gate",
        "support",
        "--candidate",
        "zai-org/GLM-5.2-FP8",
        "--reference",
        "anthropic/claude-opus-4.8",
        "--reason",
        "adoption re-gate of the optimized prompt",
        "--prompt-artifact",
        artifact.id,
      ],
      env,
    );
    expect(result.exitCode).toBe(0);
    expect(output()).toContain(`optimized prompt under test: artifact ${artifact.id}`);
    // A preview persists nothing (issue #20).
    expect(listGateResults(db)).toHaveLength(0);
  });

  test("--candidate-provider runs the model on the override (preview)", async () => {
    const { env, output } = testEnvironment();
    await importAndCurate(env);
    const result = await runCommand(
      [
        "gate",
        "support",
        "--candidate",
        "zai-org/GLM-5.2-FP8",
        "--candidate-provider",
        "openrouter",
        "--reference",
        "anthropic/claude-opus-4.8",
        "--reason",
        "provider-axis: GLM on openrouter vs opus",
      ],
      env,
    );
    expect(result.exitCode).toBe(0);
    // The candidate resolved onto the override provider (chat), shown in output.
    // (That the override joins the persisted rule is covered in decide.test.ts.)
    expect(output()).toContain("zai-org/GLM-5.2-FP8 @openrouter");
  });
});

describe("eval (CI gate)", () => {
  beforeAll(() => {
    process.env.OPENROUTER_API_KEY ??= "test-openrouter-key";
    process.env.DOUBLEWORD_API_KEY ??= "test-doubleword-key";
  });

  async function importAndCurate(env: CommandEnvironment): Promise<void> {
    await withTempFile(EXPORT_JSON, async (path) => {
      await runCommand(["import", path, "--config", "compound.yaml"], env);
    });
    await runCommand(["curate", "support"], env);
  }

  test("usage error (missing candidate/reference) exits 2", async () => {
    const { env } = testEnvironment();
    expect((await runCommand(["eval", "support"], env)).exitCode).toBe(2);
  });

  test("needs no --reason: a standing CI reason drives the preview", async () => {
    const { env, output } = testEnvironment();
    await importAndCurate(env);
    const result = await runCommand(
      [
        "eval",
        "support",
        "--candidate",
        "zai-org/GLM-5.2-FP8",
        "--reference",
        "anthropic/claude-opus-4.8",
      ],
      env,
    );
    // No cached completions on the sealed set → the gate cannot decide, and the
    // CI exit code says so (2 = undecidable), never a fabricated pass.
    expect(result.exitCode).toBe(2);
    expect(output()).toContain("CI gate check");
    expect(output()).toContain("eval verdict: INSUFFICIENT DATA (exit 2)");
  });
});

describe("suggest-assertions", () => {
  /** Seed `count` accepted cases whose output calls `tool`, in a partition. */
  function seedToolCases(
    db: ReturnType<typeof createDatabase>,
    tool: string,
    count: number,
    partition: "optimizer_validation" | "decision_test",
  ): void {
    const records = Array.from({ length: count }, (_, i) => ({
      caseId: `case-${partition}-${tool}-${i}`,
      taskKey: "support",
      sourceTraceId: `tr-${partition}-${i}`,
      contentHash: `hash-${partition}-${tool}-${i}`,
      provenance: "observed_output" as const,
      partition,
      input: { input: [{ role: "user", content: "help" }] },
      expected: {
        role: "assistant",
        content: null,
        tool_calls: [{ id: `c${i}`, name: tool, arguments: {} }],
      },
    }));
    insertCases(db, records);
  }

  test("reports when a task has no cases", async () => {
    const { env, output } = testEnvironment();
    const result = await runCommand(["suggest-assertions", "support"], env);
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("no curated cases");
  });

  test("proposes a tool_called assertion with support and paste-ready YAML", async () => {
    const { env, output, db } = testEnvironment();
    seedToolCases(db, "dispute_charge", 5, "optimizer_validation");
    const result = await runCommand(
      ["suggest-assertions", "support", "--config", "compound.yaml"],
      env,
    );
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("tool_called 'dispute_charge'");
    expect(output()).toContain("5/5 accepted outputs call 'dispute_charge'");
    // The paste block is real YAML the user can drop under assertions.support.
    expect(output()).toContain("- type: tool_called");
    expect(output()).toContain('name: "dispute_charge"');
  });

  test("never mines the sealed decision set", async () => {
    const { env, output, db } = testEnvironment();
    // Non-sealed cases all call `refund`; the sealed set calls `secret_tool`.
    seedToolCases(db, "refund", 4, "optimizer_validation");
    seedToolCases(db, "secret_tool", 4, "decision_test");
    const result = await runCommand(
      ["suggest-assertions", "support", "--config", "compound.yaml"],
      env,
    );
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("refund");
    // A suggestion derived from the held-out set would leak it.
    expect(output()).not.toContain("secret_tool");
  });
});

describe("verdictExitCode / configGateMetric", () => {
  test("the verdict maps to a CI exit code: 0 meets, 1 regresses, 2 undecidable", () => {
    expect(verdictExitCode("meets_gate")).toBe(0);
    expect(verdictExitCode("fails_gate")).toBe(1);
    expect(verdictExitCode("no_reliable_improvement")).toBe(1);
    expect(verdictExitCode("insufficient_data")).toBe(2);
    expect(verdictExitCode("judge_abstained")).toBe(2);
    expect(verdictExitCode("anything_unrecognized")).toBe(2);
  });

  test("the config's free-form gate metric maps to a decidable one", () => {
    expect(configGateMetric("task_success")).toBe("pass_rate");
    expect(configGateMetric("pass_rate")).toBe("pass_rate");
    expect(configGateMetric("mean_score")).toBe("mean_score");
    expect(configGateMetric(undefined)).toBeUndefined();
  });
});

describe("grade-batch", () => {
  beforeAll(() => {
    process.env.OPENROUTER_API_KEY ??= "test-openrouter-key";
    process.env.DOUBLEWORD_API_KEY ??= "test-doubleword-key";
  });

  const batch = (taskKey: string) =>
    JSON.stringify({
      task_key: taskKey,
      items: [{ case_id: "c1", output: { role: "assistant", content: "hello there" } }],
    });

  test("assertion-only grading (no --judge) prints per-item scores", async () => {
    const { env, output } = testEnvironment();
    await withTempFile(batch("support"), async (path) => {
      const result = await runCommand(["grade-batch", path, "--config", "compound.yaml"], env);
      expect(result.exitCode).toBe(0);
    });
    const parsed = JSON.parse(output());
    expect(parsed.items).toHaveLength(1);
    expect(parsed.items[0].case_id).toBe("c1");
    expect(typeof parsed.items[0].score).toBe("number");
  });

  test("--judge on a task with no judge configured is a clear error", async () => {
    const { env, output } = testEnvironment();
    await withTempFile(batch("data_processing"), async (path) => {
      const result = await runCommand(
        ["grade-batch", path, "--judge", "--config", "compound.yaml"],
        env,
      );
      expect(result.exitCode).toBe(1);
    });
    expect(output()).toContain("no judge configured");
  });

  test("--judge REFUSES an uncalibrated judge rather than emit weak scores", async () => {
    const { env, output } = testEnvironment();
    await withTempFile(batch("support"), async (path) => {
      const result = await runCommand(
        ["grade-batch", path, "--judge", "--config", "compound.yaml"],
        env,
      );
      // Exit 3 = judge refusal (distinct from usage/other errors).
      expect(result.exitCode).toBe(3);
    });
    expect(output()).toContain("not calibrated");
    expect(output()).toContain("judge calibrate");
  });
});

describe("optimize", () => {
  beforeAll(() => {
    process.env.OPENROUTER_API_KEY ??= "test-openrouter-key";
    process.env.DOUBLEWORD_API_KEY ??= "test-doubleword-key";
  });

  test("refuses a judge-graded task whose judge is not calibrated, naming what unblocks it", async () => {
    const { env, output } = testEnvironment();
    // 'support' is judge-graded in compound.yaml; a fresh db has no calibration.
    const result = await runCommand(
      ["optimize", "support", "--candidate", "zai-org/GLM-5.2-FP8", "--config", "compound.yaml"],
      env,
    );
    expect(result.exitCode).toBe(1);
    expect(output()).toContain("not optimizing");
    expect(output()).toContain("not calibrated");
    expect(output()).toContain("judge calibrate");
  });
});

describe("telemetry", () => {
  test("reports honestly when nothing has run", async () => {
    const { env, output } = testEnvironment();
    expect((await runCommand(["telemetry"], env)).exitCode).toBe(0);
    expect(output()).toContain("no telemetry yet");
  });

  test("--json returns an empty items array, not prose", async () => {
    const { env, output } = testEnvironment();
    expect((await runCommand(["telemetry", "--json"], env)).exitCode).toBe(0);
    expect(JSON.parse(output())).toEqual({ items: [] });
  });
});

describe("judge", () => {
  beforeAll(() => {
    process.env.OPENROUTER_API_KEY ??= "test-openrouter-key";
    process.env.DOUBLEWORD_API_KEY ??= "test-doubleword-key";
  });

  test("usage error without a subcommand", async () => {
    const { env } = testEnvironment();
    expect((await runCommand(["judge"], env)).exitCode).toBe(2);
  });

  test("errors clearly when no judge is configured for the task", async () => {
    const { env, output } = testEnvironment();
    const result = await runCommand(["judge", "calibrate", "no_such_task"], env);
    expect(result.exitCode).toBe(1);
    expect(output()).toContain("no judge configured");
  });

  test("a dry-run calibrate with no labelled cases stays uncalibrated, no calls", async () => {
    const { env, output } = testEnvironment();
    const result = await runCommand(["judge", "calibrate", "support"], env);
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("UNCALIBRATED");
  });
});

describe("curate --split", () => {
  test("a decision-heavy split lands more cases in decision_test", async () => {
    const { env, output } = testEnvironment();
    await withTempFile(EXPORT_JSON, async (path) => {
      await runCommand(["import", path, "--config", "compound.yaml"], env);
    });
    // One trace only in the fixture, so just assert the split is accepted and the
    // command reports its partitions (the ratio math is unit-tested in curation).
    const result = await runCommand(["curate", "support", "--split", "5:10:35:50"], env);
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("partitions");
  });

  test("rejects a malformed split", async () => {
    const { env, output } = testEnvironment();
    const result = await runCommand(["curate", "support", "--split", "1:2:3"], env);
    expect(result.exitCode).toBe(2);
    expect(output()).toContain("--split must be four");
  });
});

describe("status", () => {
  test("reports an empty store honestly", async () => {
    const { env, output } = testEnvironment();
    expect((await runCommand(["status"], env)).exitCode).toBe(0);
    expect(output()).toContain("traces: 0");
  });

  test("summarises traces, task keys and the diagnostic queue after an import", async () => {
    const { env, output } = testEnvironment();
    await withTempFile(EXPORT_JSON, async (path) => {
      await runCommand(["import", path, "--config", "compound.yaml"], env);
    });
    await runCommand(["status"], env);
    expect(output()).toContain("traces: 1");
    expect(output()).toContain("eval_ready: 1");
    expect(output()).toContain("support");
  });
});

describe("view", () => {
  /**
   * Seed a full reviewable scenario: one case, a candidate that FAILS it and a
   * reference that PASSES it (each with a stored completion), plus a gate that
   * decided between them — enough to exercise every subview.
   */
  function seedScenario(db: ReturnType<typeof createDatabase>) {
    insertCases(db, [
      {
        caseId: "case:abc123def456",
        taskKey: "finance.dispute_charge",
        sourceTraceId: "trace:src-1",
        contentHash: "hash-1",
        provenance: "observed_output" as const,
        partition: "decision_test" as const,
        input: { input: [{ role: "user", content: "dispute a $23 charge" }] },
        expected: {
          role: "assistant",
          content: null,
          tool_calls: [{ id: "c", name: "dispute_charge", arguments: {} }],
        },
      },
    ]);
    cacheCompletion(db, {
      fingerprint: "fp-cand",
      provider: "doubleword",
      model: "cand-model",
      params: {},
      output: { role: "assistant", content: "Could you give me the transaction ID?" },
      costUsd: 0.001,
    });
    cacheCompletion(db, {
      fingerprint: "fp-ref",
      provider: "openrouter",
      model: "ref-model",
      params: {},
      output: {
        role: "assistant",
        content: null,
        tool_calls: [{ id: "c", name: "dispute_charge", arguments: {} }],
      },
      costUsd: 0.01,
    });
    const cand = createExperiment(db, {
      taskKey: "finance.dispute_charge",
      candidateModel: "cand-model",
      provider: "doubleword",
      partition: "decision_test",
      paid: true,
    });
    const ref = createExperiment(db, {
      taskKey: "finance.dispute_charge",
      candidateModel: "ref-model",
      provider: "openrouter",
      partition: "decision_test",
      paid: true,
    });
    recordCaseResults(db, cand.id, [
      {
        caseId: "case:abc123def456",
        status: "graded",
        passed: false,
        completionFingerprint: "fp-cand",
      },
    ]);
    recordCaseResults(db, ref.id, [
      {
        caseId: "case:abc123def456",
        status: "graded",
        passed: true,
        completionFingerprint: "fp-ref",
      },
    ]);
    const spec = createGateSpec(db, {
      specHash: "spec-1",
      taskKey: "finance.dispute_charge",
      candidateModel: "cand-model",
      referenceModel: "ref-model",
      metric: "pass_rate",
      mode: "non_inferiority",
      margin: 0.02,
      confidence: 0.95,
      minCases: 20,
      judgeAbstainMax: 0,
      candidateProvider: "doubleword",
      referenceProvider: "openrouter",
      firewallReason: "review test",
    });
    const result = recordGateResult(db, {
      gateSpecId: spec.id,
      candidateExperimentId: cand.id,
      referenceExperimentId: ref.id,
      outcome: "insufficient_data",
      delta: -0.08,
      ciLo: -0.2,
      ciHi: 0,
      n: 25,
      candidateRate: 0.92,
      referenceRate: 1,
      judgeAbstainedFraction: 0,
    });
    return { candId: cand.id, gateId: result.id };
  }

  test("overview lists tasks with partition counts, gates and telemetry", async () => {
    const { env, output, db } = testEnvironment();
    seedScenario(db);
    const result = await runCommand(["view"], env);
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("COMPOUND — overview");
    expect(output()).toContain("finance.dispute_charge");
    expect(output()).toContain("1 decision_test");
    expect(output()).toContain("INSUFFICIENT_DATA");
  });

  test("gate detail shows the verdict and the candidate/reference disagreement", async () => {
    const { env, output, db } = testEnvironment();
    const { gateId } = seedScenario(db);
    const result = await runCommand(["view", "gate", gateId.slice(0, 8)], env);
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("GATE INSUFFICIENT_DATA");
    expect(output()).toContain("cand-model @doubleword   92.0%");
    expect(output()).toContain("95% CI [-20.0pp, 0.0pp]");
    expect(output()).toContain("candidate FAIL / reference PASS");
  });

  test("case detail shows the input and each model's output side by side", async () => {
    const { env, output, db } = testEnvironment();
    seedScenario(db);
    const result = await runCommand(["view", "case", "case:abc123def456"], env);
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("dispute a $23 charge");
    expect(output()).toContain("cand-model  [FAIL]");
    expect(output()).toContain("Could you give me the transaction ID?");
    expect(output()).toContain("ref-model  [PASS]");
    expect(output()).toContain("→ tool_call dispute_charge");
  });

  test("unknown subject is an error; a missing id reports cleanly", async () => {
    const { env, output } = testEnvironment();
    expect((await runCommand(["view", "frobnicate"], env)).exitCode).toBe(2);
    expect(output()).toContain("unknown view 'frobnicate'");
    const { env: e2, output: o2 } = testEnvironment();
    expect((await runCommand(["view", "case", "nope"], e2)).exitCode).toBe(1);
    expect(o2()).toContain("no case 'nope'");
  });
});

describe("view compare", () => {
  // Reuses the same shape as the `view` scenario: a candidate that fails and a
  // reference that passes, each graded with a priced completion.
  function seedTwoModels(db: ReturnType<typeof createDatabase>) {
    insertCases(db, [
      {
        caseId: "case:cmp1",
        taskKey: "finance.dispute_charge",
        sourceTraceId: "trace:cmp",
        contentHash: "hash-cmp",
        provenance: "observed_output" as const,
        partition: "decision_test" as const,
        input: { input: [{ role: "user", content: "dispute" }] },
        expected: null,
      },
    ]);
    cacheCompletion(db, {
      fingerprint: "fp-c",
      provider: "doubleword",
      model: "cand",
      params: {},
      output: { role: "assistant", content: "hedge" },
      costUsd: 0.001,
    });
    cacheCompletion(db, {
      fingerprint: "fp-r",
      provider: "openrouter",
      model: "ref",
      params: {},
      output: { role: "assistant", content: "act" },
      costUsd: 0.01,
      latencyMs: 500,
      usage: { output_tokens: 100 }, // 100 tok / 0.5s = 200 tps
    });
    const cand = createExperiment(db, {
      taskKey: "finance.dispute_charge",
      candidateModel: "cand",
      provider: "doubleword",
      partition: "decision_test",
      paid: true,
    });
    const ref = createExperiment(db, {
      taskKey: "finance.dispute_charge",
      candidateModel: "ref",
      provider: "openrouter",
      partition: "decision_test",
      paid: true,
    });
    recordCaseResults(db, cand.id, [
      {
        caseId: "case:cmp1",
        status: "graded",
        passed: false,
        score: 0,
        completionFingerprint: "fp-c",
      },
    ]);
    recordCaseResults(db, ref.id, [
      {
        caseId: "case:cmp1",
        status: "graded",
        passed: true,
        score: 1,
        completionFingerprint: "fp-r",
      },
    ]);
  }

  test("aggregated ranks best-first with pass% and per-case cost", async () => {
    const { env, output, db } = testEnvironment();
    seedTwoModels(db);
    const result = await runCommand(["view", "compare"], env);
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("aggregated (per model, across all tasks)");
    // The passing reference (100%) sorts above the failing candidate (0%).
    const out = output();
    expect(out.indexOf("ref")).toBeLessThan(out.indexOf("cand"));
    expect(out).toContain("100.0%");
    expect(out).toContain("$0.01000"); // ref per-case cost
    expect(out).toContain("$0.00100"); // cand per-case cost
    // avg latency + TPS columns are present and computed for the priced ref.
    expect(out).toContain("avg ms");
    expect(out).toContain("tps");
    expect(out).toContain("500"); // ref avg latency
    expect(out).toContain("200.0"); // ref tps (100 tok / 0.5s)
  });

  test("per-task form shows the partition and only that task", async () => {
    const { env, output, db } = testEnvironment();
    seedTwoModels(db);
    const result = await runCommand(["view", "compare", "finance.dispute_charge"], env);
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("cost / score — finance.dispute_charge");
    expect(output()).toContain("decision_test");
  });

  test("reports cleanly when a task has no graded experiments", async () => {
    const { env, output } = testEnvironment();
    const result = await runCommand(["view", "compare", "nope"], env);
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("no graded experiments for task 'nope'");
  });

  // One model id served by two OpenRouter upstreams must appear as two rows,
  // labelled by host — the provider-selection table (#9), not one collapsed row.
  function seedTwoUpstreams(db: ReturnType<typeof createDatabase>) {
    insertCases(db, [
      {
        caseId: "case:up1",
        taskKey: "finance.dispute_charge",
        sourceTraceId: "trace:up",
        contentHash: "hash-up",
        provenance: "observed_output" as const,
        partition: "optimizer_validation" as const,
        input: { input: [{ role: "user", content: "dispute" }] },
        expected: null,
      },
    ]);
    for (const [fp, upstream, latency] of [
      ["fp-fw", "Fireworks", 800],
      ["fp-tg", "Together", 1600],
    ] as const) {
      cacheCompletion(db, {
        fingerprint: fp,
        provider: "openrouter",
        model: "kimi-k3",
        upstreamProvider: upstream,
        params: { provider: { only: [upstream.toLowerCase()] } },
        output: { role: "assistant", content: "act" },
        costUsd: 0.003,
        latencyMs: latency,
        usage: { output_tokens: 40 },
      });
      const exp = createExperiment(db, {
        taskKey: "finance.dispute_charge",
        candidateModel: "kimi-k3",
        provider: "openrouter",
        partition: "optimizer_validation",
        paid: true,
      });
      recordCaseResults(db, exp.id, [
        { caseId: "case:up1", status: "graded", passed: true, score: 1, completionFingerprint: fp },
      ]);
    }
  }

  test("one model across two OpenRouter upstreams shows two host-labelled rows", async () => {
    const { env, output, db } = testEnvironment();
    seedTwoUpstreams(db);
    const result = await runCommand(["view", "compare", "finance.dispute_charge"], env);
    expect(result.exitCode).toBe(0);
    const out = output();
    // Both hosts appear, distinctly labelled — not collapsed into one "openrouter".
    expect(out).toContain("openrouter/Fireworks");
    expect(out).toContain("openrouter/Together");
  });
});
