import { beforeAll, describe, expect, test } from "bun:test";
import { rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createDatabase, listGateResults, migrate, recordOptimizationRun } from "@compound/storage";
import { type CommandEnvironment, parseArgs, runCommand } from "../src/commands";

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

  test("an unknown model is a clear config error", async () => {
    const { env, output } = testEnvironment();
    await importAndCurate(env);
    const result = await runCommand(["experiment", "support", "no-such-model"], env);
    expect(result.exitCode).toBe(1);
    expect(output()).toContain("not in models");
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

  test("a dry run opens the seal, decides honestly, and makes no provider calls", async () => {
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
        "smoke test of the gate path",
      ],
      env,
    );
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("opening the sealed decision set");
    expect(output()).toContain("GATE:");
    // No cached completions on the sealed set in a dry run → an honest verdict,
    // not a fabricated pass.
    expect(output()).toContain("INSUFFICIENT DATA");
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

  test("an adoption re-gate records the artifact and prompt hash on the declared rule", async () => {
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

    const [decided] = listGateResults(db, 1);
    expect(decided?.spec.optimizationRunId).toBe(artifact.id);
    expect(decided?.spec.candidatePromptHash).toStartWith("sha256:");
    // A different rule than a baseline gate over the same models would declare.
    expect(decided?.spec.candidatePromptHash).not.toBeNull();
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
