import { beforeAll, describe, expect, test } from "bun:test";
import { rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createDatabase, migrate } from "@compound/storage";
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
