import { describe, expect, test } from "bun:test";
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

function withTempFile(contents: string, run: (path: string) => void): void {
  const path = join(tmpdir(), `compound-cli-${crypto.randomUUID()}.json`);
  writeFileSync(path, contents);
  try {
    run(path);
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
  test("no command prints help and exits non-zero", () => {
    const { env, output } = testEnvironment();
    expect(runCommand([], env).exitCode).toBe(2);
    expect(output()).toContain("Usage:");
  });

  test("explicit help exits zero", () => {
    const { env } = testEnvironment();
    expect(runCommand(["help"], env).exitCode).toBe(0);
  });

  test("an unknown command is an error, not a silent no-op", () => {
    const { env, output } = testEnvironment();
    expect(runCommand(["frobnicate"], env).exitCode).toBe(2);
    expect(output()).toContain("unknown command 'frobnicate'");
  });
});

describe("import", () => {
  test("imports a file and reports concrete counts", () => {
    const { env, output } = testEnvironment();
    withTempFile(EXPORT_JSON, (path) => {
      const result = runCommand(["import", path, "--config", "compound.yaml"], env);
      expect(result.exitCode).toBe(0);
    });
    expect(output()).toContain("eval_ready:  1");
    expect(output()).toContain("diagnostic:  0");
    expect(output()).toContain("rejected:    0");
  });

  test("requires a file argument", () => {
    const { env, output } = testEnvironment();
    expect(runCommand(["import"], env).exitCode).toBe(2);
    expect(output()).toContain("a file to import is required");
  });

  test("reports an unreadable file without a stack trace", () => {
    const { env, output } = testEnvironment();
    expect(runCommand(["import", "/nonexistent/nope.json"], env).exitCode).toBe(1);
    expect(output()).toContain("could not read");
  });

  test("warns loudly when config is missing, since redaction is then not applied", () => {
    const { env, output } = testEnvironment();
    withTempFile(EXPORT_JSON, (path) => {
      runCommand(["import", path, "--config", "/nonexistent/compound.yaml"], env);
    });
    expect(output()).toContain("importing WITHOUT redaction rules");
  });

  test("reports duplicates on a second import rather than failing", () => {
    const { env, output } = testEnvironment();
    withTempFile(EXPORT_JSON, (path) => {
      runCommand(["import", path, "--config", "compound.yaml"], env);
      const second = runCommand(["import", path, "--config", "compound.yaml"], env);
      expect(second.exitCode).toBe(0);
    });
    expect(output()).toContain("duplicate:   1");
  });

  test("lists diagnostic reasons when traces are not replayable", () => {
    const broken = JSON.parse(EXPORT_JSON);
    broken[0].observations[0].input = 42;
    const { env, output } = testEnvironment();
    withTempFile(JSON.stringify(broken), (path) => {
      runCommand(["import", path, "--config", "compound.yaml"], env);
    });
    expect(output()).toContain("diagnostic reasons:");
    expect(output()).toContain("unparseable_generation_input");
  });

  test("rejects an unsupported importer", () => {
    const { env, output } = testEnvironment();
    withTempFile(EXPORT_JSON, (path) => {
      expect(runCommand(["import", path, "--importer", "braintrust"], env).exitCode).toBe(1);
    });
    expect(output()).toContain("unsupported importer");
  });
});

describe("curate", () => {
  test("curates imported traces into cases and names the seal", () => {
    const { env, output } = testEnvironment();
    withTempFile(EXPORT_JSON, (path) => {
      runCommand(["import", path, "--config", "compound.yaml"], env);
    });
    const result = runCommand(["curate", "support"], env);
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("cases created:   1");
    expect(output()).toContain("decision_test");
  });

  test("requires a task key", () => {
    const { env, output } = testEnvironment();
    expect(runCommand(["curate"], env).exitCode).toBe(2);
    expect(output()).toContain("a task key to curate is required");
  });

  test("re-running curate reports duplicates, not new cases", () => {
    const { env, output } = testEnvironment();
    withTempFile(EXPORT_JSON, (path) => {
      runCommand(["import", path, "--config", "compound.yaml"], env);
    });
    runCommand(["curate", "support"], env);
    const again = runCommand(["curate", "support"], env);
    expect(again.exitCode).toBe(0);
    expect(output()).toContain("duplicates:      1");
  });
});

describe("status", () => {
  test("reports an empty store honestly", () => {
    const { env, output } = testEnvironment();
    expect(runCommand(["status"], env).exitCode).toBe(0);
    expect(output()).toContain("traces: 0");
  });

  test("summarises traces, task keys and the diagnostic queue after an import", () => {
    const { env, output } = testEnvironment();
    withTempFile(EXPORT_JSON, (path) => {
      runCommand(["import", path, "--config", "compound.yaml"], env);
    });
    runCommand(["status"], env);
    expect(output()).toContain("traces: 1");
    expect(output()).toContain("eval_ready: 1");
    expect(output()).toContain("support");
  });
});
