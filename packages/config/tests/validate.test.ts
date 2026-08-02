import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ReplayPolicySchema as ContractReplayPolicySchema } from "@compound/contract";
import {
  ConfigError,
  DEFAULT_PERMISSIONS,
  defaultRedactionMarker,
  loadConfig,
  ReplayPolicySchema,
  validateConfig,
} from "../src/index";
import { baseConfig, invalidCases, validCases, withSection } from "./cases";

const REPO_ROOT = join(import.meta.dir, "..", "..", "..");

function tempConfig(text: string): string {
  const path = join(mkdtempSync(join(tmpdir(), "compound-config-")), "compound.yaml");
  writeFileSync(path, text);
  return path;
}

describe("the repository's real compound.yaml", () => {
  const config = loadConfig(join(REPO_ROOT, "compound.yaml"));

  test("validates unchanged", () => {
    expect(config.version).toBe(1);
    expect(Object.keys(config.benchmarks).sort()).toEqual(["bfcl", "ds1000", "tau_bench"]);
  });

  test("keeps the benchmark sections the Python engine reads", () => {
    expect(config.artifacts_dir).toBe("artifacts");
    expect(config.manifests_dir).toBe("benchmarks/manifests");
    expect(config.budget?.hard_limit_usd).toBe(25);
    expect(Object.keys(config.providers ?? {})).toContain("doubleword");
    expect(config.models?.candidates?.length).toBe(5);
    // The provider registry names each endpoint's wire protocol (issue #4).
    expect(config.providers?.openai?.type).toBe("openai_compatible");
    expect(config.providers?.doubleword?.type).toBe("flex");
    expect(config.flex_pricing_usd_per_million_tokens?.["deepseek-ai/DeepSeek-V4-Flash"]).toEqual({
      input: 0.07,
      output: 0.14,
    });
    expect(config.gate?.metric).toBe("task_success");
    // Per-provider wire ids (issue #19): the same logical model carries a
    // different id per host, so one entry can gate OpenAI-direct vs OpenRouter.
    const gptMini = config.models?.candidates?.find((m) => m.id === "gpt-4o-mini");
    expect(gptMini?.provider_ids).toEqual({
      openai: "gpt-4o-mini",
      openrouter: "openai/gpt-4o-mini",
    });
  });

  test("carries no product sections yet, and does not need to", () => {
    expect(config.task_keys).toBeUndefined();
    expect(config.redaction).toBeUndefined();
    expect(config.ingest).toBeUndefined();
  });
});

describe("valid cases", () => {
  for (const [name, config] of Object.entries(validCases)) {
    test(name, () => {
      const result = validateConfig(config);
      expect(result.ok).toBe(true);
    });
  }
});

describe("invalid cases", () => {
  for (const [name, config] of Object.entries(invalidCases)) {
    test(name, () => {
      const result = validateConfig(config);
      expect(result.ok).toBe(false);
      if (result.ok) return;
      expect(result.issues.length).toBeGreaterThan(0);
      for (const issue of result.issues) expect(issue.path.length).toBeGreaterThan(0);
    });
  }
});

describe("tool_call_arg matcher validation (#8)", () => {
  // Runtime (zod) enforcement only — the JSON Schema stays loose on per-type
  // params by design, and only the TS engine grades assertions. Catching this at
  // config load stops a malformed matcher from throwing mid-experiment.
  const withAssertion = (assertion: Record<string, unknown>) =>
    withSection("assertions", { "finance.dispute_charge": [assertion] });

  test("accepts exactly one matcher", () => {
    expect(
      validateConfig(withAssertion({ type: "tool_call_arg", name: "x", match: { equals: 1 } })).ok,
    ).toBe(true);
  });

  test("rejects a missing matcher", () => {
    const result = validateConfig(
      withAssertion({ type: "tool_call_arg", name: "x", arg: "amount" }),
    );
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.issues.some((i) => i.path.endsWith("match"))).toBe(true);
  });

  test("rejects two matchers at once", () => {
    expect(
      validateConfig(
        withAssertion({ type: "tool_call_arg", name: "x", match: { equals: 1, regex: "1" } }),
      ).ok,
    ).toBe(false);
  });

  test("rejects a missing tool name", () => {
    expect(validateConfig(withAssertion({ type: "tool_call_arg", match: { equals: 1 } })).ok).toBe(
      false,
    );
  });
});

describe("issues are path-qualified", () => {
  test("names the offending task key and field", () => {
    const result = validateConfig(invalidCases["unknown replay policy"]);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.issues[0]?.path).toBe("task_keys.support_agent.replay.default_tool_policy");
  });

  test("names the offending rule by index", () => {
    const result = validateConfig(
      withSection("redaction", {
        rules: [
          { name: "ok", applies_to: ["metadata.**"], detector: "secret" },
          { name: "bad", applies_to: ["metadata.**"], detector: "regex", pattern: "([a-z" },
        ],
      }),
    );
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.issues[0]?.path).toBe("redaction.rules[1].pattern");
    expect(result.issues[0]?.message).toContain("not a valid regular expression");
  });
});

describe("replay vocabulary", () => {
  test("matches the trace contract's ReplayPolicy exactly", () => {
    expect(ReplayPolicySchema.options).toEqual(ContractReplayPolicySchema.options);
  });
});

describe("defaults", () => {
  test("the trace contract's documented permission default is exported", () => {
    expect(DEFAULT_PERMISSIONS).toEqual({
      judging: true,
      optimization: true,
      fine_tuning: false,
    });
  });

  test("markers default per detector and map regex rules into custom:<name>", () => {
    expect(defaultRedactionMarker({ name: "k", applies_to: ["a"], detector: "secret" })).toBe(
      "⟦redacted:secret⟧",
    );
    expect(defaultRedactionMarker({ name: "k", applies_to: ["a"], detector: "pii" })).toBe(
      "⟦redacted:pii⟧",
    );
    expect(
      defaultRedactionMarker({
        name: "order_ids",
        applies_to: ["a"],
        detector: "regex",
        pattern: "x",
      }),
    ).toBe("⟦redacted:custom:order_ids⟧");
    expect(
      defaultRedactionMarker({
        name: "k",
        applies_to: ["a"],
        detector: "pii",
        marker: "[hidden]",
      }),
    ).toBe("[hidden]");
  });
});

describe("loadConfig", () => {
  test("parses YAML and returns typed config", () => {
    const path = tempConfig(`version: 1
artifacts_dir: artifacts
manifests_dir: benchmarks/manifests
benchmarks:
  ds1000:
    task_key: data_processing
    sample_count: 4
    partitions:
      decision_test: 4
task_keys:
  support_agent:
    description: Customer support chat agent
    replay:
      default_tool_policy: recorded
      per_tool:
        issue_refund: blocked
redaction:
  rules:
    - name: api_keys
      applies_to: ["steps[*].input"]
      detector: secret
ingest:
  default_permissions:
    judging: true
    optimization: true
    fine_tuning: false
  sources:
    - name: langfuse-prod
      importer: langfuse
      path: exports/langfuse.jsonl
`);
    const config = loadConfig(path);
    expect(config.task_keys?.support_agent?.replay.per_tool?.issue_refund).toBe("blocked");
    expect(config.ingest?.sources?.[0]?.importer).toBe("langfuse");
    expect(config.redaction?.rules[0]?.detector).toBe("secret");
  });

  test("throws a path-qualified ConfigError on invalid config", () => {
    const path = tempConfig(`version: 1
artifacts_dir: artifacts
manifests_dir: benchmarks/manifests
benchmarks:
  ds1000:
    task_key: data_processing
    sample_count: 4
    partitions:
      decision_test: 4
task_keys:
  support_agent:
    replay:
      default_tool_policy: replayed
`);
    expect(() => loadConfig(path)).toThrow(ConfigError);
    try {
      loadConfig(path);
    } catch (error) {
      expect((error as ConfigError).message).toContain(
        "task_keys.support_agent.replay.default_tool_policy",
      );
      expect((error as ConfigError).issues.length).toBe(1);
    }
  });

  test("reports unreadable files", () => {
    expect(() => loadConfig(join(REPO_ROOT, "does-not-exist.yaml"))).toThrow(/cannot read/);
  });

  test("reports YAML syntax errors", () => {
    expect(() => loadConfig(tempConfig("version: 1\n  bad: [unclosed\n"))).toThrow(
      /not valid YAML/,
    );
  });
});

describe("forward compatibility", () => {
  test("unknown top-level sections are allowed (the engine may add its own)", () => {
    const result = validateConfig({ ...baseConfig(), future_section: { anything: true } });
    expect(result.ok).toBe(true);
  });

  test("unknown keys inside benchmark sections are allowed", () => {
    const result = validateConfig(
      withSection("benchmarks", {
        tau_bench: {
          task_key: "conversational_agent",
          sample_count: 20,
          partitions: { decision_test: 20 },
          trials: 3,
          max_steps: 30,
        },
      }),
    );
    expect(result.ok).toBe(true);
  });
});
