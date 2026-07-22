/**
 * Reading and validating `compound.yaml`.
 *
 * `validateConfig` is pure (any parsed object in, result out); `loadConfig`
 * adds file reading + YAML parsing and throws on the first problem with a
 * message that always names the offending path.
 */
import { readFileSync } from "node:fs";
import { parse as parseYaml } from "yaml";
import type { z } from "zod";
import { type CompoundConfig, CompoundConfigSchema, type RedactionRule } from "./schemas";

export type ConfigIssue = {
  /** Dotted/bracketed path into the config, e.g. `task_keys.support.replay.default_tool_policy`. */
  path: string;
  message: string;
};

export type ValidateConfigResult =
  | { ok: true; config: CompoundConfig }
  | { ok: false; issues: ConfigIssue[] };

/** Thrown by `loadConfig` when a file cannot be read, parsed, or validated. */
export class ConfigError extends Error {
  readonly path: string;
  readonly issues: ConfigIssue[];

  constructor(path: string, message: string, issues: ConfigIssue[] = []) {
    super(message);
    this.name = "ConfigError";
    this.path = path;
    this.issues = issues;
  }
}

function formatPath(segments: ReadonlyArray<PropertyKey>): string {
  let out = "";
  for (const segment of segments) {
    if (typeof segment === "number") out += `[${segment}]`;
    else if (out === "") out += String(segment);
    else out += `.${String(segment)}`;
  }
  return out === "" ? "<root>" : out;
}

function toIssues(error: z.ZodError): ConfigIssue[] {
  return error.issues.map((issue) => ({
    path: formatPath(issue.path),
    message: issue.message,
  }));
}

/**
 * Semantic checks JSON Schema cannot express. Kept deliberately small: anything
 * expressible structurally belongs in the zod schema so the Python engine
 * enforces it too.
 */
function semanticIssues(config: CompoundConfig): ConfigIssue[] {
  const issues: ConfigIssue[] = [];
  const rules: RedactionRule[] = config.redaction?.rules ?? [];
  rules.forEach((rule, index) => {
    if (rule.detector !== "regex") return;
    try {
      new RegExp(rule.pattern);
    } catch (error) {
      issues.push({
        path: `redaction.rules[${index}].pattern`,
        message: `is not a valid regular expression: ${(error as Error).message}`,
      });
    }
  });
  return issues;
}

/** Validate an already-parsed object against the `compound.yaml` schema. */
export function validateConfig(input: unknown): ValidateConfigResult {
  const parsed = CompoundConfigSchema.safeParse(input);
  if (!parsed.success) return { ok: false, issues: toIssues(parsed.error) };
  const issues = semanticIssues(parsed.data);
  if (issues.length > 0) return { ok: false, issues };
  return { ok: true, config: parsed.data };
}

/** Human-readable, path-qualified rendering of validation issues. */
export function formatIssues(path: string, issues: ConfigIssue[]): string {
  const lines = issues.map((issue) => `  - ${issue.path}: ${issue.message}`);
  return `${path} is not a valid compound.yaml:\n${lines.join("\n")}`;
}

/**
 * Read, parse, and validate a `compound.yaml`.
 *
 * @throws {ConfigError} if the file is unreadable, is not valid YAML, or fails
 * validation (the message lists every issue with its config path).
 */
export function loadConfig(path = "compound.yaml"): CompoundConfig {
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch (error) {
    throw new ConfigError(path, `cannot read ${path}: ${(error as Error).message}`);
  }

  let data: unknown;
  try {
    data = parseYaml(text);
  } catch (error) {
    throw new ConfigError(path, `${path} is not valid YAML: ${(error as Error).message}`);
  }

  const result = validateConfig(data);
  if (!result.ok) throw new ConfigError(path, formatIssues(path, result.issues), result.issues);
  return result.config;
}
