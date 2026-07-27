/**
 * `compound validate` — check compound.yaml and point at the fix (issue #5).
 *
 * Schema failures already carry a config path (loadConfig throws them). This
 * command surfaces them for CI (exit 1) and adds two things the raw schema
 * cannot: a short hint per failure, and — when the schema passes — advisory
 * cross-reference WARNINGS (a model on an undeclared provider, a model with no
 * price) that would otherwise only bite at run time. Warnings do not fail the
 * check; only a schema-invalid config exits non-zero.
 *
 * The advisory checks live here, not in the shared validator, so the Python
 * engine's contract for compound.yaml is unchanged.
 */
import { type CompoundConfig, ConfigError, knownProvider, loadConfig } from "@compound/config";
import type { CommandEnvironment, CommandResult, ParsedArgs } from "./commands";
import { DEFAULT_CONFIG_PATH } from "./commands";

function stringFlag(flags: ParsedArgs["flags"], name: string): string | undefined {
  const value = flags[name];
  return typeof value === "string" ? value : undefined;
}

/** A one-line "here's how to fix it" hint for a schema issue, or null. */
function hintFor(path: string, message: string): string | null {
  if (path.endsWith(".provider")) {
    return "the provider name must match a key declared under providers:";
  }
  if (/regular expression/i.test(message)) {
    return "fix the regex so it compiles (test it against your redaction target)";
  }
  if (/invalid (enum|option)|expected one of/i.test(message)) {
    return "use one of the allowed values named in the message";
  }
  if (/required|expected .* received undefined/i.test(message)) {
    return "this field is required — add it";
  }
  return null;
}

interface ModelEntry {
  id?: string;
  provider?: string;
  role?: string;
}

function allModels(config: CompoundConfig): ModelEntry[] {
  const models = (config.models ?? {}) as {
    frontier?: ModelEntry[];
    candidates?: ModelEntry[];
  };
  return [...(models.frontier ?? []), ...(models.candidates ?? [])];
}

/** True if `modelId` has a price anywhere it could be resolved from. */
function hasPrice(
  config: CompoundConfig,
  modelId: string,
  providerName: string | undefined,
): boolean {
  const global = config.pricing_usd_per_million_tokens?.[modelId];
  const flex = config.flex_pricing_usd_per_million_tokens?.[modelId];
  const providers = (config.providers ?? {}) as Record<
    string,
    { pricing_usd_per_million_tokens?: Record<string, unknown> }
  >;
  const local = providerName
    ? providers[providerName]?.pricing_usd_per_million_tokens?.[modelId]
    : undefined;
  return global !== undefined || flex !== undefined || local !== undefined;
}

/** Advisory cross-reference checks that a valid schema still can't catch. */
function advisoryWarnings(config: CompoundConfig): string[] {
  const warnings: string[] = [];
  const providerNames = new Set(Object.keys(config.providers ?? {}));

  for (const model of allModels(config)) {
    if (model.id === undefined) continue;
    if (model.provider !== undefined && !providerNames.has(model.provider)) {
      // If the undeclared name is a KNOWN provider, point at the paste block.
      const known = knownProvider(model.provider)
        ? ` — run: compound providers ${model.provider}`
        : ` (declared: ${[...providerNames].join(", ") || "none"})`;
      warnings.push(
        `model '${model.id}' names provider '${model.provider}', which is not under providers:${known}`,
      );
    }
    if (!hasPrice(config, model.id, model.provider)) {
      warnings.push(
        `model '${model.id}' has no price — it will refuse to run. Add it under ` +
          "pricing_usd_per_million_tokens.",
      );
    }
  }
  return warnings;
}

export function runValidateCommand(args: ParsedArgs, env: CommandEnvironment): CommandResult {
  const path = stringFlag(args.flags, "config") ?? DEFAULT_CONFIG_PATH;

  let config: CompoundConfig;
  try {
    config = loadConfig(path);
  } catch (error) {
    if (error instanceof ConfigError) {
      env.write(`invalid: ${path}`);
      if (error.issues.length > 0) {
        for (const issue of error.issues) {
          env.write(`  ✗ ${issue.path}: ${issue.message}`);
          const hint = hintFor(issue.path, issue.message);
          if (hint !== null) env.write(`      → ${hint}`);
        }
      } else {
        // Unreadable file or YAML syntax error — the message is the whole story.
        env.write(`  ✗ ${error.message}`);
      }
      return { exitCode: 1 };
    }
    throw error;
  }

  const providers = Object.keys(config.providers ?? {}).length;
  const models = allModels(config).length;
  const tasks = Object.keys(config.task_keys ?? {}).length;
  env.write(`valid: ${path}`);
  env.write(`  ${providers} provider(s), ${models} model(s), ${tasks} task(s)`);

  const warnings = advisoryWarnings(config);
  if (warnings.length > 0) {
    env.write("");
    env.write(`${warnings.length} warning(s) — valid, but these will bite at run time:`);
    for (const warning of warnings) env.write(`  ! ${warning}`);
  }
  return { exitCode: 0 };
}
