/**
 * `compound suggest-assertions <task_key>` — mine a task's accepted outputs and
 * propose assertions to add (issue #5).
 *
 * It reads only NON-sealed cases (listCases keeps the decision set sealed), so
 * suggestions are never derived from the held-out set. The output is advisory:
 * it prints each candidate with its support and a ready-to-paste YAML block. A
 * human decides — nothing is written to config.
 */

import type { Assertion } from "@compound/assertions";
import { type AssertionSuggestion, suggestAssertions } from "@compound/assertions";
import { loadConfig } from "@compound/config";
import type { Message } from "@compound/contract";
import { listCases } from "@compound/storage";
import type { CommandEnvironment, CommandResult, ParsedArgs } from "./commands";
import { DEFAULT_CONFIG_PATH, DEFAULT_DATABASE_PATH } from "./commands";

function stringFlag(flags: ParsedArgs["flags"], name: string): string | undefined {
  const value = flags[name];
  return typeof value === "string" ? value : undefined;
}

/** Coerce a case's stored `expected` into a model output message, or null. */
function asMessage(expected: unknown): Message | null {
  if (typeof expected !== "object" || expected === null) return null;
  // observed_output / human_golden store the focal assistant message; other
  // provenances (deterministic outcomes, feedback arrays) are not model outputs.
  if ("role" in expected || "content" in expected || "tool_calls" in expected) {
    return expected as Message;
  }
  return null;
}

/** Render one assertion as a YAML list item under `assertions.<task>`. */
function assertionYaml(assertion: Assertion): string[] {
  const lines = [`    - type: ${assertion.type}`];
  for (const [key, value] of Object.entries(assertion)) {
    if (key === "type") continue;
    // JSON encoding of a scalar is valid YAML and safely quotes strings.
    lines.push(`      ${key}: ${JSON.stringify(value)}`);
  }
  return lines;
}

export function runSuggestAssertionsCommand(
  args: ParsedArgs,
  env: CommandEnvironment,
): CommandResult {
  const taskKey = args.positional[0];
  if (taskKey === undefined) {
    env.write("error: usage: compound suggest-assertions <task_key> [--db PATH] [--config PATH]");
    return { exitCode: 2 };
  }

  let existing: Assertion[] = [];
  try {
    const config = loadConfig(stringFlag(args.flags, "config") ?? DEFAULT_CONFIG_PATH);
    existing = (config.assertions?.[taskKey] ?? []) as Assertion[];
  } catch {
    // A missing/invalid config is not fatal here — with no declared assertions,
    // everything is a fresh suggestion. The user just sees a full slate.
    env.write("note: no config loaded; suggesting against an empty assertion set.");
  }

  const db = env.openDatabase(stringFlag(args.flags, "db") ?? DEFAULT_DATABASE_PATH);
  try {
    const rows = listCases(db, { taskKey });
    if (rows.length === 0) {
      env.write(`no curated cases for '${taskKey}'. Run: compound curate ${taskKey}`);
      return { exitCode: 0 };
    }

    const outputs = rows
      .map((row) => asMessage(row.expected))
      .filter((message): message is Message => message !== null);

    const suggestions = suggestAssertions(outputs, { existing });

    env.write(
      `suggested assertions for '${taskKey}' ` +
        `(from ${outputs.length} accepted outputs of ${rows.length} cases; sealed set untouched):`,
    );
    if (suggestions.length === 0) {
      env.write("");
      env.write(
        outputs.length < 3
          ? "  (too few captured outputs to suggest from confidently)"
          : "  (no pattern crossed the confidence bar; nothing to suggest)",
      );
      if (existing.length > 0) {
        env.write(`  note: ${existing.length} assertion(s) already declared are not re-suggested.`);
      }
      return { exitCode: 0 };
    }

    env.write("");
    for (const s of suggestions) env.write(`  ${describe(s)}`);

    env.write("");
    env.write("These are suggestions — review, then paste under assertions in compound.yaml:");
    env.write("");
    env.write("assertions:");
    env.write(`  ${taskKey}:`);
    for (const s of suggestions) for (const line of assertionYaml(s.assertion)) env.write(line);

    return { exitCode: 0 };
  } catch (error) {
    env.write(
      `error: suggest-assertions failed: ${error instanceof Error ? error.message : error}`,
    );
    return { exitCode: 1 };
  } finally {
    db.close();
  }
}

/** A one-line human description of a suggestion: what and why. */
function describe(s: AssertionSuggestion): string {
  const label =
    s.assertion.type === "tool_called"
      ? `tool_called '${s.assertion.name}'`
      : s.assertion.type === "max_length"
        ? `max_length ${s.assertion.max}`
        : s.assertion.type;
  return `${label.padEnd(28)} ${s.rationale}`;
}
