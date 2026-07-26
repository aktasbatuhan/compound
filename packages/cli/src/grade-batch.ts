/**
 * `compound grade-batch <file>` — grade a batch of model outputs against a
 * task's assertions and print JSON. This is the SINGLE grader the Python GEPA
 * adapter calls back into (docs/optimization-v1.md), so grading never forks
 * across languages: GEPA proposes prompts, this scores them with the exact same
 * `@compound/assertions` the gate uses.
 *
 * Input  JSON: { "task_key": "...", "items": [ { "case_id": "...", "output": <Message> } ] }
 * Output JSON: { "items": [ { "case_id", "passed", "score", "feedback" } ] }
 *
 * Reads the input file (or stdin when the path is "-") and writes to stdout, so
 * it is a clean subprocess contract. No model calls, no cost.
 */
import { readFileSync } from "node:fs";
import { type Assertion, evaluateAssertions } from "@compound/assertions";
import { loadConfig } from "@compound/config";
import type { Message } from "@compound/contract";
import type { CommandEnvironment, CommandResult, ParsedArgs } from "./commands";
import { DEFAULT_CONFIG_PATH } from "./commands";

interface GradeItem {
  case_id: string;
  output: Message | null;
}

function stringFlag(flags: ParsedArgs["flags"], name: string): string | undefined {
  const value = flags[name];
  return typeof value === "string" ? value : undefined;
}

export function runGradeBatchCommand(args: ParsedArgs, env: CommandEnvironment): CommandResult {
  const path = args.positional[0];
  if (path === undefined) {
    env.write(
      "error: usage: compound grade-batch <file|-> (reads {task_key, items:[{case_id, output}]})",
    );
    return { exitCode: 2 };
  }

  let raw: string;
  try {
    raw = readFileSync(path === "-" ? 0 : path, "utf8");
  } catch (error) {
    env.write(`error: could not read ${path}: ${error instanceof Error ? error.message : error}`);
    return { exitCode: 1 };
  }

  let request: { task_key?: string; items?: GradeItem[] };
  try {
    request = JSON.parse(raw);
  } catch {
    env.write("error: input must be a JSON object");
    return { exitCode: 1 };
  }
  const taskKey = request.task_key;
  const items = request.items;
  if (typeof taskKey !== "string" || !Array.isArray(items)) {
    env.write('error: input needs a string "task_key" and an "items" array');
    return { exitCode: 1 };
  }

  const config = loadConfig(stringFlag(args.flags, "config") ?? DEFAULT_CONFIG_PATH);
  const assertions = (config.assertions?.[taskKey] ?? []) as Assertion[];

  const graded = items.map((item) => {
    const report = evaluateAssertions(assertions, { output: item.output ?? null });
    // Feedback is the failing checks' details (never the raw output), which is
    // what GEPA reflects on to propose a better prompt.
    const feedback = report.results
      .filter((r) => !r.passed)
      .map((r) => `${r.type}: ${r.detail}`)
      .join("; ");
    return {
      case_id: item.case_id,
      passed: report.passed,
      score: report.score,
      feedback: feedback.length > 0 ? feedback : "all checks passed",
    };
  });

  env.write(JSON.stringify({ items: graded }));
  return { exitCode: 0 };
}
