/**
 * Assertion suggestions: mine a task's ACCEPTED outputs for high-precision,
 * explainable assertion candidates (docs/assertions-v1.md, issue #5).
 *
 * This proposes; a human accepts. Every suggestion cites its support ("31/34
 * accepted outputs call 'dispute_charge'") so the reason is auditable, and the
 * signals are deliberately conservative — a tool called almost every time, an
 * output shape that always parses, a length no accepted output has exceeded.
 * Noisy signals (common phrases, incidental substrings) are omitted on purpose.
 *
 * The miner reads outputs through the SAME readers the grader uses, so a
 * suggestion it makes is one the resulting assertion would actually grade on.
 * It must only ever be fed NON-sealed outputs: designing assertions from the
 * decision set would leak it.
 */
import type { Message } from "@compound/contract";
import { outputText, outputToolCalls, parsedOutput } from "./output";
import type { Assertion, AssertionSubject } from "./types";

export interface AssertionSuggestion {
  /** A ready-to-paste assertion. */
  assertion: Assertion;
  /** One-line human rationale citing support. */
  rationale: string;
  /** How many accepted outputs exhibit the pattern. */
  support: number;
  /** Sample size the fraction is over. */
  total: number;
  /** support / total, in [0, 1]. */
  fraction: number;
}

export interface SuggestOptions {
  /** Assertions already configured for the task; never re-suggest these. */
  existing?: Assertion[];
  /** Minimum fraction of accepted outputs that must exhibit a pattern (default 0.8). */
  minSupportFraction?: number;
  /** Minimum absolute count of supporting outputs (default 3). */
  minCases?: number;
}

/** True if `existing` already asserts `type` (matching `name` for tool assertions). */
function alreadyAsserted(existing: Assertion[], type: Assertion["type"], name?: string): boolean {
  return existing.some((a) => {
    if (a.type !== type) return false;
    if (name === undefined) return true;
    return "name" in a && a.name === name;
  });
}

/** Round a length ceiling up to a tidy value, never below 100. */
function lengthCeiling(observedMax: number): number {
  return Math.max(100, Math.ceil((observedMax * 1.5) / 50) * 50);
}

/**
 * Propose assertions from a task's accepted outputs.
 *
 * `outputs` are the accepted (reference) messages — `null` for cases with no
 * captured output. Returns an ordered list of suggestions, most-supported
 * first; empty when there is too little evidence to be confident.
 */
export function suggestAssertions(
  outputs: readonly (Message | null)[],
  opts: SuggestOptions = {},
): AssertionSuggestion[] {
  const existing = opts.existing ?? [];
  const minFraction = opts.minSupportFraction ?? 0.8;
  const minCases = opts.minCases ?? 3;

  const present = outputs.filter((o): o is Message => o !== null);
  const total = present.length;
  if (total < minCases) return [];

  const subjects: AssertionSubject[] = present.map((output) => ({ output }));
  const suggestions: AssertionSuggestion[] = [];

  // --- tool_called: a tool called in almost every accepted output ----------
  const toolCounts = new Map<string, number>();
  for (const subject of subjects) {
    const names = new Set(outputToolCalls(subject).map((call) => call.name));
    for (const name of names) toolCounts.set(name, (toolCounts.get(name) ?? 0) + 1);
  }
  const toolSuggestions: AssertionSuggestion[] = [];
  for (const [name, support] of toolCounts) {
    const fraction = support / total;
    if (fraction < minFraction || support < minCases) continue;
    if (alreadyAsserted(existing, "tool_called", name)) continue;
    toolSuggestions.push({
      assertion: { type: "tool_called", name },
      rationale: `${support}/${total} accepted outputs call '${name}'`,
      support,
      total,
      fraction,
    });
  }
  toolSuggestions.sort(
    (a, b) => b.support - a.support || a.assertion.type.localeCompare(b.assertion.type),
  );
  suggestions.push(...toolSuggestions);

  // --- valid_json: the output shape always parses --------------------------
  if (!alreadyAsserted(existing, "valid_json")) {
    const jsonCount = subjects.filter((s) => {
      const parsed = parsedOutput(s);
      return !(typeof parsed === "object" && parsed !== null && "absent" in parsed);
    }).length;
    const fraction = jsonCount / total;
    // JSON is a stronger claim than a tool call; hold it to a higher bar.
    if (fraction >= Math.max(minFraction, 0.9) && jsonCount >= minCases) {
      suggestions.push({
        assertion: { type: "valid_json" },
        rationale: `${jsonCount}/${total} accepted outputs are valid JSON`,
        support: jsonCount,
        total,
        fraction,
      });
    }
  }

  // --- max_length: a guardrail no accepted output has crossed ---------------
  if (!alreadyAsserted(existing, "max_length")) {
    const lengths = subjects
      .map((s) => outputText(s))
      .filter((t): t is string => t !== null && t.length > 0)
      .map((t) => t.length);
    // Only for text-shaped tasks: most outputs must actually carry text.
    if (lengths.length >= minCases && lengths.length / total >= minFraction) {
      const observedMax = Math.max(...lengths);
      const ceiling = lengthCeiling(observedMax);
      suggestions.push({
        assertion: { type: "max_length", max: ceiling },
        rationale: `no accepted output exceeded ${observedMax} chars — guardrail at ${ceiling}`,
        support: lengths.length,
        total,
        fraction: lengths.length / total,
      });
    }
  }

  return suggestions;
}
