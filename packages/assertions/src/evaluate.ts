/**
 * Evaluate assertions against a model output.
 *
 * Each evaluator returns `{ passed, detail }`. `detail` summarises the outcome
 * without echoing the raw output, which may be large or sensitive.
 */

import { structuralEqual } from "@compound/contract";
import Ajv from "ajv";
import {
  isAbsent,
  outputText,
  outputToolCalls,
  parsedOutput,
  resolvePath,
  walkPath,
} from "./output";
import { similarity } from "./similarity";
import type {
  Assertion,
  AssertionReport,
  AssertionResult,
  AssertionSubject,
  ToolArgMatch,
} from "./types";

const ajv = new Ajv({ allErrors: true, strict: false });

interface Outcome {
  passed: boolean;
  detail: string;
}

function stringify(value: unknown): string {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text.length > 80 ? `${text.slice(0, 77)}…` : text;
}

/** A short, value-free description of a tool-arg matcher for `detail` lines. */
function matcherLabel(match: ToolArgMatch): string {
  if ("equals" in match) return `equals ${stringify(match.equals)}`;
  if ("regex" in match) return `matches /${match.regex}/${match.flags ?? ""}`;
  if ("subset" in match) return `contains ${stringify(Object.keys(match.subset))}`;
  return "satisfies schema";
}

/**
 * Compile a tool-arg matcher into a predicate, or return the reason it could
 * not be built (an invalid regex or JSON schema), so a bad matcher fails the
 * assertion with a clear detail instead of throwing.
 */
function compileMatcher(match: unknown): ((value: unknown) => boolean) | { error: string } {
  // Fail SAFE on a malformed matcher (#8): config may not have caught a
  // tool_call_arg with a missing/blank matcher, and this runs mid-experiment
  // after completions may already be paid for — a bad matcher must fail the
  // assertion, never throw and abort the run.
  if (match === null || typeof match !== "object") {
    return { error: "tool_call_arg needs a matcher (equals/regex/subset/schema)" };
  }
  const m = match as ToolArgMatch;
  if ("equals" in m) {
    return (value) => structuralEqual(value, m.equals);
  }
  if ("regex" in m) {
    // A non-string pattern is a malformed matcher, not a regex to compile (#9):
    // reject it rather than coercing `23` into "23".
    if (typeof m.regex !== "string") {
      return { error: "tool_call_arg 'regex' must be a string pattern" };
    }
    let re: RegExp;
    try {
      re = new RegExp(m.regex, m.flags);
    } catch (error) {
      return { error: `invalid regex: ${error instanceof Error ? error.message : "error"}` };
    }
    return (value) => re.test(typeof value === "string" ? value : JSON.stringify(value));
  }
  if ("subset" in m) {
    // The subset payload MUST be an object of key/value pairs (#9). Without this
    // guard `{subset: 23}` reached Object.entries(23) → [] → `.every()` returned
    // true for ANY argument, so a malformed matcher passed vacuously.
    const subset = m.subset;
    if (subset === null || typeof subset !== "object" || Array.isArray(subset)) {
      return { error: "tool_call_arg 'subset' must be an object of key/value pairs" };
    }
    const entries = Object.entries(subset as Record<string, unknown>);
    return (value) => {
      if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
      const obj = value as Record<string, unknown>;
      return entries.every(([key, expected]) => key in obj && structuralEqual(obj[key], expected));
    };
  }
  if ("schema" in m) {
    if (m.schema === null || typeof m.schema !== "object" || Array.isArray(m.schema)) {
      return { error: "tool_call_arg 'schema' must be a JSON Schema object" };
    }
    let validateFn: ReturnType<typeof ajv.compile>;
    try {
      validateFn = ajv.compile(m.schema as object);
    } catch (error) {
      return { error: `invalid schema: ${error instanceof Error ? error.message : "error"}` };
    }
    return (value) => validateFn(value) === true;
  }
  return { error: "tool_call_arg matcher must be one of equals/regex/subset/schema" };
}

function evaluateOne(assertion: Assertion, subject: AssertionSubject): Outcome {
  switch (assertion.type) {
    case "valid_json": {
      const value = resolvePath(subject, assertion.path);
      if (isAbsent(value)) return { passed: false, detail: value.reason };
      // For a non-content path the value is already parsed JSON.
      if (assertion.path !== undefined && assertion.path !== "content") {
        return { passed: true, detail: "value present in parsed output" };
      }
      const parsed = parsedOutput(subject);
      return isAbsent(parsed)
        ? { passed: false, detail: parsed.reason }
        : { passed: true, detail: "output is valid JSON" };
    }

    case "json_schema": {
      const parsed =
        assertion.path === undefined || assertion.path === "content"
          ? parsedOutput(subject)
          : resolvePath(subject, assertion.path);
      if (isAbsent(parsed)) return { passed: false, detail: parsed.reason };
      const validateFn = ajv.compile(assertion.schema as object);
      if (validateFn(parsed)) return { passed: true, detail: "matches schema" };
      const first = validateFn.errors?.[0];
      return {
        passed: false,
        detail:
          `schema mismatch: ${first?.instancePath || "(root)"} ${first?.message ?? ""}`.trim(),
      };
    }

    case "contains":
    case "not_contains": {
      const text = outputText(subject);
      if (text === null) return { passed: false, detail: "output has no text content" };
      const haystack = assertion.ignore_case ? text.toLowerCase() : text;
      const needle = assertion.ignore_case ? assertion.value.toLowerCase() : assertion.value;
      const has = haystack.includes(needle);
      const want = assertion.type === "contains";
      return {
        passed: has === want,
        detail:
          has === want
            ? `output ${want ? "contains" : "omits"} ${stringify(assertion.value)}`
            : `output ${want ? "does not contain" : "contains"} ${stringify(assertion.value)}`,
      };
    }

    case "regex": {
      const text = outputText(subject);
      if (text === null) return { passed: false, detail: "output has no text content" };
      let re: RegExp;
      try {
        re = new RegExp(assertion.pattern, assertion.flags);
      } catch (error) {
        return {
          passed: false,
          detail: `invalid regex: ${error instanceof Error ? error.message : "error"}`,
        };
      }
      return re.test(text)
        ? { passed: true, detail: "output matches pattern" }
        : { passed: false, detail: `output does not match /${assertion.pattern}/` };
    }

    case "equals": {
      const value = resolvePath(subject, assertion.path);
      if (isAbsent(value)) return { passed: false, detail: value.reason };
      // Structural equality, so {a:1,b:2} and {b:2,a:1} are equal (#10) — a
      // JSON.stringify compare falsely fails on key order.
      const equal = structuralEqual(value, assertion.value);
      return {
        passed: equal,
        detail: equal
          ? "value equals expected"
          : `expected ${stringify(assertion.value)}, got ${stringify(value)}`,
      };
    }

    case "json_path_equals": {
      const value = resolvePath(subject, assertion.path);
      if (isAbsent(value)) return { passed: false, detail: value.reason };
      const equal = structuralEqual(value, assertion.value); // key-order-insensitive (#10)
      return {
        passed: equal,
        detail: equal
          ? `${assertion.path} equals expected`
          : `${assertion.path}: expected ${stringify(assertion.value)}, got ${stringify(value)}`,
      };
    }

    case "tool_called": {
      const min = assertion.min_times ?? 1;
      const times = outputToolCalls(subject).filter((call) => call.name === assertion.name).length;
      return times >= min
        ? { passed: true, detail: `tool '${assertion.name}' called ${times}×` }
        : {
            passed: false,
            detail: `tool '${assertion.name}' called ${times}×, need ≥ ${min}`,
          };
    }

    case "tool_not_called": {
      const called = outputToolCalls(subject).some((call) => call.name === assertion.name);
      return called
        ? { passed: false, detail: `tool '${assertion.name}' was called` }
        : { passed: true, detail: `tool '${assertion.name}' was not called` };
    }

    case "tool_arg_equals": {
      const calls = outputToolCalls(subject).filter((call) => call.name === assertion.name);
      if (calls.length === 0) {
        return { passed: false, detail: `tool '${assertion.name}' was not called` };
      }
      const match = calls.some((call) =>
        structuralEqual(call.arguments[assertion.arg], assertion.value),
      ); // key-order-insensitive (#10)
      return match
        ? { passed: true, detail: `tool '${assertion.name}' arg '${assertion.arg}' matched` }
        : {
            passed: false,
            detail: `tool '${assertion.name}' arg '${assertion.arg}' never equalled ${stringify(
              assertion.value,
            )}`,
          };
    }

    case "tool_call_arg": {
      const calls = outputToolCalls(subject).filter((call) => call.name === assertion.name);
      if (calls.length === 0) {
        return { passed: false, detail: `tool '${assertion.name}' was not called` };
      }
      const compiled = compileMatcher(assertion.match);
      if ("error" in compiled) return { passed: false, detail: compiled.error };

      const target = assertion.arg === undefined ? "arguments" : `arg '${assertion.arg}'`;
      const label = matcherLabel(assertion.match);
      // Passes if ANY call to the tool satisfies the matcher — mirrors
      // `tool_arg_equals`, which grades the tool's behaviour across all its calls.
      for (const call of calls) {
        const value =
          assertion.arg === undefined ? call.arguments : walkPath(call.arguments, assertion.arg);
        if (isAbsent(value)) continue;
        if (compiled(value)) {
          return { passed: true, detail: `tool '${assertion.name}' ${target} ${label}` };
        }
      }
      return {
        passed: false,
        detail: `tool '${assertion.name}' ${target} never ${label}`,
      };
    }

    case "max_length": {
      const text = outputText(subject);
      if (text === null) return { passed: false, detail: "output has no text content" };
      return text.length <= assertion.max
        ? { passed: true, detail: `length ${text.length} ≤ ${assertion.max}` }
        : { passed: false, detail: `length ${text.length} > ${assertion.max}` };
    }

    case "text_similarity": {
      let text: string | null;
      if (assertion.path === undefined || assertion.path === "content") {
        text = outputText(subject);
      } else {
        const value = resolvePath(subject, assertion.path);
        if (isAbsent(value)) return { passed: false, detail: value.reason };
        text = typeof value === "string" ? value : JSON.stringify(value);
      }
      if (text === null) return { passed: false, detail: "output has no text content" };
      const cand = assertion.ignore_case ? text.toLowerCase() : text;
      const ref = assertion.ignore_case ? assertion.reference.toLowerCase() : assertion.reference;
      const score = similarity(assertion.metric, cand, ref);
      const rounded = score.toFixed(3);
      return score >= assertion.pass_threshold
        ? { passed: true, detail: `${assertion.metric} ${rounded} ≥ ${assertion.pass_threshold}` }
        : { passed: false, detail: `${assertion.metric} ${rounded} < ${assertion.pass_threshold}` };
    }
  }
}

/**
 * Evaluate a list of assertions against one output.
 *
 * `passed` is true only when every REQUIRED assertion passed; `score` is the
 * weighted fraction of all assertions that passed, so an optional check can
 * lower the score without failing the case.
 */
export function evaluateAssertions(
  assertions: readonly Assertion[],
  subject: AssertionSubject,
): AssertionReport {
  const results: AssertionResult[] = assertions.map((assertion) => {
    const outcome = evaluateOne(assertion, subject);
    return {
      type: assertion.type,
      passed: outcome.passed,
      detail: outcome.detail,
      required: assertion.required ?? true,
      weight: assertion.weight ?? 1,
    };
  });

  const requiredPassed = results.every((result) => !result.required || result.passed);
  const totalWeight = results.reduce((sum, result) => sum + result.weight, 0);
  const passedWeight = results.reduce(
    (sum, result) => sum + (result.passed ? result.weight : 0),
    0,
  );

  return {
    passed: requiredPassed,
    score: totalWeight === 0 ? 1 : passedWeight / totalWeight,
    results,
  };
}
