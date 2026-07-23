/**
 * Evaluate assertions against a model output.
 *
 * Each evaluator returns `{ passed, detail }`. `detail` summarises the outcome
 * without echoing the raw output, which may be large or sensitive.
 */
import Ajv from "ajv";
import { isAbsent, outputText, outputToolCalls, parsedOutput, resolvePath } from "./output";
import type { Assertion, AssertionReport, AssertionResult, AssertionSubject } from "./types";

const ajv = new Ajv({ allErrors: true, strict: false });

interface Outcome {
  passed: boolean;
  detail: string;
}

function stringify(value: unknown): string {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text.length > 80 ? `${text.slice(0, 77)}…` : text;
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
      const equal = JSON.stringify(value) === JSON.stringify(assertion.value);
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
      const equal = JSON.stringify(value) === JSON.stringify(assertion.value);
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
      const match = calls.some(
        (call) => JSON.stringify(call.arguments[assertion.arg]) === JSON.stringify(assertion.value),
      );
      return match
        ? { passed: true, detail: `tool '${assertion.name}' arg '${assertion.arg}' matched` }
        : {
            passed: false,
            detail: `tool '${assertion.name}' arg '${assertion.arg}' never equalled ${stringify(
              assertion.value,
            )}`,
          };
    }

    case "max_length": {
      const text = outputText(subject);
      if (text === null) return { passed: false, detail: "output has no text content" };
      return text.length <= assertion.max
        ? { passed: true, detail: `length ${text.length} ≤ ${assertion.max}` }
        : { passed: false, detail: `length ${text.length} > ${assertion.max}` };
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
