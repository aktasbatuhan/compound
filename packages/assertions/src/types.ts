/**
 * Assertion types: deterministic, free checks graded against a model output.
 *
 * Spec: docs/assertions-v1.md. Every assertion is a pure function of the output
 * and its own parameters — no network, no model, no cost.
 */
import type { Message } from "@compound/contract";
import type { SimilarityMetric } from "./similarity";

export const ASSERTION_TYPES = [
  "valid_json",
  "json_schema",
  "contains",
  "not_contains",
  "regex",
  "equals",
  "tool_called",
  "tool_not_called",
  "tool_arg_equals",
  "tool_call_arg",
  "max_length",
  "json_path_equals",
  "text_similarity",
] as const;

export type AssertionType = (typeof ASSERTION_TYPES)[number];

/**
 * How a `tool_call_arg` assertion matches a resolved argument value. Exactly
 * one variant applies:
 * - `equals`   — deep-equals the expected value.
 * - `regex`    — the value (stringified) matches the pattern.
 * - `schema`   — the value satisfies a JSON schema.
 * - `subset`   — the value is an object containing each expected key/value
 *                (extra keys are allowed), so a task can pin the arguments it
 *                cares about without over-specifying the rest.
 */
export type ToolArgMatch =
  | { equals: unknown }
  | { regex: string; flags?: string }
  | { schema: unknown }
  | { subset: Record<string, unknown> };

/** One assertion, discriminated on `type`. `required` gates; `weight` scores. */
export type Assertion =
  | { type: "valid_json"; path?: string; required?: boolean; weight?: number }
  | { type: "json_schema"; schema: unknown; path?: string; required?: boolean; weight?: number }
  | {
      type: "contains";
      value: string;
      ignore_case?: boolean;
      required?: boolean;
      weight?: number;
    }
  | {
      type: "not_contains";
      value: string;
      ignore_case?: boolean;
      required?: boolean;
      weight?: number;
    }
  | { type: "regex"; pattern: string; flags?: string; required?: boolean; weight?: number }
  | { type: "equals"; value: unknown; path?: string; required?: boolean; weight?: number }
  | { type: "tool_called"; name: string; min_times?: number; required?: boolean; weight?: number }
  | { type: "tool_not_called"; name: string; required?: boolean; weight?: number }
  | {
      type: "tool_arg_equals";
      name: string;
      arg: string;
      value: unknown;
      required?: boolean;
      weight?: number;
    }
  | {
      type: "tool_call_arg";
      /** Tool whose call is graded. */
      name: string;
      /**
       * Dot-path into the tool call's arguments (e.g. `amount` or
       * `filters.min`). Omit to match against the whole arguments object.
       */
      arg?: string;
      /** Exactly one matcher; the argument passes if ANY call to `name` satisfies it. */
      match: ToolArgMatch;
      required?: boolean;
      weight?: number;
    }
  | { type: "max_length"; max: number; required?: boolean; weight?: number }
  | {
      type: "json_path_equals";
      path: string;
      value: unknown;
      required?: boolean;
      weight?: number;
    }
  | {
      type: "text_similarity";
      reference: string;
      metric: SimilarityMetric;
      pass_threshold: number;
      /** Where to read text from; defaults to the output's text content. */
      path?: string;
      ignore_case?: boolean;
      required?: boolean;
      weight?: number;
    };

/** The subject an assertion grades: one model output message. */
export interface AssertionSubject {
  /** The assistant message a model produced. `null` means no completion. */
  output: Message | null;
}

export interface AssertionResult {
  type: AssertionType;
  passed: boolean;
  /** Why it failed or a note on how it passed — never the raw output. */
  detail: string;
  required: boolean;
  weight: number;
}

export interface AssertionReport {
  /** True only when every REQUIRED assertion passed. */
  passed: boolean;
  /** Weighted fraction of assertions that passed, in [0, 1]. */
  score: number;
  results: AssertionResult[];
}
