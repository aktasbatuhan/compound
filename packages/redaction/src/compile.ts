/**
 * Config-time compilation of the `redaction` section.
 *
 * Everything that can be wrong with a rule — an uncompilable `pattern`, a
 * malformed `applies_to` path — is discovered here, once, naming the rule.
 * `redactTrace` never reports a configuration problem per trace.
 */
import type { Redaction as RedactionConfig, RedactionRule } from "@compound/config";
import { defaultRedactionMarker } from "@compound/config";
import type { DetectorPattern } from "./detectors";
import { PII_PATTERNS, SECRET_PATTERNS } from "./detectors";
import type { PathPattern } from "./paths";
import { PathPatternError, parsePathPattern } from "./paths";

export type { RedactionConfig };

/** Thrown for an unusable `redaction` config. Always names the offending rule. */
export class RedactionConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RedactionConfigError";
  }
}

/** A rule with its patterns parsed and its marker and contract rule id resolved. */
export interface CompiledRule {
  readonly name: string;
  /** The contract's `Redaction.rule` value: `secret`, `pii`, or `custom:<name>`. */
  readonly rule: string;
  readonly marker: string;
  readonly appliesTo: readonly PathPattern[];
  readonly patterns: readonly DetectorPattern[];
}

/** The compiled form of a whole `redaction` config section. */
export interface CompiledRedaction {
  readonly rules: readonly CompiledRule[];
  readonly allowlist: readonly PathPattern[];
}

function detectorPatterns(rule: RedactionRule): readonly DetectorPattern[] {
  if (rule.detector === "secret") return SECRET_PATTERNS;
  if (rule.detector === "pii") return PII_PATTERNS;
  let regex: RegExp;
  try {
    regex = new RegExp(rule.pattern, "g");
  } catch (cause) {
    const detail = cause instanceof Error ? cause.message : String(cause);
    throw new RedactionConfigError(
      `redaction rule "${rule.name}": pattern is not a valid regular expression (${detail})`,
    );
  }
  return [{ name: `custom:${rule.name}`, regex }];
}

function contractRuleId(rule: RedactionRule): string {
  return rule.detector === "regex" ? `custom:${rule.name}` : rule.detector;
}

function parsePatterns(
  source: readonly string[],
  describe: (bad: string) => string,
): PathPattern[] {
  return source.map((entry) => {
    try {
      return parsePathPattern(entry);
    } catch (cause) {
      if (cause instanceof PathPatternError) {
        throw new RedactionConfigError(`${describe(entry)}: ${cause.message}`);
      }
      throw cause;
    }
  });
}

/**
 * Compile a `redaction` config section. Throws `RedactionConfigError` for an
 * invalid regex or path pattern, naming the rule. Returns `undefined` when
 * there is nothing to do (absent config, or no rules).
 */
export function compileRedaction(
  config: RedactionConfig | undefined,
): CompiledRedaction | undefined {
  if (config === undefined || config.rules.length === 0) return undefined;

  const rules: CompiledRule[] = config.rules.map((rule) => ({
    name: rule.name,
    rule: contractRuleId(rule),
    marker: defaultRedactionMarker(rule),
    appliesTo: parsePatterns(
      rule.applies_to,
      (bad) => `redaction rule "${rule.name}" applies_to "${bad}"`,
    ),
    patterns: detectorPatterns(rule),
  }));

  const allowlist = parsePatterns(
    config.field_allowlist ?? [],
    (bad) => `redaction field_allowlist "${bad}"`,
  );

  return { rules, allowlist };
}

const cache = new WeakMap<RedactionConfig, CompiledRedaction | undefined>();

/**
 * `compileRedaction` memoized on the config object, so importing a JSONL file
 * compiles each rule once rather than once per trace.
 */
export function compileRedactionCached(
  config: RedactionConfig | undefined,
): CompiledRedaction | undefined {
  if (config === undefined) return undefined;
  if (cache.has(config)) return cache.get(config);
  const compiled = compileRedaction(config);
  cache.set(config, compiled);
  return compiled;
}
