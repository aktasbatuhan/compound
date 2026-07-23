/**
 * Built-in `secret` and `pii` detectors, plus the shared match/merge machinery
 * the `regex` detector reuses.
 *
 * Every detector is a list of named patterns over a string, optionally with a
 * validator that rejects a syntactic match (Luhn for card numbers). Detectors
 * only ever look at strings; numbers, booleans and null are never inspected.
 *
 * These are heuristics. They are documented pattern-by-pattern in the README
 * precisely because their coverage is bounded and knowable — see the honesty
 * note there.
 */

/** A half-open `[start, end)` interval of a string that must be replaced. */
export interface DetectorMatch {
  readonly start: number;
  readonly end: number;
}

export interface DetectorPattern {
  /** Stable identifier, used only in documentation and tests. */
  readonly name: string;
  /** Must carry the `g` flag; `lastIndex` is reset before every use. */
  readonly regex: RegExp;
  /** Optional second-stage check; a `false` verdict drops the match. */
  readonly validate?: (candidate: string) => boolean;
}

/** Luhn checksum over the digits of `candidate` (non-digits ignored). */
export function passesLuhn(candidate: string): boolean {
  const digits = candidate.replace(/\D/g, "");
  if (digits.length < 13 || digits.length > 19) return false;
  let sum = 0;
  let double = false;
  for (let i = digits.length - 1; i >= 0; i -= 1) {
    const code = digits.charCodeAt(i) - 48;
    let value = code;
    if (double) {
      value *= 2;
      if (value > 9) value -= 9;
    }
    sum += value;
    double = !double;
  }
  return sum % 10 === 0;
}

/**
 * Credential-shaped strings. Prefix-anchored wherever the vendor publishes a
 * prefix, because prefix-anchored patterns are the ones that do not fire on
 * ordinary prose or on numeric config values.
 */
export const SECRET_PATTERNS: readonly DetectorPattern[] = [
  /** OpenAI-style keys, including `sk-proj-` / `sk-ant-` style infixes. */
  { name: "openai_key", regex: /\bsk-[A-Za-z0-9_-]{16,}/g },
  /** GitHub personal/OAuth/server/user-to-server/refresh tokens. */
  { name: "github_token", regex: /\bgh[pousr]_[A-Za-z0-9]{20,}/g },
  /** AWS access key ids (long-term `AKIA`, temporary `ASIA`). */
  { name: "aws_access_key_id", regex: /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/g },
  /** Google API keys. */
  { name: "google_api_key", regex: /\bAIza[0-9A-Za-z_-]{35}\b/g },
  /** Slack bot/app/user/refresh/legacy tokens. */
  { name: "slack_token", regex: /\bxox[baprs]-[A-Za-z0-9-]{10,}/g },
  /** JWTs: three dot-separated base64url segments starting `eyJ`. */
  { name: "jwt", regex: /\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}/g },
  /** The token after `Bearer `; the word `Bearer` itself is left in place. */
  { name: "bearer_token", regex: /(?<=\bBearer\s)[A-Za-z0-9._~+/=-]{8,}/g },
  /** Whole PEM private key blocks, header and footer included. */
  {
    name: "pem_private_key",
    regex:
      /-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z ]+ )?PRIVATE KEY-----/g,
  },
];

/**
 * Personally identifying values. Deliberately separator-anchored for phone
 * numbers and SSNs: bare digit runs are indistinguishable from ids, timestamps
 * and token counts, and redacting those would silently destroy eval cases.
 */
export const PII_PATTERNS: readonly DetectorPattern[] = [
  /** Email addresses. */
  {
    name: "email",
    regex: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,}\b/g,
  },
  /** E.164: a leading `+` and 8-15 digits. */
  { name: "phone_e164", regex: /\+[1-9]\d{7,14}(?!\d)/g },
  /** Separated NANP-style numbers, optional country code: 555-867-5309. */
  {
    name: "phone_separated",
    regex: /(?<![\d.\-/])(?:\+?1[ .-])?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?![\d.\-/])/g,
  },
  /** 13-19 digit runs (optionally space/dash grouped) that pass Luhn. */
  {
    name: "credit_card",
    regex: /(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])/g,
    validate: passesLuhn,
  },
  /** US SSN shape; the dashes are required, for the reason above. */
  { name: "ssn_us", regex: /(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)/g },
];

function mergeMatches(matches: DetectorMatch[]): DetectorMatch[] {
  if (matches.length <= 1) return matches;
  const sorted = [...matches].sort((a, b) => a.start - b.start || b.end - a.end);
  const merged: DetectorMatch[] = [];
  for (const match of sorted) {
    const last = merged[merged.length - 1];
    if (last !== undefined && match.start <= last.end) {
      if (match.end > last.end) merged[merged.length - 1] = { start: last.start, end: match.end };
      continue;
    }
    merged.push(match);
  }
  return merged;
}

/**
 * Every region of `value` matched by any of `patterns`, merged into disjoint
 * ascending intervals. Zero-length matches are ignored (a pattern that can
 * match empty would otherwise mark the whole string).
 */
export function findMatches(patterns: readonly DetectorPattern[], value: string): DetectorMatch[] {
  const found: DetectorMatch[] = [];
  for (const pattern of patterns) {
    pattern.regex.lastIndex = 0;
    let match = pattern.regex.exec(value);
    while (match !== null) {
      const text = match[0];
      if (text.length === 0) {
        pattern.regex.lastIndex += 1;
      } else if (pattern.validate === undefined || pattern.validate(text)) {
        found.push({ start: match.index, end: match.index + text.length });
      }
      match = pattern.regex.exec(value);
    }
  }
  return mergeMatches(found);
}

/** Replace each interval with `marker`, keeping every byte in between. */
export function applyMatches(
  value: string,
  matches: readonly DetectorMatch[],
  marker: string,
): string {
  if (matches.length === 0) return value;
  let out = "";
  let cursor = 0;
  for (const match of matches) {
    out += value.slice(cursor, match.start) + marker;
    cursor = match.end;
  }
  return out + value.slice(cursor);
}
