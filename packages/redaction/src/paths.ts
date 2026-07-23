/**
 * Trace path addressing and glob matching for `redaction.rules[].applies_to`
 * and `redaction.field_allowlist`.
 *
 * A *concrete* path is the address of one node inside a trace, rendered exactly
 * the way `Redaction.path` is rendered in the contract:
 * `steps[2].input[0].content`. Object keys are dot-separated; array elements are
 * bracketed indices appended to the segment that owns them.
 *
 * A *pattern* is the same syntax with three wildcards:
 *
 * - `*` in key position matches exactly one object key (`metadata.*`).
 * - `[*]` matches exactly one array index (`steps[*]`).
 * - `**` matches zero or more segments of either kind (`metadata.**` matches
 *   `metadata` itself and everything beneath it, at any depth).
 *
 * Patterns are parsed once (at config-compile time) and matched against every
 * node the walker visits; nothing here touches trace values.
 */

/** One segment of a concrete path: an object key or an array index. */
export type PathSegment = { kind: "key"; name: string } | { kind: "index"; index: number };

/** Thrown for a syntactically invalid path pattern. Names the offending source. */
export class PathPatternError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PathPatternError";
  }
}

type PatternToken =
  | { kind: "key"; name: string }
  | { kind: "anyKey" }
  | { kind: "index"; index: number }
  | { kind: "anyIndex" }
  | { kind: "anyDepth" };

/** A parsed `applies_to` / `field_allowlist` entry. */
export interface PathPattern {
  /** The original source string, kept for error messages. */
  readonly source: string;
  readonly tokens: readonly PatternToken[];
}

export function key(name: string): PathSegment {
  return { kind: "key", name };
}

export function index(i: number): PathSegment {
  return { kind: "index", index: i };
}

/** Render a concrete path the way `Redaction.path` is rendered. */
export function formatPath(segments: readonly PathSegment[]): string {
  let out = "";
  for (const segment of segments) {
    if (segment.kind === "index") {
      out += `[${segment.index}]`;
      continue;
    }
    out = out === "" ? segment.name : `${out}.${segment.name}`;
  }
  return out;
}

const BRACKETS = /\[([^\]]*)\]/g;

function parseBrackets(source: string, raw: string, rest: string, tokens: PatternToken[]): void {
  let consumed = 0;
  BRACKETS.lastIndex = 0;
  let match = BRACKETS.exec(rest);
  while (match !== null) {
    if (match.index !== consumed) {
      throw new PathPatternError(`invalid segment "${raw}" in path pattern "${source}"`);
    }
    consumed = match.index + match[0].length;
    const body = match[1] ?? "";
    if (body === "*") {
      tokens.push({ kind: "anyIndex" });
    } else if (/^\d+$/.test(body)) {
      tokens.push({ kind: "index", index: Number(body) });
    } else {
      throw new PathPatternError(
        `array index must be a non-negative integer or "*", got "[${body}]" in path pattern "${source}"`,
      );
    }
    match = BRACKETS.exec(rest);
  }
  if (consumed !== rest.length) {
    throw new PathPatternError(`invalid segment "${raw}" in path pattern "${source}"`);
  }
}

/** Parse a path pattern. Throws `PathPatternError` on invalid syntax. */
export function parsePathPattern(source: string): PathPattern {
  if (source.trim() === "") {
    throw new PathPatternError("path pattern must not be empty");
  }
  const tokens: PatternToken[] = [];
  for (const raw of source.split(".")) {
    if (raw === "") {
      throw new PathPatternError(`empty segment in path pattern "${source}"`);
    }
    const bracketStart = raw.indexOf("[");
    const head = bracketStart === -1 ? raw : raw.slice(0, bracketStart);
    if (head === "") {
      throw new PathPatternError(`segment "${raw}" in path pattern "${source}" has no key`);
    }
    if (head === "**") {
      tokens.push({ kind: "anyDepth" });
    } else if (head === "*") {
      tokens.push({ kind: "anyKey" });
    } else if (head.includes("*")) {
      throw new PathPatternError(
        `partial wildcards are not supported: "${head}" in path pattern "${source}"`,
      );
    } else {
      tokens.push({ kind: "key", name: head });
    }
    if (bracketStart !== -1) {
      parseBrackets(source, raw, raw.slice(bracketStart), tokens);
    }
  }
  return { source, tokens };
}

function matchFrom(
  tokens: readonly PatternToken[],
  ti: number,
  segments: readonly PathSegment[],
  si: number,
): boolean {
  if (ti === tokens.length) return si === segments.length;
  const token = tokens[ti];
  if (token === undefined) return false;

  if (token.kind === "anyDepth") {
    for (let skip = si; skip <= segments.length; skip += 1) {
      if (matchFrom(tokens, ti + 1, segments, skip)) return true;
    }
    return false;
  }

  const segment = segments[si];
  if (segment === undefined) return false;

  const ok =
    token.kind === "key"
      ? segment.kind === "key" && segment.name === token.name
      : token.kind === "anyKey"
        ? segment.kind === "key"
        : token.kind === "anyIndex"
          ? segment.kind === "index"
          : segment.kind === "index" && segment.index === token.index;

  return ok && matchFrom(tokens, ti + 1, segments, si + 1);
}

/** True when `segments` is exactly the node addressed by `pattern`. */
export function matchesPath(pattern: PathPattern, segments: readonly PathSegment[]): boolean {
  return matchFrom(pattern.tokens, 0, segments, 0);
}

/**
 * True when `segments` — or any of its ancestors — matches `pattern`. Used by
 * `field_allowlist`, where allowlisting a node allowlists its whole subtree.
 */
export function matchesPathOrAncestor(
  pattern: PathPattern,
  segments: readonly PathSegment[],
): boolean {
  for (let end = 0; end <= segments.length; end += 1) {
    if (matchesPath(pattern, segments.slice(0, end))) return true;
  }
  return false;
}
