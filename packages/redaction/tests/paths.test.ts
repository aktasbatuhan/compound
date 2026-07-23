import { describe, expect, test } from "bun:test";
import type { PathSegment } from "../src/index";
import {
  formatPath,
  index,
  key,
  matchesPath,
  matchesPathOrAncestor,
  PathPatternError,
  parsePathPattern,
} from "../src/index";

/** `steps[2].input[0].content` -> segments. */
function path(source: string): PathSegment[] {
  const segments: PathSegment[] = [];
  for (const raw of source.split(".")) {
    const bracket = raw.indexOf("[");
    const head = bracket === -1 ? raw : raw.slice(0, bracket);
    if (head !== "") segments.push(key(head));
    if (bracket === -1) continue;
    for (const match of raw.slice(bracket).matchAll(/\[(\d+)\]/g)) {
      segments.push(index(Number(match[1])));
    }
  }
  return segments;
}

function matches(pattern: string, concrete: string): boolean {
  return matchesPath(parsePathPattern(pattern), path(concrete));
}

describe("formatPath", () => {
  test("renders keys and indices the way Redaction.path does", () => {
    expect(formatPath(path("steps[2].input[0].content"))).toBe("steps[2].input[0].content");
    expect(formatPath([])).toBe("");
    expect(formatPath([key("metadata"), key("customer.email")])).toBe("metadata.customer.email");
  });
});

describe("literal patterns", () => {
  test("match exactly", () => {
    expect(matches("metadata.customer", "metadata.customer")).toBe(true);
    expect(matches("metadata.customer", "metadata.other")).toBe(false);
  });

  test("do not match a prefix or an extension of the path", () => {
    expect(matches("steps[0].input", "steps[0]")).toBe(false);
    expect(matches("steps[0].input", "steps[0].input[0]")).toBe(false);
  });

  test("literal indices match only that index", () => {
    expect(matches("steps[2].input", "steps[2].input")).toBe(true);
    expect(matches("steps[2].input", "steps[3].input")).toBe(false);
  });
});

describe("* wildcard", () => {
  test("[*] matches exactly one index", () => {
    expect(matches("steps[*].input", "steps[0].input")).toBe(true);
    expect(matches("steps[*].input", "steps[17].input")).toBe(true);
    expect(matches("steps[*].input", "steps[0].input[1]")).toBe(false);
  });

  test("* matches exactly one key and not an index", () => {
    expect(matches("metadata.*", "metadata.email")).toBe(true);
    expect(matches("metadata.*", "metadata.customer.email")).toBe(false);
    expect(matches("steps.*", "steps[0]")).toBe(false);
  });

  test("chained index wildcards", () => {
    expect(matches("steps[*].input[*].content", "steps[3].input[1].content")).toBe(true);
    expect(matches("steps[*].input[*].content", "steps[3].output.content")).toBe(false);
  });
});

describe("** wildcard", () => {
  test("matches zero segments", () => {
    expect(matches("metadata.**", "metadata")).toBe(true);
  });

  test("matches any depth of keys and indices", () => {
    expect(matches("metadata.**", "metadata.customer.contact.email")).toBe(true);
    expect(matches("steps.**", "steps[1].input[0].content")).toBe(true);
  });

  test("is anchored by what follows it", () => {
    expect(matches("steps.**.content", "steps[1].input[0].content")).toBe(true);
    expect(matches("steps.**.content", "steps[1].input[0].content.text")).toBe(false);
    expect(matches("**", "steps[1].input")).toBe(true);
  });

  test("does not escape its prefix", () => {
    expect(matches("metadata.**", "steps[0].input")).toBe(false);
  });
});

describe("matchesPathOrAncestor", () => {
  test("true for the node itself and for any ancestor", () => {
    const pattern = parsePathPattern("metadata.customer");
    expect(matchesPathOrAncestor(pattern, path("metadata.customer"))).toBe(true);
    expect(matchesPathOrAncestor(pattern, path("metadata.customer.email"))).toBe(true);
    expect(matchesPathOrAncestor(pattern, path("metadata.other"))).toBe(false);
  });
});

describe("parse errors", () => {
  test("reject malformed sources, naming the pattern", () => {
    expect(() => parsePathPattern("")).toThrow(PathPatternError);
    expect(() => parsePathPattern("steps..input")).toThrow(PathPatternError);
    expect(() => parsePathPattern("steps[x]")).toThrow(/steps\[x\]/);
    expect(() => parsePathPattern("steps[0")).toThrow(PathPatternError);
    expect(() => parsePathPattern("ste*ps")).toThrow(/partial wildcards/);
    expect(() => parsePathPattern("[0]")).toThrow(PathPatternError);
  });
});
