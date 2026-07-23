/**
 * Reading values out of a model output message for assertions.
 *
 * Everything here is total: a missing target returns a typed "absent" rather
 * than throwing, so an assertion can turn it into a clean failure with a
 * reason instead of a crash.
 */
import type { Message } from "@compound/contract";
import type { AssertionSubject } from "./types";

/** The plain text of an output message: string content, or joined text parts. */
export function outputText(subject: AssertionSubject): string | null {
  const output = subject.output;
  if (output === null) return null;
  const content = output.content;
  if (content === null) return null;
  if (typeof content === "string") return content;
  // Content parts: concatenate text parts; redacted/unsupported contribute
  // nothing, since their text is deliberately absent.
  return content.map((part) => (part.type === "text" ? part.text : "")).join("");
}

export type Absent = { readonly absent: true; readonly reason: string };
export function isAbsent(value: unknown): value is Absent {
  return typeof value === "object" && value !== null && (value as Absent).absent === true;
}
function absent(reason: string): Absent {
  return { absent: true, reason };
}

/** Parse the output text as JSON, or explain why it could not be read. */
export function parsedOutput(subject: AssertionSubject): unknown | Absent {
  const text = outputText(subject);
  if (text === null) return absent("output has no text content");
  try {
    return JSON.parse(text);
  } catch {
    return absent("output text is not valid JSON");
  }
}

/**
 * Resolve a `path` against an output.
 *
 * `content` (or omitted) → the output text. Any other path is a dotted path
 * into the JSON-parsed output text, e.g. `data.items.0.name`. Numeric segments
 * index arrays.
 */
export function resolvePath(subject: AssertionSubject, path: string | undefined): unknown | Absent {
  if (path === undefined || path === "content") {
    const text = outputText(subject);
    return text === null ? absent("output has no text content") : text;
  }

  const parsed = parsedOutput(subject);
  if (isAbsent(parsed)) return parsed;

  let current: unknown = parsed;
  for (const segment of path.split(".")) {
    if (Array.isArray(current)) {
      const index = Number(segment);
      if (!Number.isInteger(index) || index < 0 || index >= current.length) {
        return absent(`path segment '${segment}' is out of range`);
      }
      current = current[index];
      continue;
    }
    if (current !== null && typeof current === "object" && segment in current) {
      current = (current as Record<string, unknown>)[segment];
      continue;
    }
    return absent(`path segment '${segment}' not found`);
  }
  return current;
}

/** Tool calls in the output, or an empty array when there are none. */
export function outputToolCalls(subject: AssertionSubject): NonNullable<Message["tool_calls"]> {
  return subject.output?.tool_calls ?? [];
}
