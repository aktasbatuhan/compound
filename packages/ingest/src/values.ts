/**
 * Small value helpers shared by the mapping modules.
 *
 * Langfuse ships the same field under two casings (API camelCase, blob
 * snake_case), so every read goes through {@link field}, which accepts the
 * aliases in order.
 */

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** First defined value among the given key aliases (`traceId` then `trace_id`, …). */
export function field(record: Record<string, unknown>, ...names: string[]): unknown {
  for (const name of names) {
    const value = record[name];
    if (value !== undefined) return value;
  }
  return undefined;
}

export function asString(value: unknown): string | null {
  return typeof value === "string" && value !== "" ? value : null;
}

export function asFiniteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** String members of an array value; non-string members are dropped. */
export function asStringArray(value: unknown): string[] | null {
  if (!Array.isArray(value)) return null;
  return value.filter((entry): entry is string => typeof entry === "string");
}

/**
 * Langfuse metadata is an object since v3 (scalars are wrapped as
 * `{"metadata": value}`); older exports may still carry a bare scalar, which we
 * wrap the same way rather than dropping it.
 */
export function asMetadataObject(value: unknown): Record<string, unknown> | null {
  if (isRecord(value)) return { ...value };
  if (value === undefined || value === null) return null;
  return { metadata: value };
}

const BLOB_TIMESTAMP = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?$/;
const HAS_ZONE = /(Z|z|[+-]\d{2}:?\d{2})$/;

/**
 * Normalize a Langfuse timestamp to a Z-terminated ISO-8601 string.
 *
 * Accepts ISO-8601 (with or without zone) and the blob text format
 * `YYYY-MM-DD HH:MM:SS.ffffff`, which the reference documents as UTC. A value
 * with no zone marker is read as UTC, never as local time. Returns `null` when
 * the value is not a parseable timestamp.
 */
export function toIsoUtc(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (trimmed === "") return null;
  let candidate = BLOB_TIMESTAMP.test(trimmed) ? trimmed.replace(" ", "T") : trimmed;
  if (!HAS_ZONE.test(candidate)) candidate = `${candidate}Z`;
  const ms = Date.parse(candidate);
  if (Number.isNaN(ms)) return null;
  return new Date(ms).toISOString();
}

/** Earliest of the given ISO timestamps, or `null` when there are none. */
export function earliest(values: (string | null)[]): string | null {
  let best: string | null = null;
  for (const value of values) {
    if (value === null) continue;
    if (best === null || value < best) best = value;
  }
  return best;
}

/** Latest of the given ISO timestamps, or `null` when there are none. */
export function latest(values: (string | null)[]): string | null {
  let best: string | null = null;
  for (const value of values) {
    if (value === null) continue;
    if (best === null || value > best) best = value;
  }
  return best;
}
