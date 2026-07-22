/**
 * Query-parameter parsing shared by list routes.
 *
 * Every parse failure is an `invalid_request` with the offending parameter
 * named. Nothing is coerced silently: `limit=banana` is an error, not 50.
 */
import { invalidRequest } from "./errors";

export const DEFAULT_LIMIT = 50;
export const MAX_LIMIT = 500;

export interface PageParams {
  limit: number;
  offset: number;
}

function parseInteger(name: string, raw: string): number {
  if (!/^-?\d+$/.test(raw)) {
    throw invalidRequest(`${name} must be an integer`, { parameter: name, value: raw });
  }
  return Number.parseInt(raw, 10);
}

/** `limit` is bounded so no caller can request an unbounded result set. */
export function parsePageParams(query: URLSearchParams): PageParams {
  const rawLimit = query.get("limit");
  const rawOffset = query.get("offset");

  let limit = DEFAULT_LIMIT;
  if (rawLimit !== null) {
    limit = parseInteger("limit", rawLimit);
    if (limit < 1 || limit > MAX_LIMIT) {
      throw invalidRequest(`limit must be between 1 and ${MAX_LIMIT}`, {
        parameter: "limit",
        value: rawLimit,
      });
    }
  }

  let offset = 0;
  if (rawOffset !== null) {
    offset = parseInteger("offset", rawOffset);
    if (offset < 0) {
      throw invalidRequest("offset must be zero or greater", {
        parameter: "offset",
        value: rawOffset,
      });
    }
  }

  return { limit, offset };
}

/**
 * Parse a value constrained to a fixed set, e.g. `validation_class`.
 * Returns `undefined` when the parameter is absent (meaning "no filter").
 */
export function parseEnumParam<T extends string>(
  query: URLSearchParams,
  name: string,
  allowed: readonly T[],
): T | undefined {
  const raw = query.get(name);
  if (raw === null) return undefined;
  if (!(allowed as readonly string[]).includes(raw)) {
    throw invalidRequest(`${name} must be one of: ${allowed.join(", ")}`, {
      parameter: name,
      value: raw,
    });
  }
  return raw as T;
}

/** Parse an ISO-8601 timestamp parameter. */
export function parseDateParam(query: URLSearchParams, name: string): Date | undefined {
  const raw = query.get(name);
  if (raw === null) return undefined;
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) {
    throw invalidRequest(`${name} must be an ISO-8601 timestamp`, { parameter: name, value: raw });
  }
  return parsed;
}

/**
 * Parse the `task_key` filter.
 *
 * Absent means "no filter"; the literal `unassigned` means the null bucket;
 * anything else is an exact task key. This mirrors the storage layer, where
 * `undefined` and `null` mean different things.
 */
export function parseTaskKeyParam(query: URLSearchParams): string | null | undefined {
  const raw = query.get("task_key");
  if (raw === null) return undefined;
  if (raw === "unassigned") return null;
  if (raw.length === 0) {
    throw invalidRequest("task_key must not be empty; use 'unassigned' for the null bucket", {
      parameter: "task_key",
    });
  }
  return raw;
}
