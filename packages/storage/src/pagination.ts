/** Shared limit/offset handling for the repositories. */

export interface Pagination {
  limit?: number;
  offset?: number;
}

interface Paginatable<T> {
  limit(value: number): T;
  offset(value: number): T;
}

/** SQLite rejects `OFFSET` without `LIMIT`; this stands in for "no limit". */
const UNBOUNDED_LIMIT = Number.MAX_SAFE_INTEGER;

/** Apply limit/offset to a dynamic drizzle query. */
export function paginate<T extends Paginatable<T>>(query: T, page: Pagination): T {
  if (page.limit === undefined && page.offset === undefined) return query;
  const limited = query.limit(page.limit ?? UNBOUNDED_LIMIT);
  return page.offset === undefined ? limited : limited.offset(page.offset);
}
