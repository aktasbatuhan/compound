/**
 * Connection and migration helpers for the Compound local store.
 *
 * The local-first CLI (`bunx compound dev`) opens a file-backed database and
 * calls `migrate()` on boot; tests open `:memory:`. Migrations are applied by
 * drizzle's migrator, which records applied hashes in `__drizzle_migrations`
 * and is therefore safe to re-run.
 */
import { Database } from "bun:sqlite";
import { dirname, join } from "node:path";
import { drizzle } from "drizzle-orm/bun-sqlite";
import { migrate as drizzleMigrate } from "drizzle-orm/bun-sqlite/migrator";
import * as schema from "./schema";

/** A connected Compound database: the drizzle handle plus the raw sqlite one. */
export interface CompoundDatabase {
  db: ReturnType<typeof drizzle<typeof schema>>;
  sqlite: Database;
  /** Close the underlying sqlite connection. */
  close(): void;
}

export interface CreateDatabaseOptions {
  /** File path, or `:memory:` for an ephemeral database. Defaults to `:memory:`. */
  path?: string;
  /**
   * Apply pending migrations immediately after connecting. Defaults to `false`
   * so callers stay in control of boot ordering.
   */
  migrate?: boolean;
}

/** Directory holding the committed, generated SQL migrations. */
export const MIGRATIONS_DIR = join(dirname(import.meta.dir), "drizzle");

/**
 * Open a database. `path` may be a file path or `:memory:`.
 *
 * Foreign keys are enabled (SQLite defaults them off) so the
 * `traces.import_batch_id` reference is actually enforced, and WAL is enabled
 * for file-backed databases so reads do not block the importer's writes.
 */
export function createDatabase(options: CreateDatabaseOptions = {}): CompoundDatabase {
  const path = options.path ?? ":memory:";
  const sqlite = new Database(path, { create: true });
  sqlite.exec("PRAGMA foreign_keys = ON;");
  if (path !== ":memory:") {
    sqlite.exec("PRAGMA journal_mode = WAL;");
  }
  const db = drizzle(sqlite, { schema });
  const handle: CompoundDatabase = {
    db,
    sqlite,
    close: () => sqlite.close(),
  };
  if (options.migrate === true) {
    migrate(handle);
  }
  return handle;
}

/**
 * Apply all pending migrations. Idempotent: re-running applies nothing and
 * leaves existing data untouched.
 */
export function migrate(handle: CompoundDatabase, migrationsFolder = MIGRATIONS_DIR): void {
  drizzleMigrate(handle.db, { migrationsFolder });
}

export { schema };
