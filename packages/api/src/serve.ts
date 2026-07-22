/**
 * Thin serving entry point around `createApp`.
 *
 * Binds to loopback by default: the local API has no auth in v1, so it must not
 * be reachable off-machine. Exposing a self-hosted deployment is a deliberate
 * later decision that ships together with an auth strategy.
 */
import { loadConfig } from "@compound/config";
import { createDatabase, migrate } from "@compound/storage";
import { createApp } from "./app";

export interface ServeOptions {
  port?: number;
  hostname?: string;
  databasePath?: string;
  configPath?: string;
}

export const DEFAULT_PORT = 4319;
export const DEFAULT_HOSTNAME = "127.0.0.1";
export const DEFAULT_DATABASE_PATH = "compound.db";

export function startServer(options: ServeOptions = {}) {
  const config = loadConfig(options.configPath ?? "compound.yaml");
  const db = createDatabase({ path: options.databasePath ?? DEFAULT_DATABASE_PATH });
  // Migrations are idempotent, so applying them on every boot is safe and keeps
  // a local-first install from ever needing a manual migration step.
  migrate(db);

  const app = createApp({ db, config });
  return Bun.serve({
    port: options.port ?? DEFAULT_PORT,
    hostname: options.hostname ?? DEFAULT_HOSTNAME,
    fetch: app.fetch,
  });
}

if (import.meta.main) {
  const server = startServer();
  console.log(`compound api listening on http://${server.hostname}:${server.port}`);
}
