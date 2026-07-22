import { readFileSync } from "node:fs";
import { join } from "node:path";
import { type CompoundConfig, loadConfig } from "@compound/config";
import { validate } from "@compound/contract";
import {
  type CompoundDatabase,
  createDatabase,
  createImportBatch,
  insertTraces,
  migrate,
  traceRecordFromValidation,
} from "@compound/storage";
import { createApp } from "../src/app";

const REPO_ROOT = join(import.meta.dir, "..", "..", "..");
const CONTRACT_FIXTURES = join(REPO_ROOT, "packages", "contract", "fixtures");

export function loadContractFixture(name: string): unknown {
  return JSON.parse(readFileSync(join(CONTRACT_FIXTURES, `${name}.json`), "utf8"));
}

/** The repository's real compound.yaml, so tests exercise a config we ship. */
export function loadRealConfig(): CompoundConfig {
  return loadConfig(join(REPO_ROOT, "compound.yaml"));
}

export function freshDatabase(): CompoundDatabase {
  const db = createDatabase();
  migrate(db);
  return db;
}

export interface SeedOptions {
  /** Fixture name -> how many copies, each given a distinct trace_id. */
  fixtures?: Array<{ name: string; count?: number; taskKey?: string | null }>;
}

/** Seed a batch of traces derived from the shared contract fixtures. */
export function seedTraces(db: CompoundDatabase, options: SeedOptions = {}) {
  const batch = createImportBatch(db, {
    importer: "test",
    importerVersion: "0",
    sourceFingerprint: "test-fingerprint",
  });

  const records = [];
  for (const spec of options.fixtures ?? []) {
    for (let index = 0; index < (spec.count ?? 1); index += 1) {
      const raw = loadContractFixture(spec.name) as Record<string, unknown>;
      raw.trace_id = `${spec.name}-${index}`;
      if (spec.taskKey !== undefined) raw.task_key = spec.taskKey;
      const record = traceRecordFromValidation(validate(raw), `hash-${spec.name}-${index}`);
      if (record === null) {
        throw new Error(`fixture ${spec.name} is rejected; cannot seed`);
      }
      records.push(record);
    }
  }
  if (records.length > 0) insertTraces(db, batch.id, records);
  return batch;
}

export function testApp(db: CompoundDatabase, config = loadRealConfig()) {
  return createApp({ db, config });
}

export async function getJson(
  app: ReturnType<typeof createApp>,
  path: string,
): Promise<{ status: number; body: any }> {
  const response = await app.request(`http://localhost${path}`);
  return { status: response.status, body: await response.json() };
}

export async function postJson(
  app: ReturnType<typeof createApp>,
  path: string,
  body: unknown,
  raw?: string,
): Promise<{ status: number; body: any }> {
  const response = await app.request(`http://localhost${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: raw ?? JSON.stringify(body),
  });
  return { status: response.status, body: await response.json() };
}
