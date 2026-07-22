import { readFileSync } from "node:fs";
import { join } from "node:path";
import { type Trace, validate } from "@compound/contract";
import {
  type CompoundDatabase,
  createDatabase,
  createImportBatch,
  type TraceRecordInput,
  traceRecordFromValidation,
} from "../src/index";

/** Contract fixtures are the shared source of trace JSON; storage invents none. */
export const CONTRACT_FIXTURES_DIR = join(import.meta.dir, "..", "..", "contract", "fixtures");

export function loadContractFixture(name: string): unknown {
  return JSON.parse(readFileSync(join(CONTRACT_FIXTURES_DIR, name), "utf8"));
}

/** A migrated in-memory database. */
export function freshDatabase(): CompoundDatabase {
  return createDatabase({ path: ":memory:", migrate: true });
}

export function newBatch(handle: CompoundDatabase, overrides: { id?: string } = {}) {
  return createImportBatch(handle, {
    importer: "langfuse",
    importerVersion: "0.1.0",
    sourceFingerprint: "sha256:fixture",
    ...overrides,
  });
}

/**
 * Build a persistable record from a contract fixture, optionally patching the
 * trace first (e.g. to give it a distinct trace_id or task_key).
 */
export function recordFromFixture(
  name: string,
  patch: (trace: Record<string, unknown>) => void = () => {},
  contentHash = `hash:${name}`,
): TraceRecordInput {
  const raw = loadContractFixture(name) as Record<string, unknown>;
  patch(raw);
  const result = validate(raw);
  const record = traceRecordFromValidation(result, contentHash);
  if (record === null) throw new Error(`fixture ${name} is rejected and cannot be persisted`);
  return record;
}

export function fixtureTrace(name: string): Trace {
  const result = validate(loadContractFixture(name));
  if (result.class === "rejected") throw new Error(`fixture ${name} is rejected`);
  return result.trace;
}
