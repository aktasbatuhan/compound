import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

export const FIXTURES_DIR = join(import.meta.dir, "..", "fixtures");

export function loadFixture(name: string): unknown {
  return JSON.parse(readFileSync(join(FIXTURES_DIR, name), "utf8"));
}

export function listFixtureNames(): string[] {
  return readdirSync(FIXTURES_DIR)
    .filter((name) => name.endsWith(".json"))
    .sort();
}

/** Deep-clone a fixture so tests can mutate copies safely. */
export function cloneFixture<T>(value: T): T {
  return structuredClone(value);
}
