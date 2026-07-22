/**
 * Proves the two config artifacts agree:
 * - the committed JSON Schema is byte-identical to a fresh generation from zod;
 * - every case zod accepts also validates against the JSON Schema, and every
 *   case zod rejects also fails it (so the Python engine enforces the same
 *   rules).
 */
import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { Ajv2020 } from "ajv/dist/2020";
import addFormats from "ajv-formats";
import { parse as parseYaml } from "yaml";
import { buildJsonSchema } from "../scripts/generate-schema";
import { validateConfig } from "../src/index";
import { invalidCases, validCases } from "./cases";

const SCHEMA_PATH = join(import.meta.dir, "..", "schema", "compound.config.v1.schema.json");
const committedSchema = JSON.parse(readFileSync(SCHEMA_PATH, "utf8"));

const ajv = new Ajv2020({ strict: false, allErrors: true });
addFormats(ajv);
const validateJsonSchema = ajv.compile(committedSchema);

describe("committed JSON Schema artifact", () => {
  test("matches a fresh generation from the zod schemas", () => {
    expect(committedSchema).toEqual(JSON.parse(JSON.stringify(buildJsonSchema())));
  });

  test("declares the 2020-12 dialect and a stable $id", () => {
    expect(committedSchema.$schema).toBe("https://json-schema.org/draft/2020-12/schema");
    expect(committedSchema.$id).toContain("compound.config.v1");
  });
});

describe("round-trip: zod and JSON Schema agree", () => {
  test("the repository's real compound.yaml", () => {
    const real = parseYaml(
      readFileSync(join(import.meta.dir, "..", "..", "..", "compound.yaml"), "utf8"),
    );
    expect(validateConfig(real).ok).toBe(true);
    expect(validateJsonSchema(real)).toBe(true);
  });

  for (const [name, config] of Object.entries(validCases)) {
    test(`accepts: ${name}`, () => {
      expect(validateConfig(config).ok).toBe(true);
      expect(validateJsonSchema(config)).toBe(true);
    });
  }

  for (const [name, config] of Object.entries(invalidCases)) {
    test(`rejects: ${name}`, () => {
      expect(validateConfig(config).ok).toBe(false);
      expect(validateJsonSchema(config)).toBe(false);
    });
  }
});
