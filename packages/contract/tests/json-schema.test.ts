/**
 * Proves the two contract artifacts agree:
 * - every fixture the zod validator accepts (eval_ready or diagnostic) must
 *   also validate against the committed JSON Schema;
 * - every fixture the zod validator rejects must fail the JSON Schema too;
 * - the committed schema file is byte-identical to a fresh generation.
 */

import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { Ajv2020 } from "ajv/dist/2020";
import addFormats from "ajv-formats";
import { buildJsonSchema } from "../scripts/generate-schema";
import { validate } from "../src/index";
import { listFixtureNames, loadFixture } from "./fixtures";

const SCHEMA_PATH = join(import.meta.dir, "..", "schema", "compound.trace.v1.schema.json");
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
    expect(committedSchema.$id).toContain("compound.trace.v1");
  });
});

describe("round-trip: zod and JSON Schema agree on every fixture", () => {
  for (const name of listFixtureNames()) {
    test(name, () => {
      const fixture = loadFixture(name);
      const result = validate(fixture);
      const jsonSchemaValid = validateJsonSchema(fixture);
      if (result.class === "rejected") {
        expect(jsonSchemaValid).toBe(false);
      } else {
        expect(jsonSchemaValid).toBe(true);
      }
    });
  }

  test("fixture set covers all three validation classes", () => {
    const classes = new Set(listFixtureNames().map((name) => validate(loadFixture(name)).class));
    expect(classes).toEqual(new Set(["eval_ready", "diagnostic", "rejected"]));
  });
});
