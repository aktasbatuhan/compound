/**
 * Generates the committed JSON Schema artifact for the trace contract:
 * packages/contract/schema/compound.trace.v1.schema.json
 *
 * The artifact lets non-TS consumers (the Python engine) validate traces
 * without importing this package. Run with:
 *   bun run generate:schema
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { z } from "zod";
import { TraceSchema } from "../src/schemas";

export function buildJsonSchema(): Record<string, unknown> {
  const generated = z.toJSONSchema(TraceSchema, { target: "draft-2020-12" });
  return {
    $schema: "https://json-schema.org/draft/2020-12/schema",
    $id: "https://compound.local/schemas/compound.trace.v1.schema.json",
    title: "Compound portable trace contract v1",
    description:
      "One trace record of the Compound portable trace contract. " +
      "Source of truth: docs/trace-contract-v1.md. " +
      "Generated from @compound/contract zod schemas; do not edit by hand.",
    ...generated,
  };
}

if (import.meta.main) {
  const outPath = join(import.meta.dir, "..", "schema", "compound.trace.v1.schema.json");
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, `${JSON.stringify(buildJsonSchema(), null, 2)}\n`);
  console.log(`wrote ${outPath}`);
}
