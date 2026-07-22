/**
 * Generates the committed JSON Schema artifact for `compound.yaml`:
 * packages/config/schema/compound.config.v1.schema.json
 *
 * The artifact lets non-TS consumers (the Python engine, via
 * `compound.config_schema`) validate the same file without importing this
 * package. Run with:
 *   bun run generate:schema
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { z } from "zod";
import { CompoundConfigSchema } from "../src/schemas";

export function buildJsonSchema(): Record<string, unknown> {
  const generated = z.toJSONSchema(CompoundConfigSchema, { target: "draft-2020-12" });
  return {
    $schema: "https://json-schema.org/draft/2020-12/schema",
    $id: "https://compound.local/schemas/compound.config.v1.schema.json",
    title: "Compound compound.yaml schema v1",
    description:
      "The single source of truth for a Compound workspace: benchmark engine " +
      "sections plus the product sections (task_keys, redaction, ingest). " +
      "Generated from @compound/config zod schemas; do not edit by hand.",
    ...generated,
  };
}

if (import.meta.main) {
  const outPath = join(import.meta.dir, "..", "schema", "compound.config.v1.schema.json");
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, `${JSON.stringify(buildJsonSchema(), null, 2)}\n`);
  console.log(`wrote ${outPath}`);
}
