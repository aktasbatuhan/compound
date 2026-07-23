import { readFileSync } from "node:fs";
import { join } from "node:path";
import { TraceSchema } from "@compound/contract";
import type { NormalizedTrace, NormalizeOptions } from "../src/index";
import { normalizeLangfuseExport } from "../src/index";

const FIXTURES = join(import.meta.dir, "..", "fixtures");

export function loadFixture(name: string): string {
  return readFileSync(join(FIXTURES, name), "utf8");
}

export const OPTIONS: NormalizeOptions = {
  defaultPermissions: { judging: true, optimization: true, fine_tuning: false },
  importerVersion: "0.1.0",
  projectId: "proj-main",
  exportedAt: "2026-07-22T12:00:00.000Z",
};

export function normalizeFixture(
  name: string,
  overrides: Partial<NormalizeOptions> = {},
): ReturnType<typeof normalizeLangfuseExport> {
  return normalizeLangfuseExport(loadFixture(name), { ...OPTIONS, ...overrides });
}

/** Every trace this package emits must be schema-valid; tests assert it everywhere. */
export function expectContractValid(traces: NormalizedTrace[]): void {
  for (const normalized of traces) {
    const parsed = TraceSchema.safeParse(normalized.trace);
    if (!parsed.success) {
      throw new Error(
        `trace ${normalized.trace.trace_id} is not contract-valid: ${JSON.stringify(parsed.error.issues, null, 2)}`,
      );
    }
  }
}

/** One trace from an inline record array, for focused cases. */
export function normalizeRecords(
  records: unknown[],
  overrides: Partial<NormalizeOptions> = {},
): ReturnType<typeof normalizeLangfuseExport> {
  return normalizeLangfuseExport(records, { ...OPTIONS, ...overrides });
}

interface GenerationOverrides {
  id?: string;
  input?: unknown;
  output?: unknown;
  usage?: unknown;
  usageDetails?: unknown;
  costDetails?: unknown;
  startTime?: string;
  endTime?: string;
}

/** Minimal camelCase trace + one GENERATION, for single-behaviour tests. */
export function traceWithGeneration(overrides: GenerationOverrides = {}): unknown[] {
  const observation: Record<string, unknown> = {
    id: overrides.id ?? "gen-1",
    traceId: "tr-inline",
    parentObservationId: null,
    type: "GENERATION",
    name: "generation",
    startTime: overrides.startTime ?? "2026-07-22T10:00:00.000Z",
    endTime: overrides.endTime ?? "2026-07-22T10:00:01.000Z",
    model: "gpt-4.1-mini",
    modelParameters: {},
    input: "input" in overrides ? overrides.input : [{ role: "user", content: "hi" }],
    output: "output" in overrides ? overrides.output : { role: "assistant", content: "hello" },
    level: "DEFAULT",
    metadata: {},
    environment: "production",
  };
  if (overrides.usage !== undefined) observation.usage = overrides.usage;
  if (overrides.usageDetails !== undefined) observation.usageDetails = overrides.usageDetails;
  if (overrides.costDetails !== undefined) observation.costDetails = overrides.costDetails;

  return [
    {
      id: "tr-inline",
      timestamp: "2026-07-22T10:00:00.000Z",
      environment: "production",
      public: false,
      metadata: {},
      tags: [],
      observations: [observation],
      scores: [],
    },
  ];
}
