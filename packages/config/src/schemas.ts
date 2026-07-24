/**
 * Zod schemas for `compound.yaml` v1 — the single source of truth for the whole
 * product (benchmark engine + product sections).
 *
 * Two zones, deliberately different in strictness:
 *
 * - **Benchmark sections** (`version`, `artifacts_dir`, `manifests_dir`, `seed`,
 *   `budget`, `providers`, `models`, `pricing_usd_per_million_tokens`,
 *   `flex_pricing_usd_per_million_tokens`, `benchmarks`, `optimization`,
 *   `gate`) are the Python engine's business. They are validated structurally
 *   (the keys `src/compound/config.py` reads must exist and have the right
 *   types) but stay permissive about extra keys, so the engine can add fields
 *   without a TypeScript release.
 * - **Product sections** (`task_keys`, `redaction`, `ingest`) are owned here and
 *   are strict: unknown keys are errors. All three are optional so a
 *   benchmark-only `compound.yaml` still validates.
 *
 * The replay vocabulary must match `docs/trace-contract-v1.md` exactly; a test
 * asserts it against `@compound/contract`'s `ReplayPolicySchema`.
 */
import { z } from "zod";

// ---------------------------------------------------------------------------
// Benchmark sections (permissive: the Python engine owns these)
// ---------------------------------------------------------------------------

const nonEmptyString = z.string().min(1);

export const BudgetSchema = z.looseObject({
  paid_runs_enabled: z.boolean().optional(),
  hard_limit_usd: z.number().nonnegative().optional(),
  smoke_cases_per_benchmark: z.int().nonnegative().optional(),
});

export const ProviderSchema = z.looseObject({
  base_url: z.url(),
  api_key_env: nonEmptyString,
});

export const ModelEntrySchema = z.looseObject({
  id: nonEmptyString,
  provider: nonEmptyString,
  role: nonEmptyString.optional(),
  /**
   * Which API surface serves this model. `flex` is Doubleword's background
   * Responses route (`background=true, service_tier=flex`) used by the cheap
   * candidates; `chat_completions` is the default OpenAI-compatible route.
   */
  backend: z.enum(["chat_completions", "flex"]).optional(),
});

export const ModelsSchema = z.looseObject({
  frontier: z.array(ModelEntrySchema).optional(),
  candidates: z.array(ModelEntrySchema).optional(),
});

export const TokenPriceSchema = z.looseObject({
  input: z.number().nonnegative(),
  output: z.number().nonnegative(),
});

export const PricingTableSchema = z.record(z.string(), TokenPriceSchema);

/**
 * Partition sizes per benchmark. The sealed-decision firewall lives in the
 * engine; the schema only guarantees the three partitions are non-negative
 * integers when present.
 */
export const PartitionsSchema = z.looseObject({
  optimizer_train: z.int().nonnegative().optional(),
  optimizer_validation: z.int().nonnegative().optional(),
  decision_test: z.int().nonnegative().optional(),
});

export const BenchmarkSchema = z.looseObject({
  task_key: nonEmptyString,
  sample_count: z.int().positive(),
  partitions: PartitionsSchema,
  revision: z.string().optional(),
  package: z.string().optional(),
  strata: z.array(z.string()).optional(),
});

export const OptimizationSchema = z.looseObject({
  algorithm: nonEmptyString.optional(),
  mutable_components: z.array(z.string()).optional(),
  max_metric_calls: z.int().positive().optional(),
  selection: z.string().optional(),
  require_textual_feedback: z.boolean().optional(),
});

export const GateSchema = z.looseObject({
  metric: nonEmptyString.optional(),
  max_regression: z.number().optional(),
  require_decision_test: z.boolean().optional(),
});

// ---------------------------------------------------------------------------
// Product section: task_keys
// ---------------------------------------------------------------------------

/**
 * Replay vocabulary — identical to the trace contract's `ReplayPolicy`.
 * A trace step's `replay_policy: null` inherits the task key's policy here.
 */
export const ReplayPolicySchema = z.enum(["recorded", "mocked", "live_read_only", "blocked"]);

export const TaskReplaySchema = z.strictObject({
  /** Applies to every tool of this task key unless `per_tool` overrides it. */
  default_tool_policy: ReplayPolicySchema,
  /** Tool name -> policy. Overrides `default_tool_policy` for that tool. */
  per_tool: z.record(z.string(), ReplayPolicySchema).optional(),
});

export const TaskKeySchema = z.strictObject({
  description: z.string().optional(),
  replay: TaskReplaySchema,
});

/** Application-supplied task key -> its declared policy. */
export const TaskKeysSchema = z.record(z.string(), TaskKeySchema);

// ---------------------------------------------------------------------------
// Product section: redaction
// ---------------------------------------------------------------------------

/**
 * Where a rule applies: trace field paths, optionally with `*` / `**` globs,
 * e.g. `steps[*].input`, `metadata.**`, `steps[*].output.content`.
 */
export const AppliesToSchema = z.array(nonEmptyString).min(1);

const redactionRuleBase = {
  name: nonEmptyString,
  applies_to: AppliesToSchema,
  /**
   * Replacement written in place of the value. Defaults are derived from the
   * detector (`⟦redacted:secret⟧`, `⟦redacted:pii⟧`,
   * `⟦redacted:custom:<name>⟧`) — see `DEFAULT_REDACTION_MARKERS`.
   */
  marker: nonEmptyString.optional(),
};

/**
 * A redaction rule. Modeled as a discriminated union on `detector` so that
 * `pattern` is required exactly when `detector: regex` and rejected otherwise —
 * a constraint that survives JSON Schema generation.
 */
export const RedactionRuleSchema = z.discriminatedUnion("detector", [
  z.strictObject({ ...redactionRuleBase, detector: z.literal("secret") }),
  z.strictObject({ ...redactionRuleBase, detector: z.literal("pii") }),
  z.strictObject({
    ...redactionRuleBase,
    detector: z.literal("regex"),
    /** JavaScript regular expression source; must compile. */
    pattern: nonEmptyString,
  }),
]);

export const RedactionSchema = z.strictObject({
  rules: z.array(RedactionRuleSchema),
  /**
   * Field paths that survive redaction untouched even if a rule matches them.
   * Escape hatch for known-safe metadata (e.g. `metadata.environment`).
   */
  field_allowlist: z.array(nonEmptyString).optional(),
});

// ---------------------------------------------------------------------------
// Product section: ingest
// ---------------------------------------------------------------------------

/**
 * Usage permissions stamped on imported traces. The trace schema deliberately
 * requires explicit values; this is where the documented default is applied as
 * import-batch policy.
 */
export const PermissionsSchema = z.strictObject({
  judging: z.boolean(),
  optimization: z.boolean(),
  fine_tuning: z.boolean(),
});

export const ImporterSchema = z.enum(["langfuse", "json"]);

export const IngestSourceSchema = z.strictObject({
  name: nonEmptyString,
  importer: ImporterSchema,
  /** File or directory the importer reads; relative paths resolve to the repo root. */
  path: nonEmptyString.optional(),
});

export const IngestSchema = z.strictObject({
  default_permissions: PermissionsSchema,
  sources: z.array(IngestSourceSchema).optional(),
});

// ---------------------------------------------------------------------------
// Product section: assertions (docs/assertions-v1.md)
// ---------------------------------------------------------------------------

/** Deterministic, free checks graded before any judge token is spent. */
export const AssertionTypeSchema = z.enum([
  "valid_json",
  "json_schema",
  "contains",
  "not_contains",
  "regex",
  "equals",
  "tool_called",
  "tool_not_called",
  "tool_arg_equals",
  "max_length",
  "json_path_equals",
]);

/**
 * One assertion. The engine (`@compound/assertions`) owns the exact per-type
 * parameters; here the shape is loose on parameters but strict on `type`,
 * `required`, and `weight`, so a config is caught for an unknown assertion type
 * without duplicating the engine's discriminated union.
 */
export const AssertionSchema = z.looseObject({
  type: AssertionTypeSchema,
  required: z.boolean().optional(),
  weight: z.number().positive().optional(),
});

/** Per-task assertion lists, keyed by task_key. */
export const AssertionsSchema = z.record(z.string(), z.array(AssertionSchema));

// ---------------------------------------------------------------------------
// Whole file
// ---------------------------------------------------------------------------

/**
 * The whole `compound.yaml`. Unknown top-level sections are allowed: the file is
 * shared with the Python engine and future product steps add sections here.
 */
export const CompoundConfigSchema = z.looseObject({
  version: z.literal(1),
  artifacts_dir: nonEmptyString,
  manifests_dir: nonEmptyString,
  seed: z.int().optional(),
  budget: BudgetSchema.optional(),
  providers: z.record(z.string(), ProviderSchema).optional(),
  models: ModelsSchema.optional(),
  pricing_usd_per_million_tokens: PricingTableSchema.optional(),
  flex_pricing_usd_per_million_tokens: PricingTableSchema.optional(),
  benchmarks: z.record(z.string(), BenchmarkSchema),
  optimization: OptimizationSchema.optional(),
  gate: GateSchema.optional(),

  // Product sections (optional; strict).
  task_keys: TaskKeysSchema.optional(),
  redaction: RedactionSchema.optional(),
  ingest: IngestSchema.optional(),
  assertions: AssertionsSchema.optional(),
});

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Documented trace-contract default, applied by `ingest.default_permissions`. */
export const DEFAULT_PERMISSIONS = {
  judging: true,
  optimization: true,
  fine_tuning: false,
} as const;

/** Marker used when a redaction rule does not set one. */
export const DEFAULT_REDACTION_MARKERS = {
  secret: "⟦redacted:secret⟧",
  pii: "⟦redacted:pii⟧",
} as const;

/**
 * Marker a rule emits when it does not declare one. `regex` rules map to the
 * contract's `custom:<name>` rule namespace.
 */
export function defaultRedactionMarker(rule: RedactionRule): string {
  if (rule.marker) return rule.marker;
  if (rule.detector === "regex") return `⟦redacted:custom:${rule.name}⟧`;
  return DEFAULT_REDACTION_MARKERS[rule.detector];
}

export const CONFIG_SCHEMA_NAME = "compound.config" as const;
export const CONFIG_SCHEMA_VERSION = 1 as const;

// ---------------------------------------------------------------------------
// Inferred TypeScript types
// ---------------------------------------------------------------------------

export type Budget = z.infer<typeof BudgetSchema>;
export type Provider = z.infer<typeof ProviderSchema>;
export type ModelEntry = z.infer<typeof ModelEntrySchema>;
export type Models = z.infer<typeof ModelsSchema>;
export type TokenPrice = z.infer<typeof TokenPriceSchema>;
export type PricingTable = z.infer<typeof PricingTableSchema>;
export type Partitions = z.infer<typeof PartitionsSchema>;
export type Benchmark = z.infer<typeof BenchmarkSchema>;
export type Optimization = z.infer<typeof OptimizationSchema>;
export type Gate = z.infer<typeof GateSchema>;
export type ReplayPolicy = z.infer<typeof ReplayPolicySchema>;
export type TaskReplay = z.infer<typeof TaskReplaySchema>;
export type TaskKey = z.infer<typeof TaskKeySchema>;
export type TaskKeys = z.infer<typeof TaskKeysSchema>;
export type RedactionRule = z.infer<typeof RedactionRuleSchema>;
export type Redaction = z.infer<typeof RedactionSchema>;
export type Permissions = z.infer<typeof PermissionsSchema>;
export type Importer = z.infer<typeof ImporterSchema>;
export type IngestSource = z.infer<typeof IngestSourceSchema>;
export type Ingest = z.infer<typeof IngestSchema>;
export type Assertions = z.infer<typeof AssertionsSchema>;
export type CompoundConfig = z.infer<typeof CompoundConfigSchema>;
