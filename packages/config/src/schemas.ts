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

export const TokenPriceSchema = z.looseObject({
  input: z.number().nonnegative(),
  output: z.number().nonnegative(),
  /**
   * USD per million CACHED input tokens (issue #34) — the discounted rate a
   * provider bills for prompt-cache hits. Optional: when absent, cached input
   * is billed at the list `input` rate and views mark the cost as list-price.
   */
  cached_input: z.number().nonnegative().optional(),
});

export const PricingTableSchema = z.record(z.string(), TokenPriceSchema);

/**
 * One provider endpoint in the registry. `type` names the wire protocol it
 * speaks — a provider endpoint speaks exactly one — so the same model can be
 * compared across providers (`experiment <task> <model> --provider X`). An
 * endpoint may carry its OWN price table: identical weights cost differently per
 * host, and the routing decision needs the right number. `openai_compatible`
 * (the default) covers OpenAI itself and the many hosts that mirror its API
 * (vLLM, Fireworks, Together, Groq, …); `flex` is Doubleword's async Responses
 * route; `anthropic`/`google` are reserved for their native adapters.
 */
export const ProviderSchema = z.looseObject({
  base_url: z.url(),
  api_key_env: nonEmptyString,
  type: z.enum(["openai_compatible", "flex", "anthropic", "google"]).optional(),
  /**
   * HTTP request timeout in milliseconds. Long-output / reasoning-heavy
   * generations can run for minutes; the default (180s chat, 1h flex) suits
   * short tool-calls, so raise this for long-running agent workloads.
   */
  request_timeout_ms: z.number().int().positive().optional(),
  /** Per-provider price overrides, keyed by model id; falls back to the global table. */
  pricing_usd_per_million_tokens: PricingTableSchema.optional(),
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
  /**
   * Per-provider wire ids for the SAME logical model, when providers disagree on
   * the id for identical weights (e.g. OpenAI `gpt-4o-mini` vs OpenRouter
   * `openai/gpt-4o-mini`). Keyed by provider name → the id sent to that provider.
   * `id` stays the LOGICAL identity for storage, the completion fingerprint, and
   * telemetry grouping, so the same model on two providers still compares as one
   * model across two rows (issue #19). Providers not listed here use `id` as-is.
   */
  provider_ids: z.record(z.string(), nonEmptyString).optional(),
});

export const ModelsSchema = z.looseObject({
  frontier: z.array(ModelEntrySchema).optional(),
  candidates: z.array(ModelEntrySchema).optional(),
});

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
  /**
   * The peeking guard (#22, #3): block a further paid decision that reuses any
   * of a task's already-decided held-out labels. Curate a fresh, non-overlapping
   * decision set to decide again, or pass --force. Defaults to warn-only.
   */
  block_repeat_decision: z.boolean().optional(),
  /** Deprecated alias for `block_repeat_decision`; both enable the guard. */
  block_repeat_after_adoption: z.boolean().optional(),
  /**
   * Coverage gate (#5): the largest fraction of the sealed decision set that may
   * be omitted (skipped/abstained on either side) before a verdict is voided to
   * insufficient_data. Asymmetric skips (the two runs dropping different cases)
   * void it regardless. Unset = report coverage but don't enforce.
   */
  max_skip_fraction: z.number().min(0).max(1).optional(),
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

/**
 * Per-task default priority vector for `view compare` (#29): relative weights
 * over the four decision axes. Unspecified axes weigh 0; the CLI normalizes the
 * weights to sum to 1, so any positive scale works. `--priority` overrides it.
 */
export const PrioritySchema = z
  .strictObject({
    quality: z.number().nonnegative().optional(),
    cost: z.number().nonnegative().optional(),
    latency: z.number().nonnegative().optional(),
    throughput: z.number().nonnegative().optional(),
  })
  .refine((p) => Object.values(p).some((w) => typeof w === "number" && w > 0), {
    message: "priority needs at least one positive weight",
  });

export const TaskKeySchema = z.strictObject({
  description: z.string().optional(),
  replay: TaskReplaySchema,
  /** Default ranking weights for `compound view compare <task_key>` (#29). */
  priority: PrioritySchema.optional(),
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

export const ImporterSchema = z.enum(["langfuse", "json", "otel"]);

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
  "tool_call_arg",
  "max_length",
  "json_path_equals",
  "text_similarity",
]);

const TOOL_ARG_MATCHER_KEYS = ["equals", "regex", "subset", "schema"] as const;

/**
 * One assertion. The engine (`@compound/assertions`) owns the exact per-type
 * parameters; here the shape is loose on parameters but strict on `type`,
 * `required`, and `weight`, so a config is caught for an unknown assertion type
 * without duplicating the engine's discriminated union.
 *
 * The one exception is `tool_call_arg` (#8): a missing or ambiguous matcher
 * would otherwise sail through config validation and throw mid-experiment, after
 * completions may already be paid for — so its matcher is checked here, up front.
 */
export const AssertionSchema = z
  .looseObject({
    type: AssertionTypeSchema,
    required: z.boolean().optional(),
    weight: z.number().positive().optional(),
  })
  .superRefine((assertion, ctx) => {
    if (assertion.type !== "tool_call_arg") return;
    const a = assertion as Record<string, unknown>;
    if (typeof a.name !== "string" || a.name.length === 0) {
      ctx.addIssue({
        code: "custom",
        message: "tool_call_arg requires a tool `name`",
        path: ["name"],
      });
    }
    const match = a.match;
    if (match === null || typeof match !== "object" || Array.isArray(match)) {
      ctx.addIssue({
        code: "custom",
        message: "tool_call_arg requires a `match` object (equals/regex/subset/schema)",
        path: ["match"],
      });
      return;
    }
    const matchObj = match as Record<string, unknown>;
    const present = TOOL_ARG_MATCHER_KEYS.filter((k) => k in matchObj);
    if (present.length !== 1) {
      ctx.addIssue({
        code: "custom",
        message: `tool_call_arg.match must have exactly one of ${TOOL_ARG_MATCHER_KEYS.join("/")}; got ${present.length}`,
        path: ["match"],
      });
      return;
    }
    // The single matcher's payload must have the right shape (#9). Without this a
    // `{subset: 23}` passed config and then vacuously passed the assertion at
    // runtime (Object.entries(23) is empty). `equals` accepts any value.
    const key = present[0] as (typeof TOOL_ARG_MATCHER_KEYS)[number];
    const payload = matchObj[key];
    const isPlainObject = (v: unknown): boolean =>
      v !== null && typeof v === "object" && !Array.isArray(v);
    if (key === "regex" && typeof payload !== "string") {
      ctx.addIssue({
        code: "custom",
        message: "tool_call_arg.match.regex must be a string pattern",
        path: ["match", "regex"],
      });
    }
    if (key === "subset" && !isPlainObject(payload)) {
      ctx.addIssue({
        code: "custom",
        message: "tool_call_arg.match.subset must be an object of key/value pairs",
        path: ["match", "subset"],
      });
    }
    // An empty subset `{}` constrains nothing and passes vacuously at runtime (#9),
    // so it is a config mistake, not a wildcard — reject it up front.
    if (key === "subset" && isPlainObject(payload) && Object.keys(payload as object).length === 0) {
      ctx.addIssue({
        code: "custom",
        message: "tool_call_arg.match.subset must not be empty",
        path: ["match", "subset"],
      });
    }
    if (key === "schema" && !isPlainObject(payload)) {
      ctx.addIssue({
        code: "custom",
        message: "tool_call_arg.match.schema must be a JSON Schema object",
        path: ["match", "schema"],
      });
    }
  });

/** Per-task assertion lists, keyed by task_key. */
export const AssertionsSchema = z.record(z.string(), z.array(AssertionSchema));

// ---------------------------------------------------------------------------
// Product section: judges (docs/judges-v1.md)
// ---------------------------------------------------------------------------

/**
 * One task's judge. The judge is trusted only after calibration against human
 * labels meets `calibration_threshold`; below it, it abstains. The model,
 * prompt_version and rubric are the PIN — changing any of them invalidates the
 * calibration until it is re-measured.
 */
export const JudgeSchema = z.strictObject({
  /** A model id that must resolve in `models`. */
  model: nonEmptyString,
  /** Bumped whenever the judge instructions change; part of the calibration pin. */
  prompt_version: nonEmptyString,
  /** The rubric text the judge scores against; part of the pin (hashed). */
  rubric: nonEmptyString,
  mode: z.enum(["pointwise", "pairwise"]).optional(),
  /** Minimum human agreement (Cohen's kappa) to leave 'uncalibrated'. */
  calibration_threshold: z.number().min(0).max(1),
  /** Score at or above which the judge's verdict counts as a pass. Default 0.5. */
  decision_point: z.number().min(0).max(1).optional(),
});

/** Per-task judges, keyed by task_key. */
export const JudgesSchema = z.record(z.string(), JudgeSchema);

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
  judges: JudgesSchema.optional(),
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
export type Priority = z.infer<typeof PrioritySchema>;
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
