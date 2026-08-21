import { describe, expect, test } from "bun:test";
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createDatabase, insertTraces, migrate } from "../src/index";
import { freshDatabase, newBatch, recordFromFixture } from "./helpers";

const DRIZZLE_DIR = join(import.meta.dir, "..", "drizzle");

function columnNames(handle: ReturnType<typeof createDatabase>, table: string): string[] {
  return handle.sqlite
    .query<{ name: string }, []>(`PRAGMA table_info(${table})`)
    .all()
    .map((row) => row.name);
}

function tableNames(handle: ReturnType<typeof freshDatabase>): string[] {
  return handle.sqlite
    .query<{ name: string }, []>(
      "select name from sqlite_master where type = 'table' order by name",
    )
    .all()
    .map((row) => row.name);
}

function indexNames(handle: ReturnType<typeof freshDatabase>): string[] {
  return handle.sqlite
    .query<{ name: string }, []>(
      "select name from sqlite_master where type = 'index' and name not like 'sqlite_%' order by name",
    )
    .all()
    .map((row) => row.name);
}

describe("migrations", () => {
  test("apply cleanly and create every table", () => {
    const handle = freshDatabase();
    const names = tableNames(handle);
    expect(names).toContain("import_batches");
    expect(names).toContain("traces");
    expect(names).toContain("cases");
    expect(names).toContain("experiments");
    expect(names).toContain("completions");
    expect(names).toContain("spend_records");
    expect(names).toContain("experiment_results");
    expect(names).toContain("gate_specs");
    expect(names).toContain("gate_results");
    expect(names).toContain("judge_calibrations");
    expect(names).toContain("optimization_runs");
    handle.close();
  });

  test("create every declared index", () => {
    const handle = freshDatabase();
    expect(indexNames(handle)).toEqual([
      "cases_case_id_unique",
      "cases_partition_idx",
      "cases_provenance_idx",
      "cases_review_state_idx",
      "cases_source_trace_id_idx",
      "cases_task_content_unique",
      "cases_task_key_idx",
      "experiment_results_experiment_id_idx",
      "experiments_candidate_model_idx",
      "experiments_task_key_idx",
      "gate_decision_cases_content_hash_idx",
      "gate_results_gate_spec_id_idx",
      "gate_results_task_lookup_idx",
      "gate_specs_spec_hash_unique",
      "gate_specs_task_key_idx",
      "judge_calibrations_pin_idx",
      "judge_calibrations_task_key_idx",
      "optimization_runs_task_key_idx",
      "spend_records_experiment_id_idx",
      "spend_records_fingerprint_unique",
      "traces_content_hash_idx",
      "traces_import_batch_id_idx",
      "traces_started_at_idx",
      "traces_task_key_idx",
      "traces_trace_id_unique",
      "traces_validation_class_idx",
    ]);
    handle.close();
  });

  test("are re-runnable and preserve data", () => {
    const handle = freshDatabase();
    const batch = newBatch(handle);
    insertTraces(handle, batch.id, [recordFromFixture("eval-ready-full.json")]);

    migrate(handle);
    migrate(handle);

    const rows = handle.sqlite.query<{ n: number }, []>("select count(*) as n from traces").get();
    expect(rows?.n).toBe(1);
    handle.close();
  });

  test("createDatabase does not migrate unless asked", () => {
    const handle = createDatabase({ path: ":memory:" });
    expect(tableNames(handle)).not.toContain("traces");
    migrate(handle);
    expect(tableNames(handle)).toContain("traces");
    handle.close();
  });

  test("0012+ add the columns and BACKFILL content_hash on a populated pre-0012 db (#5)", () => {
    // Build a migrations folder truncated at 0011 (no 0012 or later), so we can
    // reach the real legacy state — a populated experiment_results with NO
    // content_hash — and then apply 0012+ as an UPGRADE, exactly as a real
    // database would. The all-at-once `freshDatabase()` path never exercises the
    // backfill UPDATE.
    const tmp = mkdtempSync(join(tmpdir(), "compound-mig-"));
    const handle = createDatabase({ path: ":memory:" });
    try {
      cpSync(DRIZZLE_DIR, tmp, { recursive: true });
      // Drop 0012 AND everything after it: a later migration left in the folder
      // would advance the migrator's applied timestamp past 0012, making the
      // upgrade below silently skip it.
      const journal = JSON.parse(readFileSync(join(DRIZZLE_DIR, "meta", "_journal.json"), "utf8"));
      for (const e of journal.entries as { idx: number; tag: string }[]) {
        if (e.idx >= 12) rmSync(join(tmp, `${e.tag}.sql`));
      }
      journal.entries = journal.entries.filter((e: { idx: number }) => e.idx < 12);
      writeFileSync(join(tmp, "meta", "_journal.json"), JSON.stringify(journal));

      // Migrate only THROUGH 0011: the new columns must not exist yet.
      migrate(handle, tmp);
      expect(columnNames(handle, "experiment_results")).not.toContain("content_hash");
      expect(columnNames(handle, "gate_specs")).not.toContain("max_skip_fraction");
      expect(columnNames(handle, "completions")).not.toContain("cached_input_tokens");

      // Populate the exact legacy shape 0012 must upgrade: a case carrying a
      // content hash, and an experiment_result that predates the content_hash
      // column (so it is NULL) and must be backfilled from the case.
      handle.sqlite
        .query(
          "INSERT INTO cases (id, case_id, task_key, source_trace_id, content_hash, provenance, partition, input) VALUES (?,?,?,?,?,?,?,?)",
        )
        .run("case-row-1", "c1", "support", "t1", "hash-c1", "human_golden", "decision_test", "{}");
      handle.sqlite
        .query(
          "INSERT INTO experiments (id, task_key, candidate_model, provider, partition, status, started_at) VALUES (?,?,?,?,?,?,?)",
        )
        .run("exp-1", "support", "cand", "mock", "decision_test", "completed", Date.now());
      handle.sqlite
        .query("INSERT INTO experiment_results (experiment_id, case_id, status) VALUES (?,?,?)")
        .run("exp-1", "c1", "graded");

      // Apply the FULL folder → 0012 and everything after run on the populated db.
      migrate(handle, DRIZZLE_DIR);
      expect(columnNames(handle, "experiment_results")).toContain("content_hash");
      expect(columnNames(handle, "gate_specs")).toContain("max_skip_fraction");
      // 0014 (#34): nullable cached-token column; pre-existing rows stay NULL.
      expect(columnNames(handle, "completions")).toContain("cached_input_tokens");

      // The backfill copied the case's content hash onto the pre-existing row, so
      // the peeking guard can reconstruct the decided cohort without a live join.
      const row = handle.sqlite
        .query<{ content_hash: string | null }, []>(
          "SELECT content_hash FROM experiment_results WHERE experiment_id = 'exp-1' AND case_id = 'c1'",
        )
        .get();
      expect(row?.content_hash).toBe("hash-c1");
    } finally {
      handle.close();
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  test("foreign keys are enforced", () => {
    const handle = freshDatabase();
    const [row] = handle.sqlite.query<{ foreign_keys: number }, []>("PRAGMA foreign_keys").all();
    expect(row?.foreign_keys).toBe(1);
    handle.close();
  });
});
