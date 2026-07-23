import { describe, expect, test } from "bun:test";
import { createDatabase, insertTraces, migrate } from "../src/index";
import { freshDatabase, newBatch, recordFromFixture } from "./helpers";

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

  test("foreign keys are enforced", () => {
    const handle = freshDatabase();
    const [row] = handle.sqlite.query<{ foreign_keys: number }, []>("PRAGMA foreign_keys").all();
    expect(row?.foreign_keys).toBe(1);
    handle.close();
  });
});
