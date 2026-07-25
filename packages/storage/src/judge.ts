/**
 * Judge-calibration repository (docs/judges-v1.md).
 *
 * A calibration is PINNED to (task, judge_model, prompt_version, rubric_hash).
 * The latest measurement per pin is authoritative; changing any pin field means
 * there is no calibration for the new pin, so the judge is uncalibrated until
 * re-measured. Trust is never inherited across a pin change.
 */
import { and, desc, eq, sql } from "drizzle-orm";
import type { CompoundDatabase } from "./db";
import { type JudgeCalibrationRow, type JudgeMode, judgeCalibrations } from "./schema";

export interface JudgePin {
  taskKey: string;
  judgeModel: string;
  promptVersion: string;
  rubricHash: string;
}

export interface RecordCalibrationInput extends JudgePin {
  mode: JudgeMode;
  agreementKappa: number;
  kappaCiLo: number;
  kappaCiHi: number;
  n: number;
  positionBiasRate: number;
  threshold: number;
  calibrated: boolean;
}

export function recordJudgeCalibration(
  handle: CompoundDatabase,
  input: RecordCalibrationInput,
): JudgeCalibrationRow {
  const id = crypto.randomUUID();
  handle.db
    .insert(judgeCalibrations)
    .values({ id, ...input })
    .run();
  return getJudgeCalibration(handle, id) as JudgeCalibrationRow;
}

export function getJudgeCalibration(
  handle: CompoundDatabase,
  id: string,
): JudgeCalibrationRow | null {
  const [row] = handle.db
    .select()
    .from(judgeCalibrations)
    .where(eq(judgeCalibrations.id, id))
    .all();
  return row ?? null;
}

/** The most recent calibration for an exact pin, or null if never measured. */
export function latestCalibrationForPin(
  handle: CompoundDatabase,
  pin: JudgePin,
): JudgeCalibrationRow | null {
  const [row] = handle.db
    .select()
    .from(judgeCalibrations)
    .where(
      and(
        eq(judgeCalibrations.taskKey, pin.taskKey),
        eq(judgeCalibrations.judgeModel, pin.judgeModel),
        eq(judgeCalibrations.promptVersion, pin.promptVersion),
        eq(judgeCalibrations.rubricHash, pin.rubricHash),
      ),
    )
    .orderBy(desc(judgeCalibrations.measuredAt), desc(sql`rowid`))
    .limit(1)
    .all();
  return row ?? null;
}

/** The most recent calibration per task, regardless of pin (for status views). */
export function latestCalibrationForTask(
  handle: CompoundDatabase,
  taskKey: string,
): JudgeCalibrationRow | null {
  const [row] = handle.db
    .select()
    .from(judgeCalibrations)
    .where(eq(judgeCalibrations.taskKey, taskKey))
    .orderBy(desc(judgeCalibrations.measuredAt), desc(sql`rowid`))
    .limit(1)
    .all();
  return row ?? null;
}

/** The latest calibration for every task that has one, newest first. */
export function listLatestCalibrations(handle: CompoundDatabase): JudgeCalibrationRow[] {
  const rows = handle.db
    .select()
    .from(judgeCalibrations)
    .orderBy(desc(judgeCalibrations.measuredAt), desc(sql`rowid`))
    .all();
  const seen = new Set<string>();
  const latest: JudgeCalibrationRow[] = [];
  for (const row of rows) {
    if (seen.has(row.taskKey)) continue;
    seen.add(row.taskKey);
    latest.push(row);
  }
  return latest;
}
