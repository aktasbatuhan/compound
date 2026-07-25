/**
 * Shared judge types: the resolved judge config, the money-safe execution
 * context (mirrors the runner's controls), and the trust state.
 */
import type { Provider, TokenPrice } from "@compound/execution";
import type { CompoundDatabase, JudgeMode } from "@compound/storage";

/** A task's judge, resolved from config (docs/judges-v1.md). */
export interface JudgeConfig {
  taskKey: string;
  /** The model id as written in config (used for the calibration pin). */
  model: string;
  promptVersion: string;
  rubric: string;
  mode: JudgeMode;
  calibrationThreshold: number;
  /** Score at or above which the verdict is a pass. Default 0.5. */
  decisionPoint: number;
}

/** The default number of human-labelled cases below which we refuse to trust. */
export const MIN_CALIBRATION_CASES = 10;

/**
 * Everything the judge needs to make model calls under the same money-safety
 * rules as the runner: off unless paid, capped, cache-first.
 */
export interface JudgeExecutionContext {
  db: CompoundDatabase;
  provider: Provider;
  providerName: string;
  /** The provider-native model id the judge calls. */
  judgeModel: string;
  price: TokenPrice;
  paid: boolean;
  experimentCapUsd: number;
  globalHardLimitUsd: number;
  providerRevision?: string;
}

/** Whether a judge may feed a gate, and why. */
export interface JudgeTrust {
  calibrated: boolean;
  reason: string;
  kappa?: number;
  ci?: [number, number];
  n?: number;
}
