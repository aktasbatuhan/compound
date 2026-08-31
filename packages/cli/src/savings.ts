import {
  type CompoundDatabase,
  experimentScoreCost,
  experimentSpendUsd,
  taskTrafficVolume,
} from "@compound/storage";
import type { CommandEnvironment } from "./commands";

type VolumeBasis =
  | { source: "manual"; monthlyTraces: number }
  | {
      source: "telemetry";
      monthlyTraces: number;
      traceCount: number;
      firstStartedAt: Date;
      lastStartedAt: Date;
      rateWindowDays: number;
    };

export interface GateCostProjection {
  candidateCostPerTrace: number;
  referenceCostPerTrace: number;
  deltaPerTrace: number;
  monthlyImpactUsd: number;
  annualImpactUsd: number;
  candidateRecordedSpendUsd: number;
  referenceRecordedSpendUsd: number;
  volume: VolumeBasis;
}

export function parseMonthlyVolumeFlag(value: string | boolean | undefined): number | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "string") {
    throw new Error("--monthly-volume needs a positive number of task traces per month");
  }
  const monthlyVolume = Number(value);
  if (!Number.isFinite(monthlyVolume) || monthlyVolume <= 0) {
    throw new Error("--monthly-volume must be a positive number of task traces per month");
  }
  return monthlyVolume;
}

function volumeBasis(
  db: CompoundDatabase,
  taskKey: string,
  monthlyVolume: number | undefined,
): VolumeBasis | null {
  if (monthlyVolume !== undefined) return { source: "manual", monthlyTraces: monthlyVolume };
  const observed = taskTrafficVolume(db, taskKey);
  if (observed === null) return null;
  return {
    source: "telemetry",
    monthlyTraces: observed.projectedMonthlyTraces,
    traceCount: observed.traceCount,
    firstStartedAt: observed.firstStartedAt,
    lastStartedAt: observed.lastStartedAt,
    rateWindowDays: observed.rateWindowDays,
  };
}

export function gateCostProjection(
  db: CompoundDatabase,
  input: {
    taskKey: string;
    candidateExperimentId: string;
    referenceExperimentId: string;
    monthlyVolume?: number;
  },
): GateCostProjection | null {
  const volume = volumeBasis(db, input.taskKey, input.monthlyVolume);
  if (volume === null) return null;

  const candidate = experimentScoreCost(db, input.candidateExperimentId);
  const reference = experimentScoreCost(db, input.referenceExperimentId);
  if (
    candidate.meanCostUsd === null ||
    reference.meanCostUsd === null ||
    candidate.estimatedCostCount > 0 ||
    reference.estimatedCostCount > 0
  ) {
    return null;
  }

  const deltaPerTrace = reference.meanCostUsd - candidate.meanCostUsd;
  const monthlyImpactUsd = deltaPerTrace * volume.monthlyTraces;
  return {
    candidateCostPerTrace: candidate.meanCostUsd,
    referenceCostPerTrace: reference.meanCostUsd,
    deltaPerTrace,
    monthlyImpactUsd,
    annualImpactUsd: monthlyImpactUsd * 12,
    candidateRecordedSpendUsd: experimentSpendUsd(db, input.candidateExperimentId),
    referenceRecordedSpendUsd: experimentSpendUsd(db, input.referenceExperimentId),
    volume,
  };
}

const money = (value: number, unitPrecision = false): string => {
  const precision = unitPrecision ? 5 : Math.abs(value) > 0 && Math.abs(value) < 0.01 ? 5 : 2;
  return `$${Math.abs(value).toFixed(precision)}`;
};

const volume = (value: number): string =>
  value.toLocaleString("en-US", { maximumFractionDigits: 1 });

function assumptionLine(basis: VolumeBasis): string {
  if (basis.source === "manual") {
    return (
      `${volume(basis.monthlyTraces)} traces/month from --monthly-volume; ` +
      "assumes this volume continues."
    );
  }
  const traceWord = basis.traceCount === 1 ? "trace" : "traces";
  return (
    `${basis.traceCount} ingested ${traceWord} from ${basis.firstStartedAt.toISOString()} to ` +
    `${basis.lastStartedAt.toISOString()}; extrapolated to ` +
    `${volume(basis.monthlyTraces)} traces/month using a ` +
    `${basis.rateWindowDays.toFixed(1)}-day rate window (one-day minimum), and assumes that ` +
    "rate continues."
  );
}

/** Write a labelled cost projection. Returns false when no honest basis exists. */
export function writeGateCostProjection(
  db: CompoundDatabase,
  env: Pick<CommandEnvironment, "write">,
  input: {
    taskKey: string;
    candidateExperimentId: string;
    referenceExperimentId: string;
    monthlyVolume?: number;
  },
): boolean {
  const projection = gateCostProjection(db, input);
  if (projection === null) {
    const hasVolume =
      input.monthlyVolume !== undefined || taskTrafficVolume(db, input.taskKey) !== null;
    if (!hasVolume) {
      env.write(
        "  traffic basis: no ingested traces for this task; no dollar projection shown. " +
          "Pass --monthly-volume N to state a monthly trace-volume assumption.",
      );
    } else {
      env.write(
        "  cost basis: at least one gate completion has an estimated or missing cost; " +
          "no dollar projection shown.",
      );
    }
    return false;
  }

  const deltaLabel =
    projection.deltaPerTrace > 0
      ? `${money(projection.deltaPerTrace, true)} saved per trace`
      : projection.deltaPerTrace < 0
        ? `${money(projection.deltaPerTrace, true)} more per trace`
        : "$0.00000 per trace";
  env.write(
    `  cost basis:   reference ${money(projection.referenceCostPerTrace, true)} - candidate ` +
      `${money(projection.candidateCostPerTrace, true)} = ${deltaLabel} (measured completions)`,
  );
  if (projection.monthlyImpactUsd > 0) {
    env.write(
      `  projected savings: ${money(projection.monthlyImpactUsd)}/month, ` +
        `${money(projection.annualImpactUsd)}/year`,
    );
  } else if (projection.monthlyImpactUsd < 0) {
    env.write(
      `  projected added cost: ${money(projection.monthlyImpactUsd)}/month, ` +
        `${money(projection.annualImpactUsd)}/year`,
    );
  } else {
    env.write("  projected cost change: $0.00/month, $0.00/year");
  }
  env.write(`  assumption:   ${assumptionLine(projection.volume)}`);
  const recorded = projection.candidateRecordedSpendUsd + projection.referenceRecordedSpendUsd;
  env.write(
    `  measured spend: ${money(projection.candidateRecordedSpendUsd, true)} candidate + ` +
      `${money(projection.referenceRecordedSpendUsd, true)} reference = ` +
      `${money(recorded, true)} in spend_records for these gate runs`,
  );
  return true;
}
