// Shared geometry + data for the radar teaser (cold-open) and the full payoff.
export const AXES = ["quality", "reliability", "speed", "determinism", "cost", "TPS"];

export type RH = { key: string; name: string; hex: string; v: number[] };
export const RHOSTS: RH[] = [
  { key: "dwrt", name: "doubleword realtime", hex: "#1740e6", v: [1.0, 1.0, 1.0, 0.64, 0.11, 0.34] },
  { key: "dwfx", name: "doubleword flex", hex: "#0e93c0", v: [0.82, 1.0, 0.67, 0.71, 0.15, 0.23] },
  { key: "dinf", name: "deepinfra", hex: "#0d9373", v: [0.87, 1.0, 0.2, 0.86, 1.0, 0.36] },
  { key: "para", name: "parasail", hex: "#7048e8", v: [0.87, 1.0, 0.24, 0.5, 0.34, 0.51] },
  { key: "fire", name: "fireworks", hex: "#e8590c", v: [0.61, 0.64, 0.56, 0.64, 0.73, 1.0] },
];

export type Beat = { axis: number; eyebrow: string; head: string; winners: string[]; detail: string };
export const BEATS: Beat[] = [
  { axis: 2, eyebrow: "SPEED", head: "Fastest calls", winners: ["dwrt"], detail: "doubleword realtime, 4.1s median call" },
  { axis: 5, eyebrow: "THROUGHPUT", head: "Highest tokens/sec", winners: ["fire"], detail: "fireworks tops output throughput" },
  { axis: 4, eyebrow: "COST", head: "Most cost-efficient", winners: ["dinf"], detail: "deepinfra, cheapest of the six" },
  { axis: 1, eyebrow: "RELIABILITY", head: "Zero provider errors", winners: ["dwrt", "dwfx", "dinf", "para"], detail: "four clean hosts; fireworks drops on rate-limits" },
];

export const CX = 600, CY = 560, R = 300;

export const pt = (i: number, val: number): [number, number] => {
  const a = ((-90 + i * 60) * Math.PI) / 180;
  return [CX + Math.cos(a) * R * val, CY + Math.sin(a) * R * val];
};
export const polyStr = (vals: number[]) => vals.map((v, i) => pt(i, v).join(",")).join(" ");
