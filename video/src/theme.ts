import { loadFont as loadAnybody } from "@remotion/google-fonts/Anybody";
import { loadFont as loadPublicSans } from "@remotion/google-fonts/PublicSans";
import { loadFont as loadSplineMono } from "@remotion/google-fonts/SplineSansMono";

// Brand faces, same trio as compound-1js.pages.dev.
export const display = loadAnybody("normal", {
  weights: ["500", "600", "700", "800"],
  subsets: ["latin"],
}).fontFamily;

export const sans = loadPublicSans("normal", {
  weights: ["400", "500", "600"],
  subsets: ["latin"],
}).fontFamily;

export const mono = loadSplineMono("normal", {
  weights: ["400", "500"],
  subsets: ["latin"],
}).fontFamily;

// Brand tokens, same palette as the site.
export const c = {
  paper: "#faf9f5",
  panel: "#fffdf8",
  ink: "#23201a",
  ink2: "#5c564a",
  ink3: "#948d80",
  line: "#e7e1d4",
  lineSoft: "#efe9dd",
  accent: "#1740e6",
  accentSoft: "#dde3fb",
  good: "#3f7d3a",
  warn: "#9a6a15",
  bad: "#b23a2e",
} as const;

// The real terminal-bench run. deepseek-v4-flash-0731, six hosts, three trials,
// 252 episodes. Numbers match the launch report verbatim.
export type Host = {
  name: string;
  quant?: string;
  pass: number;
  rate: number; // percent
  ci: [number, number];
  status: "healthy" | "rate-limited" | "capability gap";
  note: string;
};

export const HOSTS: Host[] = [
  { name: "doubleword realtime", pass: 23, rate: 54.8, ci: [40, 69], status: "healthy", note: "fastest median latency of the five" },
  { name: "parasail", quant: "fp8", pass: 20, rate: 47.6, ci: [33, 62], status: "healthy", note: "clean episodes" },
  { name: "deepinfra", quant: "fp4", pass: 20, rate: 47.6, ci: [33, 62], status: "healthy", note: "clean episodes, cheapest of the five" },
  { name: "doubleword flex", pass: 19, rate: 45.2, ci: [31, 60], status: "healthy", note: "async queue, slower first token" },
  { name: "fireworks", pass: 14, rate: 33.3, ci: [21, 48], status: "rate-limited", note: "15 of 42 episodes killed by shared-pool 429s" },
];

// Healthy hosts carry the brand accent; the two failures get semantic alert
// colors so they read as anomalies against a field of blue.
export const statusColor = (s: Host["status"]) =>
  s === "healthy" ? c.accent : s === "rate-limited" ? c.warn : c.bad;
