import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { c, mono, sans } from "./theme";

// Warm paper ground with a faint top-light, matching the site's #faf9f5.
export const Paper: React.FC<{ children?: React.ReactNode }> = ({ children }) => (
  <AbsoluteFill
    style={{
      backgroundColor: c.paper,
      backgroundImage: `radial-gradient(120% 80% at 50% -10%, rgba(23,64,230,0.05), rgba(250,249,245,0) 60%)`,
    }}
  >
    {children}
  </AbsoluteFill>
);

// Reveal-by-character. `progress` is 0..1 across the string.
export const Typed: React.FC<{
  text: string;
  progress: number;
  style?: React.CSSProperties;
  caret?: boolean;
}> = ({ text, progress, style, caret = true }) => {
  const n = Math.round(interpolate(progress, [0, 1], [0, text.length], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }));
  const shown = text.slice(0, n);
  const done = n >= text.length;
  return (
    <span style={{ fontFamily: mono, whiteSpace: "pre", ...style }}>
      {shown}
      {caret && !done ? <span style={{ color: c.accent }}>▊</span> : null}
    </span>
  );
};

export const Pill: React.FC<{ label: string; color: string; style?: React.CSSProperties }> = ({ label, color, style }) => (
  <span
    style={{
      fontFamily: mono,
      fontSize: 20,
      fontWeight: 500,
      color,
      border: `1.5px solid ${color}`,
      borderRadius: 4,
      padding: "3px 12px 4px",
      letterSpacing: 0.2,
      lineHeight: 1,
      whiteSpace: "nowrap",
      ...style,
    }}
  >
    {label}
  </span>
);

// A framed terminal card with the site's black head bar + traffic lights.
export const CodeCard: React.FC<{
  children: React.ReactNode;
  label?: string;
  width?: number | string;
  style?: React.CSSProperties;
}> = ({ children, label, width, style }) => (
  <div
    style={{
      width,
      borderRadius: 10,
      overflow: "hidden",
      border: `1px solid ${c.line}`,
      boxShadow: "0 24px 60px -30px rgba(35,32,26,0.35)",
      background: "#17150f",
      ...style,
    }}
  >
    <div style={{ height: 40, background: "#0f0e0a", display: "flex", alignItems: "center", padding: "0 16px", gap: 8 }}>
      <Dot color="#e0564b" />
      <Dot color="#e3b73f" />
      <Dot color="#4caf50" />
      {label ? (
        <span style={{ fontFamily: mono, fontSize: 15, color: "rgba(255,255,255,0.5)", marginLeft: 12 }}>{label}</span>
      ) : null}
    </div>
    <div style={{ padding: "22px 26px", fontFamily: mono, fontSize: 24, lineHeight: 1.5, color: "#f3efe6" }}>{children}</div>
  </div>
);

const Dot: React.FC<{ color: string }> = ({ color }) => (
  <span style={{ width: 12, height: 12, borderRadius: "50%", background: color, display: "inline-block" }} />
);

// Eased fade+rise for a block, keyed to an absolute frame window.
export const useRise = (start: number, dur = 18) => {
  const frame = useCurrentFrame();
  const t = interpolate(frame, [start, start + dur], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return {
    opacity: t,
    transform: `translateY(${(1 - t) * 22}px)`,
  };
};

export const Kicker: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <div
    style={{
      fontFamily: mono,
      fontSize: 22,
      color: c.ink3,
      letterSpacing: 1.5,
      textTransform: "uppercase",
      ...style,
    }}
  >
    {children}
  </div>
);

export const Body: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <div style={{ fontFamily: sans, fontSize: 34, lineHeight: 1.4, color: c.ink2, ...style }}>{children}</div>
);
