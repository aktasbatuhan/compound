import React from "react";
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { Paper } from "../components";
import { c, display, mono, sans } from "../theme";

export const Close: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const head = spring({ frame, fps, config: { damping: 200, mass: 0.8 } });
  const url = interpolate(frame, [26, 46], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const foot = interpolate(frame, [42, 62], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  return (
    <Paper>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: 40 }}>
        <div
          style={{
            fontFamily: display,
            fontWeight: 800,
            fontSize: 92,
            lineHeight: 1.04,
            letterSpacing: -1.5,
            color: c.ink,
            textAlign: "center",
            opacity: head,
            transform: `translateY(${interpolate(head, [0, 1], [24, 0])}px)`,
          }}
        >
          Pick the right provider
          <br />
          for <span style={{ color: c.accent }}>your own workload.</span>
        </div>

        <div
          style={{
            display: "flex",
            gap: 40,
            fontFamily: mono,
            fontSize: 30,
            color: c.ink,
            opacity: url,
            transform: `translateY(${(1 - url) * 12}px)`,
          }}
        >
          <span style={{ color: c.accent }}>compound-1js.pages.dev</span>
          <span style={{ color: c.ink3 }}>github.com/aktasbatuhan/compound</span>
        </div>

        <div style={{ display: "flex", gap: 14, opacity: foot }}>
          {["Apache-2.0", "local-first", "money-safe by default"].map((t) => (
            <span
              key={t}
              style={{
                fontFamily: mono,
                fontSize: 20,
                color: c.ink2,
                border: `1px solid ${c.line}`,
                background: c.panel,
                borderRadius: 4,
                padding: "8px 16px",
              }}
            >
              {t}
            </span>
          ))}
        </div>
      </AbsoluteFill>
    </Paper>
  );
};
