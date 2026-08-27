import React from "react";
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { Paper } from "../components";
import { c, display, mono } from "../theme";

export const Hook: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const line1 = interpolate(frame, [4, 22], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const s = spring({ frame: frame - 24, fps, config: { damping: 200, mass: 0.7 } });
  const count = Math.round(interpolate(spring({ frame: frame - 34, fps, config: { damping: 200 } }), [0, 1], [33, 55]));
  const sub = interpolate(frame, [60, 82], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  return (
    <Paper>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: 34 }}>
        <div
          style={{
            fontFamily: display,
            fontWeight: 600,
            fontSize: 62,
            color: c.ink,
            letterSpacing: -0.5,
            opacity: line1,
            transform: `translateY(${(1 - line1) * 18}px)`,
          }}
        >
          One model. Five serving hosts.
        </div>

        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            gap: 40,
            fontFamily: display,
            fontWeight: 800,
            fontSize: 210,
            lineHeight: 1,
            letterSpacing: -3,
            transform: `scale(${interpolate(s, [0, 1], [0.9, 1])})`,
            opacity: s,
          }}
        >
          <span style={{ color: c.ink3 }}>33%</span>
          <span style={{ fontSize: 120, color: c.accent, fontWeight: 700 }}>→</span>
          <span style={{ color: c.ink }}>{count}%</span>
        </div>

        <div
          style={{
            fontFamily: mono,
            fontSize: 27,
            color: c.ink2,
            opacity: sub,
            transform: `translateY(${(1 - sub) * 14}px)`,
          }}
        >
          same open weights, same tasks. the serving host decided the rest.
        </div>
      </AbsoluteFill>
    </Paper>
  );
};
