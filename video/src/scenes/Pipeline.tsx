import React from "react";
import { AbsoluteFill, Easing, interpolate, useCurrentFrame } from "remotion";
import { Paper } from "../components";
import { c, display, mono, sans } from "../theme";

const STEPS = [
  { cmd: "import", sub: "traces in, redacted" },
  { cmd: "curate", sub: "cases + sealed set" },
  { cmd: "run", sub: "one model, many hosts" },
  { cmd: "optimize", sub: "close the gap on train" },
  { cmd: "gate", sub: "verdict on the sealed set" },
];

export const Pipeline: React.FC = () => {
  const frame = useCurrentFrame();
  const title = interpolate(frame, [0, 16], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  const N = STEPS.length;
  const railStart = 30;
  const railFill = interpolate(frame, [railStart, railStart + 70], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <Paper>
      <AbsoluteFill style={{ padding: "0 130px", justifyContent: "center" }}>
        <div style={{ opacity: title, transform: `translateY(${(1 - title) * 16}px)`, textAlign: "center", marginBottom: 96 }}>
          <div style={{ fontFamily: display, fontWeight: 700, fontSize: 62, color: c.ink, letterSpacing: -0.6 }}>
            From production traces to a gated verdict.
          </div>
          <div style={{ fontFamily: sans, fontSize: 30, color: c.ink2, marginTop: 12 }}>
            Five steps, one content-addressed cache. Every stage writes evidence the next one reads.
          </div>
        </div>

        <div style={{ position: "relative", height: 200 }}>
          {/* base + accent progress rail */}
          <div style={{ position: "absolute", left: 60, right: 60, top: 34, height: 3, background: c.line }} />
          <div style={{ position: "absolute", left: 60, top: 34, height: 3, width: `calc((100% - 120px) * ${railFill})`, background: c.accent }} />

          <div style={{ position: "absolute", inset: 0, display: "flex", justifyContent: "space-between" }}>
            {STEPS.map((s, i) => {
              const at = railStart + (i / (N - 1)) * 70;
              const on = interpolate(frame, [at - 4, at + 8], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
              const filled = railFill >= i / (N - 1) - 0.001;
              return (
                <div key={s.cmd} style={{ width: 260, display: "flex", flexDirection: "column", alignItems: "center", opacity: on }}>
                  <div
                    style={{
                      width: 68,
                      height: 68,
                      borderRadius: "50%",
                      background: filled ? c.accent : c.panel,
                      border: `2.5px solid ${filled ? c.accent : c.line}`,
                      color: filled ? "#fff" : c.ink3,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontFamily: mono,
                      fontSize: 26,
                      fontWeight: 500,
                    }}
                  >
                    {i + 1}
                  </div>
                  <div style={{ fontFamily: mono, fontSize: 27, color: c.ink, marginTop: 22, fontWeight: 500 }}>{s.cmd}</div>
                  <div style={{ fontFamily: sans, fontSize: 21, color: c.ink3, marginTop: 6, textAlign: "center" }}>{s.sub}</div>
                </div>
              );
            })}
          </div>
        </div>
      </AbsoluteFill>
    </Paper>
  );
};
