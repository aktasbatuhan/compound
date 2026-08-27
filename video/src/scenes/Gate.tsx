import React from "react";
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { Paper, Typed } from "../components";
import { c, display, mono, sans } from "../theme";

const VERDICTS = ["MEETS GATE", "FAILS", "INSUFFICIENT DATA", "JUDGE ABSTAINED", "NO RELIABLE GAIN"];

export const Gate: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const title = interpolate(frame, [0, 16], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const type = interpolate(frame, [22, 92], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const verdict = spring({ frame: frame - 104, fps, config: { damping: 200, mass: 0.8 } });
  const strip = interpolate(frame, [120, 138], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  return (
    <Paper>
      <AbsoluteFill style={{ padding: "0 130px", justifyContent: "center", alignItems: "center" }}>
        <div style={{ opacity: title, transform: `translateY(${(1 - title) * 16}px)`, textAlign: "center", marginBottom: 44 }}>
          <div style={{ fontFamily: display, fontWeight: 700, fontSize: 60, color: c.ink, letterSpacing: -0.6 }}>
            Ship on a verdict, not a vibe.
          </div>
          <div style={{ fontFamily: sans, fontSize: 29, color: c.ink2, marginTop: 12 }}>
            The rule is content-hashed before anyone looks. The sealed decision set cannot be reverse-fit.
          </div>
        </div>

        {/* command */}
        <div
          style={{
            width: 1360,
            borderRadius: 10,
            overflow: "hidden",
            border: `1px solid ${c.line}`,
            boxShadow: "0 24px 60px -32px rgba(35,32,26,0.35)",
          }}
        >
          <div style={{ height: 38, background: "#0f0e0a", display: "flex", alignItems: "center", padding: "0 16px", gap: 8 }}>
            <span style={{ width: 11, height: 11, borderRadius: "50%", background: "#e0564b" }} />
            <span style={{ width: 11, height: 11, borderRadius: "50%", background: "#e3b73f" }} />
            <span style={{ width: 11, height: 11, borderRadius: "50%", background: "#4caf50" }} />
          </div>
          <div style={{ background: "#17150f", padding: "24px 28px", fontSize: 25, lineHeight: 1.55 }}>
            <div>
              <span style={{ color: "#7f97ff" }}>$ </span>
              <Typed
                text={`compound gate support --candidate kimi-k3 --reference opus-5 \\`}
                progress={interpolate(type, [0, 0.6], [0, 1], { extrapolateRight: "clamp" })}
                style={{ color: "#f3efe6" }}
                caret={false}
              />
            </div>
            <div style={{ paddingLeft: 24 }}>
              <Typed
                text={`--metric task_success --max-regression 0.02 --reason "quarterly cost review"`}
                progress={interpolate(type, [0.55, 1], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })}
                style={{ color: "#f3efe6" }}
              />
            </div>
          </div>
        </div>

        {/* verdict */}
        <div
          style={{
            marginTop: 34,
            display: "flex",
            alignItems: "center",
            gap: 18,
            padding: "18px 30px",
            borderRadius: 10,
            background: c.accentSoft,
            border: `2px solid ${c.accent}`,
            opacity: verdict,
            transform: `scale(${interpolate(verdict, [0, 1], [0.92, 1])})`,
          }}
        >
          <span style={{ fontFamily: display, fontWeight: 800, fontSize: 40, color: c.accent, letterSpacing: 0.5 }}>MEETS GATE</span>
          <span style={{ fontFamily: sans, fontSize: 25, color: c.ink2 }}>candidate is non-inferior under the declared rule</span>
        </div>

        <div style={{ marginTop: 24, display: "flex", gap: 12, opacity: strip }}>
          {VERDICTS.map((v) => (
            <span
              key={v}
              style={{
                fontFamily: mono,
                fontSize: 18,
                color: v === "MEETS GATE" ? c.accent : c.ink3,
                border: `1.5px solid ${v === "MEETS GATE" ? c.accent : c.line}`,
                borderRadius: 4,
                padding: "6px 12px",
                fontWeight: v === "MEETS GATE" ? 500 : 400,
              }}
            >
              {v}
            </span>
          ))}
        </div>
        <div style={{ marginTop: 16, fontFamily: sans, fontSize: 22, color: c.ink3, opacity: strip }}>
          exactly one of five honest outcomes, chosen by a rule declared before the data is seen
        </div>
      </AbsoluteFill>
    </Paper>
  );
};
