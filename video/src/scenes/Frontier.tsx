import React from "react";
import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame } from "remotion";
import { Paper } from "../components";
import { c, display, mono, sans } from "../theme";

const Panel: React.FC<{ src: string; caption: string; show: number }> = ({ src, caption, show }) => (
  <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", opacity: show, transform: `scale(${interpolate(show, [0, 1], [0.98, 1])})` }}>
    <div style={{ width: 1160, borderRadius: 12, overflow: "hidden", border: `1px solid ${c.line}`, boxShadow: "0 26px 70px -34px rgba(35,32,26,0.35)", background: c.panel }}>
      <Img src={staticFile(src)} style={{ width: "100%", display: "block" }} />
    </div>
    <div style={{ fontFamily: mono, fontSize: 22, color: c.ink2, marginTop: 18 }}>{caption}</div>
  </div>
);

export const Frontier: React.FC = () => {
  const frame = useCurrentFrame();
  const title = interpolate(frame, [0, 14], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  // Two frontier views, one after the other.
  const a = interpolate(frame, [20, 34, 96, 110], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const b = interpolate(frame, [104, 118], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  return (
    <Paper>
      <AbsoluteFill style={{ padding: "56px 120px 40px", justifyContent: "flex-start", alignItems: "center" }}>
        <div style={{ opacity: title, transform: `translateY(${(1 - title) * 14}px)`, textAlign: "center", marginBottom: 26 }}>
          <div style={{ fontFamily: display, fontWeight: 700, fontSize: 56, color: c.ink, letterSpacing: -0.5 }}>
            Every host on the quality frontier.
          </div>
          <div style={{ fontFamily: sans, fontSize: 28, color: c.ink2, marginTop: 10 }}>
            Quality against latency, then against cost. Compound plots the trade-off from your own episodes.
          </div>
        </div>

        <div style={{ position: "relative", width: 1160, height: 720 }}>
          <Panel src="viz/speed-light.png" caption="quality vs median latency per model call" show={a} />
          <Panel src="viz/frontier-light.png" caption="quality vs cost per episode" show={b} />
        </div>
      </AbsoluteFill>
    </Paper>
  );
};
