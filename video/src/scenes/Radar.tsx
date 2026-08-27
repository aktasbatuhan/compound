import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Paper } from "../components";
import { c, display, mono, sans } from "../theme";
import { AXES, BEATS, CX, CY, pt, polyStr, RHOSTS } from "./radarData";

const B0 = 52, BL = 56;

export const Radar: React.FC = () => {
  const frame = useCurrentFrame();
  const title = interpolate(frame, [0, 14], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const draw = interpolate(frame, [16, 44], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  const beatIdx = frame < B0 ? -1 : Math.min(4, Math.floor((frame - B0) / BL)); // 4 = finale
  const beat = beatIdx >= 0 && beatIdx <= 3 ? BEATS[beatIdx] : null;
  const beatLocal = beat ? frame - (B0 + beatIdx * BL) : 0;
  const calloutOp = beat ? interpolate(beatLocal, [0, 8, BL - 10, BL], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }) : 0;
  const finale = interpolate(frame, [B0 + 4 * BL, B0 + 4 * BL + 16], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  const rings = [0.25, 0.5, 0.75, 1.0];

  return (
    <Paper>
      <AbsoluteFill>
        <div style={{ position: "absolute", top: 60, left: 120, right: 120, opacity: title, transform: `translateY(${(1 - title) * 12}px)` }}>
          <div style={{ fontFamily: display, fontWeight: 700, fontSize: 54, color: c.ink, letterSpacing: -0.5 }}>
            No host wins every axis.
          </div>
          <div style={{ fontFamily: sans, fontSize: 27, color: c.ink2, marginTop: 8 }}>
            Six axes from one terminal-bench run, normalized so the outer edge is best.
          </div>
        </div>

        <svg width={1920} height={1080} style={{ position: "absolute", inset: 0 }}>
          {/* rings */}
          {rings.map((lv, k) => (
            <polygon key={k} points={polyStr(AXES.map(() => lv * draw))} fill="none" stroke={c.line} strokeWidth={1} />
          ))}
          {/* spokes + axis labels */}
          {AXES.map((ax, i) => {
            const end = pt(i, 1 * draw);
            const lp = pt(i, 1.14);
            const activeAxis = beat && beat.axis === i;
            return (
              <g key={ax}>
                <line x1={CX} y1={CY} x2={end[0]} y2={end[1]} stroke={activeAxis ? c.accent : c.line} strokeWidth={activeAxis ? 2.5 : 1} />
                <text x={lp[0]} y={lp[1]} textAnchor="middle" dominantBaseline="middle"
                  style={{ fontFamily: mono, fontSize: 21, fontWeight: activeAxis ? 600 : 400, fill: activeAxis ? c.accent : c.ink3 }}>
                  {ax}
                </text>
              </g>
            );
          })}
          {/* host polygons */}
          {RHOSTS.map((h) => {
            const emph = beat ? (beat.winners.includes(h.key) ? 1 : 0) : 0.5;
            const op = beat ? (beat.winners.includes(h.key) ? 1 : 0.1) : 0.85 * draw;
            const sw = emph === 1 ? 3.2 : 1.6;
            const fill = emph === 1 ? 0.16 : 0.05;
            return (
              <g key={h.key} style={{ transformOrigin: `${CX}px ${CY}px`, transform: `scale(${draw})` }}>
                <polygon points={polyStr(h.v)} fill={h.hex} fillOpacity={fill} stroke={h.hex} strokeWidth={sw} strokeLinejoin="round" opacity={op} />
              </g>
            );
          })}
          {/* winner markers on active axis */}
          {beat &&
            RHOSTS.filter((h) => beat.winners.includes(h.key)).map((h) => {
              const p = pt(beat.axis, h.v[beat.axis]);
              return <circle key={h.key} cx={p[0]} cy={p[1]} r={7} fill={h.hex} stroke="#fff" strokeWidth={2} opacity={calloutOp} />;
            })}
        </svg>

        {/* right-side callout / finale */}
        <div style={{ position: "absolute", right: 130, top: 360, width: 620 }}>
          {beat ? (
            <div style={{ opacity: calloutOp }}>
              <div style={{ fontFamily: mono, fontSize: 22, letterSpacing: 2, color: c.ink3 }}>{beat.eyebrow}</div>
              <div style={{ fontFamily: display, fontWeight: 800, fontSize: 60, color: c.ink, letterSpacing: -0.5, marginTop: 6 }}>{beat.head}</div>
              <div style={{ marginTop: 20, display: "flex", flexDirection: "column", gap: 12 }}>
                {RHOSTS.filter((h) => beat.winners.includes(h.key)).map((h) => (
                  <div key={h.key} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <span style={{ width: 16, height: 16, borderRadius: 4, background: h.hex }} />
                    <span style={{ fontFamily: mono, fontSize: 26, color: c.ink }}>{h.name}</span>
                  </div>
                ))}
              </div>
              <div style={{ fontFamily: sans, fontSize: 23, color: c.ink2, marginTop: 16 }}>{beat.detail}</div>
            </div>
          ) : null}

          {finale > 0 ? (
            <div style={{ opacity: finale, transform: `translateY(${(1 - finale) * 16}px)`, position: "absolute", top: -30, width: 700 }}>
              <div style={{ fontFamily: display, fontWeight: 800, fontSize: 52, color: c.ink, letterSpacing: -0.6, lineHeight: 1.1 }}>
                Each workload has<br />its own best provider.
              </div>
              <div style={{ fontFamily: display, fontWeight: 800, fontSize: 52, color: c.accent, marginTop: 12, letterSpacing: -0.6 }}>Find yours.</div>
            </div>
          ) : null}
        </div>
      </AbsoluteFill>
    </Paper>
  );
};
