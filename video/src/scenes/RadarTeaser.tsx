import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Paper } from "../components";
import { c, display, mono } from "../theme";
import { AXES, BEATS, CX, CY, pt, polyStr, RHOSTS } from "./radarData";

// Fast, wordless cold-open: the polygons flip their winner axis by axis.
// Withholds the "find yours" conclusion — that lands in the full payoff later.
const B0 = 18, BL = 16;
const OX = 360, OY = -66; // center the CX=600 radar on the 1920 canvas

export const RadarTeaser: React.FC = () => {
  const frame = useCurrentFrame();
  const draw = interpolate(frame, [0, 16], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const beatIdx = frame < B0 ? -1 : Math.min(3, Math.floor((frame - B0) / BL));
  const beat = beatIdx >= 0 ? BEATS[beatIdx] : null;
  const beatLocal = beat ? frame - (B0 + beatIdx * BL) : 0;
  const wordOp = beat ? interpolate(beatLocal, [0, 4, BL - 4, BL], [0, 1, 1, 0.3], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }) : 0;
  const kicker = interpolate(frame, [4, 16], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  const rings = [0.25, 0.5, 0.75, 1.0];

  return (
    <Paper>
      <AbsoluteFill>
        <div style={{ position: "absolute", top: 92, left: 0, right: 0, textAlign: "center", opacity: kicker }}>
          <div style={{ fontFamily: mono, fontSize: 24, letterSpacing: 4, color: c.ink3 }}>ONE MODEL, SIX HOSTS</div>
        </div>

        <svg width={1920} height={1080} style={{ position: "absolute", inset: 0 }}>
          <g transform={`translate(${OX}, ${OY})`}>
            {rings.map((lv, k) => (
              <polygon key={k} points={polyStr(AXES.map(() => lv * draw))} fill="none" stroke={c.line} strokeWidth={1} />
            ))}
            {AXES.map((ax, i) => {
              const end = pt(i, 1 * draw);
              const lp = pt(i, 1.14);
              const activeAxis = beat && beat.axis === i;
              return (
                <g key={ax}>
                  <line x1={CX} y1={CY} x2={end[0]} y2={end[1]} stroke={activeAxis ? c.accent : c.line} strokeWidth={activeAxis ? 2.5 : 1} />
                  <text x={lp[0]} y={lp[1]} textAnchor="middle" dominantBaseline="middle" style={{ fontFamily: mono, fontSize: 21, fontWeight: activeAxis ? 600 : 400, fill: activeAxis ? c.accent : c.ink3 }}>
                    {ax}
                  </text>
                </g>
              );
            })}
            {RHOSTS.map((h) => {
              const win = beat ? beat.winners.includes(h.key) : false;
              const op = beat ? (win ? 1 : 0.12) : 0.6 * draw;
              return (
                <g key={h.key} style={{ transformOrigin: `${CX}px ${CY}px`, transform: `scale(${draw})` }}>
                  <polygon points={polyStr(h.v)} fill={h.hex} fillOpacity={win ? 0.18 : 0.05} stroke={h.hex} strokeWidth={win ? 3.4 : 1.6} strokeLinejoin="round" opacity={op} />
                </g>
              );
            })}
            {beat &&
              RHOSTS.filter((h) => beat.winners.includes(h.key)).map((h) => {
                const p = pt(beat.axis, h.v[beat.axis]);
                return <circle key={h.key} cx={p[0]} cy={p[1]} r={7} fill={h.hex} stroke="#fff" strokeWidth={2} opacity={wordOp} />;
              })}
          </g>
        </svg>

        {beat ? (
          <div style={{ position: "absolute", bottom: 88, left: 0, right: 0, textAlign: "center", opacity: wordOp }}>
            <span style={{ fontFamily: display, fontWeight: 800, fontSize: 68, color: c.accent, letterSpacing: 1 }}>{beat.eyebrow}?</span>
          </div>
        ) : null}
      </AbsoluteFill>
    </Paper>
  );
};
