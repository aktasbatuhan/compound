import React from "react";
import { AbsoluteFill, Easing, interpolate, useCurrentFrame } from "remotion";
import { Paper, Pill } from "../components";
import { c, display, HOSTS, Host, mono, sans, statusColor } from "../theme";

const TRACK = 1120;
const SCALE_MAX = 60; // percent -> px scale headroom

const BarRow: React.FC<{ host: Host; index: number }> = ({ host, index }) => {
  const frame = useCurrentFrame();
  const start = 40 + index * 9;
  const grow = interpolate(frame, [start, start + 26], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const appear = interpolate(frame, [start - 6, start + 6], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const color = statusColor(host.status);
  const w = (host.rate / SCALE_MAX) * TRACK * grow;
  const shownRate = (host.rate * grow).toFixed(1);

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 22, opacity: appear }}>
      <div style={{ width: 300, textAlign: "right", fontFamily: mono, fontSize: 25, color: c.ink }}>
        {host.name}
        {host.quant ? <span style={{ color: c.ink3 }}> {host.quant}</span> : null}
      </div>
      <div style={{ width: TRACK, height: 40, position: "relative", background: c.lineSoft, borderRadius: 5 }}>
        <div
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            height: "100%",
            width: Math.max(w, host.rate === 0 ? 0 : 3),
            background: color,
            borderRadius: 5,
          }}
        />
        <div
          style={{
            position: "absolute",
            left: Math.max(w, host.rate === 0 ? 6 : 3) + 14,
            top: 4,
            fontFamily: mono,
            fontWeight: 500,
            fontSize: 25,
            color: host.rate === 0 ? c.ink3 : c.ink,
          }}
        >
          {shownRate}%
        </div>
      </div>
      <div style={{ width: 210, opacity: interpolate(frame, [start + 18, start + 30], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }) }}>
        <Pill label={host.status} color={color} style={{ fontSize: 18 }} />
      </div>
    </div>
  );
};

export const Sweep: React.FC = () => {
  const frame = useCurrentFrame();
  const title = interpolate(frame, [0, 16], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const cap = interpolate(frame, [150, 166], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  return (
    <Paper>
      <AbsoluteFill style={{ padding: "88px 110px", justifyContent: "center" }}>
        <div style={{ opacity: title, transform: `translateY(${(1 - title) * 16}px)`, marginBottom: 12 }}>
          <div style={{ fontFamily: display, fontWeight: 700, fontSize: 58, color: c.ink, letterSpacing: -0.5 }}>
            Same weights. The serving layer is not.
          </div>
          <div style={{ fontFamily: sans, fontSize: 28, color: c.ink2, marginTop: 10 }}>
            End-to-end task success by host. The failures are infrastructure, not the model.
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 15, margin: "40px 0 24px" }}>
          {HOSTS.map((h, i) => (
            <BarRow key={h.name} host={h} index={i} />
          ))}
        </div>

        <div style={{ opacity: cap, fontFamily: mono, fontSize: 21, color: c.ink3, alignSelf: "flex-start" }}>
          fireworks loses 15 of 42 episodes to shared-pool rate limits; the four clean hosts tie at 45 to 57%.
        </div>
      </AbsoluteFill>
    </Paper>
  );
};
