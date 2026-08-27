import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Paper } from "../components";
import { c, display, mono, sans } from "../theme";

type Row = { name: string; quant: string; price: string; up: boolean };

// Real rows from `compound-bench providers deepseek/deepseek-v4-flash-0731`.
const ROWS: Row[] = [
  { name: "open-inference", quant: "fp4", price: "$0.030", up: true },
  { name: "deepinfra", quant: "fp8", price: "$0.080", up: true },
  { name: "makora", quant: "?", price: "$0.090", up: false },
  { name: "morph", quant: "bf16", price: "$0.099", up: true },
  { name: "baseten", quant: "fp8", price: "$0.130", up: true },
  { name: "parasail", quant: "fp8", price: "$0.140", up: false },
  { name: "novita", quant: "fp8", price: "$0.440", up: true },
];

const quantColor = (q: string) => (q === "fp4" ? c.warn : q === "bf16" ? c.good : q === "?" ? c.ink3 : c.accent);

export const Providers: React.FC = () => {
  const frame = useCurrentFrame();
  const title = interpolate(frame, [0, 16], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  return (
    <Paper>
      <AbsoluteFill style={{ padding: "0 130px", justifyContent: "center" }}>
        <div style={{ opacity: title, transform: `translateY(${(1 - title) * 16}px)`, marginBottom: 40 }}>
          <div style={{ fontFamily: display, fontWeight: 700, fontSize: 58, color: c.ink, letterSpacing: -0.5 }}>
            One model. Many hosts. No two alike.
          </div>
          <div style={{ fontFamily: sans, fontSize: 30, color: c.ink2, marginTop: 12 }}>
            Different quantization, price, and uptime. None of it shows on a pricing page.
          </div>
        </div>

        <div
          style={{
            background: c.panel,
            border: `1px solid ${c.line}`,
            borderRadius: 12,
            overflow: "hidden",
            width: 1180,
            boxShadow: "0 24px 60px -34px rgba(35,32,26,0.3)",
          }}
        >
          <div style={{ display: "grid", gridTemplateColumns: "1.4fr 0.8fr 0.9fr 0.8fr", padding: "16px 34px", borderBottom: `1px solid ${c.line}`, fontFamily: mono, fontSize: 20, color: c.ink3, letterSpacing: 0.6 }}>
            <div>PROVIDER</div>
            <div>QUANT</div>
            <div>$/M IN</div>
            <div>STATUS</div>
          </div>
          {ROWS.map((r, i) => {
            const on = interpolate(frame, [24 + i * 8, 24 + i * 8 + 12], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
            return (
              <div
                key={r.name}
                style={{
                  display: "grid",
                  gridTemplateColumns: "1.4fr 0.8fr 0.9fr 0.8fr",
                  alignItems: "center",
                  padding: "15px 34px",
                  borderBottom: i < ROWS.length - 1 ? `1px solid ${c.lineSoft}` : "none",
                  fontFamily: mono,
                  fontSize: 26,
                  color: c.ink,
                  opacity: on,
                  transform: `translateX(${(1 - on) * 18}px)`,
                }}
              >
                <div>openrouter/{r.name}</div>
                <div style={{ color: quantColor(r.quant), fontWeight: 500 }}>{r.quant}</div>
                <div>{r.price}</div>
                <div style={{ color: r.up ? c.good : c.bad, fontWeight: 500 }}>{r.up ? "up" : "down"}</div>
              </div>
            );
          })}
        </div>
      </AbsoluteFill>
    </Paper>
  );
};
