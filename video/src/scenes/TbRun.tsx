import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Paper, Typed } from "../components";
import { c, display, HOSTS, mono, sans, statusColor } from "../theme";

const TASKS = ["build-linux-kernel-qemu", "chess-best-move", "crack-7z-hash", "cron-broken-network", "configure-git-webserver", "count-dataset-tokens"];

export const TbRun: React.FC = () => {
  const frame = useCurrentFrame();
  const title = interpolate(frame, [0, 14], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const type1 = interpolate(frame, [18, 58], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const type2 = interpolate(frame, [50, 78], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const meta = interpolate(frame, [86, 100], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  return (
    <Paper>
      <AbsoluteFill style={{ padding: "0 120px", justifyContent: "center", alignItems: "center" }}>
        <div style={{ opacity: title, transform: `translateY(${(1 - title) * 14}px)`, marginBottom: 30, textAlign: "center" }}>
          <div style={{ fontFamily: display, fontWeight: 700, fontSize: 56, color: c.ink, letterSpacing: -0.5 }}>
            One command runs the real benchmark.
          </div>
          <div style={{ fontFamily: sans, fontSize: 28, color: c.ink2, marginTop: 10 }}>
            The official terminal-bench harness in Docker, graded by each task's own tests.
          </div>
        </div>

        <div style={{ width: 1440, borderRadius: 10, overflow: "hidden", border: `1px solid ${c.line}`, boxShadow: "0 26px 64px -32px rgba(35,32,26,0.4)" }}>
          <div style={{ height: 40, background: "#0f0e0a", display: "flex", alignItems: "center", padding: "0 16px", gap: 8 }}>
            <span style={{ width: 12, height: 12, borderRadius: "50%", background: "#e0564b" }} />
            <span style={{ width: 12, height: 12, borderRadius: "50%", background: "#e3b73f" }} />
            <span style={{ width: 12, height: 12, borderRadius: "50%", background: "#4caf50" }} />
          </div>
          <div style={{ background: "#17150f", padding: "24px 30px", fontFamily: mono, fontSize: 23, lineHeight: 1.5 }}>
            <div>
              <span style={{ color: "#7f97ff" }}>$ </span>
              <Typed text="compound-bench run terminal_bench --model deepseek/deepseek-v4-flash-0731 \" progress={type1} style={{ color: "#f3efe6" }} caret={false} />
            </div>
            <div style={{ paddingLeft: 22 }}>
              <Typed text="--providers dw/realtime,dw/flex,deepinfra,parasail,fireworks --trials 3 --go" progress={type2} style={{ color: "#f3efe6" }} />
            </div>

            <div style={{ opacity: meta, marginTop: 18, color: "#a8a094" }}>
              14 terminal-bench tasks &times; 3 trials &times; 5 hosts &nbsp;&rarr;&nbsp; 210 episodes, host verified on every call
            </div>
            <div style={{ opacity: meta, marginTop: 6, color: "#6f6a60", fontSize: 20 }}>
              {TASKS.join("   ")}   …
            </div>

            <div style={{ marginTop: 20, borderTop: "1px solid #2a2820", paddingTop: 16 }}>
              {HOSTS.map((h, i) => {
                const at = 118 + i * 12;
                const on = interpolate(frame, [at, at + 12], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
                const col = h.status === "healthy" ? "#f3efe6" : statusColor(h.status);
                return (
                  <div key={h.name} style={{ display: "flex", opacity: on, transform: `translateX(${(1 - on) * 14}px)`, lineHeight: 1.7 }}>
                    <span style={{ width: 60, color: "#4caf50" }}>{h.pass > 0 ? "✓" : "✗"}</span>
                    <span style={{ width: 380, color: "#d9d3c7" }}>{h.name}{h.quant ? ` ${h.quant}` : ""}</span>
                    <span style={{ width: 130, color: "#a8a094" }}>{h.pass}/42</span>
                    <span style={{ width: 120, color: col, fontWeight: 500 }}>{h.rate.toFixed(1)}%</span>
                    <span style={{ color: "#6f6a60", fontSize: 20 }}>{h.status}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </AbsoluteFill>
    </Paper>
  );
};
