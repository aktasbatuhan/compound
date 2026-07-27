import { createApiClient } from "../../lib/api";
import { Empty, ErrorPanel } from "../ui";

export const dynamic = "force-dynamic";

const usd = (x: number) => `$${x.toFixed(4)}`;
const ms = (x: number) => `${Math.round(x)} ms`;

export default async function TelemetryPage() {
  const api = createApiClient();
  try {
    const { items } = await api.listTelemetry();
    return (
      <div>
        <h2>Telemetry</h2>
        <p className="muted">
          The operational side of a routing decision, next to the gate’s quality verdict: latency,
          cost, and throughput per task × model × provider, aggregated from the completion cache.
          Each completion counts once, however many runs reused it. TPS is output tokens over full
          request latency — provider queueing and time-to-first-token included.
        </p>
        {items.length === 0 ? (
          <Empty>
            No telemetry yet. Run <code>compound experiment &lt;task&gt; &lt;model&gt;</code> with
            paid calls (or against a warm cache) first.
          </Empty>
        ) : (
          <div className="card table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Model</th>
                  <th>Provider</th>
                  <th>n</th>
                  <th>Latency p50</th>
                  <th>Queue p50</th>
                  <th>Decode p50</th>
                  <th>p95</th>
                  <th>Cost / case</th>
                  <th>Tokens in/out (mean)</th>
                  <th>Output TPS</th>
                </tr>
              </thead>
              <tbody>
                {items.map((row) => (
                  <tr key={`${row.task_key}-${row.model}-${row.provider}`}>
                    <td>
                      <strong>{row.task_key}</strong>
                    </td>
                    <td>{row.model}</td>
                    <td className="muted">{row.provider}</td>
                    <td>{row.completions}</td>
                    <td>{ms(row.latency_p50_ms)}</td>
                    <td className="muted">{ms(row.queue_p50_ms)}</td>
                    <td className="muted">{ms(row.decode_p50_ms)}</td>
                    <td className="muted">{ms(row.latency_p95_ms)}</td>
                    <td>{usd(row.mean_cost_usd)}</td>
                    <td className="muted">
                      {Math.round(row.mean_input_tokens)} / {Math.round(row.mean_output_tokens)}
                    </td>
                    <td>{row.output_tps.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    );
  } catch (error) {
    return (
      <div>
        <h2>Telemetry</h2>
        <ErrorPanel error={error} />
      </div>
    );
  }
}
