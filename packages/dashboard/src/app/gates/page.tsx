import { createApiClient } from "../../lib/api";
import { Empty, ErrorPanel, gateBadge } from "../ui";

export const dynamic = "force-dynamic";

const pp = (x: number) => `${x >= 0 ? "+" : ""}${(x * 100).toFixed(1)}pp`;

export default async function GatesPage() {
  const api = createApiClient();
  try {
    const { items } = await api.listGates();
    return (
      <div>
        <h2>Gate decisions</h2>
        <p className="muted">
          Each row is a pre-declared non-inferiority decision on a task’s sealed decision set,
          decided with <code>compound gate</code>. The verdict travels with its confidence interval
          — never a bare mean. The sealed cases themselves are not shown here; inspect per-case
          disagreements from the CLI, which requires stating why the seal is opened.
        </p>
        {items.length === 0 ? (
          <Empty>
            No gate decided yet. Run{" "}
            <code>
              compound gate &lt;task&gt; --candidate M --reference M --reason &quot;…&quot;
            </code>
            .
          </Empty>
        ) : (
          <div className="card table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Verdict</th>
                  <th>Candidate vs reference</th>
                  <th>Δ (cand − ref)</th>
                  <th>CI</th>
                  <th>n</th>
                  <th>Reason</th>
                  <th>Decided</th>
                </tr>
              </thead>
              <tbody>
                {items.map((gate) => (
                  <tr key={gate.id}>
                    <td>
                      <strong>{gate.task_key}</strong>
                      <div className="muted" style={{ fontSize: 12 }}>
                        {gate.metric} · {gate.mode}
                      </div>
                    </td>
                    <td>{gateBadge(gate.outcome)}</td>
                    <td>
                      <div>
                        {gate.candidate_model} {(gate.candidate_rate * 100).toFixed(0)}%
                      </div>
                      <div className="muted">
                        {gate.reference_model} {(gate.reference_rate * 100).toFixed(0)}%
                      </div>
                    </td>
                    <td>{pp(gate.delta)}</td>
                    <td className="muted">
                      [{pp(gate.ci[0])}, {pp(gate.ci[1])}]
                      <div style={{ fontSize: 12 }}>{Math.round(gate.confidence * 100)}%</div>
                    </td>
                    <td>{gate.n}</td>
                    <td className="muted">{gate.firewall_reason}</td>
                    <td className="muted">{new Date(gate.decided_at).toLocaleString()}</td>
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
        <h2>Gate decisions</h2>
        <ErrorPanel error={error} />
      </div>
    );
  }
}
