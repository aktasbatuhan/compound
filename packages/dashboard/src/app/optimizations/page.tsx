import { createApiClient } from "../../lib/api";
import { Empty, ErrorPanel } from "../ui";

export const dynamic = "force-dynamic";

const pct = (x: number) => `${(x * 100).toFixed(0)}%`;

export default async function OptimizationsPage() {
  const api = createApiClient();
  try {
    const { items } = await api.listOptimizations();
    return (
      <div>
        <h2>Optimizations</h2>
        <p className="muted">
          GEPA prompt-optimization runs, launched with <code>compound optimize</code>. Each is a{" "}
          <strong>proposal</strong>: the optimized prompt improved the validation score on the
          task’s train/val cases (never the sealed set). Adopting it means re-gating it on the
          sealed decision set and a human approval — optimization never self-certifies.
        </p>
        {items.length === 0 ? (
          <Empty>
            No optimization run yet. Run <code>compound optimize &lt;task&gt; --candidate M</code>.
          </Empty>
        ) : (
          <div className="card table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Candidate</th>
                  <th>Validation</th>
                  <th>Δ</th>
                  <th>Reason</th>
                  <th>Reflections</th>
                  <th>When</th>
                </tr>
              </thead>
              <tbody>
                {items.map((o) => {
                  const delta = o.after_val_score - o.before_val_score;
                  return (
                    <tr key={o.id}>
                      <td>
                        <strong>{o.task_key}</strong>
                      </td>
                      <td className="muted">{o.candidate_model}</td>
                      <td>
                        {pct(o.before_val_score)} → {pct(o.after_val_score)}{" "}
                        <span className="muted">({o.val_cases})</span>
                      </td>
                      <td>
                        <span className={`badge ${delta > 0 ? "good" : delta < 0 ? "bad" : ""}`}>
                          {delta >= 0 ? "+" : ""}
                          {(delta * 100).toFixed(0)}pp
                        </span>
                      </td>
                      <td className="muted">{o.eligibility_reason ?? "—"}</td>
                      <td>{o.reflection_calls}</td>
                      <td className="muted">{new Date(o.created_at).toLocaleString()}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    );
  } catch (error) {
    return (
      <div>
        <h2>Optimizations</h2>
        <ErrorPanel error={error} />
      </div>
    );
  }
}
