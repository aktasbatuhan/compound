import { createApiClient } from "../../lib/api";
import { Empty, ErrorPanel } from "../ui";

export const dynamic = "force-dynamic";

export default async function JudgesPage() {
  const api = createApiClient();
  try {
    const { items } = await api.listJudges();
    return (
      <div>
        <h2>Judges</h2>
        <p className="muted">
          A judge may feed a gate only once it is <strong>calibrated</strong> — its agreement with
          human labels (Cohen’s κ) clears the task’s threshold on enough labelled cases. Until then
          it <strong>abstains</strong>. Calibration is pinned to the exact judge model, prompt
          version, and rubric; changing any of them requires re-calibrating. Measure with{" "}
          <code>compound judge calibrate &lt;task&gt;</code>.
        </p>
        {items.length === 0 ? (
          <Empty>
            No judge calibrated yet. Configure <code>judges.&lt;task&gt;</code> and run{" "}
            <code>compound judge calibrate &lt;task&gt; --paid --cap USD</code>.
          </Empty>
        ) : (
          <div className="card table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Status</th>
                  <th>Judge model</th>
                  <th>Agreement κ</th>
                  <th>CI</th>
                  <th>n</th>
                  <th>Threshold</th>
                  <th>Measured</th>
                </tr>
              </thead>
              <tbody>
                {items.map((j) => (
                  <tr key={j.id}>
                    <td>
                      <strong>{j.task_key}</strong>
                      <div className="muted" style={{ fontSize: 12 }}>
                        {j.mode} · prompt {j.prompt_version}
                      </div>
                    </td>
                    <td>
                      <span className={`badge ${j.calibrated ? "good" : "warn"}`}>
                        {j.calibrated ? "calibrated" : "abstains"}
                      </span>
                    </td>
                    <td className="muted">{j.judge_model}</td>
                    <td>{j.agreement_kappa.toFixed(3)}</td>
                    <td className="muted">
                      [{j.kappa_ci[0].toFixed(2)}, {j.kappa_ci[1].toFixed(2)}]
                    </td>
                    <td>{j.n}</td>
                    <td className="muted">{j.threshold}</td>
                    <td className="muted">{new Date(j.measured_at).toLocaleString()}</td>
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
        <h2>Judges</h2>
        <ErrorPanel error={error} />
      </div>
    );
  }
}
