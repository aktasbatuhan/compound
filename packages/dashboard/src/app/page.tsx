import { createApiClient } from "../lib/api";
import { ErrorPanel } from "./ui";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const api = createApiClient();
  try {
    const [health, tracesStats, casesStats] = await Promise.all([
      api.getHealth(),
      api.getTracesStats(),
      api.getCasesStats(),
    ]);

    const caseTotal = casesStats.by_partition.reduce((sum, row) => sum + row.count, 0);

    return (
      <div>
        <h2>Overview</h2>
        <div className="card">
          <div className="row">
            <span className="badge good">API {health.status}</span>
            <span className="muted">v{health.version}</span>
            <span className="muted">trace schema v{health.trace_schema_version}</span>
          </div>
        </div>

        <div className="card">
          <h3>Traces</h3>
          <div className="row">
            <span className="badge good">
              {tracesStats.by_validation_class.eval_ready} eval-ready
            </span>
            <span className="badge warn">
              {tracesStats.by_validation_class.diagnostic} diagnostic
            </span>
          </div>
        </div>

        <div className="card">
          <h3>Cases</h3>
          <div className="row">
            <span className="badge">{caseTotal} total</span>
            {casesStats.by_partition.map((row) => (
              <span
                key={row.partition}
                className={`badge ${row.partition === "decision_test" ? "sealed" : ""}`}
              >
                {row.count} {row.partition}
              </span>
            ))}
          </div>
          <p className="muted" style={{ marginBottom: 0 }}>
            The <strong>decision_test</strong> partition is sealed — its cases never appear in any
            list, only in these counts.
          </p>
        </div>
      </div>
    );
  } catch (error) {
    return (
      <div>
        <h2>Overview</h2>
        <ErrorPanel error={error} />
      </div>
    );
  }
}
