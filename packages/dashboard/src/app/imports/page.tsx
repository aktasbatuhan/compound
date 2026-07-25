import { createApiClient } from "../../lib/api";
import { Empty, ErrorPanel } from "../ui";

export const dynamic = "force-dynamic";

function num(report: unknown, path: string[]): number | undefined {
  let current: unknown = report;
  for (const key of path) {
    if (current && typeof current === "object" && key in current) {
      current = (current as Record<string, unknown>)[key];
    } else {
      return undefined;
    }
  }
  return typeof current === "number" ? current : undefined;
}

export default async function ImportsPage() {
  const api = createApiClient();
  try {
    const page = await api.listImports(100);
    if (page.items.length === 0) {
      return (
        <div>
          <h2>Imports</h2>
          <Empty>No imports yet. Run `compound import &lt;file&gt;`.</Empty>
        </div>
      );
    }

    return (
      <div>
        <h2>Imports</h2>
        <div className="card table-scroll">
          <table>
            <thead>
              <tr>
                <th>Batch</th>
                <th>Importer</th>
                <th>Status</th>
                <th>Eval-ready</th>
                <th>Diagnostic</th>
                <th>Rejected</th>
                <th>Redactions</th>
                <th>Started</th>
              </tr>
            </thead>
            <tbody>
              {page.items.map((batch) => {
                const report = batch.report;
                const redactions =
                  report && typeof report === "object" && "redactions_by_rule" in report
                    ? Object.values(
                        (report as { redactions_by_rule: Record<string, number> })
                          .redactions_by_rule,
                      ).reduce((sum, n) => sum + n, 0)
                    : undefined;
                return (
                  <tr key={batch.id}>
                    <td className="muted">{batch.id.slice(0, 8)}</td>
                    <td>{batch.importer}</td>
                    <td>
                      <span
                        className={`badge ${
                          batch.status === "completed"
                            ? "good"
                            : batch.status === "failed"
                              ? "bad"
                              : "warn"
                        }`}
                      >
                        {batch.status}
                      </span>
                    </td>
                    <td>{num(report, ["counts", "eval_ready"]) ?? "—"}</td>
                    <td>{num(report, ["counts", "diagnostic"]) ?? "—"}</td>
                    <td>{num(report, ["counts", "rejected"]) ?? "—"}</td>
                    <td>{redactions ?? "—"}</td>
                    <td className="muted">{new Date(batch.started_at).toLocaleString()}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  } catch (error) {
    return (
      <div>
        <h2>Imports</h2>
        <ErrorPanel error={error} />
      </div>
    );
  }
}
