import type { TraceListFilters } from "../../lib/api";
import { createApiClient } from "../../lib/api";
import { Empty, ErrorPanel } from "../ui";

export const dynamic = "force-dynamic";

export default async function TracesPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | undefined>>;
}) {
  const params = await searchParams;
  const api = createApiClient();
  const filters: TraceListFilters = {
    task_key: params.task_key || undefined,
    validation_class:
      (params.validation_class as TraceListFilters["validation_class"]) || undefined,
    limit: 100,
  };

  try {
    const [page, stats] = await Promise.all([api.listTraces(filters), api.getTracesStats()]);

    return (
      <div>
        <h2>Traces</h2>

        <div className="card">
          <h3>Diagnostic queue</h3>
          {stats.by_diagnostic_reason.length === 0 ? (
            <div className="muted">No diagnostic traces — nothing needs attention.</div>
          ) : (
            <table>
              <tbody>
                {stats.by_diagnostic_reason.map((row) => (
                  <tr key={row.reason}>
                    <td>
                      <span className="badge warn">{row.count}</span>
                    </td>
                    <td>{row.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="muted" style={{ marginBottom: 0 }}>
            A trace is counted once per reason it carries, so these can sum to more than the
            diagnostic total.
          </p>
        </div>

        <form className="filters" method="get">
          <input name="task_key" placeholder="task key" defaultValue={params.task_key ?? ""} />
          <select name="validation_class" defaultValue={params.validation_class ?? ""}>
            <option value="">class: any</option>
            <option value="eval_ready">eval_ready</option>
            <option value="diagnostic">diagnostic</option>
          </select>
          <button type="submit">Filter</button>
        </form>

        {page.items.length === 0 ? (
          <Empty>No traces match.</Empty>
        ) : (
          <div className="card table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Trace</th>
                  <th>Task</th>
                  <th>Class</th>
                  <th>Reasons</th>
                </tr>
              </thead>
              <tbody>
                {page.items.map((t) => {
                  const traceId = String(t.trace.trace_id ?? "");
                  return (
                    <tr key={traceId}>
                      <td className="muted">{traceId}</td>
                      <td>{String(t.trace.task_key ?? "(unassigned)")}</td>
                      <td>
                        <span
                          className={`badge ${t.validation_class === "eval_ready" ? "good" : "warn"}`}
                        >
                          {t.validation_class}
                        </span>
                      </td>
                      <td className="muted">{t.diagnostic_reasons.join(", ")}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="muted">
              Showing {page.items.length} of {page.total}.
            </p>
          </div>
        )}
      </div>
    );
  } catch (error) {
    return (
      <div>
        <h2>Traces</h2>
        <ErrorPanel error={error} />
      </div>
    );
  }
}
