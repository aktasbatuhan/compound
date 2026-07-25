import type { CaseListFilters } from "../../lib/api";
import { createApiClient } from "../../lib/api";
import { Empty, ErrorPanel, partitionBadge } from "../ui";

export const dynamic = "force-dynamic";

const PARTITIONS = ["optimization_train", "optimizer_validation", "judge_calibration"] as const;
const PROVENANCES = [
  "observed_output",
  "human_golden",
  "deterministic_outcome",
  "user_feedback",
  "synthetic_label",
] as const;
const REVIEW_STATES = ["unreviewed", "approved", "rejected", "needs_edit"] as const;

export default async function CasesPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | undefined>>;
}) {
  const params = await searchParams;
  const api = createApiClient();
  const filters: CaseListFilters = {
    task_key: params.task_key || undefined,
    partition: (params.partition as CaseListFilters["partition"]) || undefined,
    provenance: (params.provenance as CaseListFilters["provenance"]) || undefined,
    review_state: (params.review_state as CaseListFilters["review_state"]) || undefined,
    limit: 100,
  };

  try {
    const page = await api.listCases(filters);
    return (
      <div>
        <h2>Cases</h2>
        <p className="muted">
          Sealed <strong>decision_test</strong> cases are never listed — the API does not return
          them.
        </p>

        <form className="filters" method="get">
          <input name="task_key" placeholder="task key" defaultValue={params.task_key ?? ""} />
          <Select name="partition" value={params.partition} options={PARTITIONS} />
          <Select name="provenance" value={params.provenance} options={PROVENANCES} />
          <Select name="review_state" value={params.review_state} options={REVIEW_STATES} />
          <button type="submit">Filter</button>
        </form>

        {page.items.length === 0 ? (
          <Empty>No cases match.</Empty>
        ) : (
          <div className="card table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Case</th>
                  <th>Task</th>
                  <th>Partition</th>
                  <th>Provenance</th>
                  <th>Review</th>
                  <th>Source trace</th>
                </tr>
              </thead>
              <tbody>
                {page.items.map((c) => (
                  <tr key={c.case_id}>
                    <td>
                      <a href={`/cases/${encodeURIComponent(c.case_id)}`}>
                        {c.case_id.slice(0, 20)}…
                      </a>
                    </td>
                    <td>{c.task_key}</td>
                    <td>{partitionBadge(c.partition)}</td>
                    <td>{c.provenance}</td>
                    <td>{c.review_state}</td>
                    <td className="muted">{c.source_trace_id}</td>
                  </tr>
                ))}
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
        <h2>Cases</h2>
        <ErrorPanel error={error} />
      </div>
    );
  }
}

function Select({
  name,
  value,
  options,
}: {
  name: string;
  value: string | undefined;
  options: readonly string[];
}) {
  return (
    <select name={name} defaultValue={value ?? ""}>
      <option value="">{name}: any</option>
      {options.map((option) => (
        <option key={option} value={option}>
          {option}
        </option>
      ))}
    </select>
  );
}
