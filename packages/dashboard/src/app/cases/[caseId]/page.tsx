import { createApiClient } from "../../../lib/api";
import { Assertions, Conversation, ErrorPanel, partitionBadge } from "../../ui";

export const dynamic = "force-dynamic";

export default async function CaseDetailPage({ params }: { params: Promise<{ caseId: string }> }) {
  const { caseId } = await params;
  const api = createApiClient();

  try {
    const [caseData, assertions] = await Promise.all([
      api.getCase(caseId),
      api.getCaseAssertions(caseId).catch(() => null),
    ]);

    return (
      <div>
        <h2>Case</h2>
        <div className="card">
          <div className="row" style={{ marginBottom: 8 }}>
            <span className="badge">{caseData.provenance}</span>
            {partitionBadge(caseData.partition)}
            <span className="badge">{caseData.review_state}</span>
          </div>
          <table>
            <tbody>
              <tr>
                <th>Case id</th>
                <td>{caseData.case_id}</td>
              </tr>
              <tr>
                <th>Task</th>
                <td>{caseData.task_key}</td>
              </tr>
              <tr>
                <th>Source trace</th>
                <td>
                  <a href={`/traces?trace_id=${encodeURIComponent(caseData.source_trace_id)}`}>
                    {caseData.source_trace_id}
                  </a>
                </td>
              </tr>
              <tr>
                <th>Content hash</th>
                <td className="muted">{caseData.content_hash}</td>
              </tr>
              <tr>
                <th>Duplicates collapsed</th>
                <td>{caseData.duplicate_count}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="card">
          <h3>Input</h3>
          <Conversation messages={caseData.input.input} />
        </div>

        <div className="card">
          <h3>Expected output</h3>
          <pre className="msg">
            {caseData.expected === null
              ? "(none — assertion-gradeable without an expected output)"
              : JSON.stringify(caseData.expected, null, 2)}
          </pre>
        </div>

        <div className="card">
          <h3>Assertions</h3>
          <Assertions report={assertions} />
        </div>
      </div>
    );
  } catch (error) {
    return (
      <div>
        <h2>Case</h2>
        <ErrorPanel error={error} />
      </div>
    );
  }
}
