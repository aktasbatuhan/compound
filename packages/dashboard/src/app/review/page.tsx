import { createApiClient } from "../../lib/api";
import { Assertions, Conversation, Empty, ErrorPanel, partitionBadge } from "../ui";
import { ReviewForm } from "./ReviewForm";

export const dynamic = "force-dynamic";

/** Choose a task to review: the flag, or the largest task by case count. */
async function resolveTask(
  api: ReturnType<typeof createApiClient>,
  requested: string | undefined,
): Promise<string | null> {
  if (requested) return requested;
  // Any task with cases works; pick the first from a broad listing.
  const page = await api.listCases({ limit: 1 });
  return page.items[0]?.task_key ?? null;
}

export default async function ReviewPage({
  searchParams,
}: {
  searchParams: Promise<{ task_key?: string }>;
}) {
  const { task_key: requested } = await searchParams;
  const api = createApiClient();

  try {
    const taskKey = await resolveTask(api, requested);
    if (taskKey === null) {
      return (
        <div>
          <h2>Review</h2>
          <Empty>No cases yet. Import traces and curate a task first.</Empty>
        </div>
      );
    }

    const [unreviewed, all] = await Promise.all([
      api.listCases({ task_key: taskKey, review_state: "unreviewed", limit: 1 }),
      api.listCases({ task_key: taskKey, limit: 500 }),
    ]);

    const reviewedCount = all.items.filter((c) => c.review_state !== "unreviewed").length;
    const progressPct = all.total > 0 ? Math.round((reviewedCount / all.total) * 100) : 0;

    const current = unreviewed.items[0];

    return (
      <div>
        <h2>Review — {taskKey}</h2>
        <div className="card">
          <div className="row">
            <span className="badge">{unreviewed.total} unreviewed</span>
            <span className="badge good">{reviewedCount} reviewed</span>
            <span className="muted">of {all.total} in non-sealed partitions</span>
          </div>
          <div className="progress">
            <span style={{ width: `${progressPct}%` }} />
          </div>
        </div>

        {current === undefined ? (
          <Empty>Nothing left to review for this task. 🎉</Empty>
        ) : (
          <CaseReview api={api} caseId={current.case_id} />
        )}
      </div>
    );
  } catch (error) {
    return (
      <div>
        <h2>Review</h2>
        <ErrorPanel error={error} />
      </div>
    );
  }
}

async function CaseReview({
  api,
  caseId,
}: {
  api: ReturnType<typeof createApiClient>;
  caseId: string;
}) {
  const [caseData, assertions] = await Promise.all([
    api.getCase(caseId),
    api.getCaseAssertions(caseId).catch(() => null),
  ]);

  return (
    <>
      <div className="card">
        <div className="row" style={{ marginBottom: 8 }}>
          <span className="badge">{caseData.provenance}</span>
          {partitionBadge(caseData.partition)}
          <span className="muted">{caseData.case_id}</span>
        </div>
        <h3>Input</h3>
        <Conversation messages={caseData.input.input} />
      </div>

      <div className="card">
        <h3>Assertions</h3>
        <Assertions report={assertions} />
      </div>

      <ReviewForm caseData={caseData} />
    </>
  );
}
