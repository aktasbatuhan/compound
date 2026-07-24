"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import type { CaseResponse } from "../../lib/api";
import { submitReview } from "./actions";

/**
 * The labeling control. Promotion to golden is ONLY enabled with an approval,
 * and the copy states plainly that this is the human validation that turns
 * observed/synthetic output into ground truth (docs/curation-v1.md).
 */
export function ReviewForm({ caseData }: { caseData: CaseResponse }) {
  const router = useRouter();
  const [expected, setExpected] = useState(JSON.stringify(caseData.expected, null, 2));
  const [promote, setPromote] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function act(reviewState: "approved" | "rejected" | "needs_edit") {
    setBusy(true);
    setError(null);

    let parsedExpected: unknown;
    let sendExpected = false;
    const trimmed = expected.trim();
    const original = JSON.stringify(caseData.expected, null, 2);
    if (trimmed !== original.trim()) {
      try {
        parsedExpected = trimmed.length === 0 ? null : JSON.parse(trimmed);
        sendExpected = true;
      } catch {
        setError("expected output is not valid JSON");
        setBusy(false);
        return;
      }
    }

    const result = await submitReview(caseData.case_id, {
      review_state: reviewState,
      ...(sendExpected ? { expected: parsedExpected } : {}),
      ...(reviewState === "approved" && promote ? { promote_to_golden: true } : {}),
    });

    setBusy(false);
    if (!result.ok) {
      setError(result.error ?? "review failed");
      return;
    }
    // Advance to the next unreviewed case.
    router.refresh();
  }

  return (
    <div className="card">
      <h3>Review</h3>
      <label className="muted" htmlFor="expected">
        Expected output (edit to correct it)
      </label>
      <textarea
        id="expected"
        value={expected}
        onChange={(event) => setExpected(event.target.value)}
      />

      <label className="row" style={{ margin: "12px 0" }}>
        <input
          type="checkbox"
          checked={promote}
          onChange={(event) => setPromote(event.target.checked)}
        />
        <span>
          Promote to <strong>human_golden</strong> on approval — this is the human validation that
          lets this output count as ground truth. Only applied with “Approve”.
        </span>
      </label>

      {error ? <div className="error">{error}</div> : null}

      <div className="row" style={{ marginTop: 12 }}>
        <button type="button" className="primary" disabled={busy} onClick={() => act("approved")}>
          Approve
        </button>
        <button type="button" disabled={busy} onClick={() => act("needs_edit")}>
          Needs edit
        </button>
        <button type="button" disabled={busy} onClick={() => act("rejected")}>
          Reject
        </button>
      </div>
    </div>
  );
}
