/**
 * Parsing a `POST /api/cases/:caseId/review` body.
 *
 * Promotion to `human_golden` is deliberately explicit: the client must ask for
 * it AND the review must be an approval. This is the human validation SPEC.md
 * requires before observed/synthetic output is treated as ground truth.
 */
import type { CaseReviewState, ReviewCaseInput } from "@compound/storage";
import { CASE_REVIEW_STATES } from "@compound/storage";
import { invalidRequest } from "./errors";

export function parseReviewRequest(body: unknown): ReviewCaseInput {
  if (body === null || typeof body !== "object" || Array.isArray(body)) {
    throw invalidRequest("request body must be a JSON object");
  }
  const record = body as Record<string, unknown>;

  const reviewState = record.review_state;
  if (
    typeof reviewState !== "string" ||
    !(CASE_REVIEW_STATES as readonly string[]).includes(reviewState)
  ) {
    throw invalidRequest(`review_state must be one of: ${CASE_REVIEW_STATES.join(", ")}`, {
      parameter: "review_state",
    });
  }

  const promoteToGolden = record.promote_to_golden;
  if (promoteToGolden !== undefined && typeof promoteToGolden !== "boolean") {
    throw invalidRequest("promote_to_golden must be a boolean", {
      parameter: "promote_to_golden",
    });
  }

  return {
    reviewState: reviewState as CaseReviewState,
    // `expected` may be any JSON, including null to explicitly clear it; only
    // an absent key means "leave unchanged".
    ...("expected" in record ? { expected: record.expected } : {}),
    ...(promoteToGolden !== undefined ? { promoteToGolden } : {}),
  };
}
