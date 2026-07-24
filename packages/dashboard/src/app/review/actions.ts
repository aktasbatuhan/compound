"use server";

import { createApiClient, type ReviewRequest } from "../../lib/api";

export interface ReviewActionResult {
  ok: boolean;
  error?: string;
}

/**
 * Submit a review through the API. Promotion to golden is only sent with an
 * approval; the API is the final authority and refuses an invalid promotion,
 * so this action forwards the intent and surfaces any refusal.
 */
export async function submitReview(
  caseId: string,
  review: ReviewRequest,
): Promise<ReviewActionResult> {
  const api = createApiClient();
  try {
    await api.reviewCase(caseId, review);
    return { ok: true };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : "review failed" };
  }
}
