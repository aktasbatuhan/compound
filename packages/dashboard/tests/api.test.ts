import { afterEach, describe, expect, test } from "bun:test";
import { ApiError, createApiClient } from "../src/lib/api";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

/** Capture the URL/method/body of the next call and return `payload`. */
function mockFetch(payload: unknown, status = 200) {
  const calls: Array<{ url: string; method: string; body: unknown }> = [];
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    calls.push({
      url,
      method: init?.method ?? "GET",
      body: init?.body ? JSON.parse(init.body as string) : undefined,
    });
    return new Response(JSON.stringify(payload), { status });
  }) as unknown as typeof fetch;
  return calls;
}

const BASE = "http://127.0.0.1:4319";

describe("createApiClient — URLs and methods", () => {
  test("listCases builds the query string and hits /api/cases", async () => {
    const calls = mockFetch({ items: [], total: 0, limit: 100, offset: 0 });
    const api = createApiClient(BASE);
    await api.listCases({ task_key: "support", review_state: "unreviewed", limit: 100 });
    expect(calls[0]?.url).toBe(
      `${BASE}/api/cases?task_key=support&review_state=unreviewed&limit=100`,
    );
    expect(calls[0]?.method).toBe("GET");
  });

  test("reviewCase POSTs the review body", async () => {
    const calls = mockFetch({ case_id: "c1", review_state: "approved" });
    const api = createApiClient(BASE);
    await api.reviewCase("c1", { review_state: "approved", promote_to_golden: true });
    expect(calls[0]?.url).toBe(`${BASE}/api/cases/c1/review`);
    expect(calls[0]?.method).toBe("POST");
    expect(calls[0]?.body).toEqual({ review_state: "approved", promote_to_golden: true });
  });

  test("getCaseAssertions encodes the case id", async () => {
    const calls = mockFetch({
      caseId: "case:1",
      graded: true,
      passed: true,
      score: 1,
      results: [],
    });
    const api = createApiClient(BASE);
    await api.getCaseAssertions("case:1");
    expect(calls[0]?.url).toBe(`${BASE}/api/cases/case%3A1/assertions`);
  });

  test("listExperiments hits /api/experiments with filters", async () => {
    const calls = mockFetch({ items: [], total: 0, limit: 500, offset: 0 });
    const api = createApiClient(BASE);
    await api.listExperiments({ task_key: "support", limit: 500 });
    expect(calls[0]?.url).toBe(`${BASE}/api/experiments?task_key=support&limit=500`);
  });

  test("listGates hits /api/gates with an optional task_key", async () => {
    const calls = mockFetch({ items: [] });
    const api = createApiClient(BASE);
    await api.listGates("support");
    expect(calls[0]?.url).toBe(`${BASE}/api/gates?task_key=support`);
    await api.listGates();
    expect(calls[1]?.url).toBe(`${BASE}/api/gates`);
  });

  test("omits undefined filter params", async () => {
    const calls = mockFetch({ items: [], total: 0, limit: 100, offset: 0 });
    const api = createApiClient(BASE);
    await api.listCases({ task_key: "support" });
    expect(calls[0]?.url).toBe(`${BASE}/api/cases?task_key=support`);
  });
});

describe("createApiClient — error handling", () => {
  test("throws ApiError with the envelope's code and message on a 4xx", async () => {
    mockFetch({ error: { code: "not_found", message: "no case with id x" } }, 404);
    const api = createApiClient(BASE);
    await expect(api.getCase("x")).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
      code: "not_found",
    });
  });

  test("throws api_unreachable when fetch itself fails", async () => {
    globalThis.fetch = (async () => {
      throw new Error("ECONNREFUSED");
    }) as unknown as typeof fetch;
    const api = createApiClient(BASE);
    await expect(api.getHealth()).rejects.toMatchObject({
      name: "ApiError",
      code: "api_unreachable",
    });
  });

  test("falls back to http_error when a non-2xx has no envelope", async () => {
    mockFetch("upstream boom", 502);
    const api = createApiClient(BASE);
    const error = await api.getHealth().catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("http_error");
  });
});
