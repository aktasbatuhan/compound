import { describe, expect, test } from "bun:test";
import { applyMatches, findMatches, PII_PATTERNS, passesLuhn, SECRET_PATTERNS } from "../src/index";

function secretHits(value: string): string[] {
  return findMatches(SECRET_PATTERNS, value).map((m) => value.slice(m.start, m.end));
}

function piiHits(value: string): string[] {
  return findMatches(PII_PATTERNS, value).map((m) => value.slice(m.start, m.end));
}

describe("secret detector", () => {
  test("OpenAI-style keys", () => {
    expect(secretHits("key=sk-abcdefghij0123456789")).toEqual(["sk-abcdefghij0123456789"]);
    expect(secretHits("sk-proj-Aa0_bb-CCddEEffGGhh1122")).toEqual([
      "sk-proj-Aa0_bb-CCddEEffGGhh1122",
    ]);
  });

  test("GitHub tokens", () => {
    expect(secretHits("ghp_0123456789abcdefghijABCDEFGHIJ0123")).toHaveLength(1);
    expect(secretHits("gho_0123456789abcdefghijABCD")).toHaveLength(1);
    expect(secretHits("ghs_0123456789abcdefghijABCD")).toHaveLength(1);
  });

  test("AWS access key ids", () => {
    expect(secretHits("AKIAIOSFODNN7EXAMPLE")).toEqual(["AKIAIOSFODNN7EXAMPLE"]);
    expect(secretHits("ASIAIOSFODNN7EXAMPLE")).toEqual(["ASIAIOSFODNN7EXAMPLE"]);
  });

  test("Google API keys", () => {
    const value = `AIza${"a".repeat(35)}`;
    expect(secretHits(value)).toEqual([value]);
  });

  test("Slack tokens", () => {
    expect(secretHits("xoxb-123456789012-abcdefABCDEF")).toHaveLength(1);
    expect(secretHits("xoxp-123456789012-abcdefABCDEF")).toHaveLength(1);
  });

  test("JWTs", () => {
    const jwt =
      "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1";
    expect(secretHits(`Authorization: ${jwt}`)).toEqual([jwt]);
  });

  test("Bearer tokens keep the scheme word and redact only the credential", () => {
    const hits = secretHits("Authorization: Bearer abcdef0123456789");
    expect(hits).toEqual(["abcdef0123456789"]);
  });

  test("PEM private key blocks, header to footer", () => {
    const pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAK\n-----END RSA PRIVATE KEY-----";
    expect(secretHits(`before ${pem} after`)).toEqual([pem]);
  });

  test("near misses are left alone", () => {
    expect(secretHits("max_tokens: 4096")).toEqual([]);
    expect(secretHits("the sk- prefix is documented")).toEqual([]);
    expect(secretHits("AKIASHORT")).toEqual([]);
    expect(secretHits("Bearer x")).toEqual([]);
    expect(secretHits("please ask the bearer of this note")).toEqual([]);
    expect(secretHits("model gpt-4.1-mini finished with stop")).toEqual([]);
  });
});

describe("pii detector", () => {
  test("email addresses", () => {
    expect(piiHits("write to ada@example.com please")).toEqual(["ada@example.com"]);
    expect(piiHits("ada.lovelace+billing@mail.example.co.uk")).toEqual([
      "ada.lovelace+billing@mail.example.co.uk",
    ]);
  });

  test("phone numbers", () => {
    expect(piiHits("call +14155552671 now")).toEqual(["+14155552671"]);
    expect(piiHits("call (415) 555-2671")).toEqual(["(415) 555-2671"]);
    expect(piiHits("call 415.555.2671")).toEqual(["415.555.2671"]);
    expect(piiHits("call +1 415-555-2671")).toEqual(["+1 415-555-2671"]);
  });

  test("credit cards that pass Luhn", () => {
    expect(piiHits("card 4111111111111111 on file")).toEqual(["4111111111111111"]);
    expect(piiHits("card 4111 1111 1111 1111 on file")).toEqual(["4111 1111 1111 1111"]);
  });

  test("a card-shaped number that fails Luhn is not redacted", () => {
    expect(passesLuhn("4111111111111112")).toBe(false);
    expect(piiHits("order 4111111111111112 shipped")).toEqual([]);
  });

  test("US SSN shape", () => {
    expect(piiHits("ssn 123-45-6789")).toEqual(["123-45-6789"]);
  });

  test("near misses are left alone", () => {
    expect(piiHits("max_tokens: 4096")).toEqual([]);
    expect(piiHits("temperature 0.7, top_p 1")).toEqual([]);
    expect(piiHits("started at 2026-07-20T14:03:11.000Z")).toEqual([]);
    expect(piiHits("invoice INV-2291 and order 1234567890")).toEqual([]);
    expect(piiHits("version app-v2.14.0")).toEqual([]);
    expect(piiHits("@mentions and user@ are not addresses")).toEqual([]);
  });
});

describe("passesLuhn", () => {
  test("bounds the digit count to real card lengths", () => {
    expect(passesLuhn("4111111111111111")).toBe(true);
    expect(passesLuhn("0")).toBe(false);
    expect(passesLuhn("4111 1111 1111 1111")).toBe(true);
    expect(passesLuhn("41111111111111111111")).toBe(false);
  });
});

describe("findMatches / applyMatches", () => {
  test("overlapping matches from different patterns merge into one interval", () => {
    const value = "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.abcdefghij";
    expect(findMatches(SECRET_PATTERNS, value)).toHaveLength(1);
  });

  test("replacement keeps the text between matches", () => {
    const value = "from ada@example.com to grace@example.com";
    const out = applyMatches(value, findMatches(PII_PATTERNS, value), "X");
    expect(out).toBe("from X to X");
  });

  test("patterns are reusable: lastIndex never leaks between calls", () => {
    const value = "sk-abcdefghij0123456789";
    expect(findMatches(SECRET_PATTERNS, value)).toEqual(findMatches(SECRET_PATTERNS, value));
  });
});
