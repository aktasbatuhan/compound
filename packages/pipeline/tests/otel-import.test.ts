import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import type { CompoundDatabase } from "@compound/storage";
import { createDatabase, getTraceByTraceId, migrate } from "@compound/storage";
import { runImport } from "../src/index";

let db: CompoundDatabase;

beforeEach(() => {
  db = createDatabase();
  migrate(db);
});

afterEach(() => {
  db.close();
});

type Attr = { key: string; value: Record<string, unknown> };
const str = (key: string, s: string): Attr => ({ key, value: { stringValue: s } });
const int = (key: string, n: number): Attr => ({ key, value: { intValue: String(n) } });

/** One OTLP/JSON export with a single, fully-populated GenAI chat span. */
function otelExport(attrs: Attr[]): string {
  return JSON.stringify({
    resourceSpans: [
      {
        resource: { attributes: [] },
        scopeSpans: [
          {
            scope: { name: "openllmetry" },
            spans: [
              {
                traceId: "t-otel-1",
                spanId: "span-1",
                name: "openai.chat",
                startTimeUnixNano: "1700000000000000000",
                endTimeUnixNano: "1700000001000000000",
                attributes: attrs,
              },
            ],
          },
        ],
      },
    ],
  });
}

const CHAT_ATTRS: Attr[] = [
  str("gen_ai.system", "openai"),
  str("gen_ai.request.model", "gpt-4o"),
  str("gen_ai.prompt.0.role", "user"),
  str("gen_ai.prompt.0.content", "what is our refund window?"),
  str("gen_ai.completion.0.role", "assistant"),
  str("gen_ai.completion.0.content", "Thirty days."),
  int("gen_ai.usage.input_tokens", 12),
  int("gen_ai.usage.output_tokens", 4),
  str("gen_ai.compound.task_key", "support"),
];

describe("runImport — otel importer", () => {
  test("imports an OTLP GenAI span end to end into a stored, eval-ready trace", () => {
    const { batch, report } = runImport(db, { importer: "otel", content: otelExport(CHAT_ATTRS) });

    expect(batch.status).toBe("completed");
    expect(batch.importer).toBe("otel");
    expect(report.counts).toMatchObject({ eval_ready: 1, diagnostic: 0, rejected: 0 });

    const stored = getTraceByTraceId(db, "otel:t-otel-1");
    expect(stored).not.toBeNull();
    expect(stored?.validationClass).toBe("eval_ready");
    expect(stored?.taskKey).toBe("support");
    expect(stored?.focalModel).toBe("gpt-4o");
    expect(stored?.contentHash).toMatch(/^[0-9a-f]{64}$/);
  });

  test("a span with no recoverable prompt lands in the diagnostic queue, not as an eval case", () => {
    const noPrompt = CHAT_ATTRS.filter((a) => !a.key.startsWith("gen_ai.prompt."));
    const { report } = runImport(db, { importer: "otel", content: otelExport(noPrompt) });
    expect(report.counts?.eval_ready).toBe(0);
    expect(report.counts?.diagnostic).toBe(1);

    const stored = getTraceByTraceId(db, "otel:t-otel-1");
    expect(stored?.validationClass).toBe("diagnostic");
  });
});
