import { describe, expect, test } from "bun:test";
import { type Trace, validate } from "@compound/contract";
import type { NormalizeOptions } from "../src/index";
import { normalizeOtelExport, OTEL_REJECTION_REASONS } from "../src/index";

const OPTIONS: NormalizeOptions = {
  defaultPermissions: { judging: true, optimization: true, fine_tuning: false },
  importerVersion: "0.1.0",
  exportedAt: "2026-07-24T09:00:00.000Z",
};

// --- OTLP/JSON builders ----------------------------------------------------

type Attr = { key: string; value: Record<string, unknown> };
const str = (key: string, s: string): Attr => ({ key, value: { stringValue: s } });
// int64 arrives as a string in OTLP/JSON — exercise that on purpose.
const int = (key: string, n: number): Attr => ({ key, value: { intValue: String(n) } });
const dbl = (key: string, n: number): Attr => ({ key, value: { doubleValue: n } });

interface SpanInit {
  traceId?: string;
  spanId: string;
  parentSpanId?: string;
  name?: string;
  attributes?: Attr[];
  status?: Record<string, unknown>;
  omitTraceId?: boolean;
}

function span(init: SpanInit): Record<string, unknown> {
  const s: Record<string, unknown> = {
    spanId: init.spanId,
    name: init.name ?? "chat",
    startTimeUnixNano: "1700000000000000000",
    endTimeUnixNano: "1700000001500000000",
    attributes: init.attributes ?? [],
  };
  if (!init.omitTraceId) s.traceId = init.traceId ?? "t1";
  if (init.parentSpanId !== undefined) s.parentSpanId = init.parentSpanId;
  if (init.status !== undefined) s.status = init.status;
  return s;
}

function otlp(
  spans: Record<string, unknown>[],
  resourceAttrs: Attr[] = [],
): Record<string, unknown> {
  return {
    resourceSpans: [
      {
        resource: { attributes: resourceAttrs },
        scopeSpans: [{ scope: { name: "test" }, spans }],
      },
    ],
  };
}

/** A complete flat-attribute GenAI chat span (the common OpenLLMetry/Logfire shape). */
function chatSpanAttrs(): Attr[] {
  return [
    str("gen_ai.system", "openai"),
    str("gen_ai.request.model", "gpt-4o"),
    str("gen_ai.response.model", "gpt-4o-2026-01"),
    dbl("gen_ai.request.temperature", 0.2),
    int("gen_ai.request.max_tokens", 512),
    str("gen_ai.prompt.0.role", "system"),
    str("gen_ai.prompt.0.content", "You are a support agent."),
    str("gen_ai.prompt.1.role", "user"),
    str("gen_ai.prompt.1.content", "Where is my order?"),
    str("gen_ai.completion.0.role", "assistant"),
    str("gen_ai.completion.0.content", "Let me check that for you."),
    str("gen_ai.completion.0.finish_reason", "stop"),
    int("gen_ai.usage.input_tokens", 18),
    int("gen_ai.usage.output_tokens", 7),
    str("gen_ai.compound.task_key", "support"),
  ];
}

describe("normalizeOtelExport", () => {
  test("a flat-attribute GenAI chat span becomes an eval-ready model_call", () => {
    const { traces, report } = normalizeOtelExport(
      otlp([span({ spanId: "s1", attributes: chatSpanAttrs() })]),
      OPTIONS,
    );
    expect(traces).toHaveLength(1);
    const { trace } = traces[0] as { trace: Trace };
    expect(trace.trace_id).toBe("otel:t1");
    expect(trace.task_key).toBe("support");
    expect(trace.started_at).toBe("2023-11-14T22:13:20.000Z");
    expect(trace.steps).toHaveLength(1);

    const step = trace.steps[0];
    expect(step?.type).toBe("model_call");
    if (step?.type !== "model_call") throw new Error("expected model_call");
    expect(step.provider).toBe("openai");
    expect(step.model).toBe("gpt-4o");
    expect(step.resolved_model).toBe("gpt-4o-2026-01");
    expect(step.params).toEqual({ temperature: 0.2, max_tokens: 512 });
    expect(step.input.map((m) => m.role)).toEqual(["system", "user"]);
    expect(step.input[1]?.content).toBe("Where is my order?");
    expect(step.output?.content).toBe("Let me check that for you.");
    expect(step.finish_reason).toBe("stop");
    expect(step.usage).toEqual({ input_tokens: 18, output_tokens: 7 });

    // The focal call is the sole model_call, and the whole trace is eval-ready.
    expect(trace.focal_step_id).toBe("s1");
    expect(validate(trace).class).toBe("eval_ready");
    expect(report.counts.tracesNormalized).toBe(1);
    expect(report.counts.observationRecords).toBe(1);
  });

  test("structured input/output messages carry a tool call through to the contract", () => {
    const attrs: Attr[] = [
      str("gen_ai.system", "anthropic"),
      str("gen_ai.request.model", "claude"),
      str(
        "gen_ai.input.messages",
        JSON.stringify([{ role: "user", content: "Dispute the $23 charge." }]),
      ),
      str(
        "gen_ai.output.messages",
        JSON.stringify([
          {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_1",
                type: "function",
                function: { name: "dispute_charge", arguments: '{"amount":23}' },
              },
            ],
          },
        ]),
      ),
      str("task_key", "finance.dispute_charge"),
    ];
    const { traces } = normalizeOtelExport(
      otlp([span({ spanId: "s1", attributes: attrs })]),
      OPTIONS,
    );
    const { trace } = traces[0] as { trace: Trace };
    const step = trace.steps[0];
    if (step?.type !== "model_call") throw new Error("expected model_call");
    expect(step.input[0]?.content).toBe("Dispute the $23 charge.");
    expect(step.output?.tool_calls?.[0]?.name).toBe("dispute_charge");
    expect(step.output?.tool_calls?.[0]?.arguments).toEqual({ amount: 23 });
    expect(validate(trace).class).toBe("eval_ready");
  });

  test("a GenAI span with no recoverable prompt is diagnostic, not a silent eval case", () => {
    const attrs: Attr[] = [
      str("gen_ai.system", "openai"),
      str("gen_ai.request.model", "gpt-4o"),
      // Only a completion — no prompt of any form.
      str("gen_ai.completion.0.role", "assistant"),
      str("gen_ai.completion.0.content", "hi"),
    ];
    const { traces } = normalizeOtelExport(
      otlp([span({ spanId: "s1", attributes: attrs })]),
      OPTIONS,
    );
    const { trace } = traces[0] as { trace: Trace };
    const step = trace.steps[0];
    if (step?.type !== "model_call") throw new Error("expected model_call");
    expect(step.input).toEqual([]);
    const result = validate(trace);
    expect(result.class).toBe("diagnostic");
    if (result.class !== "diagnostic") throw new Error("expected diagnostic");
    expect(result.diagnostic_reasons).toContain("focal_step_missing_input");
  });

  test("a non-GenAI span in the same trace is kept as an 'other' step with parent linkage", () => {
    const doc = otlp([
      span({ spanId: "root", name: "handle_request", attributes: [str("http.method", "POST")] }),
      span({ spanId: "llm", parentSpanId: "root", attributes: chatSpanAttrs() }),
    ]);
    const { traces } = normalizeOtelExport(doc, OPTIONS);
    const { trace } = traces[0] as { trace: Trace };
    expect(trace.steps.map((s) => s.type)).toEqual(["other", "model_call"]);
    const modelCall = trace.steps[1];
    if (modelCall?.type !== "model_call") throw new Error("expected model_call");
    expect(modelCall.parent_step_id).toBe("root");
    // The model_call is still the focal call despite the sibling span.
    expect(trace.focal_step_id).toBe("llm");
  });

  test("task_key falls back to a resource attribute when the span carries none", () => {
    const doc = otlp(
      [
        span({
          spanId: "s1",
          attributes: chatSpanAttrs().filter((a) => a.key !== "gen_ai.compound.task_key"),
        }),
      ],
      [str("compound.task_key", "support"), str("deployment.environment.name", "production")],
    );
    const { traces } = normalizeOtelExport(doc, OPTIONS);
    const { trace } = traces[0] as { trace: Trace };
    expect(trace.task_key).toBe("support");
    expect(trace.environment).toBe("production");
  });

  test("legacy llm.* aliases and prompt/completion token names are read", () => {
    const attrs: Attr[] = [
      str("llm.system", "openai"),
      str("llm.request.model", "gpt-4o-mini"),
      str("llm.prompt.0.role", "user"),
      str("llm.prompt.0.content", "hi"),
      str("llm.completion.0.role", "assistant"),
      str("llm.completion.0.content", "hello"),
      int("gen_ai.usage.prompt_tokens", 3),
      int("gen_ai.usage.completion_tokens", 1),
    ];
    const { traces } = normalizeOtelExport(
      otlp([span({ spanId: "s1", attributes: attrs })]),
      OPTIONS,
    );
    const { trace } = traces[0] as { trace: Trace };
    const step = trace.steps[0];
    if (step?.type !== "model_call") throw new Error("expected model_call");
    expect(step.provider).toBe("openai");
    expect(step.model).toBe("gpt-4o-mini");
    expect(step.usage).toEqual({ input_tokens: 3, output_tokens: 1 });
  });

  test("an ERROR span status is recorded on the model call", () => {
    const { traces } = normalizeOtelExport(
      otlp([
        span({
          spanId: "s1",
          attributes: chatSpanAttrs(),
          status: { code: 2, message: "rate limited" },
        }),
      ]),
      OPTIONS,
    );
    const { trace } = traces[0] as { trace: Trace };
    const step = trace.steps[0];
    if (step?.type !== "model_call") throw new Error("expected model_call");
    expect(step.error).toBe("rate limited");
  });

  test("spans across a JSONL export split into one trace per traceId", () => {
    const line1 = JSON.stringify(
      otlp([span({ traceId: "ta", spanId: "s1", attributes: chatSpanAttrs() })]),
    );
    const line2 = JSON.stringify(
      otlp([span({ traceId: "tb", spanId: "s2", attributes: chatSpanAttrs() })]),
    );
    const { traces, report } = normalizeOtelExport(`${line1}\n${line2}\n`, OPTIONS);
    expect(report.format).toBe("jsonl");
    expect(traces.map((t) => t.trace.trace_id).sort()).toEqual(["otel:ta", "otel:tb"]);
  });

  test("a span missing ids is rejected with a line number, not dropped silently", () => {
    const { traces, report } = normalizeOtelExport(
      otlp([
        span({ omitTraceId: true, spanId: "s0", attributes: chatSpanAttrs() }),
        span({ spanId: "s1", attributes: chatSpanAttrs() }),
      ]),
      OPTIONS,
    );
    expect(traces).toHaveLength(1);
    expect(report.rejected).toEqual([{ line: 1, reason: OTEL_REJECTION_REASONS.spanMissingIds }]);
  });

  test("non-OTLP input is rejected as a whole rather than throwing", () => {
    expect(normalizeOtelExport("this is not json", OPTIONS).report.rejected).toEqual([
      { line: 1, reason: OTEL_REJECTION_REASONS.fileNotValidJson },
    ]);
    expect(normalizeOtelExport({ hello: "world" }, OPTIONS).report.rejected).toEqual([
      { line: 1, reason: OTEL_REJECTION_REASONS.noResourceSpans },
    ]);
  });

  test("the trace_id prefix folds in a project id when supplied", () => {
    const { traces } = normalizeOtelExport(
      otlp([span({ spanId: "s1", attributes: chatSpanAttrs() })]),
      {
        ...OPTIONS,
        projectId: "proj7",
      },
    );
    expect(traces[0]?.trace.trace_id).toBe("otel:proj7:t1");
  });
});
