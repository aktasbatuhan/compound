/**
 * Build the judge prompt and parse its verdict.
 *
 * Two design rules from docs/judges-v1.md are enforced here:
 * - **Blinding**: the prompt presents "the response" neutrally. It never carries
 *   the model id, the provider, or which side is the incumbent/candidate — that
 *   metadata is simply never put into the prompt.
 * - **Structured output**: the judge must return `{ "score": 0..1, "reasoning" }`
 *   (the score_model shape from the OpenAI-graders research), so the verdict is
 *   machine-readable and its rationale is auditable, never a free-text blob.
 */
import { createHash } from "node:crypto";
import type { Message } from "@compound/contract";

/** The rubric is part of the calibration pin; hash it so a change is detectable. */
export function hashRubric(rubric: string): string {
  return `sha256:${createHash("sha256").update(rubric).digest("hex")}`;
}

/** Best-effort plain text of an assistant message, for presenting to the judge. */
export function messageText(message: Message | null): string {
  if (message === null) return "";
  const content = (message as { content?: unknown }).content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) =>
        typeof part === "string"
          ? part
          : typeof (part as { text?: unknown })?.text === "string"
            ? (part as { text: string }).text
            : JSON.stringify(part),
      )
      .join("\n");
  }
  return content === undefined ? "" : JSON.stringify(content);
}

/**
 * Build the pointwise judge messages: a system message carrying the rubric and
 * the strict-JSON contract, and a user message with the (blinded) response.
 */
export function buildPointwiseMessages(rubric: string, responseText: string): Message[] {
  const system =
    "You are a strict evaluator. Score the response below against the rubric.\n\n" +
    `RUBRIC:\n${rubric}\n\n` +
    "Judge only the response's quality against the rubric. You are not told which " +
    "model produced it and must not speculate. Reply with ONLY a JSON object of the " +
    'form {"score": <number between 0 and 1>, "reasoning": "<one or two sentences>"}. ' +
    "1 means the response fully satisfies the rubric; 0 means it fails it. No prose " +
    "outside the JSON.";
  const user = `RESPONSE TO SCORE:\n${responseText}`;
  return [
    { role: "system", content: system } as Message,
    { role: "user", content: user } as Message,
  ];
}

export interface JudgeVerdict {
  score: number;
  reasoning: string;
}

/**
 * Parse the judge's reply into a verdict. Returns null if the reply is not the
 * agreed JSON or the score is out of range — a judge that won't answer in the
 * contract is treated as no verdict (the caller abstains on that case), never
 * coerced into a number.
 */
export function parseJudgeVerdict(reply: Message | null): JudgeVerdict | null {
  const text = messageText(reply).trim();
  if (text.length === 0) return null;
  const json = extractJsonObject(text);
  if (json === null) return null;
  const score = (json as { score?: unknown }).score;
  if (typeof score !== "number" || Number.isNaN(score) || score < 0 || score > 1) return null;
  const reasoning = (json as { reasoning?: unknown }).reasoning;
  return { score, reasoning: typeof reasoning === "string" ? reasoning : "" };
}

/** Pull the first balanced JSON object out of a string, tolerating stray prose. */
function extractJsonObject(text: string): unknown {
  const start = text.indexOf("{");
  if (start === -1) return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        try {
          return JSON.parse(text.slice(start, i + 1));
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}
