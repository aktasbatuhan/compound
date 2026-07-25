import { describe, expect, test } from "bun:test";
import type { Message } from "@compound/contract";
import { buildPointwiseMessages, hashRubric, messageText, parseJudgeVerdict } from "../src/prompt";

function assistant(content: unknown): Message {
  return { role: "assistant", content } as Message;
}

describe("hashRubric", () => {
  test("is stable and changes with the rubric", () => {
    expect(hashRubric("be helpful")).toBe(hashRubric("be helpful"));
    expect(hashRubric("be helpful")).not.toBe(hashRubric("be terse"));
  });
});

describe("buildPointwiseMessages", () => {
  test("carries the rubric and the response but never a model identity", () => {
    const messages = buildPointwiseMessages("RUBRIC TEXT", "the model output");
    const joined = messages.map((m) => messageText(m)).join("\n");
    expect(joined).toContain("RUBRIC TEXT");
    expect(joined).toContain("the model output");
    // Blinding: the judge is told nothing about which model produced it.
    expect(joined.toLowerCase()).not.toContain("candidate");
    expect(joined.toLowerCase()).not.toContain("gpt");
  });
});

describe("parseJudgeVerdict", () => {
  test("parses a clean verdict", () => {
    const v = parseJudgeVerdict(assistant('{"score": 0.8, "reasoning": "solid"}'));
    expect(v).toEqual({ score: 0.8, reasoning: "solid" });
  });

  test("extracts JSON embedded in stray prose", () => {
    const v = parseJudgeVerdict(
      assistant('Here is my answer: {"score": 0.4, "reasoning": "meh"} done'),
    );
    expect(v?.score).toBe(0.4);
  });

  test("rejects a score out of range", () => {
    expect(parseJudgeVerdict(assistant('{"score": 1.5}'))).toBeNull();
    expect(parseJudgeVerdict(assistant('{"score": -0.1}'))).toBeNull();
  });

  test("rejects non-JSON or a missing score", () => {
    expect(parseJudgeVerdict(assistant("it was pretty good honestly"))).toBeNull();
    expect(parseJudgeVerdict(assistant('{"reasoning": "no score here"}'))).toBeNull();
    expect(parseJudgeVerdict(null)).toBeNull();
  });

  test("tolerates a missing reasoning field", () => {
    expect(parseJudgeVerdict(assistant('{"score": 0.5}'))).toEqual({ score: 0.5, reasoning: "" });
  });
});

describe("messageText", () => {
  test("handles string, array, and structured content", () => {
    expect(messageText(assistant("plain"))).toBe("plain");
    expect(messageText(assistant([{ text: "a" }, { text: "b" }]))).toBe("a\nb");
    expect(messageText(null)).toBe("");
  });
});
