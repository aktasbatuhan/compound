import { describe, expect, test } from "bun:test";
import { structuralEqual } from "../src/equal";

describe("structuralEqual", () => {
  test("is insensitive to object key order (the JSON.stringify trap)", () => {
    expect(structuralEqual({ a: 1, b: 2 }, { b: 2, a: 1 })).toBe(true);
    expect(structuralEqual({ a: { x: 1, y: 2 } }, { a: { y: 2, x: 1 } })).toBe(true);
  });

  test("is sensitive to array order", () => {
    expect(structuralEqual([1, 2, 3], [1, 2, 3])).toBe(true);
    expect(structuralEqual([1, 2, 3], [3, 2, 1])).toBe(false);
  });

  test("compares nested structures and primitives", () => {
    expect(structuralEqual({ a: [1, { b: 2 }] }, { a: [1, { b: 2 }] })).toBe(true);
    expect(structuralEqual({ a: [1, { b: 2 }] }, { a: [1, { b: 3 }] })).toBe(false);
    expect(structuralEqual("x", "x")).toBe(true);
    expect(structuralEqual(1, "1")).toBe(false);
  });

  test("distinguishes null, missing keys, and differing key counts", () => {
    expect(structuralEqual(null, null)).toBe(true);
    expect(structuralEqual(null, {})).toBe(false);
    expect(structuralEqual({ a: 1 }, { a: 1, b: 2 })).toBe(false);
    expect(structuralEqual({ a: undefined }, { b: undefined })).toBe(false);
  });
});
