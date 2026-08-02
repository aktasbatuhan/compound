/**
 * Structural equality for JSON-shaped values.
 *
 * `JSON.stringify(a) === JSON.stringify(b)` is a tempting shortcut but it is
 * key-ORDER sensitive: `{a:1,b:2}` and `{b:2,a:1}` serialize differently and
 * compare unequal, which produces false assertion failures and spurious
 * recorded-result misses. This compares by structure — arrays by order, objects
 * by key membership — so semantically identical values are equal regardless of
 * how their keys were ordered.
 */
export function structuralEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a === null || b === null || typeof a !== "object" || typeof b !== "object") {
    return a === b;
  }
  const aArray = Array.isArray(a);
  const bArray = Array.isArray(b);
  if (aArray !== bArray) return false;
  if (aArray && bArray) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i += 1) {
      if (!structuralEqual(a[i], b[i])) return false;
    }
    return true;
  }
  const ao = a as Record<string, unknown>;
  const bo = b as Record<string, unknown>;
  const aKeys = Object.keys(ao);
  const bKeys = Object.keys(bo);
  if (aKeys.length !== bKeys.length) return false;
  for (const key of aKeys) {
    if (!Object.hasOwn(bo, key)) return false;
    if (!structuralEqual(ao[key], bo[key])) return false;
  }
  return true;
}
