/**
 * Secret-free rendering of the config for HTTP responses.
 *
 * Credentials are supposed to live in `.env`, never in `compound.yaml` — the
 * config only references them by environment-variable *name* (`api_key_env`).
 * This module is the defence in depth for when someone pastes a real key in
 * anyway: such keys are **omitted entirely**, not masked in place, so a
 * response can never carry a secret value nor advertise its length or shape.
 */

/**
 * Key names treated as carrying a secret VALUE.
 *
 * Deliberately precise rather than broad. A bare `token` would match
 * `max_tokens` and `pricing_usd_per_million_tokens` — LLM token counts and
 * prices, which the dashboard must be able to show. Only token names that
 * denote a credential are listed. `api_key_env` and friends are excluded
 * separately below: they name an environment variable, which is a pointer the
 * dashboard needs, not a credential.
 */
const SECRET_KEY_PATTERN =
  /(^|_)(api_?keys?|secrets?|passwords?|credentials?|private_keys?|(access|auth|bearer|refresh|session|id)_tokens?)$/i;

/** Names that reference a secret without containing one. */
const SECRET_REFERENCE_PATTERN = /_env$/i;

export function isSecretKey(key: string): boolean {
  if (SECRET_REFERENCE_PATTERN.test(key)) return false;
  return SECRET_KEY_PATTERN.test(key);
}

/**
 * Deep-copy `value`, dropping every property whose key names a secret.
 * Returns the omitted key paths so the response can be honest about the fact
 * that something was withheld.
 */
export function stripSecrets(value: unknown): { value: unknown; omitted: string[] } {
  const omitted: string[] = [];

  function walk(node: unknown, path: string): unknown {
    if (Array.isArray(node)) {
      return node.map((item, index) => walk(item, `${path}[${index}]`));
    }
    if (node !== null && typeof node === "object") {
      const result: Record<string, unknown> = {};
      for (const [key, child] of Object.entries(node)) {
        const childPath = path === "" ? key : `${path}.${key}`;
        if (isSecretKey(key)) {
          omitted.push(childPath);
          continue;
        }
        result[key] = walk(child, childPath);
      }
      return result;
    }
    return node;
  }

  return { value: walk(value, ""), omitted };
}
