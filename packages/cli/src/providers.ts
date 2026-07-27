/**
 * `compound providers [name]` — the known-provider registry (issues #8–#18).
 *
 * With no argument it lists every known OpenAI-compatible/flex host with its
 * base_url, API-key env var, and tool-calling support. With a name it emits a
 * paste-ready `providers:` block for that host. Nothing is written — the user
 * copies the block into their own compound.yaml and adds models + a price.
 */
import { KNOWN_PROVIDERS, knownProvider, providerBlockYaml } from "@compound/config";
import type { CommandEnvironment, CommandResult, ParsedArgs } from "./commands";

export function runProvidersCommand(args: ParsedArgs, env: CommandEnvironment): CommandResult {
  const name = args.positional[0];

  if (name !== undefined) {
    const provider = knownProvider(name);
    if (provider === undefined) {
      env.write(
        `error: no known provider '${name}'. Known: ${KNOWN_PROVIDERS.map((p) => p.name).join(", ")}.`,
      );
      env.write(
        "Any OpenAI-compatible host works via the 'self_hosted' template — point base_url at it.",
      );
      return { exitCode: 1 };
    }
    env.write(
      `# ${provider.name}: tool calling ${provider.tools}${provider.note ? ` — ${provider.note}` : ""}`,
    );
    env.write("providers:");
    env.write(providerBlockYaml(provider));
    env.write("# Then add models under models.candidates/frontier and a price for each.");
    return { exitCode: 0 };
  }

  env.write("known providers (declare one in compound.yaml, or run: compound providers <name>):");
  env.write("");
  for (const provider of KNOWN_PROVIDERS) {
    env.write(`  ${provider.name.padEnd(12)} ${provider.type.padEnd(18)} tools: ${provider.tools}`);
    env.write(`  ${" ".repeat(12)} ${provider.baseUrl}   [${provider.apiKeyEnv}]`);
    if (provider.note) env.write(`  ${" ".repeat(12)} ${provider.note}`);
    env.write("");
  }
  env.write(
    "anthropic and google are not here yet — they need native adapters (#11, #12); until then a",
  );
  env.write("gate/experiment on them fails honestly rather than sending wrong-shaped requests.");
  return { exitCode: 0 };
}
