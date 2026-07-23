#!/usr/bin/env bun
/**
 * CLI entry point. `serve` lives here rather than in commands.ts because it
 * binds a port and never returns; everything else is a pure command.
 */
import { startServer } from "@compound/api/src/serve";
import { defaultEnvironment, parseArgs, runCommand } from "./commands";

const argv = process.argv.slice(2);

if (argv[0] === "serve" || argv[0] === "dev") {
  const { flags } = parseArgs(argv);
  const port = typeof flags.port === "string" ? Number.parseInt(flags.port, 10) : undefined;
  const server = startServer({
    port,
    hostname: typeof flags.host === "string" ? flags.host : undefined,
    databasePath: typeof flags.db === "string" ? flags.db : undefined,
    configPath: typeof flags.config === "string" ? flags.config : undefined,
  });
  console.log(`compound listening on http://${server.hostname}:${server.port}`);
} else {
  const { exitCode } = runCommand(argv, defaultEnvironment());
  process.exit(exitCode);
}
