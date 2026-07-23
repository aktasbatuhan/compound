/**
 * Parsing and validating a `POST /api/imports` body.
 *
 * Exactly one of `path` or `content` must be given: a path for the CLI and
 * local files, inline content for a dashboard upload. Accepting both would
 * leave the precedence ambiguous, so it is an error rather than a silent
 * preference.
 */
import { invalidRequest } from "./errors";

export interface ImportRequest {
  importer: string;
  path?: string;
  content?: string;
  project_id?: string;
}

function optionalString(body: Record<string, unknown>, key: string): string | undefined {
  const value = body[key];
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "string") {
    throw invalidRequest(`${key} must be a string`, { parameter: key });
  }
  return value;
}

export function parseImportRequest(body: unknown): ImportRequest {
  if (body === null || typeof body !== "object" || Array.isArray(body)) {
    throw invalidRequest("request body must be a JSON object");
  }
  const record = body as Record<string, unknown>;

  const importer = record.importer;
  if (typeof importer !== "string" || importer.length === 0) {
    throw invalidRequest("importer is required", { parameter: "importer" });
  }

  const path = optionalString(record, "path");
  const content = optionalString(record, "content");

  if (path === undefined && content === undefined) {
    throw invalidRequest("exactly one of path or content is required", {
      parameter: "path|content",
    });
  }
  if (path !== undefined && content !== undefined) {
    throw invalidRequest("path and content are mutually exclusive", {
      parameter: "path|content",
    });
  }

  return {
    importer,
    path,
    content,
    project_id: optionalString(record, "project_id"),
  };
}
