/**
 * One error envelope for every route: `{ error: { code, message, details? } }`.
 *
 * `code` is a stable snake_case string that clients may branch on; `message` is
 * for humans and may change. Validation failures carry path-qualified details.
 */
import type { Context } from "hono";
import type { ContentfulStatusCode } from "hono/utils/http-status";

export const ERROR_CODES = [
  "not_found",
  "invalid_request",
  "invalid_config",
  "internal_error",
] as const;

export type ErrorCode = (typeof ERROR_CODES)[number];

export interface ErrorBody {
  error: {
    code: ErrorCode;
    message: string;
    details?: unknown;
  };
}

const STATUS_BY_CODE: Record<ErrorCode, ContentfulStatusCode> = {
  not_found: 404,
  invalid_request: 400,
  invalid_config: 400,
  internal_error: 500,
};

/** An error that carries its own wire representation. */
export class ApiError extends Error {
  readonly code: ErrorCode;
  readonly details: unknown;

  constructor(code: ErrorCode, message: string, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.details = details;
  }

  get status(): ContentfulStatusCode {
    return STATUS_BY_CODE[this.code];
  }

  toBody(): ErrorBody {
    return {
      error: {
        code: this.code,
        message: this.message,
        ...(this.details === undefined ? {} : { details: this.details }),
      },
    };
  }
}

export function notFound(message: string, details?: unknown): ApiError {
  return new ApiError("not_found", message, details);
}

export function invalidRequest(message: string, details?: unknown): ApiError {
  return new ApiError("invalid_request", message, details);
}

/** Render any thrown value as the standard envelope. */
export function errorResponse(c: Context, error: unknown): Response {
  if (error instanceof ApiError) {
    return c.json(error.toBody(), error.status);
  }
  const message = error instanceof Error ? error.message : "unexpected error";
  return c.json(new ApiError("internal_error", message).toBody(), 500);
}
