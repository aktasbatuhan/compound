/**
 * Small shared render helpers. Kept deliberately minimal — this is an internal
 * tool, and honesty about empty/error states matters more than polish.
 */
import type { ReactNode } from "react";
import type { AssertionReportResponse, ConversationMessage } from "../lib/api";
import { ApiError } from "../lib/api";

/** Render an API failure as a message, never a crash — an offline API is a state. */
export function ErrorPanel({ error }: { error: unknown }) {
  const message =
    error instanceof ApiError
      ? error.message
      : error instanceof Error
        ? error.message
        : "unexpected error";
  return <div className="error">{message}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

function contentToText(content: unknown): string {
  if (typeof content === "string") return content;
  if (content === null || content === undefined) return "";
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (part && typeof part === "object" && "text" in part) {
          return String((part as { text: unknown }).text ?? "");
        }
        if (part && typeof part === "object" && "marker" in part) {
          return String((part as { marker: unknown }).marker ?? "");
        }
        return JSON.stringify(part);
      })
      .join("");
  }
  return JSON.stringify(content, null, 2);
}

/** Render a case's replayable input conversation. Redaction markers show as-is. */
export function Conversation({ messages }: { messages: ConversationMessage[] | undefined }) {
  if (!messages || messages.length === 0) {
    return <div className="muted">no input messages</div>;
  }
  return (
    <div>
      {messages.map((message, index) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: messages have no stable id
        <div className="msg" key={index}>
          <div className="role">{message.role ?? "?"}</div>
          <pre>{contentToText(message.content)}</pre>
          {Array.isArray((message as { tool_calls?: unknown }).tool_calls) ? (
            <pre className="muted">
              tool_calls: {JSON.stringify((message as { tool_calls: unknown }).tool_calls, null, 2)}
            </pre>
          ) : null}
        </div>
      ))}
    </div>
  );
}

/** Render the deterministic assertion result for a case. */
export function Assertions({ report }: { report: AssertionReportResponse | null }) {
  if (report === null) return <div className="muted">assertions unavailable</div>;
  if (!report.graded) {
    return <div className="muted">not graded (no observed output to check yet)</div>;
  }
  return (
    <div>
      <div className="row" style={{ marginBottom: 8 }}>
        <span className={`badge ${report.passed ? "good" : "bad"}`}>
          {report.passed ? "assertions pass" : "assertions fail"}
        </span>
        <span className="muted">score {report.score.toFixed(2)}</span>
      </div>
      {report.results.length === 0 ? (
        <div className="muted">no assertions configured for this task</div>
      ) : (
        <table>
          <tbody>
            {report.results.map((result) => (
              <tr key={`${result.type}-${result.detail}`}>
                <td>
                  <span className={`badge ${result.passed ? "good" : "bad"}`}>{result.type}</span>
                </td>
                <td className="muted">{result.detail}</td>
                <td className="muted">{result.required ? "required" : "optional"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export function partitionBadge(partition: string) {
  const sealed = partition === "decision_test";
  return <span className={`badge ${sealed ? "sealed" : ""}`}>{partition}</span>;
}
