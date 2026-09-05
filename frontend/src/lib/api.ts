import type { DecisionResult } from "../types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export class ResearchApiError extends Error {}

export async function askResearchQuestion(
  question: string
): Promise<DecisionResult> {
  const response = await fetch(`${API_URL}/research`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail =
      (body && typeof body.detail === "string" && body.detail) ||
      `The server returned an error (${response.status}).`;
    throw new ResearchApiError(detail);
  }

  return response.json();
}
