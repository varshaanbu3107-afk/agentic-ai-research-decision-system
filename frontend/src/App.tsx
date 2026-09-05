import { useState } from "react";
import QuestionForm from "./components/QuestionForm";
import DecisionBanner from "./components/DecisionBanner";
import { askResearchQuestion, ResearchApiError } from "./lib/api";
import type { DecisionResult } from "./types";

type Status = "idle" | "loading" | "error";

export default function App() {
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<DecisionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(question: string) {
    setStatus("loading");
    setError(null);
    setResult(null);

    try {
      const data = await askResearchQuestion(question);
      setResult(data);
      setStatus("idle");
    } catch (err) {
      const message =
        err instanceof ResearchApiError
          ? err.message
          : "Couldn't reach the research pipeline. Check that the backend is running.";
      setError(message);
      setStatus("error");
    }
  }

  return (
    <main className="min-h-screen px-6 py-16 sm:py-24">
      <div className="mx-auto flex max-w-memo flex-col gap-10">
        <header className="flex flex-col gap-3">
          <h1 className="font-serif text-3xl sm:text-4xl leading-tight text-paper">
            Ask a question. Get an evidence-backed decision.
          </h1>
          <p className="text-[15px] leading-relaxed text-paper/60 max-w-[52ch]">
            Every decision below is produced by a multi-agent pipeline that
            plans, retrieves evidence, verifies it against the question, and
            only then decides — so you can see why it concluded what it did,
            not just what it concluded.
          </p>
        </header>

        <QuestionForm onSubmit={handleSubmit} isLoading={status === "loading"} />

        {status === "loading" && (
          <div
            className="flex items-center gap-3 rounded-lg border border-white/10 bg-ink-raised px-5 py-4 text-[15px] text-paper/60"
            role="status"
          >
            <span className="h-2 w-2 animate-pulse rounded-full bg-teal-bright" />
            Running the research pipeline — this can take 10–30 seconds.
          </div>
        )}

        {status === "error" && error && (
          <div
            role="alert"
            className="rounded-lg border border-status-low/30 bg-status-low/10 px-5 py-4"
          >
            <p className="text-[15px] font-medium text-status-low">
              The pipeline couldn't complete.
            </p>
            <p className="mt-1 text-sm text-paper/60">{error}</p>
          </div>
        )}

        {result && <DecisionBanner result={result} />}
      </div>
    </main>
  );
}
