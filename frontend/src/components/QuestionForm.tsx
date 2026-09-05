import { useState, FormEvent } from "react";

interface Props {
  onSubmit: (question: string) => void;
  isLoading: boolean;
}

export default function QuestionForm({ onSubmit, isLoading }: Props) {
  const [question, setQuestion] = useState("");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || isLoading) return;
    onSubmit(trimmed);
  }

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <label htmlFor="research-question" className="sr-only">
        Research question
      </label>
      <div className="flex flex-col sm:flex-row gap-3">
        <textarea
          id="research-question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="How can AI improve customer support response times?"
          rows={2}
          disabled={isLoading}
          className="flex-1 resize-none rounded-lg border border-white/15 bg-ink-raised px-4 py-3 text-[15px] text-paper placeholder:text-paper/35 outline-none focus:border-teal-bright disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={isLoading || !question.trim()}
          className="shrink-0 rounded-lg bg-teal px-5 py-3 text-[15px] font-medium text-paper transition-colors hover:bg-teal-bright disabled:cursor-not-allowed disabled:opacity-40 sm:self-start"
        >
          {isLoading ? "Working…" : "Get decision"}
        </button>
      </div>
    </form>
  );
}
