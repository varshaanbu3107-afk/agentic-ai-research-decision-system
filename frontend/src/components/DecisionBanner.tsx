import type { DecisionResult } from "../types";
import { confidenceTier } from "../types";

const TIER_STYLES = {
  high: {
    dot: "bg-status-high",
    text: "text-status-high",
    ring: "ring-status-high/30",
  },
  medium: {
    dot: "bg-status-medium",
    text: "text-status-medium",
    ring: "ring-status-medium/30",
  },
  low: {
    dot: "bg-status-low",
    text: "text-status-low",
    ring: "ring-status-low/30",
  },
} as const;

const TIER_LABEL = {
  high: "High confidence",
  medium: "Moderate confidence",
  low: "Low confidence",
} as const;

export default function DecisionBanner({ result }: { result: DecisionResult }) {
  const score = result.decision.average_evidence_quality;
  const tier = confidenceTier(score);
  const styles = TIER_STYLES[tier];

  const reasons =
    result.decision.decision_reasons?.length > 0
      ? result.decision.decision_reasons
      : result.final_report.decision_reasons;

  const topReasons = reasons.slice(0, 3);

  return (
    <section
      className={`rounded-lg border border-white/10 bg-ink-raised p-6 sm:p-8 ring-1 ${styles.ring}`}
      aria-live="polite"
    >
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <p className="font-serif text-xl sm:text-2xl leading-snug text-paper max-w-[46ch]">
          {result.decision.decision}
        </p>

        <div
          className={`flex items-center gap-2 rounded-full border border-white/10 bg-black/20 px-3 py-1.5 shrink-0`}
        >
          <span className={`h-2 w-2 rounded-full ${styles.dot}`} />
          <span className={`text-sm font-medium ${styles.text}`}>
            {TIER_LABEL[tier]}
          </span>
          <span className="text-sm text-paper/50">·</span>
          <span className="text-sm tabular-nums text-paper/80">
            {score.toFixed(0)}%
          </span>
        </div>
      </div>

      {topReasons.length > 0 && (
        <ul className="mt-5 space-y-2 border-t border-white/10 pt-4">
          {topReasons.map((reason, i) => (
            <li
              key={i}
              className="flex gap-2.5 text-[15px] leading-relaxed text-paper/75"
            >
              <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-paper/40" />
              <span>{reason}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
