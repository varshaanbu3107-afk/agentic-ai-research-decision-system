// Mirrors the JSON shape returned by POST /research
// (see app/core/orchestrator.py -> run_research_system's return value).
// Keep this in sync with the backend — do not invent fields here that
// the backend doesn't actually send.

export interface DecisionResult {
  research_question: string;
  decision: {
    decision: string;
    average_evidence_quality: number; // 0-100
    decision_reasons: string[];
    limitations: string[];
  };
  verification: {
    overall_relevance: "High" | "Medium" | "Low" | string;
    verification_confidence: "High" | "Medium" | "Low" | string;
  };
  report_quality: {
    average_evidence_quality: number;
    supporting_evidence: number;
    partially_supporting_evidence: number;
    contradicting_evidence: number;
    neutral_evidence: number;
    total_verified_evidence: number;
  };
  final_report: {
    decision_reasons: string[];
    limitations: string[];
  };
}

export type ConfidenceTier = "high" | "medium" | "low";

export function confidenceTier(score: number): ConfidenceTier {
  if (score > 70) return "high";
  if (score >= 40) return "medium";
  return "low";
}
