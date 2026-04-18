import type { ConfidenceScores } from "@/types/flower";

interface Props {
  scores: ConfidenceScores;
}

const THRESHOLD = 0.72;

export function ConfidenceScoresView({ scores }: Props) {
  const entries = Object.entries(scores);
  if (!entries.length) return null;

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-5">
      <h3 className="text-sm font-semibold mb-4">Confidence Scores</h3>
      <div className="space-y-3">
        {entries.map(([field, { llm_score, embedding_score }]) => {
          const avg = (llm_score + embedding_score) / 2;
          const pct = Math.round(avg * 100);
          const isLow = avg < THRESHOLD;
          return (
            <div key={field}>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-gray-600 capitalize">{field.replace(/_/g, " ")}</span>
                <span className={isLow ? "text-red-500 font-medium" : "text-gray-500"}>
                  {pct}% {isLow && "(low)"}
                </span>
              </div>
              <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${isLow ? "bg-red-400" : "bg-emerald-400"}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-gray-400 mt-0.5">
                <span>LLM: {Math.round(llm_score * 100)}%</span>
                <span>Embedding: {Math.round(embedding_score * 100)}%</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
