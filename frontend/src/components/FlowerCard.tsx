import Link from "next/link";
import type { Flower } from "@/types/flower";

const STATUS_COLORS: Record<string, string> = {
  pending: "bg-gray-100 text-gray-600",
  scraping: "bg-blue-100 text-blue-700",
  scraped: "bg-yellow-100 text-yellow-700",
  embedding: "bg-purple-100 text-purple-700",
  enriched: "bg-green-100 text-green-700",
  images_done: "bg-teal-100 text-teal-700",
  complete: "bg-emerald-100 text-emerald-700",
  scrape_failed: "bg-red-100 text-red-700",
};

interface Props {
  flower: Flower;
}

export function FlowerCard({ flower }: Props) {
  const colorDot = flower.petal_color_hex ?? "#d1d5db";
  const slug = encodeURIComponent(flower.latin_name);

  return (
    <Link href={`/flowers/${slug}`} className="block rounded-xl border border-gray-200 bg-white p-5 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <div
            className="w-4 h-4 rounded-full flex-shrink-0 border border-gray-200"
            style={{ background: colorDot }}
          />
          <div className="min-w-0">
            <p className="font-medium text-sm truncate">{flower.latin_name}</p>
            {flower.common_name && (
              <p className="text-xs text-gray-500 truncate">{flower.common_name}</p>
            )}
          </div>
        </div>
        <span className={`text-xs px-2 py-0.5 rounded-full font-medium flex-shrink-0 ${STATUS_COLORS[flower.status] ?? "bg-gray-100 text-gray-600"}`}>
          {flower.status}
        </span>
      </div>

      {flower.description && (
        <p className="mt-3 text-xs text-gray-600 line-clamp-2">{flower.description}</p>
      )}

      {flower.confidence_scores && (
        <div className="mt-3 flex gap-2 flex-wrap">
          {Object.entries(flower.confidence_scores).slice(0, 3).map(([field, scores]) => {
            const avg = ((scores.llm_score + scores.embedding_score) / 2) * 100;
            return (
              <span key={field} className="text-xs text-gray-400">
                {field.replace("_", " ")}: {avg.toFixed(0)}%
              </span>
            );
          })}
        </div>
      )}
    </Link>
  );
}
