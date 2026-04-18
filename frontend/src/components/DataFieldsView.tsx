import type { Flower } from "@/types/flower";

interface Props {
  flower: Flower;
}

const NOT_AVAILABLE = "Information not available.";

function Field({ label, value }: { label: string; value: string | null | undefined }) {
  if (!value || value === NOT_AVAILABLE) return null;
  return (
    <div>
      <dt className="text-xs font-medium text-gray-500 uppercase tracking-wide">{label}</dt>
      <dd className="mt-1 text-sm text-gray-800">{value}</dd>
    </div>
  );
}

function RatingBar({ label, value, max = 5 }: { label: string; value: number | null; max?: number }) {
  if (value === null) return null;
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-gray-500 w-28">{label}</span>
      <div className="flex gap-1">
        {Array.from({ length: max }).map((_, i) => (
          <div
            key={i}
            className={`w-3 h-3 rounded-sm ${i < value ? "bg-green-400" : "bg-gray-200"}`}
          />
        ))}
      </div>
      <span className="text-xs text-gray-500">{value}/{max}</span>
    </div>
  );
}

export function DataFieldsView({ flower }: Props) {
  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-gray-200 bg-white p-5">
        <h3 className="text-sm font-semibold mb-4">Generated Fields</h3>
        <dl className="space-y-4">
          <Field label="Description" value={flower.description} />
          <Field label="Fun Fact" value={flower.fun_fact} />
          <Field label="Wikipedia Summary" value={flower.wiki_description} />
          <Field label="Habitat" value={flower.habitat} />
          <Field label="Etymology" value={flower.etymology} />
          <Field label="Cultural Info" value={flower.cultural_info} />
        </dl>
      </div>

      <div className="rounded-xl border border-gray-200 bg-white p-5">
        <h3 className="text-sm font-semibold mb-4">PFAF Ratings</h3>
        <div className="space-y-2">
          <RatingBar label="Edibility" value={flower.edibility_rating} />
          <RatingBar label="Medicinal" value={flower.medicinal_rating} />
          <RatingBar label="Other Uses" value={flower.other_uses_rating} />
        </div>
        {flower.weed_potential && (
          <p className="mt-3 text-xs text-gray-500">Weed potential: {flower.weed_potential}</p>
        )}
      </div>

      {flower.care_info && Object.keys(flower.care_info).length > 0 && (
        <div className="rounded-xl border border-gray-200 bg-white p-5">
          <h3 className="text-sm font-semibold mb-4">Care Info</h3>
          <dl className="grid grid-cols-2 gap-3">
            {Object.entries(flower.care_info).slice(0, 10).map(([k, v]) => (
              <div key={k}>
                <dt className="text-xs text-gray-400 capitalize">{k}</dt>
                <dd className="text-sm text-gray-700">{v}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}
    </div>
  );
}
