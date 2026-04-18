"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { FlowerCard } from "@/components/FlowerCard";
import type { Flower } from "@/types/flower";

const STATUS_FILTERS = ["all", "pending", "scraped", "enriched", "complete", "scrape_failed"];

export default function DashboardPage() {
  const [flowers, setFlowers] = useState<Flower[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState("all");
  const [newLatinName, setNewLatinName] = useState("");
  const [adding, setAdding] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const data = await api.flowers.list(statusFilter === "all" ? undefined : statusFilter);
      setFlowers(data);
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [statusFilter]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newLatinName.trim()) return;
    setAdding(true);
    try {
      await api.flowers.create({ latin_name: newLatinName.trim() });
      setNewLatinName("");
      await load();
    } catch (e) {
      setError(String(e));
    } finally {
      setAdding(false);
    }
  };

  const counts = flowers.reduce<Record<string, number>>((acc, f) => {
    acc[f.status] = (acc[f.status] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold">Flower Library</h2>
          <p className="text-sm text-gray-500 mt-0.5">{flowers.length} flowers</p>
        </div>
        <form onSubmit={handleAdd} className="flex gap-2">
          <input
            type="text"
            value={newLatinName}
            onChange={(e) => setNewLatinName(e.target.value)}
            placeholder="Latin name (e.g. Rosa canina)"
            className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm w-64 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="submit"
            disabled={adding || !newLatinName.trim()}
            className="bg-blue-600 text-white text-sm px-4 py-1.5 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {adding ? "Adding..." : "Add Flower"}
          </button>
        </form>
      </div>

      {/* Status summary */}
      <div className="flex gap-2 flex-wrap mb-6">
        {STATUS_FILTERS.map((s) => (
          <button
            key={s}
            onClick={() => setStatusFilter(s)}
            className={`text-xs px-3 py-1 rounded-full border transition-colors ${
              statusFilter === s
                ? "bg-blue-600 text-white border-blue-600"
                : "bg-white text-gray-600 border-gray-200 hover:border-gray-300"
            }`}
          >
            {s === "all" ? `All (${flowers.length})` : `${s} (${counts[s] ?? 0})`}
          </button>
        ))}
      </div>

      {error && (
        <div className="rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700 mb-4">
          {error}
        </div>
      )}

      {loading ? (
        <div className="text-sm text-gray-400 text-center py-12">Loading...</div>
      ) : flowers.length === 0 ? (
        <div className="text-sm text-gray-400 text-center py-12">
          No flowers found. Add one above to get started.
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {flowers.map((f) => <FlowerCard key={f.id} flower={f} />)}
        </div>
      )}
    </div>
  );
}
