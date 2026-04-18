import type { Flower, FlowerCreate, ScrapeResult, EnrichResult } from "@/types/flower";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  flowers: {
    list: (status?: string): Promise<Flower[]> => {
      const qs = status ? `?status=${encodeURIComponent(status)}` : "";
      return request<Flower[]>(`/flowers${qs}`);
    },
    get: (id: number): Promise<Flower> => request<Flower>(`/flowers/${id}`),
    create: (body: FlowerCreate): Promise<Flower> =>
      request<Flower>("/flowers", { method: "POST", body: JSON.stringify(body) }),
    delete: (id: number): Promise<void> => request<void>(`/flowers/${id}`, { method: "DELETE" }),
  },
  scrape: {
    run: (flowerId: number): Promise<ScrapeResult> =>
      request<ScrapeResult>(`/scrape/${flowerId}/sync`, { method: "POST" }),
  },
  enrich: {
    run: (flowerId: number): Promise<EnrichResult> =>
      request<EnrichResult>(`/enrich/${flowerId}/sync`, { method: "POST" }),
    chunks: (flowerId: number, deduplicated?: boolean): Promise<unknown[]> => {
      const qs = deduplicated ? "?deduplicated=true" : "";
      return request<unknown[]>(`/enrich/${flowerId}/chunks${qs}`);
    },
  },
};
