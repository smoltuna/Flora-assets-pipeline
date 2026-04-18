export type FlowerStatus =
  | "pending"
  | "scraping"
  | "scraped"
  | "embedding"
  | "enriched"
  | "images_done"
  | "complete"
  | "scrape_failed";

export interface Flower {
  id: number;
  latin_name: string;
  common_name: string | null;
  status: FlowerStatus;
  description: string | null;
  fun_fact: string | null;
  wiki_description: string | null;
  habitat: string | null;
  etymology: string | null;
  cultural_info: string | null;
  petal_color_hex: string | null;
  care_info: Record<string, string> | null;
  edibility_rating: number | null;
  medicinal_rating: number | null;
  other_uses_rating: number | null;
  weed_potential: string | null;
  info_image_path: string | null;
  info_image_author: string | null;
  main_image_path: string | null;
  lock_image_path: string | null;
  feature_year: number | null;
  feature_month: number | null;
  feature_day: number | null;
  confidence_scores: ConfidenceScores | null;
  wikipedia_url: string | null;
}

export interface ConfidenceScores {
  [field: string]: {
    llm_score: number;
    embedding_score: number;
  };
}

export interface FlowerCreate {
  latin_name: string;
  common_name?: string;
}

export interface ScrapeResult {
  flower_id: number;
  latin_name: string;
  sources_scraped: string[];
  sources_failed: string[];
}

export interface EnrichResult {
  flower_id: number;
  latin_name: string;
  status: string;
  confidence_scores: ConfidenceScores | null;
}
