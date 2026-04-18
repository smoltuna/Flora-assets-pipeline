# Flora Asset Pipeline v2

> Automated botanical data enrichment and image pipeline for the [Flora iOS app](https://apps.apple.com/ca/app/flora-flower-of-the-day/id6759986494)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org)
[![pgvector](https://img.shields.io/badge/pgvector-HNSW-orange.svg)](https://github.com/pgvector/pgvector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What It Does

Given a plant's latin name (e.g. `Rosa canina`), the pipeline:

1. Scrapes four authoritative botanical sources (PFAF, Wikipedia, Wikidata, GBIF)
2. Embeds source chunks into pgvector (768-dim HNSW index)
3. Retrieves relevant context via hybrid BM25 + dense vector search (Reciprocal Rank Fusion)
4. Removes semantic near-duplicates (0.92 cosine threshold)
5. Grades retrieval quality per-field using **Corrective RAG (CRAG)**
6. Synthesizes structured botanical content via a local or cloud LLM
7. Fact-checks generated fields against sources using **Self-RAG**
8. Translates into DE/FR/ES/IT (MarianMT) and ZH/JA (Llama)
9. Exports xcassets-compatible JSON for the Flora iOS app

**Output:** description, fun fact, habitat, etymology, cultural info, care data, petal color hex, confidence scores, and 3 processed images (info, transparent-background blossom, lock screen).

---

## Architecture

```
                          ┌─────────────────────────────────────────┐
                          │            FastAPI Backend               │
                          │                                         │
  Latin Name  ──────────► │  /scrape   →  PFAF · Wikipedia          │
                          │             Wikidata · GBIF              │
                          │                 │                       │
                          │                 ▼                       │
                          │  Embedder  →  pgvector (HNSW 768d)      │
                          │                 │                       │
                          │  Retriever (BM25 + dense, RRF fusion)   │
                          │                 │                       │
                          │  Deduplicator (cosine 0.92 threshold)   │
                          │                 │                       │
                          │  CRAG Grader  →  per-field quality check │
                          │                 │                       │
                          │  Synthesizer  →  LLM (Ollama/Groq)      │
                          │                 │                       │
                          │  Self-RAG Verifier  →  confidence scores │
                          │                 │                       │
                          │  /images   →  Wikimedia → rembg → lock  │
                          │  /translate →  MarianMT + Llama CJK     │
                          │  /export   →  xcassets JSON             │
                          └─────────────────────────────────────────┘
                                          │
                          ┌──────────────────────────────┐
                          │      Next.js Dashboard        │
                          │  Library · Detail · Scores    │
                          └──────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Node.js 20+ (for frontend dev)

### 1. Clone and configure

```bash
git clone https://github.com/yourusername/Flora-RAG-Pipeline.git
cd Flora-RAG-Pipeline
cp .env.example .env
# Edit .env — set LLM_PROVIDER and the matching API key
```

### 2. Start all services

```bash
docker-compose up -d
# PostgreSQL + pgvector → localhost:5432
# Ollama (local LLM)    → localhost:11434
# MLflow tracking UI    → localhost:5000
# FastAPI backend       → localhost:8000  (Swagger: /docs)
# Next.js frontend      → localhost:3000
```

### 3. Run database migrations

```bash
cd backend
uv sync --all-extras
uv run alembic upgrade head
```

### 4. Seed and enrich flowers

```bash
# Add demo flowers
uv run python scripts/seed_flowers.py --file data/demo_flowers.txt

# Run full pipeline on first 5 pending flowers
uv run python scripts/run_batch.py --limit 5
```

### 5. Browse the dashboard

Open [http://localhost:3000](http://localhost:3000) — add flowers, run scrape/enrich, inspect confidence scores.

### 6. Export for Flora iOS app

```bash
# Export all enriched flowers
python -m cli export --all --output ./exports
```

---

## Pipeline Stages

| Stage | Description | Implementation |
|-------|-------------|----------------|
| **Scrape** | PFAF (care info + ratings), Wikipedia (extract + taxobox), Wikidata SPARQL (range, conservation), GBIF (taxonomy, distributions) | `services/scraper/` |
| **Embed** | Chunk source text → `nomic-embed-text` → pgvector (768d, HNSW, m=16, ef=64) | `services/rag/embedder.py` |
| **Retrieve** | Hybrid BM25 (`tsvector GIN`) + cosine similarity (`HNSW <=>`) fused via RRF | `services/rag/retriever.py` |
| **Deduplicate** | Remove semantically near-identical chunks (cosine ≥ 0.92); prefer longer/structured source | `services/rag/deduplicator.py` |
| **Grade (CRAG)** | LLM scores retrieved chunks per field; re-routes synthesis if quality is insufficient | `services/rag/grader.py` |
| **Synthesize** | Source-attributed prompt → LLM → structured JSON (description, fun_fact, habitat, etc.) | `services/rag/synthesizer.py` |
| **Verify (Self-RAG)** | LLM fact-checks each generated field against the original source material; returns confidence 0–1 | `services/rag/verifier.py` |
| **Images** | Wikimedia Commons (CC0/CC-BY only) → rembg background removal → lock screen posterize | `services/images/` |
| **Translate** | MarianMT for EU languages (de/fr/es/it); Llama via Ollama for CJK (zh/ja) | `services/translation/` |
| **Export** | Merge all fields → xcassets-compatible JSON with image references | `routers/export.py`, `cli/export.py` |

---

## Project Structure

```
Flora-RAG-Pipeline/
├── backend/
│   ├── main.py                  # FastAPI app, OTel + Prometheus setup
│   ├── config.py                # Pydantic settings (env-driven)
│   ├── models.py                # SQLAlchemy ORM (Flower, RawSource, SourceEmbedding, Translation)
│   ├── alembic/                 # DB migrations (hybrid_search SQL fn, HNSW/GIN indexes)
│   ├── routers/                 # flowers · scrape · enrich · images · translate · export
│   ├── services/
│   │   ├── scraper/             # pfaf · wikipedia · wikidata · gbif
│   │   ├── rag/                 # embedder · retriever · deduplicator · grader · synthesizer · verifier
│   │   ├── llm/                 # provider abstraction · ollama · groq · together
│   │   ├── images/              # wikimedia · processor · lock_gen
│   │   └── translation/         # translator (MarianMT + Llama CJK)
│   └── tasks/
│       └── pipeline.py          # 9-stage sequential orchestration + MLflow tracking
├── frontend/
│   └── src/
│       ├── app/                 # Dashboard + flower detail page (Next.js App Router)
│       ├── components/          # FlowerCard · DataFieldsView · ConfidenceScores
│       ├── lib/api.ts           # Typed API client
│       └── types/flower.ts      # TypeScript interfaces
├── tests/
│   ├── test_rag_pipeline.py     # Unit: deduplication, synthesis parsing
│   ├── test_scrapers.py         # Unit: Wikipedia + GBIF with respx mocks
│   ├── test_deduplication.py    # Unit: extended deduplication scenarios
│   └── test_integration.py      # Integration: FastAPI + PostgreSQL end-to-end
├── experiments/
│   ├── embedding_eval.py        # MLflow: compare nomic-embed-text vs mxbai-embed-large on recall@5
│   └── chunk_strategy.py        # MLflow: compare whole_doc / paragraph / sentence chunking strategies
├── data/
│   └── demo_flowers.txt         # 15 botanically diverse demo entries
├── scripts/
│   ├── seed_flowers.py          # Bulk-insert latin names
│   └── run_batch.py             # Process pending flowers through full pipeline
├── cli/
│   ├── __main__.py              # Entry point: export / status subcommands
│   └── export.py                # xcassets JSON export
├── data/
│   └── demo_flowers.txt         # 15 botanically diverse demo entries
├── infra/                       # Terraform: AWS S3 + VPC + RDS PostgreSQL 17
├── k8s/                         # Kubernetes: Deployment · Service · Ingress · Secrets
├── .github/workflows/ci.yml     # GitHub Actions: lint → typecheck → test → Docker build
└── docker-compose.yml           # PostgreSQL · Ollama · MLflow · backend · frontend
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.11, FastAPI, SQLAlchemy 2 async, Alembic |
| **Vector DB** | PostgreSQL 17 + pgvector (HNSW index, 768d), hybrid `hybrid_search()` SQL function |
| **LLM (local)** | Ollama — `llama3.1:8b` (synthesis) + `nomic-embed-text` (embedding) |
| **LLM (cloud)** | Groq, Together.ai, OpenAI — provider-agnostic `LLMProvider` Protocol |
| **RAG patterns** | CRAG, Self-RAG, semantic deduplication, hybrid BM25+dense retrieval |
| **Translation** | Hugging Face `transformers` (MarianMT EU), Llama CJK via Ollama |
| **Image processing** | rembg (background removal), Pillow (resize/crop/posterize) |
| **Frontend** | Next.js 15 App Router, TypeScript, Tailwind CSS |
| **Observability** | OpenTelemetry (FastAPI auto-instrumentation), Prometheus `/metrics`, MLflow experiment tracking |
| **Infrastructure** | Docker Compose (local), Terraform (AWS S3 + RDS), Kubernetes manifests |
| **CI/CD** | GitHub Actions (ruff lint, mypy, pytest, Docker build) |

---

## LLM Providers

The backend abstracts LLM calls behind a `LLMProvider` Protocol. Configure via `LLM_PROVIDER` env var:

| Provider | `LLM_PROVIDER` value | Required env var |
|----------|---------------------|------------------|
| Ollama (local, default) | `ollama` | — |
| Groq (free tier, fast) | `groq` | `GROQ_API_KEY` |
| Together.ai | `together` | `TOGETHER_API_KEY` |
| OpenAI | `openai` | `OPENAI_API_KEY` |

---

## API Reference

FastAPI auto-generates Swagger docs at [http://localhost:8000/docs](http://localhost:8000/docs).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/flowers` | GET | List flowers (filter by `?status=`) |
| `/flowers` | POST | Add a new flower |
| `/flowers/{id}` | GET | Get flower detail |
| `/scrape/{id}/sync` | POST | Scrape all sources (sync) |
| `/enrich/{id}/sync` | POST | Run full RAG pipeline (sync) |
| `/enrich/{id}/chunks` | GET | Inspect retrieved + deduplicated chunks |
| `/images/{id}` | POST | Run image pipeline (async) |
| `/images/{id}/serve/{type}` | GET | Serve image file (`info`/`main`/`lock`) |
| `/translate/{id}` | POST | Run translation for all languages |
| `/export/{id}` | GET | Export xcassets-compatible JSON |
| `/metrics` | GET | Prometheus metrics scrape endpoint |
| `/health` | GET | Health check |

---

## MLflow Experiment Tracking

Each pipeline run logs to MLflow under the `flora-enrichment` experiment:

- **Parameters:** `latin_name`, `sources_scraped`, `llm_provider`
- **Metrics:** `pipeline_duration_s`, `chunks_retrieved`, `chunks_after_dedup`, per-field `confidence_llm_*` scores

View the MLflow UI at [http://localhost:5000](http://localhost:5000).

### Embedding Model Evaluation

Compare retrieval quality across embedding models on a hand-labeled botanical query set:

```bash
# Ensure Ollama models are pulled
ollama pull nomic-embed-text
ollama pull mxbai-embed-large

uv run python experiments/embedding_eval.py
```

Logs recall@5 and precision@5 per model to the `embedding-model-eval` experiment.

### Chunking Strategy Evaluation

Compare whole-document, paragraph, and sentence-level chunking on recall@5:

```bash
uv run python experiments/chunk_strategy.py
```

Logs recall@5, index size, and build time per strategy to `chunk-strategy-eval`.

---

## Running Tests

```bash
cd backend

# Unit tests (no DB required)
uv run pytest tests/ -v -m "not integration"

# Integration tests (requires PostgreSQL)
DATABASE_URL=postgresql+asyncpg://flora:flora@localhost:5432/flora \
  uv run pytest tests/test_integration.py -v -m integration
```

---

## Cloud Deployment

### Terraform (AWS)

```bash
cd infra
terraform init
terraform plan -var="db_password=your-password"
terraform apply
```

Provisions: S3 artifact bucket, VPC (2 private subnets), RDS PostgreSQL 17 with pgvector.

### Kubernetes

```bash
# Apply secrets (edit k8s/secrets.yaml first)
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/
```

---

## Design Decisions

- **No LangChain** — all RAG components are plain Python for full control and simpler debugging
- **Sequential pipeline** — no parallelism; simpler reasoning, easier to trace
- **Provider-agnostic LLM** — swap Ollama → Groq → Together with a single env var
- **Semantic dedup threshold 0.92** — calibrated to collapse paraphrase without losing complementary sources
- **CC0/CC-BY images only** — no licensing friction for the Flora iOS app

---

## License

MIT
