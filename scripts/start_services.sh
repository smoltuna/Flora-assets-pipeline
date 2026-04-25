#!/usr/bin/env bash
# Start all Flora services via Docker Compose.
#
# Usage:
#   ./scripts/start_services.sh            # start everything
#   ./scripts/start_services.sh --no-ui    # infrastructure + backend only (no Next.js)
#   ./scripts/start_services.sh --stop     # stop and remove all containers
#
# Services started:
#   postgres  — pgvector database (port 5432)
#   ollama    — local embedding inference (port 11434)
#   mlflow    — experiment tracking UI (port 5001)
#   backend   — FastAPI server (port 8000)
#   frontend  — Next.js dashboard (port 3000)

set -euo pipefail

COMPOSE_FILE="$(dirname "$0")/../docker-compose.yml"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ── Parse arguments ────────────────────────────────────────────────────────
NO_UI=false
STOP=false
for arg in "$@"; do
  case "$arg" in
    --no-ui)   NO_UI=true ;;
    --stop)    STOP=true ;;
    --help|-h)
      sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      echo "Unknown option: $arg  (use --help for usage)" >&2
      exit 1 ;;
  esac
done

# ── Stop mode ─────────────────────────────────────────────────────────────
if $STOP; then
  echo "Stopping all Flora services …"
  docker compose -f "$COMPOSE_FILE" down
  echo "Done."
  exit 0
fi

# ── Pre-flight checks ─────────────────────────────────────────────────────
if [ ! -f "$ROOT_DIR/.env" ]; then
  echo "WARNING: .env file not found at $ROOT_DIR/.env"
  echo "  Copy .env.example to .env and fill in API keys before running the pipeline."
  echo ""
fi

if [ ! -d "$ROOT_DIR/.venv" ]; then
  echo "WARNING: Python virtualenv not found at $ROOT_DIR/.venv"
  echo "  Run:  uv sync  (in $ROOT_DIR) to create it."
  echo ""
fi

echo "Pulling latest Docker images (this may take a while on first run)…"
docker compose -f "$COMPOSE_FILE" pull

# ── Start mode ────────────────────────────────────────────────────────────
INFRA_SERVICES="postgres ollama mlflow"
APP_SERVICES="backend"
UI_SERVICES="frontend"

echo "Starting infrastructure …"
docker compose -f "$COMPOSE_FILE" up -d $INFRA_SERVICES

echo "Waiting for Postgres to be healthy …"
until docker compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U flora -d flora &>/dev/null; do
  sleep 1
done
echo "Postgres ready."

echo "Starting backend …"
docker compose -f "$COMPOSE_FILE" up -d $APP_SERVICES

if ! $NO_UI; then
  echo "Starting frontend …"
  docker compose -f "$COMPOSE_FILE" up -d $UI_SERVICES
fi

echo ""
echo "All services running:"
echo "  FastAPI  → http://localhost:8000/docs"
echo "  MLflow   → http://localhost:5001"
if ! $NO_UI; then
  echo "  Next.js  → http://localhost:3000"
fi
