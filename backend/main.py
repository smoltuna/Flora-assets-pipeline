from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace, metrics as otel_metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import make_asgi_app

from config import settings
from database import create_tables
from log_config import configure_logging
from routers import flowers, scrape, enrich, images, translate, export


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    await create_tables()
    yield


app = FastAPI(
    title="Flora Asset Pipeline",
    description="Automated botanical data and image pipeline for the Flora iOS app",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenTelemetry — traces
tracer_provider = TracerProvider()
trace.set_tracer_provider(tracer_provider)

# OpenTelemetry — metrics exported to Prometheus
_prometheus_reader = PrometheusMetricReader()
_meter_provider = MeterProvider(metric_readers=[_prometheus_reader])
otel_metrics.set_meter_provider(_meter_provider)

FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)

# Prometheus scrape endpoint (consumed by the OTel metrics reader above)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Routers
app.include_router(flowers.router, prefix="/flowers", tags=["flowers"])
app.include_router(scrape.router, prefix="/scrape", tags=["scrape"])
app.include_router(enrich.router, prefix="/enrich", tags=["enrich"])
app.include_router(images.router, prefix="/images", tags=["images"])
app.include_router(translate.router, prefix="/translate", tags=["translate"])
app.include_router(export.router, prefix="/export", tags=["export"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "environment": settings.environment}
