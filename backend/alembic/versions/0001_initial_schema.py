"""Initial schema: extensions, tables, indexes, hybrid_search function.

Revision ID: 0001
Revises:
Create Date: 2026-04-18
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "flowers",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("latin_name", sa.Text(), unique=True, nullable=False),
        sa.Column("common_name", sa.Text()),
        sa.Column("status", sa.Text(), server_default="pending"),
        sa.Column("description", sa.Text()),
        sa.Column("fun_fact", sa.Text()),
        sa.Column("wiki_description", sa.Text()),
        sa.Column("habitat", sa.Text()),
        sa.Column("etymology", sa.Text()),
        sa.Column("cultural_info", sa.Text()),
        sa.Column("petal_color_hex", sa.Text()),
        sa.Column("care_info", JSONB()),
        sa.Column("edibility_rating", sa.Integer()),
        sa.Column("medicinal_rating", sa.Integer()),
        sa.Column("other_uses_rating", sa.Integer()),
        sa.Column("weed_potential", sa.Text()),
        sa.Column("info_image_path", sa.Text()),
        sa.Column("info_image_author", sa.Text()),
        sa.Column("main_image_path", sa.Text()),
        sa.Column("lock_image_path", sa.Text()),
        sa.Column("feature_year", sa.Integer()),
        sa.Column("feature_month", sa.Integer()),
        sa.Column("feature_day", sa.Integer()),
        sa.Column("confidence_scores", JSONB()),
        sa.Column("wikipedia_url", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "raw_sources",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("flower_id", sa.BigInteger(), sa.ForeignKey("flowers.id", ondelete="CASCADE")),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("raw_content", sa.Text()),
        sa.Column("parsed_content", JSONB()),
        sa.Column("scraped_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("flower_id", "source", name="uq_raw_sources_flower_source"),
    )

    op.create_table(
        "source_embeddings",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("raw_source_id", sa.BigInteger(), sa.ForeignKey("raw_sources.id", ondelete="CASCADE")),
        sa.Column("flower_id", sa.BigInteger(), sa.ForeignKey("flowers.id", ondelete="CASCADE")),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("metadata", JSONB()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Add vector column separately (pgvector DDL)
    op.execute("ALTER TABLE source_embeddings ADD COLUMN embedding vector(768)")

    # tsvector GENERATED ALWAYS column for BM25 hybrid search
    op.execute("""
        ALTER TABLE source_embeddings
        ADD COLUMN chunk_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED
    """)

    # HNSW index for dense vector search
    op.execute("""
        CREATE INDEX ix_source_embeddings_hnsw
        ON source_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # GIN index for BM25 full-text search
    op.execute("""
        CREATE INDEX ix_source_embeddings_gin_tsv
        ON source_embeddings USING gin (chunk_tsv)
    """)

    op.create_table(
        "translations",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("flower_id", sa.BigInteger(), sa.ForeignKey("flowers.id", ondelete="CASCADE")),
        sa.Column("language", sa.Text(), nullable=False),
        sa.Column("name", sa.Text()),
        sa.Column("description", sa.Text()),
        sa.Column("fun_fact", sa.Text()),
        sa.Column("wiki_description", sa.Text()),
        sa.Column("habitat", sa.Text()),
        sa.Column("etymology", sa.Text()),
        sa.Column("cultural_info", sa.Text()),
        sa.Column("source_method", sa.Text()),
        sa.UniqueConstraint("flower_id", "language", name="uq_translations_flower_language"),
    )

    # BM25 + vector hybrid search function (Reciprocal Rank Fusion)
    op.execute("""
        CREATE OR REPLACE FUNCTION hybrid_search(
            query_text TEXT,
            query_embedding vector(768),
            match_count INT DEFAULT 10,
            rrf_k INT DEFAULT 60
        ) RETURNS TABLE(chunk_id INT, chunk_text TEXT, rrf_score FLOAT) AS $$
            WITH vector_results AS (
                SELECT id, 1.0 / (rrf_k + ROW_NUMBER() OVER (
                    ORDER BY embedding <=> query_embedding
                )) AS score
                FROM source_embeddings
                ORDER BY embedding <=> query_embedding
                LIMIT match_count * 2
            ),
            text_results AS (
                SELECT id, 1.0 / (rrf_k + ROW_NUMBER() OVER (
                    ORDER BY ts_rank(chunk_tsv, websearch_to_tsquery(query_text)) DESC
                )) AS score
                FROM source_embeddings
                WHERE chunk_tsv @@ websearch_to_tsquery(query_text)
                LIMIT match_count * 2
            )
            SELECT
                se.id, se.chunk_text,
                COALESCE(v.score, 0) + COALESCE(t.score, 0) AS rrf_score
            FROM source_embeddings se
            LEFT JOIN vector_results v ON se.id = v.id
            LEFT JOIN text_results t ON se.id = t.id
            WHERE v.id IS NOT NULL OR t.id IS NOT NULL
            ORDER BY rrf_score DESC
            LIMIT match_count;
        $$ LANGUAGE sql;
    """)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS hybrid_search")
    op.drop_table("translations")
    op.drop_table("source_embeddings")
    op.drop_table("raw_sources")
    op.drop_table("flowers")
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")
    op.execute("DROP EXTENSION IF EXISTS vector")
