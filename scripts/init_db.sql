-- SkillSync Postgres initialization
-- Runs once on first container start (mounted via docker-compose)
--
-- Philosophy: only enable extensions + create base schemas here.
-- Actual tables are created via Alembic migrations (reproducible).

-- pgvector for hybrid workloads (we primarily use Qdrant, but having pgvector
-- available is useful for small lookups and joins)
CREATE EXTENSION IF NOT EXISTS vector;

-- UUID generation for primary keys
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Trigram search for fuzzy skill name matching
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Full-text search (we'll use English for now; ESCO supports multilingual later)
-- tsvector is built-in, no extension needed

-- Schemas to namespace different subsystems
CREATE SCHEMA IF NOT EXISTS ontology;      -- ESCO skills, taxonomies
CREATE SCHEMA IF NOT EXISTS curriculum;    -- courses, syllabi, CO-PO mappings
CREATE SCHEMA IF NOT EXISTS market;        -- job descriptions, skill trends
CREATE SCHEMA IF NOT EXISTS audit;         -- track every recommendation + evidence

-- Grant usage
GRANT ALL PRIVILEGES ON SCHEMA ontology, curriculum, market, audit TO skillsync;

-- Basic ping table for verification
CREATE TABLE IF NOT EXISTS public.system_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO public.system_info (key, value) VALUES
    ('schema_version', '0.1.0'),
    ('initialized_at', NOW()::TEXT),
    ('domain_focus', 'CS_IT'),
    ('accreditation', 'NBA_SAR_2025')
ON CONFLICT (key) DO NOTHING;
