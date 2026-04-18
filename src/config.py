"""Central configuration for SkillSync. All config comes from .env + YAML files."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LLMStrategy = Literal["hybrid_cloud", "groq_only", "gemini_only", "offline_only"]


class Settings(BaseSettings):
    """Application settings loaded from .env with sensible defaults."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths ---
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    cache_dir: Path = Field(default=PROJECT_ROOT / "data" / "embeddings_cache")
    configs_dir: Path = Field(default=PROJECT_ROOT / "configs")

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # --- PostgreSQL ---
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "skillsync"
    postgres_user: str = "skillsync"
    postgres_password: str = "change_me_locally"

    # --- Redis ---
    redis_host: str = "localhost"
    redis_port: int = 6379

    # --- LLM: Groq (primary cloud) ---
    groq_api_key: str = "gsk_paste_your_key_here"
    groq_model_heavy: str = "llama-3.3-70b-versatile"
    groq_model_fast: str = "llama-3.1-8b-instant"

    # --- LLM: Google AI Studio (secondary cloud) ---
    google_api_key: str = "paste_your_key_here"
    gemini_model: str = "gemini-2.5-flash"

    # --- LLM: Ollama (offline fallback) ---
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:1.5b"

    # --- LLM routing strategy ---
    llm_strategy: LLMStrategy = "hybrid_cloud"

    # --- Embeddings (CPU-friendly) ---
    embed_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    # --- Langfuse ---
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = "pk-lf-xxx"
    langfuse_secret_key: str = "sk-lf-xxx"

    # --- MinIO ---
    minio_endpoint: str = "localhost:9000"
    minio_root_user: str = "skillsync"
    minio_root_password: str = "change_me_locally"
    minio_bucket_raw: str = "skillsync-raw"
    minio_bucket_reports: str = "skillsync-reports"

    # --- Domain scope ---
    domain_focus: str = "CS_IT"
    accreditation_version: str = "SAR-2025"

    # --- App ---
    log_level: str = "INFO"
    environment: str = "development"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/0"

    @property
    def groq_configured(self) -> bool:
        return self.groq_api_key and not self.groq_api_key.startswith("gsk_paste")

    @property
    def gemini_configured(self) -> bool:
        return self.google_api_key and self.google_api_key != "paste_your_key_here"

    def ensure_dirs(self) -> None:
        """Create required data directories if missing."""
        for sub in ("raw/curricula", "raw/job_descriptions", "raw/accreditation",
                    "raw/industry_reports", "raw/skill_taxonomies", "processed"):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


@lru_cache(maxsize=8)
def load_yaml_config(name: str) -> dict:
    """Load a YAML config file from the configs/ directory."""
    path = get_settings().configs_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Convenience accessors ------------------------------------------------------
def nba_pos() -> dict:
    """Active NBA Program Outcomes configuration."""
    return load_yaml_config("nba_po_2025")


def blooms_taxonomy() -> dict:
    """Bloom's Taxonomy verb mappings."""
    return load_yaml_config("blooms_taxonomy")


def cs_it_pillars() -> dict:
    """CS/IT skill pillar definitions."""
    return load_yaml_config("cs_it_pillars")
