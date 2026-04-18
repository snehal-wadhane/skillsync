"""SQLAlchemy ORM for the skill ontology tables in Postgres.

Schema is namespaced under `ontology.*`. Tables are created via the loader
on first run (no Alembic migration yet — we'll add that in Week 6).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from src.config import get_settings


class Base(DeclarativeBase):
    pass


class Skill(Base):
    """One ESCO skill, filtered to CS/IT relevance."""

    __tablename__ = "skills"
    __table_args__ = (
        Index("ix_skills_pillar", "cs_it_pillar"),
        Index("ix_skills_source", "source"),
        {"schema": "ontology"},
    )

    # ESCO URI is the stable ID. e.g. http://data.europa.eu/esco/skill/abc-123
    uri: Mapped[str] = mapped_column(String(512), primary_key=True)
    preferred_label: Mapped[str] = mapped_column(String(512), nullable=False)
    alt_labels: Mapped[str | None] = mapped_column(Text)   # newline-separated
    description: Mapped[str | None] = mapped_column(Text)

    # ESCO classification
    skill_type: Mapped[str | None] = mapped_column(String(64))      # "skill/competence", "knowledge"
    reuse_level: Mapped[str | None] = mapped_column(String(64))     # "cross-sector", "occupation-specific"
    is_digital: Mapped[bool] = mapped_column(default=False)          # ESCO's digital skill tag

    # Our classification
    cs_it_pillar: Mapped[str | None] = mapped_column(String(64))     # one of the 12 CS/IT pillars
    classification_method: Mapped[str | None] = mapped_column(String(32))  # "keyword" | "llm" | "manual"
    classification_confidence: Mapped[float | None] = mapped_column(Float)

    source: Mapped[str] = mapped_column(String(32), default="esco_v1_1_1")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relations (navigable both ways for convenience)
    outgoing: Mapped[list["SkillRelation"]] = relationship(
        foreign_keys="SkillRelation.source_uri",
        back_populates="source_skill",
    )
    incoming: Mapped[list["SkillRelation"]] = relationship(
        foreign_keys="SkillRelation.target_uri",
        back_populates="target_skill",
    )

    def __repr__(self) -> str:
        return f"<Skill {self.preferred_label} pillar={self.cs_it_pillar}>"


class SkillRelation(Base):
    """Skill-to-skill relationship from ESCO.

    Types: 'essential_for', 'optional_for', 'broader', 'narrower', 'related'.
    """

    __tablename__ = "skill_relations"
    __table_args__ = (
        Index("ix_skill_rel_source", "source_uri"),
        Index("ix_skill_rel_target", "target_uri"),
        Index("ix_skill_rel_type", "relation_type"),
        {"schema": "ontology"},
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_uri: Mapped[str] = mapped_column(
        String(512), ForeignKey("ontology.skills.uri", ondelete="CASCADE"), nullable=False
    )
    target_uri: Mapped[str] = mapped_column(
        String(512), ForeignKey("ontology.skills.uri", ondelete="CASCADE"), nullable=False
    )
    relation_type: Mapped[str] = mapped_column(String(32), nullable=False)

    source_skill: Mapped["Skill"] = relationship(
        foreign_keys=[source_uri], back_populates="outgoing"
    )
    target_skill: Mapped["Skill"] = relationship(
        foreign_keys=[target_uri], back_populates="incoming"
    )


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def get_engine():
    """Create a SQLAlchemy engine for our Postgres."""
    return create_engine(get_settings().postgres_dsn, pool_pre_ping=True, future=True)


def get_session_factory():
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def create_tables(engine=None) -> None:
    """Create all ontology tables. Idempotent."""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine, checkfirst=True)


def drop_tables(engine=None) -> None:
    """Drop all ontology tables. Use with care."""
    if engine is None:
        engine = get_engine()
    Base.metadata.drop_all(engine, checkfirst=True)