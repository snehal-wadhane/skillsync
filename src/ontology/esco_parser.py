"""Parse raw ESCO CSVs into typed Python dataclasses.

Pure parsing — no database, no embeddings. Kept separate so we can unit-test
this without a running Postgres.

This parser is case-insensitive and tries multiple known column-name
conventions:
  - Tabiya format:         ORIGINURI, PREFERREDLABEL, ALTLABELS, DESCRIPTION
  - Official ESCO format:  conceptUri, preferredLabel, altLabels, description
  - Snake case:            concept_uri, preferred_label, alt_labels, description
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from src.config import get_settings


@dataclass
class RawSkill:
    """One ESCO skill as loaded from CSV, pre-enrichment."""

    uri: str
    preferred_label: str
    alt_labels: list[str] = field(default_factory=list)
    description: str | None = None
    skill_type: str | None = None
    reuse_level: str | None = None

    @property
    def search_text(self) -> str:
        """Text used for embedding + keyword matching."""
        parts = [self.preferred_label] + self.alt_labels
        if self.description:
            first_sentence = self.description.split(".")[0].strip()
            if first_sentence:
                parts.append(first_sentence)
        return " | ".join(parts)


@dataclass
class RawRelation:
    source_uri: str
    target_uri: str
    relation_type: str


# ---------------------------------------------------------------------------
# Column name resolution
# ---------------------------------------------------------------------------
# We build a case-insensitive lookup from each row. Candidate column names
# are tried in order; first match wins.

_URI_COLS = ("originuri", "concepturi", "concept_uri", "uri", "skilluri", "skill_uri")
_LABEL_COLS = ("preferredlabel", "preferred_label", "label", "name")
_ALT_COLS = ("altlabels", "alt_labels", "alternatives", "alternate_labels")
_DESC_COLS = ("description", "definition", "desc")
_TYPE_COLS = ("skilltype", "skill_type", "concepttype", "concept_type")
_REUSE_COLS = ("reuselevel", "reuse_level", "reuse")

# Relation file columns
_REL_SRC_COLS = (
    "originalskilluri", "original_skill_uri",
    "originalconcepturi", "original_concept_uri",
    "sourceuri", "source_uri", "source", "from",
)
_REL_TGT_COLS = (
    "relatedskilluri", "related_skill_uri",
    "relatedconcepturi", "related_concept_uri",
    "targeturi", "target_uri", "target", "to",
)
_REL_TYPE_COLS = ("relationtype", "relation_type", "type", "relation")


def _ci_get(row: dict[str, str], candidates: tuple[str, ...]) -> str:
    """Case-insensitive lookup — try each candidate against the row's keys."""
    # Build a one-time lower -> actual key map
    lower_keys = {k.lower(): k for k in row.keys()}
    for c in candidates:
        actual = lower_keys.get(c.lower())
        if actual is not None:
            return (row[actual] or "").strip()
    return ""


# ---------------------------------------------------------------------------
def _skills_dir() -> Path:
    return get_settings().data_dir / "raw" / "skill_taxonomies"


def parse_skills_csv(path: Path | None = None) -> list[RawSkill]:
    """Parse an ESCO skills CSV (any common mirror format)."""
    path = path or (_skills_dir() / "skills.csv")
    if not path.exists():
        raise FileNotFoundError(f"ESCO skills file not found: {path}")

    skills: list[RawSkill] = []
    skipped_no_uri = 0
    skipped_no_label = 0

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames:
            logger.debug(f"skills.csv columns: {reader.fieldnames}")
        for row in reader:
            uri = _ci_get(row, _URI_COLS)
            label = _ci_get(row, _LABEL_COLS)
            if not uri:
                skipped_no_uri += 1
                continue
            if not label:
                skipped_no_label += 1
                continue
            alt_raw = _ci_get(row, _ALT_COLS)
            alts = [a.strip() for a in alt_raw.split("\n") if a.strip()]
            skills.append(
                RawSkill(
                    uri=uri,
                    preferred_label=label,
                    alt_labels=alts[:10],
                    description=_ci_get(row, _DESC_COLS) or None,
                    skill_type=_ci_get(row, _TYPE_COLS) or None,
                    reuse_level=_ci_get(row, _REUSE_COLS) or None,
                )
            )

    if skipped_no_uri or skipped_no_label:
        logger.warning(
            f"Skipped rows: {skipped_no_uri} missing URI, {skipped_no_label} missing label"
        )
    logger.info(f"Parsed {len(skills):,} ESCO skills from {path.name}")
    return skills


def parse_skill_relations_csv(path: Path | None = None) -> list[RawRelation]:
    """Parse ESCO skill_skill_relations CSV (any common mirror format)."""
    path = path or (_skills_dir() / "skill_skill_relations.csv")
    if not path.exists():
        logger.warning(f"Skill relations file not found: {path}, returning empty")
        return []

    rels: list[RawRelation] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames:
            logger.debug(f"skill_skill_relations.csv columns: {reader.fieldnames}")
        for row in reader:
            src = _ci_get(row, _REL_SRC_COLS)
            tgt = _ci_get(row, _REL_TGT_COLS)
            rel_type = _ci_get(row, _REL_TYPE_COLS) or "related"
            if src and tgt:
                rels.append(RawRelation(source_uri=src, target_uri=tgt, relation_type=rel_type))
    logger.info(f"Parsed {len(rels):,} skill-skill relations")
    return rels