"""Smoke tests for the ontology module. No real DB / Qdrant needed for most."""
from __future__ import annotations

import pytest


def test_parse_skills_csv():
    """ESCO skills file parses and has reasonable count."""
    from src.ontology.esco_parser import parse_skills_csv
    skills = parse_skills_csv()
    assert len(skills) > 10_000, f"Expected >10k ESCO skills, got {len(skills)}"
    assert len(skills) < 20_000, f"Suspiciously many skills: {len(skills)}"
    # Every skill has a URI and label
    for s in skills[:100]:
        assert s.uri.startswith("http"), f"Bad URI: {s.uri}"
        assert s.preferred_label


def test_cs_it_filter_keyword_coverage():
    """Keyword filter should classify at least 800 skills without LLM."""
    from src.ontology.cs_it_filter import classify_skills
    from src.ontology.esco_parser import parse_skills_csv
    skills = parse_skills_csv()
    classified = classify_skills(skills, use_llm_fallback=False)
    assert len(classified) >= 800, f"Too few CS/IT skills: {len(classified)}"
    assert len(classified) <= 5000, f"Too permissive filter: {len(classified)}"
    # Every classified skill has a valid pillar
    from src.config import cs_it_pillars
    valid_pillars = set(cs_it_pillars()["pillars"].keys())
    for c in classified[:200]:
        assert c.pillar in valid_pillars, f"Unknown pillar {c.pillar}"


def test_pillar_distribution_is_balanced():
    """No single pillar should dominate (>80% of classified skills)."""
    from collections import Counter
    from src.ontology.cs_it_filter import classify_skills
    from src.ontology.esco_parser import parse_skills_csv
    classified = classify_skills(parse_skills_csv(), use_llm_fallback=False)
    by_pillar = Counter(c.pillar for c in classified)
    total = sum(by_pillar.values())
    for pillar, count in by_pillar.items():
        pct = count / total
        assert pct < 0.8, f"Pillar {pillar} too dominant at {pct:.0%}"


@pytest.mark.slow
def test_famous_skills_classified_correctly():
    """Spot-check: well-known skills should land in expected pillars."""
    from src.ontology.cs_it_filter import classify_skills
    from src.ontology.esco_parser import parse_skills_csv
    classified = classify_skills(parse_skills_csv(), use_llm_fallback=False)
    by_label = {c.raw.preferred_label.lower(): c for c in classified}

    expectations = [
        ("python", "programming_fundamentals"),
        ("javascript", {"programming_fundamentals", "web_development"}),
        ("docker", "cloud_devops"),
        ("sql", "databases"),
    ]
    for label, expected in expectations:
        # Find any skill whose label contains the keyword
        match = next((c for lbl, c in by_label.items() if label in lbl), None)
        if match is None:
            pytest.skip(f"No ESCO skill matching '{label}' found")
        if isinstance(expected, set):
            assert match.pillar in expected, f"{label} -> {match.pillar}, expected one of {expected}"
        else:
            assert match.pillar == expected, f"{label} -> {match.pillar}, expected {expected}"