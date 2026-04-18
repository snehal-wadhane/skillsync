"""Smoke tests for configuration loading."""
from __future__ import annotations


def test_settings_load():
    from src.config import get_settings
    s = get_settings()
    assert s.data_dir.exists()
    assert s.cache_dir.exists()
    assert s.domain_focus == "CS_IT"
    # Cloud-first config
    assert s.groq_model_heavy
    assert s.groq_model_fast
    assert s.gemini_model
    assert s.llm_strategy in {"hybrid_cloud", "groq_only", "gemini_only", "offline_only"}


def test_default_keys_not_configured():
    """Verify the default placeholder values are correctly identified as unconfigured."""
    from src.config import get_settings
    s = get_settings()
    # These should be False with placeholder values
    if s.groq_api_key == "gsk_paste_your_key_here":
        assert not s.groq_configured
    if s.google_api_key == "paste_your_key_here":
        assert not s.gemini_configured


def test_nba_config_structure():
    from src.config import nba_pos
    cfg = nba_pos()
    assert cfg["version"] == "SAR-2025"
    assert cfg["total_pos"] == 11
    assert len(cfg["program_outcomes"]) == 11
    for po_code, po in cfg["program_outcomes"].items():
        assert po_code.startswith("PO")
        assert po["title"]
        assert po["description"]
        assert "bloom_expected_min" in po
    assert "program_specific_outcomes_cs_it" in cfg


def test_blooms_config_structure():
    from src.config import blooms_taxonomy
    cfg = blooms_taxonomy()
    assert len(cfg["levels"]) == 6
    expected = ["1_remember", "2_understand", "3_apply", "4_analyze", "5_evaluate", "6_create"]
    assert list(cfg["levels"].keys()) == expected
    for key, level in cfg["levels"].items():
        assert level["label"]
        assert len(level["verbs"]) > 5
    assert cfg["engineering_hints"]["debug"] == "4_analyze"


def test_cs_it_pillars():
    from src.config import cs_it_pillars
    cfg = cs_it_pillars()
    assert cfg["domain"] == "CS_IT"
    assert len(cfg["pillars"]) >= 10
    for pillar_id, p in cfg["pillars"].items():
        assert p["label"]
        assert p["stability"] in {"stable", "evolving", "volatile"}
        assert len(p["example_skills"]) >= 3
        assert all(po.startswith("PO") for po in p["nba_pos_primary"])
    assert cfg["stability_weights"]["stable"] > cfg["stability_weights"]["volatile"]


def test_pillar_po_references_are_valid():
    """Every PO referenced in pillars must exist in NBA config."""
    from src.config import cs_it_pillars, nba_pos
    nba = nba_pos()
    valid_pos = set(nba["program_outcomes"].keys())
    pillars = cs_it_pillars()
    for pillar_id, p in pillars["pillars"].items():
        for po in p["nba_pos_primary"]:
            assert po in valid_pos, f"Pillar {pillar_id} references unknown {po}"
