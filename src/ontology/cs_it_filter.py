"""Filter ESCO's ~13k skills down to CS/IT-relevant ones (~2k).

Strategy — cheap first, expensive as fallback:
  1. KEYWORD MATCH: skill label contains any example skill from a pillar  (fast)
  2. SUBSTRING MATCH against a curated CS/IT vocabulary                   (fast)
  3. LLM CLASSIFICATION for borderline candidates                         (slow, costs Groq tokens)

Skills classified into one of 12 pillars from configs/cs_it_pillars.yaml.
Skills that don't match any pillar are dropped (~11k skills excluded).

Why keyword-first: ~90% of CS/IT skills contain obvious keywords (Python,
Docker, SQL). Only ambiguous cases ("system administration", "data
processing") need LLM classification, costing ~200-300 Groq requests total.
"""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from src.config import cs_it_pillars
from src.llm.router import LLMRouter, TaskType
from src.ontology.esco_parser import RawSkill


@dataclass
class ClassifiedSkill:
    """A skill assigned to a CS/IT pillar."""

    raw: RawSkill
    pillar: str
    method: str          # "keyword" | "vocabulary" | "llm"
    confidence: float    # 0.0 - 1.0


# Curated seed vocabulary per pillar — extends the `example_skills` from YAML.
# Keep lowercase; matching is case-insensitive.
_EXTRA_VOCAB: dict[str, set[str]] = {
    "cs_foundations": {
        "algorithm", "data structure", "complexity", "discrete math", "graph theory",
        "computational thinking", "automata", "turing", "big-o",
    },
    "programming_fundamentals": {
        "python", "java", "c++", "c#", "javascript", "typescript", "golang", "rust",
        "kotlin", "scala", "ruby", "object-oriented", "functional programming",
        "debugging", "code review", "design patterns", "programming language",
    },
    "systems_architecture": {
        "operating system", "linux", "unix", "windows server", "network", "tcp/ip",
        "compiler", "computer architecture", "assembly", "memory management",
        "process scheduling", "concurrency", "distributed system",
    },
    "databases": {
        "sql", "database", "postgres", "postgresql", "mysql", "oracle", "mongodb",
        "redis", "cassandra", "dynamodb", "data modeling", "indexing",
        "query optimization", "vector database", "nosql", "rdbms",
    },
    "software_engineering": {
        "software engineering", "git", "agile", "scrum", "kanban", "unit test",
        "integration test", "test-driven", "ci/cd", "code review", "refactoring",
        "software design", "uml", "software architecture", "devops",
    },
    "web_development": {
        "html", "css", "react", "angular", "vue", "svelte", "nextjs", "node.js",
        "nodejs", "express", "rest api", "graphql", "fastapi", "django", "flask",
        "spring boot", "webpack", "frontend", "backend", "full-stack", "full stack",
    },
    "mobile_development": {
        "android", "ios", "kotlin", "swift", "objective-c", "flutter",
        "react native", "xamarin", "mobile app",
    },
    "ai_ml": {
        "machine learning", "deep learning", "neural network", "pytorch", "tensorflow",
        "scikit", "sklearn", "transformers", "llm", "large language model",
        "natural language processing", "nlp", "computer vision", "reinforcement",
        "supervised learning", "unsupervised", "mlops", "data science",
        "artificial intelligence", "generative ai", "rag ", "prompt engineering",
    },
    "data_engineering": {
        "etl", "elt", "apache spark", "pyspark", "airflow", "kafka", "pandas",
        "data warehouse", "data lake", "data pipeline", "bigquery", "snowflake",
        "tableau", "power bi", "data analytics", "business intelligence",
        "data visualization",
    },
    "cloud_devops": {
        "aws", "amazon web services", "azure", "google cloud", "gcp", "docker",
        "kubernetes", "terraform", "ansible", "jenkins", "gitlab", "github actions",
        "helm", "istio", "prometheus", "grafana", "cloud computing",
        "infrastructure as code", "site reliability",
    },
    "cybersecurity": {
        "cybersecurity", "cyber security", "information security", "cryptography",
        "penetration test", "pen test", "ethical hacking", "owasp", "vulnerability",
        "firewall", "intrusion detection", "siem", "zero trust", "malware",
        "encryption", "authentication", "oauth",
    },
    "professional_skills": {
        "technical writing", "technical documentation", "presentation",
        "communication", "teamwork", "leadership", "project management",
        "stakeholder", "collaboration", "problem solving", "critical thinking",
    },
}


def _build_pillar_keywords() -> dict[str, set[str]]:
    """Combine example_skills from YAML with our extra curated vocab, all lowercase."""
    pillars = cs_it_pillars()["pillars"]
    result: dict[str, set[str]] = {}
    for pid, pdata in pillars.items():
        kws = {s.replace("_", " ").lower() for s in pdata["example_skills"]}
        kws.update(_EXTRA_VOCAB.get(pid, set()))
        # Also include the pillar's focus terms from description
        result[pid] = kws
    return result


# Skills to EXCLUDE even if they match — too generic / unrelated to CS curriculum
_BLACKLIST_SUBSTRINGS = {
    "veterinary", "nursing", "midwif", "medicine", "dental", "agricult", "fishing",
    "mining", "construction", "welding", "plumb", "carpentr", "cookery", "culinary",
    "hairdress", "cosmet", "tailor", "textile", "leather", "wood", "metal casting",
}


def _is_blacklisted(label: str, description: str | None) -> bool:
    text = (label + " " + (description or "")).lower()
    return any(bad in text for bad in _BLACKLIST_SUBSTRINGS)


def _keyword_match(skill: RawSkill, pillar_kws: dict[str, set[str]]) -> tuple[str, float] | None:
    """Return (pillar, confidence) if skill matches any pillar's keywords."""
    search_text = (skill.preferred_label + " " + " ".join(skill.alt_labels)).lower()
    best_pillar = None
    best_score = 0.0
    for pillar, kws in pillar_kws.items():
        # Count matches; longer keywords give higher confidence
        score = 0.0
        for kw in kws:
            if kw in search_text:
                # Longer multi-word keywords are more specific
                score += 1.0 + 0.3 * (kw.count(" "))
        if score > best_score:
            best_score = score
            best_pillar = pillar
    if best_pillar and best_score >= 1.0:
        # Normalize confidence to [0, 1]; 3+ matches or long multi-word = high confidence
        confidence = min(1.0, 0.5 + best_score / 6.0)
        return (best_pillar, confidence)
    return None


# ---------------------------------------------------------------------------
def classify_skills(
    skills: list[RawSkill],
    use_llm_fallback: bool = True,
    llm_budget: int = 300,
) -> list[ClassifiedSkill]:
    """Classify each skill into a CS/IT pillar, or drop it.

    Args:
        skills: Raw ESCO skills
        use_llm_fallback: Use Groq for borderline cases (costs tokens)
        llm_budget: Max LLM calls to make (respect free tier limits)
    """
    pillar_kws = _build_pillar_keywords()
    classified: list[ClassifiedSkill] = []
    llm_candidates: list[RawSkill] = []
    llm_calls = 0

    for skill in skills:
        if _is_blacklisted(skill.preferred_label, skill.description):
            continue
        match = _keyword_match(skill, pillar_kws)
        if match:
            pillar, conf = match
            method = "keyword" if conf >= 0.7 else "vocabulary"
            classified.append(ClassifiedSkill(raw=skill, pillar=pillar, method=method, confidence=conf))
        elif use_llm_fallback and _is_potentially_digital(skill):
            # Queue for LLM classification
            llm_candidates.append(skill)

    logger.info(
        f"Keyword classification: {len(classified):,} skills placed. "
        f"{len(llm_candidates):,} candidates queued for LLM."
    )

    # LLM classification pass (bounded)
    if use_llm_fallback and llm_candidates:
        llm_candidates = llm_candidates[:llm_budget]
        router = LLMRouter()
        pillar_ids = list(pillar_kws.keys()) + ["NONE"]
        for skill in llm_candidates:
            result = _llm_classify_one(skill, pillar_ids, router)
            llm_calls += 1
            if result and result != "NONE":
                classified.append(ClassifiedSkill(raw=skill, pillar=result, method="llm", confidence=0.6))

    logger.info(
        f"Final classification: {len(classified):,} CS/IT skills kept "
        f"from {len(skills):,} total ESCO skills. LLM calls: {llm_calls}"
    )
    return classified


def _is_potentially_digital(skill: RawSkill) -> bool:
    """Cheap heuristic — is this skill worth spending an LLM call on?"""
    text = (skill.preferred_label + " " + (skill.description or "")).lower()
    digital_hints = {
        "software", "computer", "system", "digital", "information technology",
        "program", "algorithm", "data", "network", "security", "web", "cloud",
        "application", "interface", "server", "technical", "engineering",
    }
    return any(h in text for h in digital_hints)


def _llm_classify_one(
    skill: RawSkill,
    pillar_ids: list[str],
    router: LLMRouter,
) -> str | None:
    """Ask the LLM to pick one of the pillar IDs for this skill."""
    pillars_str = "\n".join(f"- {p}" for p in pillar_ids)
    prompt = (
        f"Classify this skill into one of the following CS/IT pillars, or NONE if not CS/IT:\n\n"
        f"{pillars_str}\n\n"
        f"Skill: {skill.preferred_label}\n"
        f"Description: {skill.description or '(none)'}\n\n"
        f"Reply with just the pillar ID (e.g. 'ai_ml') or 'NONE'. Nothing else."
    )
    try:
        reply = router.complete(
            task=TaskType.CLASSIFICATION,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
        ).strip().lower()
        # Extract the pillar name from reply
        for pid in pillar_ids:
            if pid.lower() in reply:
                return pid
        return "NONE"
    except Exception as e:  # noqa: BLE001
        logger.warning(f"LLM classification failed for {skill.preferred_label}: {e}")
        return None