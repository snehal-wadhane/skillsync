"""Orchestrate the full ontology load:
   parse CSVs -> classify -> embed -> write to Postgres + Qdrant.

Run:
   python -m src.ontology.loader                    # keyword-only, ~5 min
   python -m src.ontology.loader --with-llm         # + LLM borderline cases, ~15 min
   python -m src.ontology.loader --llm-budget=100   # cap LLM calls
"""
from __future__ import annotations

import sys
import time

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sqlalchemy import text

from src.config import get_settings
from src.ontology.cs_it_filter import ClassifiedSkill, classify_skills
from src.ontology.esco_parser import parse_skill_relations_csv, parse_skills_csv
from src.ontology.schema import (
    Skill,
    SkillRelation,
    create_tables,
    get_engine,
    get_session_factory,
)

console = Console()

QDRANT_COLLECTION = "skills"


def _get_embedder():
    """Load BGE-small. First call downloads ~133 MB."""
    from sentence_transformers import SentenceTransformer
    settings = get_settings()
    console.print(f"[dim]Loading embedder: {settings.embed_model} (first run downloads ~133 MB)...[/dim]")
    return SentenceTransformer(settings.embed_model, device="cpu")


def _ensure_qdrant_collection(client: QdrantClient, vector_size: int, recreate: bool = True) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        if recreate:
            logger.info(f"Recreating Qdrant collection '{QDRANT_COLLECTION}'")
            client.delete_collection(QDRANT_COLLECTION)
        else:
            logger.info(f"Qdrant collection '{QDRANT_COLLECTION}' exists, will upsert")
            return
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    logger.info(f"Created Qdrant collection '{QDRANT_COLLECTION}' (dim={vector_size})")


def _write_postgres(classified: list[ClassifiedSkill], relations: list) -> None:
    """Dual-transaction: skills first, then relations (FK dependency)."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS ontology"))
    create_tables(engine)

    Session = get_session_factory()
    kept_uris = {c.raw.uri for c in classified}

    with Session() as s:
        s.execute(text("TRUNCATE TABLE ontology.skill_relations CASCADE"))
        s.execute(text("TRUNCATE TABLE ontology.skills CASCADE"))
        s.commit()

        console.print(f"[cyan]Writing {len(classified):,} skills to Postgres...[/cyan]")
        batch = []
        for c in classified:
            batch.append(Skill(
                uri=c.raw.uri,
                preferred_label=c.raw.preferred_label,
                alt_labels="\n".join(c.raw.alt_labels) if c.raw.alt_labels else None,
                description=c.raw.description,
                skill_type=c.raw.skill_type,
                reuse_level=c.raw.reuse_level,
                cs_it_pillar=c.pillar,
                classification_method=c.method,
                classification_confidence=c.confidence,
                source="esco_v1_1_1",
            ))
            if len(batch) >= 500:
                s.add_all(batch); s.commit(); batch = []
        if batch:
            s.add_all(batch); s.commit()

        kept_rels = [r for r in relations if r.source_uri in kept_uris and r.target_uri in kept_uris]
        if kept_rels:
            console.print(f"[cyan]Writing {len(kept_rels):,} skill-skill relations...[/cyan]")
            rel_batch = []
            for r in kept_rels:
                rel_batch.append(SkillRelation(
                    source_uri=r.source_uri,
                    target_uri=r.target_uri,
                    relation_type=r.relation_type,
                ))
                if len(rel_batch) >= 1000:
                    s.add_all(rel_batch); s.commit(); rel_batch = []
            if rel_batch:
                s.add_all(rel_batch); s.commit()
        else:
            console.print("[yellow]No matching skill-skill relations — skipping relation load[/yellow]")


def _write_qdrant(classified: list[ClassifiedSkill], embedder) -> None:
    """Embed skills and upsert to Qdrant with a visible progress bar."""
    settings = get_settings()
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=60)

    sample_vec = embedder.encode(["probe"], show_progress_bar=False)
    vector_size = sample_vec.shape[1]
    _ensure_qdrant_collection(client, vector_size, recreate=True)

    batch_size = 32
    total = len(classified)

    t0 = time.time()
    with Progress(
        TextColumn("[cyan]Embedding skills[/cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("embed", total=total)
        for i in range(0, total, batch_size):
            chunk = classified[i : i + batch_size]
            texts = [c.raw.search_text for c in chunk]
            vectors = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            points = [
                PointStruct(
                    id=abs(hash(c.raw.uri)) % (10**15),
                    vector=vectors[j].tolist(),
                    payload={
                        "uri": c.raw.uri,
                        "label": c.raw.preferred_label,
                        "pillar": c.pillar,
                        "method": c.method,
                        "confidence": c.confidence,
                        "alt_labels": c.raw.alt_labels[:5],
                    },
                )
                for j, c in enumerate(chunk)
            ]
            client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=False)
            progress.update(task, advance=len(chunk))

    elapsed = time.time() - t0
    console.print(f"[green]Qdrant upsert complete in {elapsed:.1f}s ({total/elapsed:.0f} skills/sec)[/green]")


# ---------------------------------------------------------------------------
def main(use_llm: bool = False, llm_budget: int = 300) -> None:
    console.rule("[bold cyan]SkillSync Ontology Loader[/bold cyan]")

    # 1. Parse
    console.print("[cyan]Step 1/4: Parsing ESCO skills.csv...[/cyan]")
    skills = parse_skills_csv()
    console.print(f"  Parsed [bold]{len(skills):,}[/bold] skills")

    console.print("[cyan]Step 2/4: Parsing skill-skill relations...[/cyan]")
    relations = parse_skill_relations_csv()
    if len(relations) == 0:
        console.print("  [yellow]No relations loaded (Tabiya internal IDs don't match URIs — non-blocking)[/yellow]")
    else:
        console.print(f"  Parsed [bold]{len(relations):,}[/bold] relations")

    # 2. Classify
    mode = "keyword + LLM" if use_llm else "keyword-only"
    console.print(f"[cyan]Step 3/4: Classifying skills ({mode})...[/cyan]")
    t0 = time.time()
    classified = classify_skills(skills, use_llm_fallback=use_llm, llm_budget=llm_budget)
    console.print(f"  Classified [bold]{len(classified):,}[/bold] CS/IT skills in {time.time()-t0:.1f}s")

    # 3. Load embedder (BGE-small), then write Postgres + Qdrant
    console.print("[cyan]Step 4/4: Loading embedder, writing to Postgres + Qdrant...[/cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("Loading BGE-small (first run downloads ~133 MB)..."),
        console=console,
        transient=True,
    ) as p:
        p.add_task("load", total=None)
        embedder = _get_embedder()

    _write_postgres(classified, relations)
    _write_qdrant(classified, embedder)

    # Summary
    from collections import Counter
    by_pillar = Counter(c.pillar for c in classified)
    by_method = Counter(c.method for c in classified)
    console.rule("[bold green]Load complete[/bold green]")
    console.print("\n[bold]Skills per pillar:[/bold]")
    for pillar, count in sorted(by_pillar.items(), key=lambda x: -x[1]):
        console.print(f"  {pillar:<30} {count:>4}")
    console.print("\n[bold]Classification method:[/bold]")
    for method, count in sorted(by_method.items(), key=lambda x: -x[1]):
        console.print(f"  {method:<30} {count:>4}")
    console.print(f"\n[bold]Total: {len(classified):,} CS/IT skills loaded[/bold]")


if __name__ == "__main__":
    use_llm = "--with-llm" in sys.argv
    budget = 300
    for arg in sys.argv[1:]:
        if arg.startswith("--llm-budget="):
            budget = int(arg.split("=")[1])
    main(use_llm=use_llm, llm_budget=budget)