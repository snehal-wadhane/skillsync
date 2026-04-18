"""Verify the ontology load worked — check Qdrant, Postgres, and semantic search.

Run: python scripts/verify_ontology.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

console = Console()


def check_qdrant_collection():
    from qdrant_client import QdrantClient
    from src.config import get_settings
    s = get_settings()
    client = QdrantClient(host=s.qdrant_host, port=s.qdrant_port, timeout=10)
    info = client.get_collection("skills")
    return info.points_count, info.config.params.vectors.size


def check_postgres_counts():
    from sqlalchemy import text
    from src.ontology.schema import get_engine
    engine = get_engine()
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM ontology.skills")).scalar()
        by_pillar = conn.execute(text(
            "SELECT cs_it_pillar, COUNT(*) FROM ontology.skills "
            "GROUP BY cs_it_pillar ORDER BY 2 DESC"
        )).all()
    return total, by_pillar


def semantic_search_demo(queries: list[str]) -> None:
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    from src.config import get_settings

    s = get_settings()
    console.print("[dim]Loading embedder...[/dim]")
    embedder = SentenceTransformer(s.embed_model, device="cpu")
    client = QdrantClient(host=s.qdrant_host, port=s.qdrant_port, timeout=10)

    for query in queries:
        console.rule(f"[bold cyan]Query: {query!r}[/bold cyan]")
        q_vec = embedder.encode(query, normalize_embeddings=True).tolist()
        results = client.query_points(
            collection_name="skills",
            query=q_vec,
            limit=5,
        ).points

        table = Table(show_header=True, header_style="bold")
        table.add_column("Score", style="dim", width=6)
        table.add_column("Skill")
        table.add_column("Pillar", style="cyan")
        table.add_column("Method", style="dim")
        for p in results:
            table.add_row(
                f"{p.score:.3f}",
                p.payload.get("label", "?"),
                p.payload.get("pillar", "?"),
                p.payload.get("method", "?"),
            )
        console.print(table)


def main() -> None:
    console.rule("[bold green]SkillSync Ontology Verification[/bold green]")

    # Qdrant check
    try:
        points, dim = check_qdrant_collection()
        console.print(f"[green]✓ Qdrant 'skills' collection:[/green] {points:,} points, dim={dim}")
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]✗ Qdrant check failed: {e}[/red]")
        return

    # Postgres check
    try:
        total, by_pillar = check_postgres_counts()
        console.print(f"[green]✓ Postgres ontology.skills:[/green] {total:,} rows")
        pt = Table(title="Skills per pillar", show_header=True, header_style="bold")
        pt.add_column("Pillar", style="cyan")
        pt.add_column("Count", justify="right")
        for pillar, count in by_pillar:
            pt.add_row(pillar, f"{count:,}")
        console.print(pt)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]✗ Postgres check failed: {e}[/red]")
        return

    # Semantic search demo — 5 classic CS/IT queries
    console.rule("[bold cyan]Semantic search smoke tests[/bold cyan]")
    semantic_search_demo([
        "container orchestration",
        "machine learning for images",
        "secure web application development",
        "manage relational database",
        "write automated tests",
    ])


if __name__ == "__main__":
    main()