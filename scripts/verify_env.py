"""
Week 1 Day 1 verification script — Hybrid Cloud + Docker stack.

Checks:
 1. Python version (3.11+)
 2. PyTorch installed (CPU is fine for this hardware)
 3. Data directories present
 4. Config files load (NBA + Blooms + CS/IT pillars)
 5. Groq API reachable (primary cloud LLM)
 6. Google AI Studio / Gemini reachable (secondary, optional if not configured)
 7. Qdrant reachable (Docker)
 8. PostgreSQL reachable (Docker)
 9. Redis reachable (Docker)
10. MinIO reachable (Docker)
11. Langfuse reachable (Docker, optional)
12. Ollama (optional — offline fallback, skip if not installed)

Run:  python scripts/verify_env.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

console = Console()


def check(name: str, fn, optional: bool = False) -> tuple[str, str, str]:
    """Run a check and return (name, status, detail)."""
    try:
        detail = fn()
        return (name, "[green]✓ OK[/green]", detail or "")
    except Exception as e:  # noqa: BLE001
        status = "[yellow]⚠ MISSING[/yellow]" if optional else "[red]✗ FAIL[/red]"
        return (name, status, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
def check_python():
    v = sys.version_info
    if v < (3, 11):
        raise RuntimeError(f"Python 3.11+ required, got {v.major}.{v.minor}")
    return f"Python {v.major}.{v.minor}.{v.micro}"


def check_torch():
    import torch
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    return f"PyTorch {torch.__version__} ({device} mode — expected for this hardware)"


def check_directories():
    from src.config import get_settings
    s = get_settings()
    required = [
        s.data_dir / "raw" / "curricula",
        s.data_dir / "raw" / "job_descriptions",
        s.data_dir / "raw" / "accreditation",
        s.data_dir / "raw" / "skill_taxonomies",
        s.data_dir / "processed",
        s.cache_dir,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing dirs: {missing}")
    return f"{len(required)} data directories present"


def check_configs():
    from src.config import blooms_taxonomy, cs_it_pillars, nba_pos
    nba = nba_pos()
    blooms = blooms_taxonomy()
    pillars = cs_it_pillars()
    assert len(nba["program_outcomes"]) == 11
    assert len(blooms["levels"]) == 6
    assert len(pillars["pillars"]) >= 10
    return (
        f"NBA v{nba['version']} ({len(nba['program_outcomes'])} POs), "
        f"Bloom's {len(blooms['levels'])} levels, "
        f"CS/IT {len(pillars['pillars'])} pillars"
    )


def check_groq():
    """Primary LLM — required."""
    from src.config import get_settings
    s = get_settings()
    if not s.groq_configured:
        raise RuntimeError(
            "GROQ_API_KEY not set. Sign up at https://console.groq.com/ (free, no credit card)"
        )
    # Lightweight ping — 1 token completion
    from groq import Groq
    client = Groq(api_key=s.groq_api_key)
    resp = client.chat.completions.create(
        model=s.groq_model_fast,
        messages=[{"role": "user", "content": "Reply with only: pong"}],
        max_tokens=5,
        temperature=0,
    )
    reply = resp.choices[0].message.content.strip()
    return f"Groq OK ({s.groq_model_fast}): got '{reply}'"


def check_gemini():
    """Secondary LLM — optional. Warns if not configured."""
    from src.config import get_settings
    s = get_settings()
    if not s.gemini_configured:
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Optional but recommended — https://aistudio.google.com/apikey"
        )
    import google.generativeai as genai
    genai.configure(api_key=s.google_api_key)
    model = genai.GenerativeModel(s.gemini_model)
    resp = model.generate_content("Reply with only: pong")
    return f"Gemini OK ({s.gemini_model}): got '{resp.text.strip()[:10]}'"


def check_qdrant():
    from qdrant_client import QdrantClient
    from src.config import get_settings
    s = get_settings()
    client = QdrantClient(host=s.qdrant_host, port=s.qdrant_port, timeout=5)
    collections = client.get_collections()
    return f"Qdrant up, {len(collections.collections)} collections"


def check_postgres():
    import psycopg2
    from src.config import get_settings
    s = get_settings()
    conn = psycopg2.connect(
        host=s.postgres_host, port=s.postgres_port, dbname=s.postgres_db,
        user=s.postgres_user, password=s.postgres_password, connect_timeout=5,
    )
    with conn.cursor() as cur:
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        cur.execute("SELECT value FROM public.system_info WHERE key='domain_focus';")
        row = cur.fetchone()
        domain = row[0] if row else "unknown"
    conn.close()
    return f"{version.split(',')[0]} | domain={domain}"


def check_redis():
    import redis
    from src.config import get_settings
    s = get_settings()
    r = redis.Redis(host=s.redis_host, port=s.redis_port, socket_connect_timeout=5)
    if not r.ping():
        raise RuntimeError("Redis ping returned False")
    info = r.info(section="server")
    return f"Redis {info.get('redis_version', '?')}"


def check_minio():
    from minio import Minio
    from src.config import get_settings
    s = get_settings()
    client = Minio(
        s.minio_endpoint,
        access_key=s.minio_root_user,
        secret_key=s.minio_root_password,
        secure=False,
    )
    for bucket in (s.minio_bucket_raw, s.minio_bucket_reports):
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
    return f"MinIO up, buckets: {s.minio_bucket_raw}, {s.minio_bucket_reports}"


def check_langfuse():
    import httpx
    from src.config import get_settings
    s = get_settings()
    r = httpx.get(f"{s.langfuse_host}/api/public/health", timeout=3.0)
    r.raise_for_status()
    return f"Langfuse up at {s.langfuse_host}"


def check_ollama():
    """Optional — only needed for offline demos."""
    import httpx
    from src.config import get_settings
    s = get_settings()
    r = httpx.get(f"{s.ollama_host}/api/tags", timeout=3.0)
    r.raise_for_status()
    models = [m["name"] for m in r.json().get("models", [])]
    if not models:
        raise RuntimeError(f"Ollama up but no models. Run: ollama pull {s.ollama_model}")
    return f"Ollama up, {len(models)} models available"


def check_embeddings():
    from src.config import get_settings
    s = get_settings()
    return f"Will use {s.embed_model} (~133 MB download, CPU-friendly)"


# ---------------------------------------------------------------------------
def main():
    console.rule("[bold cyan]SkillSync — Environment Verification (Hybrid Cloud)[/bold cyan]")

    required_checks = [
        ("Python version", check_python),
        ("PyTorch", check_torch),
        ("Data directories", check_directories),
        ("Configs (NBA + Blooms + CS/IT pillars)", check_configs),
        ("Groq API (primary cloud LLM)", check_groq),
        ("Qdrant (docker)", check_qdrant),
        ("PostgreSQL (docker)", check_postgres),
        ("Redis (docker)", check_redis),
        ("MinIO (docker)", check_minio),
        ("Embeddings model", check_embeddings),
    ]
    optional_checks = [
        ("Gemini API (secondary, recommended)", check_gemini),
        ("Langfuse (observability, optional)", check_langfuse),
        ("Ollama (offline demo fallback, optional)", check_ollama),
    ]

    results = [check(n, fn) for n, fn in required_checks]
    results += [check(n, fn, optional=True) for n, fn in optional_checks]

    table = Table(show_lines=False, expand=True)
    table.add_column("Check", style="bold", width=42)
    table.add_column("Status", width=14)
    table.add_column("Detail")
    for row in results:
        table.add_row(*row)
    console.print(table)

    fails = [r for r in results if "FAIL" in r[1]]
    if fails:
        console.print(f"\n[red]{len(fails)} required check(s) failed.[/red] See docs/WEEK1_SETUP.md")
        sys.exit(1)
    console.print("\n[green bold]All required checks passed — ready for Week 1 ingestion.[/green bold]")


if __name__ == "__main__":
    main()
