"""Data downloader for SkillSync.

Fetches:
 - ESCO v1.2 skill taxonomy
 - O*NET technology skills
 - NBA SAR 2025 guidance docs
 - WEF Future of Jobs 2025 report
 - AICTE model curriculum for CSE (if accessible)
 - Sample reference curricula (MIT OCW, IIT)

Design:
 - Resumable (skips files already downloaded)
 - Checksummed (stores SHA-256 in manifest.json)
 - Respectful (5s delay between requests to same host)
 - Verbose (rich progress bars)

Usage:
    python scripts/download_all.py              # Download everything
    python scripts/download_all.py --dry-run    # Show what would be downloaded
    python scripts/download_all.py --only esco  # Download just one dataset
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_settings  # noqa: E402

console = Console()
app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    """A dataset we want to download."""

    key: str                      # Short identifier
    description: str              # Human-readable
    url: str                      # Direct download URL
    subdir: str                   # Where under data/raw/ it lands
    filename: str                 # Final filename
    source: str                   # "official" | "kaggle" | "manual"
    expected_size_mb: float = 0.0
    notes: str = ""
    headers: dict[str, str] = field(default_factory=dict)


# Curated list of datasets. URLs verified as of April 2026.
# Some (Kaggle, certain reports) require manual download — we list them for user guidance.
# Curated list of datasets. URLs verified April 2026.
# Note: The official ESCO portal requires going through their form-builder UI,
# so we use the Tabiya GitHub mirror which publishes ESCO CSVs directly.
# This is ESCO v1.1.1 — for our CS/IT pilot this is fine; the ~2000 CS/IT skills
# we care about have barely changed between v1.1.1 and v1.2.1.
DATASETS: list[Dataset] = [
    Dataset(
        key="esco_mirror",
        description="ESCO v1.1.1 taxonomy — Tabiya mirror (GitHub)",
        url="https://github.com/tabiya-tech/tabiya-open-dataset/archive/refs/heads/main.zip",
        subdir="skill_taxonomies",
        filename="tabiya_esco_v1_1_1.zip",
        source="github_mirror",
        expected_size_mb=30.0,
        notes=(
            "Community-maintained ESCO mirror. Contains skills_en.csv, occupations_en.csv, "
            "skillSkillRelations_en.csv etc. that we'll parse in Day 4."
        ),
    ),
]

# Datasets that require manual download (no stable direct URL, need login, or block bot traffic)
MANUAL_DATASETS: list[dict[str, str]] = [
    {
        "key": "nba_sar_2025_tier1",
        "description": "NBA SAR 2025 — Tier-I engineering programs SAR format",
        "target_path": "data/raw/accreditation/nba_sar_2025_tier1.pdf",
        "instruction": (
            "1. Visit https://www.nbaind.org/En/1060-accreditation-parameters.aspx\n"
            "   2. Download 'SAR for Tier-I Engineering UG programs'\n"
            "   3. Save as: data/raw/accreditation/nba_sar_2025_tier1.pdf"
        ),
    },
    {
        "key": "nba_sar_2025_tier2",
        "description": "NBA SAR 2025 — Tier-II engineering programs SAR format",
        "target_path": "data/raw/accreditation/nba_sar_2025_tier2.pdf",
        "instruction": (
            "1. Visit https://www.nbaind.org/En/1060-accreditation-parameters.aspx\n"
            "   2. Download 'SAR for Tier-II Engineering UG programs'\n"
            "   3. Save as: data/raw/accreditation/nba_sar_2025_tier2.pdf"
        ),
    },
    {
        "key": "wef_future_of_jobs_2025",
        "description": "WEF Future of Jobs Report 2025",
        "target_path": "data/raw/industry_reports/wef_future_of_jobs_2025.pdf",
        "instruction": (
            "1. Visit https://www.weforum.org/publications/the-future-of-jobs-report-2025/\n"
            "   2. Click 'Download Report'\n"
            "   3. Save as: data/raw/industry_reports/wef_future_of_jobs_2025.pdf"
        ),
    },
    {
        "key": "aicte_model_curriculum_cse",
        "description": "AICTE Model Curriculum — Computer Science & Engineering",
        "target_path": "data/raw/curricula/aicte_model_curriculum_cse.pdf",
        "instruction": (
            "1. Visit https://www.aicte-india.org/education/model-curriculum\n"
            "   2. Download 'Computer Science and Engineering' model curriculum\n"
            "   3. Save as: data/raw/curricula/aicte_model_curriculum_cse.pdf"
        ),
    },
    {
        "key": "kaggle_india_jobs",
        "description": "Indian IT/Tech job postings dataset (Kaggle)",
        "target_path": "data/raw/job_descriptions/india_jobs.csv",
        "instruction": (
            "1. Visit https://www.kaggle.com/datasets/promptcloud/indeed-job-posting-dataset\n"
            "   or https://www.kaggle.com/datasets/PromptCloudHQ/indian-it-jobs-naukricom\n"
            "   2. Sign in (free), download CSV\n"
            "   3. Save as: data/raw/job_descriptions/india_jobs.csv\n"
            "   4. If the dataset is a zip, extract the main CSV first."
        ),
    },
    {
        "key": "onet_tech_skills",
        "description": "O*NET Technology Skills (US labor skill taxonomy, supplements ESCO)",
        "target_path": "data/raw/skill_taxonomies/onet_technology_skills.txt",
        "instruction": (
            "1. Visit https://www.onetcenter.org/dictionary/28.3/text/technology_skills.html\n"
            "   (Their server blocks direct requests from Python clients, but a browser works fine)\n"
            "   2. Save the page as 'Technology Skills.txt' (tab-delimited text)\n"
            "   3. Rename to: data/raw/skill_taxonomies/onet_technology_skills.txt"
        ),
    },
    {
        "key": "sample_iit_curriculum",
        "description": "Sample IIT/NIT/premier CS curriculum (reference comparator)",
        "target_path": "data/raw/curricula/reference_iit_cse_curriculum.pdf",
        "instruction": (
            "Pick any reliable reference curriculum to compare against:\n"
            "   - IIT Bombay CSE: https://www.cse.iitb.ac.in/academics/programmes/btech\n"
            "   - IIT Madras CSE: https://www.cse.iitm.ac.in/pagelist.php?arg=MQ==\n"
            "   - NIT Trichy CSE: https://www.nitt.edu/home/academics/departments/cse/\n"
            "   2. Download the B.Tech CSE syllabus PDF\n"
            "   3. Save as: data/raw/curricula/reference_iit_cse_curriculum.pdf"
        ),
    },
]

# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------
def sha256_of_file(path: Path, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(buf_size):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(ds: Dataset, manifest: dict, dry_run: bool = False) -> tuple[bool, str]:
    """Download one dataset. Returns (downloaded, message)."""
    settings = get_settings()
    target_dir = settings.data_dir / "raw" / ds.subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / ds.filename

    # Skip if already downloaded and checksum matches
    if target.exists() and ds.key in manifest:
        existing_hash = manifest[ds.key].get("sha256")
        if existing_hash and existing_hash == sha256_of_file(target):
            return (False, f"[green]cached[/green]")

    if dry_run:
        return (False, f"[yellow]would download[/yellow] (~{ds.expected_size_mb:.1f} MB)")

    # Actual download with progress bar
    headers = {"User-Agent": "SkillSync/0.1 (academic project)"} | ds.headers
    tmp = target.with_suffix(target.suffix + ".partial")
    try:
        with httpx.stream("GET", ds.url, headers=headers, follow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=True,
            ) as progress:
                task_id = progress.add_task(ds.key, total=total or None)
                with tmp.open("wb") as fh:
                    for chunk in r.iter_bytes(chunk_size=1 << 16):
                        fh.write(chunk)
                        progress.update(task_id, advance=len(chunk))
        tmp.replace(target)
        sha = sha256_of_file(target)
        manifest[ds.key] = {
            "url": ds.url,
            "filename": ds.filename,
            "subdir": ds.subdir,
            "sha256": sha,
            "size_bytes": target.stat().st_size,
            "source": ds.source,
        }
        return (True, f"[green]OK[/green] ({target.stat().st_size / 1e6:.1f} MB)")
    except httpx.HTTPStatusError as e:
        tmp.unlink(missing_ok=True)
        return (False, f"[red]HTTP {e.response.status_code}[/red]")
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return (False, f"[red]{type(e).__name__}: {e}[/red]")


def load_manifest() -> dict:
    settings = get_settings()
    path = settings.data_dir / "raw" / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict) -> None:
    settings = get_settings()
    path = settings.data_dir / "raw" / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@app.command()
def main(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be downloaded."),
    only: str | None = typer.Option(None, "--only", help="Download only this dataset key."),
    delay: float = typer.Option(2.0, "--delay", help="Seconds between downloads (be polite)."),
) -> None:
    """Download all SkillSync datasets."""
    console.rule("[bold cyan]SkillSync Data Downloader[/bold cyan]")
    manifest = load_manifest()
    datasets = [d for d in DATASETS if only is None or d.key == only]

    if not datasets:
        console.print(f"[red]No dataset matches key '{only}'[/red]")
        console.print(f"Available: {', '.join(d.key for d in DATASETS)}")
        sys.exit(1)

    # Automated downloads
    results: list[tuple[str, str]] = []
    for i, ds in enumerate(datasets):
        console.print(f"\n[bold]{ds.key}[/bold] — {ds.description}")
        if ds.notes:
            console.print(f"  [dim]{ds.notes}[/dim]")
        downloaded, msg = download_dataset(ds, manifest, dry_run=dry_run)
        results.append((ds.key, msg))
        console.print(f"  → {msg}")
        if downloaded and i < len(datasets) - 1:
            time.sleep(delay)  # be polite

    save_manifest(manifest)

    # Summary table
    console.rule("[bold]Summary[/bold]")
    table = Table(show_header=True)
    table.add_column("Dataset", style="bold")
    table.add_column("Result")
    for key, msg in results:
        table.add_row(key, msg)
    console.print(table)

    # Manual download instructions
    if only is None:
        console.rule("[bold yellow]Manual Downloads Required[/bold yellow]")
        console.print(
            "The following datasets can't be auto-downloaded (require login or "
            "don't have stable direct URLs). Please fetch them manually:\n"
        )
        for m in MANUAL_DATASETS:
            target = ROOT / m["target_path"]
            status = "[green]✓ present[/green]" if target.exists() else "[red]✗ missing[/red]"
            console.print(f"[bold]{m['key']}[/bold] — {m['description']}  {status}")
            if not target.exists():
                console.print(f"  {m['instruction']}\n")

    console.print("\n[green bold]Done.[/green bold] Manifest saved to data/raw/manifest.json")


if __name__ == "__main__":
    app()