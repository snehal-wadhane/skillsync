from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_settings  # noqa: E402

console = Console()

DATE_CANDIDATES = {
    "posted_date", "date_posted", "created_at", "created_date", "post_date",
    "date", "job_posted_date", "listed_date", "crawled_date", "scraped_date",
    "Date Posted", "Posted Date", "Date", "Created At",
}

SIZE_THRESHOLD_MB = 500  # don't fully load files bigger than this


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def inspect_csv(path: Path) -> None:
    size_mb = path.stat().st_size / 1e6
    display_name = f"{path.parent.name}/{path.name}" if path.parent.name else path.name
    console.rule(f"[bold cyan]{display_name}[/bold cyan]")
    console.print(f"File size: {human_bytes(path.stat().st_size)}")

    # For huge files, sample instead of loading all
    nrows = None
    sampled = False
    if size_mb > SIZE_THRESHOLD_MB:
        nrows = 10_000
        sampled = True
        console.print(f"[yellow]File > {SIZE_THRESHOLD_MB} MB — loading first {nrows:,} rows only[/yellow]")

    # Try multiple encodings; Kaggle files often aren't UTF-8
    df = None
    last_err = None
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, nrows=nrows, encoding=enc, on_bad_lines="skip", low_memory=False)
            console.print(f"Encoding: {enc}")
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    if df is None:
        console.print(f"[red]Failed to read: {last_err}[/red]")
        return

    console.print(f"Shape: [bold]{df.shape[0]:,} rows × {df.shape[1]} columns[/bold]{' (sampled)' if sampled else ''}")

    # Column report
    col_table = Table(title="Columns", show_header=True, header_style="bold")
    col_table.add_column("#", style="dim")
    col_table.add_column("Name")
    col_table.add_column("Dtype")
    col_table.add_column("Non-null %")
    col_table.add_column("Sample", overflow="fold", max_width=50)
    for i, col in enumerate(df.columns):
        non_null_pct = df[col].notna().mean() * 100
        sample_val = df[col].dropna().head(1)
        sample_str = str(sample_val.iloc[0])[:60] if len(sample_val) > 0 else ""
        col_table.add_row(
            str(i + 1),
            col,
            str(df[col].dtype),
            f"{non_null_pct:.0f}%",
            sample_str,
        )
    console.print(col_table)

    # Try to find a date column
    date_col = None
    for col in df.columns:
        if col in DATE_CANDIDATES or col.lower() in {c.lower() for c in DATE_CANDIDATES}:
            date_col = col
            break
    if date_col:
        try:
            parsed = pd.to_datetime(df[date_col], errors="coerce", utc=False)
            valid = parsed.dropna()
            if len(valid) > 0:
                console.print(f"Date column: [bold]{date_col}[/bold]")
                console.print(
                    f"  Range: {valid.min()} → {valid.max()}  "
                    f"({(valid.max() - valid.min()).days} days, {len(valid):,} valid dates)"
                )
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]Could not parse dates in {date_col}: {e}[/yellow]")
    else:
        console.print("[yellow]No recognizable date column found[/yellow]")

    # Show 3 sample rows
    console.print("\n[bold]Sample rows:[/bold]")
    sample_cols = df.columns[:6]  # limit columns for readability
    console.print(df[sample_cols].head(3).to_string())
    if len(df.columns) > 6:
        console.print(f"[dim]... and {len(df.columns) - 6} more columns[/dim]")


def main() -> None:
    settings = get_settings()
    jd_dir = settings.data_dir / "raw" / "job_descriptions"
    if not jd_dir.exists():
        console.print(f"[red]Directory not found: {jd_dir}[/red]")
        sys.exit(1)

    # Find all CSV/TSV/XLSX files recursively
    files = []
    for pattern in ("*.csv", "*.CSV", "*.tsv", "*.xlsx", "*.xls"):
        files.extend(jd_dir.rglob(pattern))
    files = sorted(set(files))

    if not files:
        console.print(Panel(
            f"No CSV/Excel files found under {jd_dir}.\n\n"
            "Expected structure:\n"
            "  data/raw/job_descriptions/\n"
            "    ├── linkedin_india/linkedin_jobs.csv\n"
            "    ├── naukri_datascience_12k/naukri_ds.csv\n"
            "    ├── naukri_postings/jobs.csv\n"
            "    ├── linkedin_global/linkedin_global.csv\n"
            "    └── linkedin_1_3m_2024/postings.csv",
            title="No files found",
            border_style="red",
        ))
        sys.exit(1)

    console.print(f"Found {len(files)} dataset file(s). Inspecting each...\n")
    for f in files:
        try:
            inspect_csv(f)
            console.print()
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Failed inspecting {f.name}: {e}[/red]")

    console.rule("[bold green]Inspection complete[/bold green]")
    console.print(
        "\n[bold]Next step:[/bold] share this output with Claude so we can design the "
        "schema mapper in Day 6. The column names tell us which datasets need which "
        "transformations.\n"
    )


if __name__ == "__main__":
    main()