# SkillSync — RAG-Based Curriculum Gap Analyzer

> AI-powered curriculum alignment for Indian CS/IT engineering departments (NBA SAR 2025 compliant)
> **Hybrid architecture**: local orchestration + free cloud LLMs. No credit card. No GPU required.

## What this is

SkillSync compares a CS/IT department's current curriculum against:
- **Live Indian job market demand** (Naukri, LinkedIn India, 100k+ JDs)
- **NBA SAR 2025 accreditation** (11 POs, GAPC v4.0)
- **Industry skill reports** (NASSCOM FutureSkills, WEF)
- **AICTE model curriculum for CSE/IT**

...and produces a data-grounded gap analysis, auto-generated CO-PO mapping matrix, and interactive "what-if" simulator.

## Unique features

1. **Temporal skill decay** — Skills weighted by trajectory (rising/stable/declining)
2. **Bloom's Taxonomy depth analysis** — Detects shallow coverage ("mentions Docker, doesn't teach deployment")
3. **Multi-agent debate (LangGraph)** — Industry + Pedagogy + Accreditation + Sustainability agents negotiate
4. **Counterfactual what-if simulator** — Live alignment score as you add/remove courses
5. **Auto CO-PO-PSO mapping** — NBA-ready matrix with evidence trail

## Tech stack

| Layer | Tool | Where |
|---|---|---|
| Orchestration | Python 3.12 + LangChain + LangGraph | Local |
| Primary LLM | Groq `llama-3.3-70b-versatile` | Free cloud (30 RPM, no card) |
| Fast LLM | Groq `llama-3.1-8b-instant` | Free cloud (14.4k RPD) |
| Long-context LLM | Google Gemini 2.0 Flash | Free cloud (1M TPM) |
| Offline fallback | Ollama `qwen2.5:1.5b` | Local (optional) |
| Embeddings | BGE-small-en-v1.5 | Local CPU (133 MB) |
| Vector DB | Qdrant | Docker |
| Relational DB | PostgreSQL 16 + pgvector | Docker |
| Cache | Redis | Docker |
| Observability | Langfuse | Docker (self-hosted) |
| Object storage | MinIO | Docker |
| Heavy compute (one-time) | Google Colab (T4) | Cloud (free) |
| Frontend | Streamlit → Next.js | Local |

## Hardware requirements

- Any modern laptop with 16 GB RAM, 200 GB free disk
- No GPU required
- Stable internet (for cloud LLMs)

## Quick start

```powershell
git clone <your-repo-url> skillsync
cd skillsync
copy .env.example .env
# Edit .env — paste your Groq API key (free, no card: console.groq.com)
notepad .env

docker compose up -d

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
python -m spacy download en_core_web_sm

python scripts\verify_env.py
```

See `docs/WEEK1_SETUP.md` for the full walkthrough.

## Project structure

```
skillsync/
├── docker-compose.yml
├── configs/
│   ├── nba_po_2025.yaml
│   ├── blooms_taxonomy.yaml
│   └── cs_it_pillars.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── config.py
│   ├── ingestion/    # PDF/CSV parsers
│   ├── ontology/     # ESCO skill graph
│   ├── retrieval/    # Hybrid search
│   ├── agents/       # LangGraph
│   ├── analysis/     # Gap, temporal, Bloom's, counterfactual
│   ├── mapping/      # CO-PO auto-mapper
│   ├── api/          # FastAPI
│   └── report/       # PDF reports
├── scripts/
├── tests/
├── evaluation/
└── docs/
```

## Status

🚧 **Week 1 — Foundation & Environment**
- [x] Docker stack
- [x] Configs (NBA, Bloom's, CS/IT pillars)
- [x] Hybrid cloud LLM router design
- [x] Environment verification
- [ ] Data download scripts
- [ ] First ingestion pipeline

## License

MIT (planned)
