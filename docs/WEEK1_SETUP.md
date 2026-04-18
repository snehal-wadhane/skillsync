# Week 1 Setup — Hybrid Cloud Architecture (CPU laptop friendly)

This setup is optimized for laptops WITHOUT NVIDIA GPUs. Heavy AI compute happens in the cloud (free tiers); your laptop runs the orchestration, storage, and UI.

Estimated time: 1–1.5 hours.

---

## Architecture at a glance

```
YOUR LAPTOP (AMD Ryzen + integrated graphics, 16 GB RAM)
 ├─ Python 3.12 venv — your code runs here
 ├─ Docker Compose — 5 services
 │   ├─ Qdrant      (vector DB)
 │   ├─ PostgreSQL  (relational)
 │   ├─ Redis       (cache)
 │   ├─ MinIO       (object storage)
 │   └─ Langfuse+DB (observability)
 └─ Ollama (optional, for offline demos only)

FREE CLOUD APIs (no credit card needed)
 ├─ Groq          — Llama 3.3 70B, 30 RPM, 6K TPM free
 ├─ Google AI     — Gemini 2.0 Flash, 1500 RPD, 1M TPM free
 └─ Google Colab  — T4 GPU for occasional heavy compute
```

---

## Prerequisites

Open PowerShell and check:

```powershell
python --version       # 3.11 or 3.12
git --version
docker --version
docker compose version
```

Install anything missing:
- Python: python.org/downloads ("Add to PATH")
- Docker Desktop: docker.com/products/docker-desktop
  - After install, start Docker Desktop (whale icon in tray)
  - Settings → Resources: allocate 6 GB RAM, 4 CPUs (leave room for everything else)
- Git: git-scm.com

You do NOT need CUDA, NVIDIA drivers, or a GPU. PyTorch will install CPU wheels automatically.

---

## Step 1 — Get free API keys (5 minutes)

### Groq (required, primary LLM)
1. Go to https://console.groq.com/
2. Sign up with Google or email (no credit card)
3. Click "API Keys" → "Create API Key"
4. Copy the key (starts with `gsk_...`) — you'll paste it into `.env` later

### Google AI Studio (recommended, secondary)
1. Go to https://aistudio.google.com/apikey
2. Sign in with Google
3. Click "Create API key" → "Create API key in new project"
4. Copy the key — paste into `.env`

> **Why both?** Groq is faster but has a 6K TPM limit. Gemini has 1M TPM — better for long documents. We use Groq for interactive calls, Gemini for bulk report generation.

---

## Step 2 — Clone and configure

```powershell
cd C:\
git clone <your-repo-url> skillsync
cd skillsync
copy .env.example .env
notepad .env
```

Edit `.env` — at minimum fill in:
- `GROQ_API_KEY` (from Step 1)
- `GOOGLE_API_KEY` (from Step 1, optional)
- `POSTGRES_PASSWORD` (any strong string)
- `MINIO_ROOT_PASSWORD` (any strong string)
- `LANGFUSE_NEXTAUTH_SECRET` (run `python -c "import secrets; print(secrets.token_urlsafe(32))"`)
- `LANGFUSE_SALT` (another random string)

Leave `LANGFUSE_PUBLIC_KEY` / `SECRET_KEY` as placeholders — set after Step 3.

---

## Step 3 — Start Docker services

```powershell
docker compose up -d
```

First run downloads ~2 GB of images (5–10 min).

**Verify:**
```powershell
docker compose ps
```

6 containers, all `Up` or `healthy`:
- skillsync-qdrant, skillsync-postgres, skillsync-redis
- skillsync-langfuse-db, skillsync-langfuse
- skillsync-minio

**Dashboards** (optional sanity):
- Qdrant: http://localhost:6333/dashboard
- Langfuse: http://localhost:3000 (sign up, create org+project, get API keys → paste into `.env`)
- MinIO: http://localhost:9001

---

## Step 4 — Python environment

```powershell
cd C:\skillsync

python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If blocked: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

python -m pip install --upgrade pip setuptools wheel

# CPU-only PyTorch (fast install, ~200 MB)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Everything else
pip install -e ".[dev]"

# spaCy model
python -m spacy download en_core_web_sm
```

**Sanity check:**
```powershell
python -c "import torch, groq; print('torch:', torch.__version__, '| groq:', groq.__version__)"
```

---

## Step 5 — Install Ollama (OPTIONAL — only for offline demos)

Skip this step if you always have internet. You can come back later.

If you want offline fallback:
1. Download: https://ollama.com/download/windows
2. Install (runs as Windows service)
3. Pull tiny model:
```powershell
ollama pull qwen2.5:1.5b
```
Only ~1 GB, CPU-friendly. Don't bother with larger models on this hardware.

---

## Step 6 — Verify everything

```powershell
python scripts\verify_env.py
```

Expected:
- **10 required checks green** ✓ (Python, PyTorch, dirs, configs, Groq, Qdrant, Postgres, Redis, MinIO, embeddings)
- **3 optional checks** — any mix of green/yellow:
  - Gemini: green if key set, yellow if skipped
  - Langfuse: green after you create keys in UI, yellow otherwise
  - Ollama: green if installed, yellow if skipped

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `docker: command not found` | Docker Desktop not installed or not running |
| `docker compose up` very slow | First-time image pull. Wait 10 min, check `docker compose logs` |
| Qdrant unhealthy | Port 6333 in use. `netstat -ano | findstr :6333` → kill process or change port |
| Postgres auth failed | `.env` password differs from initial setup. Run `docker compose down -v && docker compose up -d` to reset (DELETES all data) |
| Groq check fails | Check key starts with `gsk_` and has no extra spaces. Test: `curl -H "Authorization: Bearer YOUR_KEY" https://api.groq.com/openai/v1/models` |
| Torch install fails | Use CPU-only wheel: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `pip install -e .` slow | Normal — first install downloads ~1.5 GB of deps. Go get coffee. |
| Langfuse check fails first time | Takes ~60s to finish DB migrations on initial start. Try again. |

---

## What's next (Day 3-7)

Once verification passes:
- **Day 3**: `scripts/download_all.py` — ESCO, NBA SAR, AICTE, Kaggle Indian jobs
- **Day 4**: ESCO skill ontology loader (Postgres + Qdrant dual-write)
- **Day 5**: PDF ingestion pipeline with smart course-boundary chunking
- **Day 6**: Job description ingestion + skill extractor
- **Day 7**: First end-to-end CLI query

Report back with `verify_env.py` output — all green or paste errors.
