# rag — grounded, cited Q&A over a local document folder

LangGraph query pipeline, Qdrant hybrid search, Arize Phoenix tracing and evaluation.
Work in progress: see `PLAN.md` (architecture, phases) and `PROGRESS.md` (status).
The full README (architecture diagram, config reference, eval guide) arrives in Phase 6.

## Quick start (development)

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and Docker.

```bash
cp .env.example .env      # fill in ANTHROPIC_API_KEY; change ports if they clash
make install              # uv sync
make up                   # Qdrant + Phoenix (Docker), waits until healthy
uv run rag doctor         # verify services
uv run rag config         # show resolved settings (secrets masked)
make check                # lint + strict types + unit tests
```

## Indexing documents

```bash
uv run rag ingest                     # indexes ./data/docs (docs_dir)
uv run rag ingest path/to/folder      # or any folder / single file
uv run rag ingest --no-contextualize  # skip LLM contextual enrichment (no API key needed)
uv run rag ingest --force             # re-index everything
uv run rag ingest --retry-failed      # retry files that failed to parse earlier
```

Re-running is a no-op unless files changed: a SQLite manifest (`.rag/rag.db`) tracks content hashes.
Changed files are re-indexed, deleted files have their chunks removed. Supported: PDF, DOCX, PPTX,
XLSX, HTML, Markdown, TXT. First run downloads models (Qwen3-Embedding, Docling layout models).

Embedding and reranking models run natively on the host (MPS on Apple Silicon, CUDA on Linux),
not in Docker. Choose a profile with `HARDWARE_PROFILE=mac|gpu|cpu`.

Configuration precedence: environment > `.env` > `config.yaml` > built-in defaults.
