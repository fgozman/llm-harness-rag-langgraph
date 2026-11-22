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

Embedding and reranking models run natively on the host (MPS on Apple Silicon, CUDA on Linux),
not in Docker. Choose a profile with `HARDWARE_PROFILE=mac|gpu|cpu`.

Configuration precedence: environment > `.env` > `config.yaml` > built-in defaults.
