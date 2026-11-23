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

Contextual enrichment uses the configured provider's cheap model: Anthropic (prompt caching, and the
Message Batches API for large runs), OpenAI or a local Ollama model (Responses API, synchronous).
Set `LLM__PROVIDER=openai|ollama` and optionally `LLM__CHEAP_MODEL`.

## Searching

```bash
uv run rag search "What was Uber's revenue in Q3 2026?"          # hybrid + rerank, top 6
uv run rag search "seal kit K-2219-B" --mode sparse --no-rerank   # compare retrieval modes
uv run rag search "risk factors" --folder data/docs/sec --file report.pdf --after 2026-07-01
uv run rag search "..." --show-context                            # print the parent passages sent to the LLM
```

Date filters use each document's date: taken from the file name (e.g. `report_2026-11-04.pdf`), else from the
first date written near the start of the document, else the file's modification time.

Hybrid search fuses dense (Qwen3-Embedding) and BM25 rankings with Reciprocal Rank Fusion, reranks the top
candidates with Qwen3-Reranker, then expands child chunks to their parent sections within a token budget.
Latency benchmark: `uv run python scripts/bench_retrieval.py`.

## Asking questions

```bash
uv run rag ask "What was Uber's revenue in Q3 2026?"     # streamed answer + sources table
uv run rag chat                                         # multi-turn; follow-ups use the conversation
uv run rag chat --thread-id <id>                        # resume a conversation (SQLite checkpointer)
```

Answers cite passages inline as [1], [2] and list the file, page and section of each. If the documents don't
contain the answer, the reply is "Not found in the documents." Text inside documents is treated as untrusted data.

## Test corpus (post-cutoff documents for evaluation)

```bash
uv run python scripts/build_test_corpus.py            # builds data/test_corpus/ (gitignored)
uv run python scripts/build_test_corpus.py --only arxiv,github --refresh
```

Sources are fixed in `scripts/corpus_sources.yaml` (SEC EDGAR 10-Qs/8-Ks, arXiv CC BY/CC0 papers, GAO
report highlights, Federal Register rules, GitHub release notes, and locally generated fiction "canary"
documents whose facts are written to `eval/canary_facts.jsonl`). Provenance and licenses land in
`data/test_corpus/SOURCES.md`. Re-runs are idempotent and offline (HTTP cache under `data/test_corpus/.cache`).
gao.gov blocks scripted downloads: save full GAO report PDFs by hand into `data/test_corpus/gao/` as
`gao-XX-XXXXXX.pdf` and re-run the builder to register them.

Embedding and reranking models run natively on the host (MPS on Apple Silicon, CUDA on Linux),
not in Docker. Choose a profile with `HARDWARE_PROFILE=mac|gpu|cpu`.

Configuration precedence: environment > `.env` > `config.yaml` > built-in defaults.
