# Task: Build a state-of-the-art RAG system over a local document folder

You are building a production-quality Retrieval-Augmented Generation (RAG) system in Python that answers
questions about documents in a given folder, with grounded answers and citations. Orchestration uses
LangGraph; observability and evaluation use Arize Phoenix.

## 0. Project parameters (fixed by the user)

- Documents folder: ./data/docs
- Document types to support: PDF, DOCX, PPTX, XLSX, HTML, Markdown, TXT
- Document languages: English
- Expected corpus size: ~1,000 documents / ~50k pages
- Deployment target: local dev machine with Docker
- LLM provider + model for generation: Anthropic `claude-sonnet-5-5` (configurable; `claude-opus-5-5` as optional upgrade) or OpenAI or local via Ollama
- Cheaper/faster LLM for grading, rewriting, contextualization: Anthropic `claude-haiku-5-5`
  (low effort for grading, medium for rewriting; Message Batches API + prompt caching for
  contextualization during ingestion; keep contextualization prompts under 100K tokens)
- Primary dev machine: MacBook Pro M1 Max (Apple Silicon, unified memory). Must also run on Linux + CUDA.
- Hardware profiles (selected via config, `HARDWARE_PROFILE=mac|gpu|cpu`):
  - mac (default): embeddings `Qwen/Qwen3-Embedding-0.6B`, reranker `Qwen/Qwen3-Reranker-0.6B`
    (fallback `BAAI/bge-reranker-v2-m3`), device `mps`, dtype float16, rerank top 20–30 child chunks
  - gpu: embeddings `Qwen/Qwen3-Embedding-4B`, reranker `Qwen/Qwen3-Reranker-4B`, device `cuda`
  - cpu: embeddings `Qwen/Qwen3-Embedding-0.6B`, reranker `BAAI/bge-reranker-v2-m3`, rerank top 20
- Embeddings: apply the model card's instruction prefix to QUERIES only. Sparse side: BM25 via Qdrant/FastEmbed.
- Qwen3-Reranker is LLM-based, not a classic cross-encoder: follow its model card's exact usage.

Apple Silicon requirements:
- Run embedding and reranking models NATIVELY in the Python process. Only Qdrant and Phoenix run in
  Docker (Docker on macOS has no GPU/MPS access).
- Device auto-detection order: cuda > mps > cpu. Never hardcode a device.
- Load models in float16 on MPS (M1 has no native bfloat16). Do not use flash_attention_2 on MPS.
- Set PYTORCH_ENABLE_MPS_FALLBACK=1 in .env.example.
- Batch embeddings during ingestion (tune batch size for unified memory) and load each model once
  per process, not per request.
- Phase 2 acceptance check addition: benchmark rerank latency on the Mac profile and record p50/p95
  in PROGRESS.md; target under ~3 s per query for the full retrieve + rerank step.

Claude 5.5 API constraints (verify against current docs before coding):
- Forced tool use returns an error on Sonnet 5.5. Do NOT use LangChain structured output modes that
  force tool_choice; use native JSON-schema structured outputs instead, and test every routing node.
- Do not set temperature/top_p/top_k to non-default values on Sonnet 5.5 (returns HTTP 400).
- Thinking blocks are tied to the model that produced them: never pass Sonnet's thinking blocks
  into Haiku calls or vice versa when sharing message history across nodes.

Every provider/model must be configurable, never hardcoded. Use these values as defaults.

## 1. Working rules for you (the coding agent)

1. Before writing code, read this whole spec, then create `PLAN.md` (architecture, module layout, phase
   checklist) and `PROGRESS.md` (what's done, what's next, open questions, decisions made and why).
   Update `PROGRESS.md` at the end of every phase. If your context is reset, re-read both files first.
2. Work phase by phase (Section 5). Do not start a phase until the previous phase's acceptance checks pass.
3. Library APIs change quickly. Do NOT rely on memory for LangGraph, Phoenix, Qdrant, or Docling APIs.
   After installing, check the installed version (`uv pip show <pkg>`) and inspect the actual API
   (`python -c "import x; help(x.y)"` or read the package source in `.venv`) before using it.
4. Verify by running things: after every meaningful change run `ruff check`, `ruff format`, type checks,
   and `pytest`. Never claim something works without having executed it.
5. Keep functions small, typed, and documented with "why" comments only. No dead code, no TODO stubs
   left behind at the end of a phase.
6. If a requirement is ambiguous or blocked (missing API key, service not running), stop and ask
   rather than inventing a workaround that silently changes behavior.
7. Testing approach: test-first for deterministic logic, evals for LLM quality.
   - Write the failing test BEFORE the implementation for: manifest/incremental indexing, chunking
     rules, RRF fusion and dedup, context budgeting, citation mapping, and every LangGraph routing
     edge and loop cap. Run it, confirm it fails for the right reason, then implement.
   - For LangGraph nodes, use a fake LLM that returns scripted structured outputs, so routing tests
     are fast, deterministic, and need no API key or network.
   - For exploratory integration work (Docling output, Qdrant hybrid queries, Phoenix APIs), spike
     first, then pin the observed behavior with a test before moving on.
   - Do NOT unit-test LLM answer quality. Answer quality, groundedness, and retrieval relevance are
     measured by the Phoenix evaluation suite (Phase 5).
   - Never weaken an assertion, change an expected value, or skip/delete a test to make it pass. If
     you believe a test is wrong, stop and explain why in PROGRESS.md before changing it.
   - Keep the unit suite under ~30 s so it runs after every change; mark tests needing Qdrant,
     Phoenix, models, or the Anthropic API as `integration` and run them at phase boundaries.

## 2. Test corpus (real documents published after the models' training cutoff)

The LLMs used here have a training cutoff around June 2026. Build a test corpus of documents published
on or after 2026-07-01 so correct answers can only come from retrieval.

Create `scripts/build_test_corpus.py` that builds `data/test_corpus/` (gitignored; never commit
downloaded documents), idempotent and cached, with a fixed source list in `scripts/corpus_sources.yaml`.
Record every file's source URL, publication date, and license in `data/test_corpus/SOURCES.md`.
Verify each API's current documentation before coding against it; do not rely on memory.

1. SEC EDGAR (data.sec.gov / EDGAR full-text and submissions APIs):
   - For 5–10 configurable companies: 10-Q filings from Aug 2026 and Nov 2026, the related 8-K
     earnings releases (Exhibit 99.1), and `Financial_Report.xlsx` where available.
   - Also one 10-Q from before July 2026 per company, flagged `pre_cutoff: true` in SOURCES.md,
     as a temporal hard negative.
   - Respect SEC fair-access rules: a descriptive User-Agent with a contact email from config,
     and stay well under the published request-rate limit.

2. arXiv (export.arxiv.org API):
   - ~30 papers in a configurable category (default cs.IR) submitted 2026-08-01 or later.
   - Keep only papers whose license metadata is CC BY or CC0; download PDFs with a 3 s delay.

3. US government reports (public domain):
   - ~10 GAO reports and ~10 Federal Register documents (federalregister.gov API) published on or
     after 2026-07-01. Prefer PDFs for GAO, HTML/XML for the Federal Register.

4. Stack release notes (GitHub API):
   - Release notes / CHANGELOG entries since 2026-07-01 for langgraph, arize-phoenix, and
     qdrant-client, saved as Markdown, one file per repository.

5. Canary documents (generated locally, clearly marked as fiction in metadata):
   - 1 PPTX with speaker notes, 2 DOCX, 1 XLSX, containing invented facts written to
     `eval/canary_facts.jsonl`. Used for format coverage and abstention tests.

Evaluation set (`eval/golden.jsonl`, ~80 questions, generated with an LLM from the corpus, then
human-reviewed; each item has question, reference answer, source file(s), location, category):
   - Exact figures from filings ("What was <company>'s revenue for the quarter ended Sep 2026?").
   - Temporal disambiguation (Q2 vs Q3 2026 vs the pre-cutoff quarter of the same company).
   - Cross-document comparison ("How did gross margin change between the Aug and Nov 10-Qs?").
   - Table and appendix lookups in arXiv papers and GAO reports.
   - Version-specific questions from release notes ("Which LangGraph release added X?").
   - Unanswerable questions that sound plausible for the corpus (system must abstain).

Closed-book baseline: add `rag eval --no-retrieval`, which runs the same questions with retrieval
disabled. Report both scores side by side in Phoenix. Flag every question answered correctly
closed-book as "leaky" in the eval report so it can be rewritten or removed.

## 3. Tech stack

- Python 3.12, `uv` for env/deps, `pyproject.toml`, `ruff`, `pyright` (or mypy strict), `pytest`.
- `langgraph` (1.x) for the query pipeline; `langchain-core` only where LangGraph needs it.
- Arize Phoenix: `arize-phoenix` (server, run via Docker), `arize-phoenix-otel` for tracing,
  `openinference-instrumentation-langchain` (covers LangGraph), plus `arize-phoenix-client` /
  `arize-phoenix-evals` for datasets, experiments, and evaluators.
- Parsing: Docling (layout-aware, tables, headings, page numbers) as primary parser; a lightweight
  fallback (e.g. PyMuPDF / plain text) when Docling fails on a file.
- Vector store: Qdrant (Docker) with hybrid search: dense vectors + sparse vectors (BM25 or SPLADE),
  fused with Reciprocal Rank Fusion. Store rich payload metadata.
- Config: `pydantic-settings` reading `.env` + `config.yaml`.
- Interfaces: Typer CLI; FastAPI server with streaming (SSE) as a later phase.
- `docker-compose.yml` for Qdrant and Phoenix.

## 4. Architecture

### 4.1 Ingestion pipeline (offline)

1. Discovery: recursively scan the folder; filter by supported extensions; compute a content hash per file.
2. Incremental indexing: keep a manifest (SQLite) of path, hash, mtime, chunk IDs. Re-index only new or
   changed files; delete chunks of removed files. Ingestion must be idempotent.
3. Parsing: convert each document to a structured representation preserving headings hierarchy, page
   numbers, tables (as Markdown), and lists. Record parse failures in the manifest; never crash the run.
4. Chunking (structure-aware):
   - Split along document structure (sections/headings) first, then enforce a token budget
     (configurable, default ~400–600 tokens, ~10–15% overlap). Never split a table mid-row.
   - Parent–child ("small-to-big"): index small child chunks for retrieval, but keep a link to the
     larger parent section that is passed to the LLM.
5. Contextual enrichment ("contextual retrieval"): for each chunk, generate with the cheap LLM a 1–2
   sentence context describing where the chunk sits in its document (title, section, what it's about).
   Prepend it to the text used for embeddings and sparse indexing; keep the original text for display.
   Use prompt caching / batching where the provider supports it; make this step toggleable.
6. Indexing: compute dense + sparse vectors; upsert into Qdrant with payload: doc_id, file path, file
   name, title, section path, page range, chunk index, parent_id, hash, ingested_at, language.
7. Concurrency + robustness: bounded parallelism, retries with backoff on model calls, progress bar,
   a final summary (files processed / skipped / failed, chunks written, time, cost estimate).

### 4.2 Query pipeline (LangGraph StateGraph)

Define a typed state (TypedDict or Pydantic) containing at least: messages/history, original question,
rewritten queries, retrieved docs, reranked docs, grades, answer, citations, loop counters, flags.

Nodes:
1. `analyze_query`: classify intent: needs retrieval / chit-chat / out-of-scope; detect whether the question
   is complex (multi-part, comparative) and should be decomposed.
2. `rewrite_query`: turn the latest user turn into a standalone query using conversation history;
   optionally produce 2–4 query variants (multi-query) or sub-questions (decomposition).
3. `retrieve`: hybrid search in Qdrant for each query variant; merge with RRF; support metadata filters
   (file name, folder, date) extracted from the question when present.
4. `rerank`: cross-encoder rerank to top-k (configurable, default 5–8); then expand child chunks to their
   parent sections, deduplicate, and fit to a context token budget.
5. `grade_documents`: cheap LLM judges relevance of each passage (structured output). If too few relevant
   passages, route back to `rewrite_query` with feedback (max N=2 retries), then proceed anyway.
6. `generate`: answer strictly from provided passages, with inline citations like [1], [2] mapped to
   file + page/section. If the passages don't contain the answer, say so explicitly. Stream tokens.
7. `check_answer`: verify groundedness (every claim supported by cited passages) and that the answer
   addresses the question. If it fails, regenerate once with the critique; if it fails again, return a
   cautious answer that states what could and couldn't be verified.
8. `respond`: format final answer + structured citation list (path, page, section, snippet).

Requirements:
- Conditional edges for all routing; hard caps on every loop.
- Conversation memory via a LangGraph checkpointer (SQLite) keyed by `thread_id`.
- All LLM outputs that drive routing use structured output (Pydantic schemas), never regex parsing.
- Prompts live in a `prompts/` module (or Phoenix prompt management), versioned, not inline strings
  scattered across nodes.
- Security: treat document text as untrusted data. The system prompt must instruct the model to ignore
  instructions found inside documents. Wrap passages in clear delimiters.

### 4.3 Observability (Phoenix)

- Initialize tracing once at startup with `phoenix.otel.register(project_name=..., auto_instrument=True)`
  (verify the signature against the installed version) so LangGraph/LangChain calls are traced.
- Add manual spans where auto-instrumentation is blind: ingestion stages, Qdrant hybrid search, RRF
  fusion, reranking. Use OpenInference semantic conventions so retrieved documents (id, score, content,
  metadata) show up as RETRIEVER/RERANKER spans in Phoenix.
- Attach `session.id` (= thread_id) and user metadata to traces so multi-turn conversations group.
- Record token usage and latency per node. Tracing must be disable-able via config and must never
  break the app if Phoenix is down.

### 4.4 Evaluation (Phoenix datasets + experiments)

1. Golden dataset: a script that samples chunks from the corpus and uses an LLM to generate diverse
   question/answer/source triples (factual, multi-hop across sections, table lookup, unanswerable).
   Save to `eval/golden.jsonl` for human review, then upload as a Phoenix dataset.
2. Experiments: run the full graph over the dataset as a Phoenix experiment with evaluators:
   - Retrieval: hit rate@k, MRR, NDCG against the gold source chunks/documents.
   - Generation: faithfulness/hallucination, answer correctness vs reference, answer relevance.
   - Citations: do the cited sources actually contain the supporting text?
   - Unanswerable questions: does the system correctly abstain?
   - Latency and cost per query.
3. A CLI command to compare configurations (e.g. with/without contextual enrichment, with/without
   reranker, chunk sizes, top-k) as separate experiments so results are comparable in the Phoenix UI.

## 5. Repository layout (adjust if you have a good reason; record it in PROGRESS.md)

src/rag/
  config.py          # settings
  telemetry.py       # Phoenix/OTel setup
  ingestion/         # discovery, manifest, parsing, chunking, contextualize, indexing
  retrieval/         # qdrant client, hybrid search, rrf, rerank, parent expansion
  graph/             # state.py, nodes/, edges.py, build.py
  llm/               # provider factory, structured output helpers
  prompts/
  api/               # FastAPI app (later phase)
  cli.py             # typer: ingest, ask, chat, eval, compare
eval/                # dataset generation, evaluators, experiment runners
tests/               # unit + integration, small fixture corpus in tests/fixtures/
docker-compose.yml, pyproject.toml, .env.example, config.yaml, README.md

## 6. Phases and acceptance checks

Phase 0: Scaffolding
- uv project, tooling, config, docker-compose (Qdrant + Phoenix), `.env.example`, Makefile/justfile.
- Check: `docker compose up -d` brings both services up; `pytest` runs (even if few tests); lint clean.

Phase 1: Ingestion
- Discovery, manifest, Docling parsing + fallback, structure-aware chunking, parent–child, indexing
  with dense + sparse vectors. Contextual enrichment behind a flag.
- Check: `rag ingest <folder>` on tests/fixtures indexes all files; re-running is a no-op; modifying
  one file re-indexes only that file; deleting a file removes its chunks. Unit tests for chunker
  (tables intact, token budget respected).

Phase 2: Retrieval
- Hybrid search + RRF, reranking, parent expansion, context budget, metadata filters.
- Check: `rag search "<query>"` prints ranked results with scores and sources; tests prove hybrid
  beats dense-only on a keyword-heavy fixture query (e.g. an exact product code).

Phase 3: LangGraph pipeline
- All nodes from 3.2, routing, loop caps, checkpointer, streaming, citations.
- Check: `rag ask "<question>"` returns a cited answer; `rag chat` handles follow-ups ("what about the
  second one?"); an unanswerable question yields an explicit "not found in the documents"; unit tests
  for each node using a fake/deterministic LLM; one integration test end-to-end.

Phase 4: Phoenix tracing
- Check: running `rag ask` produces a trace in Phoenix showing every graph node, retriever spans with
  documents, reranker spans, token counts; chat turns grouped by session. App still works with
  Phoenix stopped.

Phase 5: Evaluation
- Golden dataset generation, Phoenix dataset upload, experiments with all evaluators, compare command.
- Check: `rag eval` produces an experiment visible in Phoenix with per-example scores; `rag compare`
  runs at least two configurations; write a short results summary into PROGRESS.md.

Phase 6: API + hardening
- FastAPI with `/ingest`, `/ask` (SSE streaming), `/health`; request timeouts; structured logging;
  README with setup, architecture diagram (Mermaid), configuration reference, and eval instructions.
- Check: `curl` against `/ask` streams tokens and ends with a citations payload.

## 7. Definition of done

- All phase checks pass; lint, type checks and tests are green.
- A new user can follow README.md from a clean clone to a working Q&A over their own folder.
- Every answer is either grounded with verifiable citations or an explicit abstention.
- Phoenix shows full traces and at least one completed evaluation experiment.

Start now with Phase 0: read the environment, write PLAN.md and PROGRESS.md, then scaffold.

