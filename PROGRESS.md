# PROGRESS

Read `SPEC.md` and `PLAN.md` first after any context reset.

## Status

| Phase | State |
|---|---|
| 0 — Scaffolding | ✅ done (2026-11-22) |
| 1 — Ingestion | ✅ done (2026-11-22) |
| 2 — Retrieval | ⏭ next |
| Test corpus (Section 2) | not started — blocked on SEC contact email |
| 3–6 | not started |

## Phase 1 — done

Modules (`src/rag/`): `ingestion/{models,discovery,manifest,parsing,chunking,contextualize,pipeline}.py`,
`retrieval/{types,qdrant_index}.py`, `embeddings.py`, `devices.py`, `factory.py`, `prompts/contextualize.py`,
CLI `rag ingest [FOLDER] [--force] [--retry-failed] [--contextualize/--no-contextualize]`.

Acceptance (all executed):
- `rag ingest` on a copy of `tests/fixtures/docs` (7 files, all 7 formats): 7 indexed / 20 chunks; re-run → 0
  indexed, 7 unchanged; appending to one file → only it re-indexed; deleting `shipments.xlsx` → its chunks
  removed (20 → 18 points). Automated in `tests/test_ingest_integration.py`.
- With contextualization (auto → sync, Haiku 5.5): 20/20 chunks enriched, $0.0017, 9.5 s.
- Message Batches round-trip test passed (2 requests, 48 s). Prompt caching verified: write 2,707 → read 2,707.
- Chunker unit tests: tables intact (row boundaries, header repeated), token budget, overlap, parent/child links.
- `make check`: ruff + pyright strict clean; 92 unit tests in ~2.5 s. Integration: 10 pass (+1 slow batch test).

Test-first was followed for chunking, discovery, manifest, incremental pipeline, contextualizer logic, device
selection, config profiles. One test bug fixed before first green (`test_second_run_is_a_noop` didn't reset the
fake index's call log between runs; fix made the assertion check what it intended — stricter, not weaker).

### Verified versions / API facts (Phase 1)
docling 2.151.1 (docling-core 2.108.0), torch 2.15.0 (MPS available), transformers 5.24.0 (`dtype=` replaces
`torch_dtype`), sentence-transformers 6.3.0 (`processor_kwargs`, `get_embedding_dimension()`), fastembed 0.9.1,
anthropic 2.12.0, pymupdf 2.1, typer 0.27.3 (docling caps typer < 0.28).
- Docling: no TXT input format; PPTX speaker notes are in `ContentLayer.NOTES`; HTML `<title>` duplicates into
  FURNITURE; XLSX sheet names only via `GroupItem(label=sheet)` with `with_groups=True`; pages exist for
  PDF/PPTX(slide)/XLSX(sheet) only; PDF titles often come as section headers.
- Qwen3-Embedding: query prompt `"Instruct: {task}\nQuery:"` (no space), none for documents; 1024-d; left padding.
- Anthropic (platform.claude.com docs): `output_config={"effort": ...}`; Haiku 5.5 allows `thinking: disabled`
  at effort ≤ high; cache min 512 tokens (Haiku/Sonnet 5.5); `cache_control.ttl` "5m"|"1h"; batches ≤ 100k
  requests / 256 MB; Haiku 5.5 price tier jumps 5× above 100K prompt tokens.

## Decisions (and why)

Phase 0:
1. `device` defaults to `auto` (cuda > mps > cpu) for every profile; profile picks models/dtype/batch size.
2. dtype per profile: mac float16, gpu bfloat16, cpu float32; also auto-corrected per device (bf16 on MPS → fp16,
   half on CPU → fp32).
3. Mac rerank candidates = 25 (spec 20–30), gpu 30, cpu 20.
4. API keys are top-level settings (standard `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` names).
5. Heavy deps added only in the phase that uses them.
6. `rag doctor` fails only if Qdrant is down; Phoenix is optional.
7. Services bound to 127.0.0.1; host ports configurable via `.env`.

Phase 1:
8. **Tokenizer for chunk budgets = embedding model's tokenizer** (Qwen3, via `tokenizers`), so budgets match what
   the embedder sees. (Open question from Phase 0 resolved with the proposed default; user said "yes".)
9. **Own chunker over a neutral `Block` model** instead of Docling's HybridChunker: spec rules (never split table
   rows, parent/child, overlap) are pinned by unit tests independent of Docling. Children never cross sections;
   overlap is a verbatim sentence-aligned suffix (≤ 12% of target); tables get no overlap but repeat their header.
   Parents = consecutive sections under the same top-level heading packed to ≤ 2000 tokens.
10. **Parents stored in the SQLite manifest** (docstore table), not in Qdrant: fetched by ID only, no vectors.
11. IDs: `doc_id = uuid5(abs path)`, chunk/parent IDs `uuid5(doc_id:index)` → deterministic; re-index =
    delete-by-doc_id then upsert, idempotent even after a crash mid-file.
12. Parse failures → recorded in manifest, stale chunks deleted, run continues; not retried until the file changes
    or `--retry-failed`. Infra errors (Qdrant, embedder) abort the run. If the Qdrant collection is missing while the
    manifest has records, everything is re-indexed (index is the source of truth).
13. Fallback parsers: PyMuPDF for PDF, plain text for MD; DOCX/PPTX/XLSX/HTML have no fallback (failure recorded).
    `.txt` is always read directly (Docling has no TXT format).
14. Contextualization: Anthropic-only for now (prompt caching + Batches are provider-specific); a clear config error
    if provider ≠ anthropic. `auto` mode = Batches API (50% off, 1h cache TTL) at ≥ 200 chunks per window, else
    concurrent sync calls (4) with cache "primer" requests first. Thinking disabled, effort low, no sampling params.
    Long docs: outline + window of neighboring sections within 70K local tokens (margin under Haiku's 100K tier).
    Per-chunk failures leave context empty (counted in summary); truncated outputs trimmed to last sentence.
15. Batch size per profile from measurement (M1 Max, 500-token chunks): ~10 chunks/s flat for batch 8–64; batch 16
    = peak speed at ~3.3 GiB MPS memory (64 → 6.3 GiB, slower). ⇒ ~50k pages (~150k chunks) ≈ 4 h embedding on Mac.
16. Parsing is sequential (Docling parallelizes internally and its converter isn't documented as thread-safe);
    bounded parallelism applies to LLM calls; files are processed in windows of 50 to bound memory.
17. Supported targets in uv resolution: macOS arm64 + Linux (`tool.uv.environments`); Intel macOS has no torch wheels.
18. Pyright strict everywhere; only the adapters wrapping untyped libraries (`embeddings.py`, `parsing.py`) relax the
    "unknown type" checks via file-level pragmas. Ruff `PLC0415` disabled: lazy imports of heavy deps are deliberate.

## Environment notes (this machine)

- Another project's Phoenix (`file-organizer-phoenix`) holds 6006/4317; this project's Phoenix runs on **6007/4318**
  via `.env`. Compose defaults stay 6006/4317.
- ⚠️ The Anthropic API key was accidentally printed in the session log during Phase 1 setup (macOS `sed` redaction
  failed). Recommend rotating it.
- HF downloads are unauthenticated (rate-limit warning); optional `HF_TOKEN` in `.env` speeds them up.

## Next (Phase 2 — Retrieval)

- Inspect qdrant-client 1.20 Query API (prefetch + `FusionQuery(RRF)`) — spiked in Phase 1, works.
- Decide: server-side RRF (Qdrant) vs client-side RRF across multi-query variants (test-first RRF + dedup).
- Qwen3-Reranker-0.6B per model card (LLM yes/no logits), fallback bge-reranker-v2-m3; benchmark p50/p95 on mac.
- Parent expansion from manifest docstore, dedup, context budget (test-first); metadata filters (file name, folder,
  date) — may need a `folders` ancestor list in payload for prefix filtering.
- `rag search "<query>"`; test: hybrid beats dense-only on an exact product code (fixture: `NRX-4471-B`, `E-217`).

## Open questions

1. `CORPUS__SEC_USER_AGENT_EMAIL` — contact email for SEC EDGAR User-Agent (needed before building the test corpus).
2. Contextualization for non-Anthropic providers (OpenAI/Ollama): implement a sync-only path later, or keep it
   Anthropic-only? Currently errors clearly and suggests `--no-contextualize`.
