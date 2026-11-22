# PROGRESS

Read `SPEC.md` and `PLAN.md` first after any context reset.

## Status

| Phase | State |
|---|---|
| 0 — Scaffolding | ✅ done (2026-11-22) |
| 1 — Ingestion | ⏭ next |
| 2–6 | not started |

## Phase 0 — done

- uv project (`src/rag`, Python 3.12.15, uv 0.13.0), `rag` console script.
- Tooling: ruff 0.17.6 (lint + format), pyright 1.1.414 **strict**, pytest 9.2.0 with `integration` marker
  (deselected by default; `make test-integration` runs them).
- `src/rag/config.py`: pydantic-settings, precedence init > env > `.env` > `config.yaml` (`RAG_CONFIG_FILE` to
  override path) > defaults; nested env via `__`; hardware profiles mac/gpu/cpu fill model/dtype/candidate defaults
  only where unset.
- `src/rag/cli.py`: `rag config` (resolved settings, secrets masked), `rag doctor` (Qdrant/Phoenix health).
- `docker-compose.yml`: `qdrant/qdrant:v1.20.2`, `arizephoenix/phoenix:version-22.0.1`, named volumes,
  healthchecks, bound to 127.0.0.1 only. `Makefile`, `.env.example`, `config.yaml`, `.gitignore`, stub README.
- Acceptance: `docker compose up -d --wait` → both services healthy; `make check` green (14 unit tests);
  integration test `rag doctor` against live services passes.

### Verified versions (installed)
langgraph 1.2.24, langchain-core 1.8.4, qdrant-client 1.20.0, arize-phoenix-otel 0.17.3,
openinference-instrumentation-langchain 0.1.90, pydantic 2.14.4, pydantic-settings 2.16.3, typer 0.29.2.
Anthropic model IDs confirmed on platform.claude.com models overview: `claude-sonnet-5-5`, `claude-opus-5-5`,
`claude-haiku-5-5` (Sonnet 5.5: thinking cannot be disabled; default effort high; Haiku default effort medium).

## Decisions (and why)

1. **`device` defaults to `auto`** for every profile instead of the profile's device. The spec forbids hardcoding a
   device and requires cuda > mps > cpu detection; the profile still picks models/dtype. Explicit `DEVICE=` wins.
2. **dtype per profile**: mac float16 (no native bf16 on M1), gpu bfloat16, cpu float32 (half-precision CPU kernels
   are slow/unsupported). Overridable via `DTYPE`.
3. **Mac rerank candidates = 25** (spec range 20–30), gpu 30, cpu 20.
4. **API keys are top-level settings** so the standard `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` env names work.
5. **Heavy deps deferred** (torch, transformers, docling, fastembed, anthropic SDK, phoenix client/evals, FastAPI) to
   the phase that uses them, so each is version-checked and API-inspected right before use (rule 3).
6. **`rag doctor` fails only if Qdrant is down**; Phoenix is optional (app must work without tracing).
7. **Services bound to 127.0.0.1**; host ports configurable via `.env` (`QDRANT_HTTP_PORT`, `PHOENIX_PORT`, …).
8. `data/` and `.rag/` (runtime state: manifest, checkpointer) are gitignored.

## Environment notes (this machine)

- Another project's Phoenix container (`file-organizer-phoenix`) already holds host ports 6006/4317. It was left
  untouched; the local `.env` maps this project's Phoenix to **6007 (UI/OTLP-HTTP) and 4318 (gRPC)** and sets
  `PHOENIX__COLLECTOR_ENDPOINT=http://localhost:6007`. Compose defaults remain 6006/4317 for other machines.
- No `ANTHROPIC_API_KEY` in the environment yet; needed from Phase 1 (contextualization) / Phase 3.

## Next (Phase 1)

- Add + inspect: docling, pymupdf, torch, transformers / sentence-transformers, fastembed, tiktoken-or-HF tokenizer.
- `devices.py` (cuda > mps > cpu; set `PYTORCH_ENABLE_MPS_FALLBACK` before torch import — pydantic-settings does not
  export `.env` into `os.environ`).
- Test-first: manifest/incremental indexing, chunking rules (tables intact, token budget, overlap, parent/child).
- Fixture corpus in `tests/fixtures/` covering all 7 formats incl. a table and an exact product code (for Phase 2).

## Open questions

1. `ANTHROPIC_API_KEY` — please add it to `.env` before Phase 1's contextualization step (it is toggleable; ingestion
   can be tested with `INGESTION__CONTEXTUALIZE=false` meanwhile).
2. `CORPUS__SEC_USER_AGENT_EMAIL` — contact email for the SEC User-Agent (required before building the test corpus).
3. Token counting for chunk budgets: plan is to use the embedding model's own tokenizer (Qwen3) so budgets match what
   the embedder sees; confirm acceptable vs. a model-agnostic tokenizer.
