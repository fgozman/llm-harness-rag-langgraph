"""Application settings.

Precedence (highest first): explicit init args > environment > `.env` > YAML config > field defaults.
The YAML path defaults to `config.yaml` and can be overridden with `RAG_CONFIG_FILE`.
Nested fields use `__` in env vars, e.g. `LLM__GENERATION_MODEL`.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

Device = Literal["auto", "cuda", "mps", "cpu"]
DType = Literal["float16", "bfloat16", "float32"]
# Order of all-numeric dates (05/06/2026): day-month-year (Europe) or month-day-year (US).
NumericDateOrder = Literal["dmy", "mdy"]


class HardwareProfile(StrEnum):
    MAC = "mac"
    GPU = "gpu"
    CPU = "cpu"


class _ProfileDefaults(BaseModel):
    embedding_model: str
    embedding_batch_size: int
    reranker_model: str
    rerank_candidates: int
    dtype: DType


# float16 on mac because M1 has no native bfloat16; bfloat16 on CUDA for range safety; float32 on CPU
# because half precision is slow or unsupported for many CPU kernels.
PROFILE_DEFAULTS: dict[HardwareProfile, _ProfileDefaults] = {
    HardwareProfile.MAC: _ProfileDefaults(
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        # Measured on M1 Max: ~10 chunks/s at 500 tokens for batch 8-64; 16 is peak speed at ~3 GiB.
        embedding_batch_size=16,
        reranker_model="Qwen/Qwen3-Reranker-0.6B",
        # Measured on M1 Max (test corpus, warm): 20 candidates -> retrieve+rerank p50 2.8 s / p95 3.2 s;
        # rerank cost is ~140 ms per candidate, so 25-30 would exceed the ~3 s budget.
        rerank_candidates=20,
        dtype="float16",
    ),
    HardwareProfile.GPU: _ProfileDefaults(
        embedding_model="Qwen/Qwen3-Embedding-4B",
        embedding_batch_size=64,
        reranker_model="Qwen/Qwen3-Reranker-4B",
        rerank_candidates=30,
        dtype="bfloat16",
    ),
    HardwareProfile.CPU: _ProfileDefaults(
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        embedding_batch_size=8,
        reranker_model="BAAI/bge-reranker-v2-m3",
        rerank_candidates=20,
        dtype="float32",
    ),
}


class ModelPricing(BaseModel):
    """USD per million tokens; used only for cost estimates in run summaries."""

    input: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float
    output: float


# List prices checked 2026-11-22 (platform.claude.com / platform.openai.com pricing). Anthropic Haiku: <=100K
# prompt tier; OpenAI: short-context tier, "cache_write_5m" holds OpenAI's 30-minute write price.
_DEFAULT_PRICING: dict[str, ModelPricing] = {
    "claude-haiku-5-5": ModelPricing(
        input=0.10, cache_write_5m=0.125, cache_write_1h=0.20, cache_read=0.01, output=0.50
    ),
    "claude-sonnet-5-5": ModelPricing(
        input=2.0, cache_write_5m=2.50, cache_write_1h=4.0, cache_read=0.10, output=10.0
    ),
    "claude-opus-5-5": ModelPricing(
        input=4.0, cache_write_5m=5.0, cache_write_1h=8.0, cache_read=0.20, output=20.0
    ),
    "gpt-6-luna": ModelPricing(
        input=0.10, cache_write_5m=0.125, cache_write_1h=0.25, cache_read=0.01, output=0.50
    ),
    "gpt-6.2-sol": ModelPricing(
        input=2.0, cache_write_5m=2.50, cache_write_1h=5.0, cache_read=0.10, output=10.0
    ),
    "gpt-6.2-astra": ModelPricing(
        input=8.0, cache_write_5m=10.0, cache_write_1h=20.0, cache_read=0.40, output=40.0
    ),
}


# Graph nodes + evaluation (golden-set generation, LLM-as-judge evaluators, closed-book baseline).
LLMTask = Literal["analyze", "rewrite", "grade", "check", "generate", "golden", "judge", "closed_book"]

# Chat-model options passed to LangChain's `init_chat_model`; names are the integration's own (e.g. Anthropic
# `effort`/`thinking`, OpenAI `reasoning_effort`, Ollama `reasoning`).
ModelParams = dict[str, Any]


class ProviderPreset(BaseModel):
    """Defaults for a LangChain chat-model provider. Any other provider works with explicit settings."""

    generation: str
    upgrade: str
    cheap: str
    # Options for every chat model of the provider (timeouts, retries, API flavour).
    params: ModelParams = Field(default_factory=dict)
    # Per-task options: cheap routing calls skip thinking, verification calls think harder.
    task_params: dict[str, ModelParams] = Field(default_factory=dict)
    contextualize_params: ModelParams = Field(default_factory=dict)
    # Extra `with_structured_output` options (e.g. OpenAI strict schema adherence).
    structured_output_params: ModelParams = Field(default_factory=dict)
    # The integration's name for the output-token limit (LangChain's standard name is max_tokens).
    max_tokens_param: str = "max_tokens"
    # Whether the provider needs an explicit cache_control marker on the shared prompt prefix.
    prompt_cache_marker: bool = False


_NO_THINKING_ANTHROPIC: ModelParams = {"effort": "low", "thinking": {"type": "disabled"}}

# OpenAI IDs per platform.openai.com/docs/models (2026-11-22): Sol = balanced, Astra = flagship, Luna = cheap.
# Ollama: a model the user has pulled; qwen3:8b is Ollama's own documented example. `reasoning: false` is
# accepted by models without thinking too; effort levels are not, so other Ollama tasks keep model defaults.
PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "anthropic": ProviderPreset(
        generation="claude-sonnet-5-5",
        upgrade="claude-opus-5-5",
        cheap="claude-haiku-5-5",
        params={"timeout": 120.0, "max_retries": 3},
        task_params={
            "analyze": _NO_THINKING_ANTHROPIC,
            "rewrite": {"effort": "medium"},
            "grade": _NO_THINKING_ANTHROPIC,
            "check": {"effort": "medium"},
            # Sonnet 5.5 cannot disable thinking: tasks on the generation model keep its default.
            "golden": {"effort": "medium"},
            "judge": {"effort": "medium"},
        },
        contextualize_params=_NO_THINKING_ANTHROPIC,
        prompt_cache_marker=True,
    ),
    "openai": ProviderPreset(
        generation="gpt-6.2-sol",
        upgrade="gpt-6.2-astra",
        cheap="gpt-6-luna",
        # Responses API without server-side storage: documents are untrusted/private.
        params={
            "timeout": 120.0,
            "max_retries": 3,
            "use_responses_api": True,
            "store": False,
            "stream_usage": True,
        },
        task_params={
            "analyze": {"reasoning_effort": "none"},
            "rewrite": {"reasoning_effort": "medium"},
            "grade": {"reasoning_effort": "none"},
            "check": {"reasoning_effort": "medium"},
            "golden": {"reasoning_effort": "medium"},
            "judge": {"reasoning_effort": "medium"},
        },
        # Reasoning tokens would eat the tiny output budget of a 1-2 sentence blurb.
        contextualize_params={"reasoning_effort": "none"},
        structured_output_params={"strict": True},
    ),
    "ollama": ProviderPreset(
        generation="qwen3:8b",
        upgrade="qwen3:32b",
        cheap="qwen3:8b",
        params={"client_kwargs": {"timeout": 120.0}},
        task_params={"analyze": {"reasoning": False}, "grade": {"reasoning": False}},
        contextualize_params={"reasoning": False},
        max_tokens_param="num_predict",
    ),
}


class TaskSettings(BaseModel):
    """Which model a graph node uses, its output budget, and provider-specific options."""

    # Unknown keys are errors: stale options (e.g. the former `effort`/`thinking`) must not be ignored.
    model_config = ConfigDict(extra="forbid")

    model: Literal["cheap", "generation", "upgrade"]
    max_tokens: int = Field(default=2048, ge=64)
    # None = the provider preset's options for this task.
    params: ModelParams | None = None


def _default_tasks() -> dict[str, TaskSettings]:
    return {
        "analyze": TaskSettings(model="cheap", max_tokens=512),
        "rewrite": TaskSettings(model="cheap", max_tokens=2048),
        "grade": TaskSettings(model="cheap", max_tokens=2048),
        "check": TaskSettings(model="cheap", max_tokens=4096),
        "generate": TaskSettings(model="generation", max_tokens=8192),
        "golden": TaskSettings(model="generation", max_tokens=4096),
        "judge": TaskSettings(model="cheap", max_tokens=4096),
        # Same model as `generate`: leaky-question detection must reflect what the answering model knows.
        "closed_book": TaskSettings(model="generation", max_tokens=4096),
    }


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Any LangChain chat-model provider (`init_chat_model` model_provider: anthropic, openai, ollama,
    # google_genai, mistralai, bedrock_converse, ...). Its integration package must be installed, e.g.
    # `uv sync --extra anthropic` or `uv add langchain-mistralai`. API keys come from the provider's standard
    # environment variable (ANTHROPIC_API_KEY, OPENAI_API_KEY, ...; `.env` is exported by the CLI).
    provider: str = "anthropic"
    # None = the provider preset's model (required for providers without a preset).
    generation_model: str | None = None
    upgrade_model: str | None = None
    cheap_model: str | None = None
    # Endpoint override (Ollama host, OpenAI-compatible gateway, ...); None = the integration's default.
    base_url: str | None = None
    # Options for every chat model, merged over the provider preset's (a key set here wins; the preset's other
    # options, e.g. OpenAI `store: false`, stay unless overridden explicitly).
    params: ModelParams = Field(default_factory=dict)
    # None = the provider preset's name for the output-token limit (else "max_tokens").
    max_tokens_param: str | None = None
    # LangChain `with_structured_output` method; function_calling for models without native JSON schema.
    structured_output_method: Literal["json_schema", "function_calling", "json_mode"] = "json_schema"
    # Extra `with_structured_output` options, merged over the provider preset's.
    structured_output_params: ModelParams = Field(default_factory=dict)
    # Mark the shared document prefix of contextualization prompts with cache_control (explicit prompt
    # caching, Anthropic-style); None = the provider preset's.
    prompt_cache_marker: bool | None = None
    pricing: dict[str, ModelPricing] = Field(default_factory=lambda: dict(_DEFAULT_PRICING))
    tasks: dict[str, TaskSettings] = Field(default_factory=_default_tasks)

    @field_validator("tasks", mode="before")
    @classmethod
    def _merge_default_tasks(cls, value: Any) -> Any:
        # A config entry overrides only the fields it sets: `generate: {model: upgrade}` keeps the default
        # max_tokens, and every other task keeps its defaults (a plain dict field would replace them all).
        # Values of the wrong shape pass through unchanged so pydantic reports them as validation errors.
        if not isinstance(value, dict):
            return value
        overrides = cast(dict[str, Any], value)
        merged: dict[str, Any] = {name: t.model_dump() for name, t in _default_tasks().items()}
        for name, override in overrides.items():
            if name not in merged:
                raise ValueError(f"unknown LLM task {name!r}; expected one of {', '.join(get_args(LLMTask))}")
            fields: Any = (
                override.model_dump(exclude_unset=True) if isinstance(override, TaskSettings) else override
            )
            merged[name] = (
                {**merged[name], **cast(dict[str, Any], fields)} if isinstance(fields, dict) else fields
            )
        return merged

    @model_validator(mode="after")
    def _apply_provider_preset(self) -> Self:
        preset = PROVIDER_PRESETS.get(self.provider)
        if preset is None:
            missing = [
                f"llm.{f}"
                for f in ("generation_model", "upgrade_model", "cheap_model")
                if not getattr(self, f)
            ]
            if missing:
                raise ValueError(
                    f"provider {self.provider!r} has no preset: set {', '.join(missing)} "
                    f"(presets: {', '.join(PROVIDER_PRESETS)})"
                )
            preset = ProviderPreset(generation="", upgrade="", cheap="")
        self.generation_model = self.generation_model or preset.generation
        self.upgrade_model = self.upgrade_model or preset.upgrade
        self.cheap_model = self.cheap_model or preset.cheap
        self.params = {**preset.params, **self.params}
        self.structured_output_params = {**preset.structured_output_params, **self.structured_output_params}
        if self.max_tokens_param is None:
            self.max_tokens_param = preset.max_tokens_param
        if self.prompt_cache_marker is None:
            self.prompt_cache_marker = preset.prompt_cache_marker
        for name, task in self.tasks.items():
            if task.params is None:
                task.params = dict(preset.task_params.get(name, {}))
        return self


class EmbeddingSettings(BaseModel):
    # None means "use the hardware profile default".
    model: str | None = None
    batch_size: int | None = Field(default=None, ge=1)
    max_length: int = 8192
    # Qwen3-Embedding format: "Instruct: {task}\nQuery:" prepended to queries only (per model card).
    query_instruction: str = (
        "Given a question about a document collection, retrieve relevant passages that answer the question"
    )


class RerankerSettings(BaseModel):
    enabled: bool = True
    model: str | None = None
    fallback_model: str = "BAAI/bge-reranker-v2-m3"
    # How many fused candidates are reranked (latency scales linearly with this).
    candidates: int | None = Field(default=None, ge=1)
    top_k: int = Field(default=6, ge=1)
    batch_size: int = Field(default=8, ge=1)
    max_length: int = Field(default=8192, ge=256)
    # Qwen3-Reranker <Instruct> text (model-card format); ignored by classic cross-encoders.
    instruction: str = "Given a question, judge whether the passage contains information that answers it"


class RetrievalSettings(BaseModel):
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    # Hits fetched per ranking (each vector type x each query variant) before fusion.
    candidates_per_ranking: int = Field(default=50, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    context_token_budget: int = Field(default=12_000, ge=500)


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    api_key: SecretStr | None = None
    collection: str = "rag_chunks"
    sparse_model: str = "Qdrant/bm25"


class PhoenixSettings(BaseModel):
    enabled: bool = True
    collector_endpoint: str = "http://localhost:6006"
    project_name: str = "rag"
    # Recorded as user.id on traces; None = the OS user name.
    user_id: str | None = None
    # Needed only when Phoenix authentication is enabled.
    api_key: SecretStr | None = None


class ChunkingSettings(BaseModel):
    child_target_tokens: int = Field(default=500, ge=50)
    child_max_tokens: int = Field(default=600, ge=50)
    overlap_ratio: float = Field(default=0.12, ge=0.0, le=0.5)
    parent_max_tokens: int = Field(default=2000, ge=100)

    @model_validator(mode="after")
    def _target_within_max(self) -> Self:
        if self.child_target_tokens > self.child_max_tokens:
            raise ValueError("child_target_tokens must be <= child_max_tokens")
        return self


class IngestionSettings(BaseModel):
    # Unknown keys are errors: the former batch/effort options must not be silently ignored.
    model_config = ConfigDict(extra="forbid")

    contextualize: bool = True
    contextualize_concurrency: int = Field(default=4, ge=1)
    # Chat-model options for contextualization (cheap model); None = the LLM provider preset's (thinking off:
    # reasoning tokens would eat the tiny output budget).
    contextualize_params: ModelParams | None = None
    contextualize_max_output_tokens: int = Field(default=150, ge=16)
    # Spec: keep contextualization prompts under 100K tokens. Long documents get a per-section window
    # (outline + neighbouring sections) and each distinct window is a separate cache prefix: 8K keeps the test
    # corpus at ~$2.4-3.8 vs $5-21 at 70K (cost model 2026-11-23), ample for a 1-2 sentence blurb.
    contextualize_max_prompt_tokens: int = Field(default=8_000, ge=1_000, le=100_000)
    # Files are processed in windows so one huge run doesn't hold every chunk in memory at once.
    window_files: int = Field(default=50, ge=1)
    pdf_ocr: bool = True
    language: str = "en"
    # How all-numeric dates in document text (05/06/2026) are read when dating documents: "dmy" (Europe) or
    # "mdy" (US). None = they are ambiguous and skipped.
    numeric_date_order: NumericDateOrder | None = None


class GraphSettings(BaseModel):
    max_rewrite_retries: int = Field(default=2, ge=0)
    max_regenerations: int = Field(default=1, ge=0)
    min_relevant_passages: int = Field(default=2, ge=1)
    # Extra queries (variants + sub-questions) fused with the standalone query.
    max_query_variants: int = Field(default=4, ge=0, le=6)
    history_turns: int = Field(default=6, ge=0)
    # End-to-end deadline for one question (CLI and API); the turn is cancelled when it expires.
    turn_timeout_s: float = Field(default=180.0, gt=0)


class ApiSettings(BaseModel):
    # Loopback only by default: the API has no authentication.
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    # Concurrent /ask turns; more get HTTP 429 (models run in-process and share one device).
    max_concurrent_turns: int = Field(default=4, ge=1)
    # Folders /ingest may index besides docs_dir (requests outside these are rejected).
    ingest_roots: list[Path] = Field(default_factory=list[Path])
    max_question_chars: int = Field(default=4000, ge=1)


class LogSettings(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # text for terminals; json = one object per line for log shippers (the server's recommended format).
    format: Literal["text", "json"] = "text"


class EvaluationSettings(BaseModel):
    golden_path: Path = Path("eval/golden.jsonl")
    canary_facts_path: Path = Path("eval/canary_facts.jsonl")
    corpus_dir: Path = Path("data/test_corpus")
    results_dir: Path = Path("eval/results")
    dataset_prefix: str = "rag-golden"
    # Experiments trace into their own Phoenix project so interactive traces stay readable.
    project_name: str = "rag-eval"
    seed: int = 7
    unanswerable: int = Field(default=8, ge=0)
    canary_answerable: int = Field(default=8, ge=0)
    canary_unanswerable: int = Field(default=4, ge=0)


class CorpusSettings(BaseModel):
    # Test-corpus sources live in scripts/corpus_sources.yaml; only the secret-ish contact stays here (.env).
    sec_user_agent_email: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        # `.env.example` ships keys as `KEY=`; an empty value must mean "not set", not an empty secret.
        env_ignore_empty=True,
        extra="ignore",
    )

    hardware_profile: HardwareProfile = HardwareProfile.MAC
    device: Device = "auto"
    dtype: DType | None = None

    docs_dir: Path = Path("./data/docs")
    state_dir: Path = Path("./.rag")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    phoenix: PhoenixSettings = Field(default_factory=PhoenixSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    corpus: CorpusSettings = Field(default_factory=CorpusSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    log: LogSettings = Field(default_factory=LogSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)

    @model_validator(mode="after")
    def _apply_profile_defaults(self) -> Self:
        # Fill only fields the user left unset, so explicit config always wins over the profile.
        prof = PROFILE_DEFAULTS[self.hardware_profile]
        self.embedding.model = self.embedding.model or prof.embedding_model
        self.embedding.batch_size = self.embedding.batch_size or prof.embedding_batch_size
        self.reranker.model = self.reranker.model or prof.reranker_model
        self.reranker.candidates = self.reranker.candidates or prof.rerank_candidates
        self.dtype = self.dtype or prof.dtype
        if self.ingestion.contextualize_params is None:
            preset = PROVIDER_PRESETS.get(self.llm.provider)
            self.ingestion.contextualize_params = dict(preset.contextualize_params) if preset else {}
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_path = Path(os.environ.get("RAG_CONFIG_FILE", "config.yaml"))
        yaml_settings = YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path)
        return (init_settings, env_settings, dotenv_settings, yaml_settings, file_secret_settings)


def get_settings() -> Settings:
    return Settings()
