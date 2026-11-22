"""Application settings.

Precedence (highest first): explicit init args > environment > `.env` > YAML config > field defaults.
The YAML path defaults to `config.yaml` and can be overridden with `RAG_CONFIG_FILE`.
Nested fields use `__` in env vars, e.g. `LLM__GENERATION_MODEL`.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

Device = Literal["auto", "cuda", "mps", "cpu"]
DType = Literal["float16", "bfloat16", "float32"]
LLMProvider = Literal["anthropic", "openai", "ollama"]
Effort = Literal["low", "medium", "high"]


class HardwareProfile(StrEnum):
    MAC = "mac"
    GPU = "gpu"
    CPU = "cpu"


class _ProfileDefaults(BaseModel):
    embedding_model: str
    reranker_model: str
    rerank_candidates: int
    dtype: DType


# float16 on mac because M1 has no native bfloat16; bfloat16 on CUDA for range safety; float32 on CPU
# because half precision is slow or unsupported for many CPU kernels.
PROFILE_DEFAULTS: dict[HardwareProfile, _ProfileDefaults] = {
    HardwareProfile.MAC: _ProfileDefaults(
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        reranker_model="Qwen/Qwen3-Reranker-0.6B",
        rerank_candidates=25,
        dtype="float16",
    ),
    HardwareProfile.GPU: _ProfileDefaults(
        embedding_model="Qwen/Qwen3-Embedding-4B",
        reranker_model="Qwen/Qwen3-Reranker-4B",
        rerank_candidates=30,
        dtype="bfloat16",
    ),
    HardwareProfile.CPU: _ProfileDefaults(
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        reranker_model="BAAI/bge-reranker-v2-m3",
        rerank_candidates=20,
        dtype="float32",
    ),
}


class LLMSettings(BaseModel):
    provider: LLMProvider = "anthropic"
    generation_model: str = "claude-sonnet-5-5"
    upgrade_model: str = "claude-opus-5-5"
    cheap_model: str = "claude-haiku-5-5"
    grading_effort: Effort = "low"
    rewriting_effort: Effort = "medium"
    ollama_base_url: str = "http://localhost:11434"
    request_timeout_s: float = 120.0
    max_retries: int = 3


class EmbeddingSettings(BaseModel):
    # None means "use the hardware profile default".
    model: str | None = None
    batch_size: int = 32
    max_length: int = 8192


class RerankerSettings(BaseModel):
    model: str | None = None
    fallback_model: str = "BAAI/bge-reranker-v2-m3"
    candidates: int | None = Field(default=None, ge=1)
    top_k: int = Field(default=6, ge=1)


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    api_key: SecretStr | None = None
    collection: str = "rag_chunks"
    sparse_model: str = "Qdrant/bm25"


class PhoenixSettings(BaseModel):
    enabled: bool = True
    collector_endpoint: str = "http://localhost:6006"
    project_name: str = "rag"


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
    contextualize: bool = True
    max_workers: int = Field(default=4, ge=1)
    # Spec: keep contextualization prompts under 100K tokens.
    contextualize_max_prompt_tokens: int = Field(default=100_000, le=100_000)


class GraphSettings(BaseModel):
    max_rewrite_retries: int = Field(default=2, ge=0)
    max_regenerations: int = Field(default=1, ge=0)
    min_relevant_passages: int = Field(default=2, ge=1)
    context_token_budget: int = Field(default=12_000, ge=1000)
    max_query_variants: int = Field(default=4, ge=1, le=4)


class CorpusSettings(BaseModel):
    sec_user_agent_email: str | None = None
    arxiv_category: str = "cs.IR"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    hardware_profile: HardwareProfile = HardwareProfile.MAC
    device: Device = "auto"
    dtype: DType | None = None

    # Top-level so the providers' standard env var names (ANTHROPIC_API_KEY, OPENAI_API_KEY) work.
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    docs_dir: Path = Path("./data/docs")
    state_dir: Path = Path("./.rag")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    phoenix: PhoenixSettings = Field(default_factory=PhoenixSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    corpus: CorpusSettings = Field(default_factory=CorpusSettings)

    @model_validator(mode="after")
    def _apply_profile_defaults(self) -> Self:
        # Fill only fields the user left unset, so explicit config always wins over the profile.
        prof = PROFILE_DEFAULTS[self.hardware_profile]
        self.embedding.model = self.embedding.model or prof.embedding_model
        self.reranker.model = self.reranker.model or prof.reranker_model
        self.reranker.candidates = self.reranker.candidates or prof.rerank_candidates
        self.dtype = self.dtype or prof.dtype
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
