"""Device and dtype selection for the native embedding/reranking models."""

from __future__ import annotations

import os

# Must be set before torch initializes MPS so unsupported ops fall back to CPU instead of raising.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# HF tokenizers' parallel batch encoding (Rayon) deadlocked mid-ingest next to Docling's thread pools;
# sequential encoding costs little compared to the model forward pass.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging

import torch

from rag.config import Device, DType

log = logging.getLogger(__name__)

ConcreteDevice = str
# flash_attention_2 is CUDA-only, an optional extra, and must never be used on MPS; SDPA works everywhere.
ATTN_IMPLEMENTATION = "sdpa"
_DTYPES: dict[DType, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _available(device: str) -> bool:
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return torch.backends.mps.is_available()
    return device == "cpu"


def resolve_device(requested: Device) -> ConcreteDevice:
    if requested != "auto":
        if not _available(requested):
            raise RuntimeError(f"device {requested!r} was requested but is not available")
        return requested
    return next(d for d in ("cuda", "mps", "cpu") if _available(d))


def resolve_dtype(device: ConcreteDevice, requested: DType) -> torch.dtype:
    if device == "mps" and requested == "bfloat16":
        log.info("bfloat16 is not native on Apple M1; using float16")
        return torch.float16
    if device == "cpu" and requested != "float32":
        log.info("half precision is slow on CPU; using float32")
        return torch.float32
    return _DTYPES[requested]
