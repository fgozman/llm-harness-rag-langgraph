"""Process-wide LLM usage accounting (tokens per model), used for per-query cost in evaluations."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace

from rag.config import ModelPricing


@dataclass(frozen=True, slots=True)
class ModelUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class UsageMeter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_model: dict[str, ModelUsage] = {}

    def record(
        self,
        model: str,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        with self._lock:
            u = self._by_model.get(model, ModelUsage())
            self._by_model[model] = replace(
                u,
                calls=u.calls + 1,
                input_tokens=u.input_tokens + input_tokens,
                output_tokens=u.output_tokens + output_tokens,
                cache_read_tokens=u.cache_read_tokens + cache_read_tokens,
                cache_write_tokens=u.cache_write_tokens + cache_write_tokens,
            )

    def snapshot(self) -> dict[str, ModelUsage]:
        with self._lock:
            return dict(self._by_model)

    def reset(self) -> None:
        with self._lock:
            self._by_model.clear()

    def total_tokens(self) -> int:
        return sum(
            u.input_tokens + u.output_tokens + u.cache_read_tokens + u.cache_write_tokens
            for u in self.snapshot().values()
        )

    def cost_usd(self, pricing: Mapping[str, ModelPricing]) -> float:
        total = 0.0
        for model, u in self.snapshot().items():
            p = pricing.get(model)
            if p is None:
                continue  # local/unpriced models
            # Cache writes priced at the 5-minute rate: interactive calls don't request the 1h TTL.
            total += (
                u.input_tokens * p.input
                + u.output_tokens * p.output
                + u.cache_read_tokens * p.cache_read
                + u.cache_write_tokens * p.cache_write_5m
            )
        return total / 1_000_000


# Shared by all LLM adapters in the process.
METER = UsageMeter()
