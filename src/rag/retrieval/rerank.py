# transformers/sentence-transformers are partially untyped; relax only the "unknown" checks in this adapter.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Rerankers: Qwen3-Reranker scores P("yes") from an LLM (model-card recipe); BGE is a cross-encoder."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

from rag import devices  # sets PYTORCH_ENABLE_MPS_FALLBACK / TOKENIZERS_PARALLELISM before torch loads
from rag.config import Device, DType

log = logging.getLogger(__name__)

# Verbatim from the Qwen3-Reranker model card.
_QWEN_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_qwen_pair(instruction: str, query: str, document: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


class Qwen3Reranker:
    def __init__(
        self,
        model_name: str,
        *,
        device: Device,
        dtype: DType,
        batch_size: int,
        max_length: int,
        instruction: str,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.device = devices.resolve_device(device)
        torch_dtype = devices.resolve_dtype(self.device, dtype)
        log.info("loading %s on %s (%s)", model_name, self.device, torch_dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            attn_implementation=devices.ATTN_IMPLEMENTATION,
            device_map=self.device,
        ).eval()
        self._yes = self._tokenizer.convert_tokens_to_ids("yes")
        self._no = self._tokenizer.convert_tokens_to_ids("no")
        self._prefix = self._tokenizer.encode(_QWEN_PREFIX, add_special_tokens=False)
        self._suffix = self._tokenizer.encode(_QWEN_SUFFIX, add_special_tokens=False)
        self._batch_size = batch_size
        self._max_length = max_length
        self._instruction = instruction
        # Shared across request threads; HF fast tokenizers fail ("Already borrowed") under concurrent use.
        self._lock = threading.Lock()

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        pairs = [format_qwen_pair(self._instruction, query, d) for d in documents]
        # Score similar lengths together to minimise padding, then restore input order.
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i]))
        scores = [0.0] * len(pairs)
        with self._lock:
            for start in range(0, len(order), self._batch_size):
                batch = order[start : start + self._batch_size]
                for i, score in zip(batch, self._score([pairs[i] for i in batch]), strict=True):
                    scores[i] = score
        return scores

    def _score(self, pairs: list[str]) -> list[float]:
        torch = self._torch
        enc = self._tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self._max_length - len(self._prefix) - len(self._suffix),
        )
        enc["input_ids"] = [self._prefix + ids + self._suffix for ids in enc["input_ids"]]
        # The model card also passes max_length here; it is ignored with padding=True (and only warns).
        inputs = self._tokenizer.pad(enc, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits[:, -1, :]
        pair = torch.stack([logits[:, self._no], logits[:, self._yes]], dim=1).float()
        return torch.nn.functional.log_softmax(pair, dim=1)[:, 1].exp().tolist()


class CrossEncoderReranker:
    def __init__(
        self, model_name: str, *, device: Device, dtype: DType, batch_size: int, max_length: int
    ) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.device = devices.resolve_device(device)
        torch_dtype = devices.resolve_dtype(self.device, dtype)
        log.info("loading %s on %s (%s)", model_name, self.device, torch_dtype)
        self._model = CrossEncoder(
            model_name, device=self.device, max_length=max_length, model_kwargs={"dtype": torch_dtype}
        )
        self._batch_size = batch_size
        self._sigmoid = torch.nn.Sigmoid()
        self._lock = threading.Lock()

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        with self._lock:
            scores = self._model.predict(
                [(query, d) for d in documents], batch_size=self._batch_size, activation_fn=self._sigmoid
            )
        return [float(s) for s in scores]


def is_qwen3_reranker(model_name: str) -> bool:
    return "qwen3-reranker" in model_name.lower()
