import threading
from typing import Any

import numpy as np

from rag.embeddings import QwenDenseEmbedder


class FakeModel:
    def __init__(self, lock: threading.Lock) -> None:
        self.lock = lock
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], **_: Any) -> np.ndarray:
        assert self.lock.locked(), "encode must run under the device lock"
        self.calls.append(texts)
        return np.array([[float(len(t))] for t in texts], dtype=np.float16)


def test_documents_are_embedded_one_locked_batch_at_a_time() -> None:
    embedder = QwenDenseEmbedder.__new__(QwenDenseEmbedder)
    embedder._lock = threading.Lock()  # pyright: ignore[reportPrivateUsage]
    model = FakeModel(embedder._lock)  # pyright: ignore[reportPrivateUsage]
    embedder._model = model  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    embedder._batch_size = 2  # pyright: ignore[reportPrivateUsage]
    vectors = embedder.embed_documents(["a", "bb", "ccc", "dddd", "eeeee"])
    assert model.calls == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert not embedder._lock.locked()  # pyright: ignore[reportPrivateUsage]
