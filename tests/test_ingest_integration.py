"""Phase 1 acceptance, automated: real parsers, embedders and Qdrant (no LLM)."""

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from rag.config import Settings
from rag.factory import build_dense, build_index, build_ingestion_pipeline
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.qdrant_index import QdrantChunkIndex

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "docs"

pytestmark = pytest.mark.integration


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Settings]:
    monkeypatch.chdir(REPO)
    s = Settings()
    s.qdrant.collection = f"test_ingest_{uuid.uuid4().hex[:8]}"
    s.state_dir = tmp_path / "state"
    s.ingestion.contextualize = False
    yield s
    index = build_index(s)
    if index.exists():
        index.client.delete_collection(s.qdrant.collection)


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    shutil.copytree(FIXTURES, root)
    return root


def file_counts(index: QdrantChunkIndex) -> dict[str, int]:
    points, _ = index.client.scroll(index.collection, limit=10_000, with_payload=["file_name"])
    counts: dict[str, int] = {}
    for p in points:
        name = str((p.payload or {})["file_name"])
        counts[name] = counts.get(name, 0) + 1
    return counts


def test_ingest_is_incremental_end_to_end(settings: Settings, docs: Path) -> None:
    pipeline: IngestionPipeline = build_ingestion_pipeline(settings)
    index = build_index(settings)

    first = pipeline.run(docs)
    assert (first.indexed, first.failed) == (7, 0)
    counts = file_counts(index)
    assert set(counts) == {p.name for p in docs.rglob("*") if p.is_file()}
    assert sum(counts.values()) == first.chunks_written

    again = pipeline.run(docs)
    assert (again.indexed, again.unchanged, again.removed) == (0, 7, 0)
    assert file_counts(index) == counts

    (docs / "release_notes.txt").write_text("Version 9.9.9\nA brand new release.\n")
    changed = pipeline.run(docs)
    assert (changed.indexed, changed.unchanged) == (1, 6)
    after = file_counts(index)
    assert {k: v for k, v in after.items() if k != "release_notes.txt"} == {
        k: v for k, v in counts.items() if k != "release_notes.txt"
    }

    (docs / "shipments.xlsx").unlink()
    removed = pipeline.run(docs)
    assert removed.removed == 1
    assert "shipments.xlsx" not in file_counts(index)


def test_query_instruction_applies_to_queries_only(settings: Settings) -> None:
    dense = build_dense(settings)
    text = "payload of the NRX-4471-B gripper"
    as_query = np.array(dense.embed_query(text))
    as_doc = np.array(dense.embed_documents([text])[0])
    assert as_query.shape == as_doc.shape == (dense.dimension,)
    assert float(as_query @ as_doc) < 0.99, "the instruction prefix must change the query embedding"
