from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from rag.config import ChunkingSettings, IngestionSettings
from rag.ingestion.contextualize import ContextualizeResult
from rag.ingestion.manifest import FileStatus, Manifest
from rag.ingestion.models import Block, BlockKind, Chunk, ParsedDocument
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.types import IndexPoint, SparseVec


class FakeParser:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def parse(self, path: Path) -> ParsedDocument:
        self.calls.append(path)
        text = path.read_text()
        if "CORRUPT" in text:
            raise ValueError("cannot parse")
        return ParsedDocument(
            title=path.stem,
            blocks=[
                Block(BlockKind.HEADING, "Section", level=1, page=1),
                Block(BlockKind.TEXT, text, page=1),
            ],
            parser="fake",
        )


class FakeDense:
    dimension = 3

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.seen.extend(texts)
        return [[float(len(t)), 0.0, 1.0] for t in texts]


class FakeSparse:
    def embed_documents(self, texts: Sequence[str]) -> list[SparseVec]:
        return [SparseVec(indices=[len(t)], values=[1.0]) for t in texts]


class FakeIndex:
    def __init__(self) -> None:
        self.points: dict[str, IndexPoint] = {}
        self.upserts: list[list[IndexPoint]] = []
        self.deleted_docs: list[str] = []
        self.dim: int | None = None

    def exists(self) -> bool:
        return self.dim is not None

    def ensure_collection(self, dense_dim: int) -> None:
        self.dim = dense_dim

    def upsert(self, points: Sequence[IndexPoint]) -> None:
        self.upserts.append(list(points))
        self.points.update({p.id: p for p in points})

    def delete_doc(self, doc_id: str) -> None:
        self.deleted_docs.append(doc_id)
        self.points = {k: p for k, p in self.points.items() if p.payload["doc_id"] != doc_id}


class FakeContextualizer:
    def contextualize(self, docs: Sequence[tuple[ParsedDocument, list[Chunk]]]) -> ContextualizeResult:
        out = [[replace(c, context=f"CTX about {doc.title}") for c in chunks] for doc, chunks in docs]
        return ContextualizeResult(chunks=out, input_tokens=10, output_tokens=5)


class Harness:
    def __init__(self, tmp_path: Path, contextualizer: FakeContextualizer | None = None) -> None:
        self.root = tmp_path / "docs"
        self.root.mkdir()
        self.manifest = Manifest(tmp_path / "state" / "rag.db")
        self.parser = FakeParser()
        self.dense = FakeDense()
        self.index = FakeIndex()
        self.pipeline = IngestionPipeline(
            manifest=self.manifest,
            parser=self.parser,
            dense=self.dense,
            sparse=FakeSparse(),
            index=self.index,
            count_tokens=lambda s: len(s.split()),
            chunking=ChunkingSettings(),
            ingestion=IngestionSettings(contextualize=contextualizer is not None),
            contextualizer=contextualizer,
        )

    def write(self, name: str, text: str) -> Path:
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def doc_ids_in_index(self) -> set[str]:
        return {str(p.payload["doc_id"]) for p in self.index.points.values()}


@pytest.fixture
def h(tmp_path: Path) -> Harness:
    hh = Harness(tmp_path)
    hh.write("a.md", "alpha content")
    hh.write("sub/b.txt", "beta content")
    return hh


def test_first_run_indexes_every_file(h: Harness) -> None:
    stats = h.pipeline.run(h.root)
    assert (stats.indexed, stats.unchanged, stats.failed, stats.removed) == (2, 0, 0, 0)
    assert stats.chunks_written == len(h.index.points) == 2
    assert h.index.dim == 3


def test_second_run_is_a_noop(h: Harness) -> None:
    h.pipeline.run(h.root)
    h.parser.calls.clear()
    h.index.deleted_docs.clear()
    upserts_before = len(h.index.upserts)
    stats = h.pipeline.run(h.root)
    assert (stats.indexed, stats.unchanged) == (0, 2)
    assert h.parser.calls == []
    assert len(h.index.upserts) == upserts_before
    assert h.index.deleted_docs == []


def test_modified_file_is_the_only_one_reindexed(h: Harness) -> None:
    h.pipeline.run(h.root)
    h.parser.calls.clear()
    changed = h.write("a.md", "alpha content revised")
    stats = h.pipeline.run(h.root)
    assert stats.indexed == 1 and stats.unchanged == 1
    assert h.parser.calls == [changed.resolve()]
    texts = [str(p.payload["text"]) for p in h.index.points.values()]
    assert "alpha content revised" in texts and "alpha content" not in texts
    assert len(h.index.points) == 2


def test_deleted_file_chunks_are_removed(h: Harness) -> None:
    h.pipeline.run(h.root)
    gone = h.root / "sub" / "b.txt"
    rec = h.manifest.get(gone.resolve())
    assert rec is not None
    gone.unlink()
    stats = h.pipeline.run(h.root)
    assert stats.removed == 1
    assert rec.doc_id in h.index.deleted_docs
    assert rec.doc_id not in h.doc_ids_in_index()
    assert h.manifest.get(gone.resolve()) is None


def test_parse_failure_is_recorded_and_does_not_stop_the_run(h: Harness) -> None:
    h.write("bad.md", "CORRUPT")
    stats = h.pipeline.run(h.root)
    assert (stats.indexed, stats.failed) == (2, 1)
    rec = h.manifest.get((h.root / "bad.md").resolve())
    assert rec is not None and rec.status is FileStatus.FAILED and "cannot parse" in (rec.error or "")
    assert h.pipeline.run(h.root).skipped_failed == 1


def test_file_that_starts_failing_loses_its_stale_chunks(h: Harness) -> None:
    h.pipeline.run(h.root)
    rec = h.manifest.get((h.root / "a.md").resolve())
    assert rec is not None
    h.write("a.md", "CORRUPT now")
    h.pipeline.run(h.root)
    assert rec.doc_id not in h.doc_ids_in_index()


def test_payload_has_required_metadata(h: Harness) -> None:
    h.pipeline.run(h.root)
    point = next(p for p in h.index.points.values() if p.payload["file_name"] == "a.md")
    payload = point.payload
    for key in (
        "doc_id",
        "file_path",
        "file_name",
        "folder",
        "title",
        "section_path",
        "page_start",
        "page_end",
        "chunk_index",
        "parent_id",
        "content_hash",
        "ingested_at",
        "language",
        "text",
        "context",
    ):
        assert key in payload, key
    assert payload["file_path"] == str((h.root / "a.md").resolve())
    assert payload["section_path"] == ["Section"]
    assert payload["language"] == "en"
    assert h.manifest.get_parents([str(payload["parent_id"])]) != {}


def test_contextualization_feeds_embeddings_but_not_display_text(tmp_path: Path) -> None:
    h = Harness(tmp_path, contextualizer=FakeContextualizer())
    h.write("a.md", "alpha content")
    stats = h.pipeline.run(h.root)
    (point,) = h.index.points.values()
    assert point.payload["text"] == "alpha content"
    assert point.payload["context"] == "CTX about a"
    assert any(t.startswith("CTX about a") for t in h.dense.seen)
    assert stats.llm_input_tokens == 10 and stats.llm_output_tokens == 5


def test_contextualization_disabled_leaves_context_empty(h: Harness) -> None:
    h.pipeline.run(h.root)
    assert all(p.payload["context"] is None for p in h.index.points.values())


def test_missing_collection_forces_full_reindex(h: Harness) -> None:
    h.pipeline.run(h.root)
    h.index.points.clear()
    h.index.dim = None  # collection dropped while the manifest survived
    stats = h.pipeline.run(h.root)
    assert stats.indexed == 2
    assert len(h.index.points) == 2


def test_force_reindexes_unchanged_files(h: Harness) -> None:
    h.pipeline.run(h.root)
    assert h.pipeline.run(h.root, force=True).indexed == 2
