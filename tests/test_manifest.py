from pathlib import Path

import pytest

from rag.ingestion.discovery import DiscoveredFile, doc_id_for
from rag.ingestion.manifest import FileStatus, Manifest
from rag.ingestion.models import ParentSection


def df(root: Path, name: str, content_hash: str = "h1") -> DiscoveredFile:
    path = (root / name).resolve()
    return DiscoveredFile(path=path, doc_id=doc_id_for(path), content_hash=content_hash, mtime=1.0, size=10)


def parent(pid: str, text: str = "parent text") -> ParentSection:
    return ParentSection(parent_id=pid, text=text, section_path=("A", "B"), page_start=1, page_end=2)


@pytest.fixture
def manifest(tmp_path: Path) -> Manifest:
    return Manifest(tmp_path / "state" / "rag.db")


def test_new_files_are_planned_for_indexing(manifest: Manifest, tmp_path: Path) -> None:
    files = [df(tmp_path, "a.md"), df(tmp_path, "b.md")]
    plan = manifest.plan(files, root=tmp_path)
    assert [f.path.name for f in plan.to_index] == ["a.md", "b.md"]
    assert plan.unchanged == [] and plan.removed == []


def test_indexed_unchanged_files_are_skipped(manifest: Manifest, tmp_path: Path) -> None:
    a = df(tmp_path, "a.md")
    manifest.record_indexed(a, parser="docling", parents=[parent("p1")], chunk_ids=["c1", "c2"])
    plan = manifest.plan([a], root=tmp_path)
    assert plan.to_index == []
    assert [f.path.name for f in plan.unchanged] == ["a.md"]


def test_changed_hash_is_reindexed(manifest: Manifest, tmp_path: Path) -> None:
    manifest.record_indexed(df(tmp_path, "a.md", "h1"), parser="docling", parents=[], chunk_ids=["c1"])
    plan = manifest.plan([df(tmp_path, "a.md", "h2")], root=tmp_path)
    assert [f.content_hash for f in plan.to_index] == ["h2"]


def test_missing_files_under_root_are_removed_but_other_roots_untouched(
    manifest: Manifest, tmp_path: Path
) -> None:
    other = tmp_path.parent / f"{tmp_path.name}-other"
    manifest.record_indexed(df(tmp_path, "gone.md"), parser="docling", parents=[], chunk_ids=["c1"])
    manifest.record_indexed(df(other, "elsewhere.md"), parser="docling", parents=[], chunk_ids=["c9"])
    plan = manifest.plan([], root=tmp_path)
    assert [r.path.name for r in plan.removed] == ["gone.md"]
    assert plan.removed[0].chunk_ids == ["c1"]


def test_root_prefix_does_not_match_sibling_dirs(manifest: Manifest, tmp_path: Path) -> None:
    sibling = tmp_path.parent / f"{tmp_path.name}2"
    manifest.record_indexed(df(sibling, "x.md"), parser="docling", parents=[], chunk_ids=[])
    assert manifest.plan([], root=tmp_path).removed == []


def test_single_file_root_never_removes_other_files(manifest: Manifest, tmp_path: Path) -> None:
    a, b = df(tmp_path, "a.md"), df(tmp_path, "b.md")
    manifest.record_indexed(b, parser="docling", parents=[], chunk_ids=[])
    plan = manifest.plan([a], root=a.path)
    assert plan.removed == []


def test_failed_files_are_recorded_and_not_retried_until_changed(manifest: Manifest, tmp_path: Path) -> None:
    a = df(tmp_path, "bad.pdf")
    manifest.record_failed(a, error="boom")
    rec = manifest.get(a.path)
    assert rec is not None and rec.status is FileStatus.FAILED and rec.error == "boom"
    assert manifest.plan([a], root=tmp_path).to_index == []
    assert manifest.plan([a], root=tmp_path, retry_failed=True).to_index == [a]
    assert manifest.plan([df(tmp_path, "bad.pdf", "h2")], root=tmp_path).to_index != []


def test_failure_after_success_clears_old_chunks_and_parents(manifest: Manifest, tmp_path: Path) -> None:
    a = df(tmp_path, "a.md")
    manifest.record_indexed(a, parser="docling", parents=[parent("p1")], chunk_ids=["c1"])
    manifest.record_failed(df(tmp_path, "a.md", "h2"), error="boom")
    rec = manifest.get(a.path)
    assert rec is not None and rec.chunk_ids == []
    assert manifest.get_parents(["p1"]) == {}


def test_reindex_replaces_chunks_and_parents(manifest: Manifest, tmp_path: Path) -> None:
    manifest.record_indexed(df(tmp_path, "a.md"), parser="docling", parents=[parent("p1")], chunk_ids=["c1"])
    manifest.record_indexed(
        df(tmp_path, "a.md", "h2"), parser="fallback", parents=[parent("p2", "new")], chunk_ids=["c2", "c3"]
    )
    rec = manifest.get((tmp_path / "a.md").resolve())
    assert rec is not None
    assert rec.chunk_ids == ["c2", "c3"] and rec.parser == "fallback" and rec.content_hash == "h2"
    assert set(manifest.get_parents(["p1", "p2"])) == {"p2"}


def test_parents_roundtrip(manifest: Manifest, tmp_path: Path) -> None:
    p = parent("p1", "full section text")
    manifest.record_indexed(df(tmp_path, "a.md"), parser="docling", parents=[p], chunk_ids=["c1"])
    assert manifest.get_parents(["p1", "missing"]) == {"p1": p}


def test_remove_deletes_record_and_parents(manifest: Manifest, tmp_path: Path) -> None:
    a = df(tmp_path, "a.md")
    manifest.record_indexed(a, parser="docling", parents=[parent("p1")], chunk_ids=["c1"])
    manifest.remove(a.path)
    assert manifest.get(a.path) is None
    assert manifest.get_parents(["p1"]) == {}


def test_state_persists_across_instances(tmp_path: Path) -> None:
    db = tmp_path / "rag.db"
    a = df(tmp_path, "a.md")
    Manifest(db).record_indexed(a, parser="docling", parents=[], chunk_ids=["c1"])
    assert Manifest(db).plan([a], root=tmp_path).to_index == []


def test_force_plans_everything(manifest: Manifest, tmp_path: Path) -> None:
    a = df(tmp_path, "a.md")
    manifest.record_indexed(a, parser="docling", parents=[], chunk_ids=[])
    manifest.record_failed(df(tmp_path, "b.md"), error="x")
    plan = manifest.plan([a, df(tmp_path, "b.md")], root=tmp_path, force=True)
    assert len(plan.to_index) == 2 and plan.unchanged == [] and plan.skipped_failed == []
