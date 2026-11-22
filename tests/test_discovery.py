import hashlib
from pathlib import Path

from rag.ingestion.discovery import SUPPORTED_EXTENSIONS, discover, doc_id_for


def touch(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_finds_supported_files_recursively_in_stable_order(tmp_path: Path) -> None:
    touch(tmp_path / "b.md")
    touch(tmp_path / "sub" / "deeper" / "a.PDF")
    touch(tmp_path / "c.txt")
    touch(tmp_path / "notes.json")
    found = [f.path.relative_to(tmp_path).as_posix() for f in discover(tmp_path)]
    assert found == ["b.md", "c.txt", "sub/deeper/a.PDF"]


def test_skips_hidden_and_office_lock_files(tmp_path: Path) -> None:
    touch(tmp_path / ".hidden.md")
    touch(tmp_path / ".git" / "x.md")
    touch(tmp_path / "~$report.docx")
    touch(tmp_path / "ok.docx")
    assert [f.path.name for f in discover(tmp_path)] == ["ok.docx"]


def test_hash_and_metadata(tmp_path: Path) -> None:
    p = touch(tmp_path / "a.md", "hello")
    (f,) = discover(tmp_path)
    assert f.path == p.resolve()
    assert f.content_hash == hashlib.sha256(b"hello").hexdigest()
    assert f.size == 5
    assert f.doc_id == doc_id_for(p.resolve())


def test_doc_id_is_stable_per_path(tmp_path: Path) -> None:
    assert doc_id_for(tmp_path / "a.md") == doc_id_for(tmp_path / "a.md")
    assert doc_id_for(tmp_path / "a.md") != doc_id_for(tmp_path / "b.md")


def test_accepts_a_single_file(tmp_path: Path) -> None:
    p = touch(tmp_path / "a.md")
    assert [f.path for f in discover(p)] == [p.resolve()]


def test_all_spec_formats_supported() -> None:
    assert {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md", ".txt"} <= SUPPORTED_EXTENSIONS
