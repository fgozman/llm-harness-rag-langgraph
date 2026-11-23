"""Phase 3 end-to-end: real LLM (configured provider), embeddings, reranker, Qdrant, SQLite checkpointer."""

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rag.config import Settings
from rag.factory import build_index, build_ingestion_pipeline, build_qa_graph
from rag.graph.build import run_turn
from rag.graph.citations import NOT_FOUND

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def graph(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    mp = pytest.MonkeyPatch()
    mp.chdir(REPO)
    s = Settings()
    s.qdrant.collection = f"test_graph_{uuid.uuid4().hex[:8]}"
    s.state_dir = tmp_path_factory.mktemp("state")
    s.ingestion.contextualize = False
    assert build_ingestion_pipeline(s).run(REPO / "tests" / "fixtures" / "docs").failed == 0
    yield build_qa_graph(s)
    build_index(s).client.delete_collection(s.qdrant.collection)
    mp.undo()


def ask(graph: Any, question: str, thread: str) -> tuple[Any, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    state = run_turn(graph, question, thread_id=thread, on_event=events.append)
    return state["final"], events


def test_answer_is_grounded_cited_and_streamed(graph: Any) -> None:
    final, events = ask(graph, "What is the payload of the NRX-4471-B gripper?", "e2e-1")
    assert final.status == "answered"
    assert "7.5" in final.text
    assert any(c.file_name == "roadmap_q3.pptx" and c.page_start == 2 for c in final.citations)
    assert any(e["type"] == "token" for e in events)


def test_unanswerable_question_says_not_found(graph: Any) -> None:
    final, _ = ask(graph, "What is the warranty period of the NRX-9000 vacuum pump?", "e2e-2")
    assert final.status == "abstained"
    assert NOT_FOUND in final.text


def test_follow_up_resolves_reference_from_history(graph: Any) -> None:
    ask(graph, "Which firmware versions are described in the release notes, newest first?", "e2e-3")
    final, _ = ask(graph, "What changed in the second one?", "e2e-3")
    assert final.status == "answered"
    assert "collision" in final.text.lower()  # 4.3.0 added collision detection sensitivity levels
    assert any(c.file_name == "release_notes.txt" for c in final.citations)
