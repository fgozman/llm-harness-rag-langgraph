# LangGraph's overloads contain partially unknown types (e.g. CachePolicy[Unknown]) and ship no stubs for some
# modules; relax only those checks in this wiring module.
# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
"""Graph assembly, the SQLite checkpointer, and a helper that runs one conversational turn."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rag.graph.edges import route_after_analyze, route_after_check, route_after_generate, route_after_grade
from rag.graph.nodes import GraphDeps, Nodes
from rag.graph.state import STATE_MODELS, GraphState, Turn

__all__ = ["GraphDeps", "build_checkpointer", "build_graph", "run_turn", "state_serializer"]

EventCallback = Callable[[dict[str, Any]], None]


def build_graph(
    deps: GraphDeps, *, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> CompiledStateGraph[Any]:
    nodes = Nodes(deps)
    cfg = deps.settings
    graph = StateGraph(GraphState)
    graph.add_node("analyze_query", nodes.analyze_query)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("rerank", nodes.rerank)
    graph.add_node("grade_documents", nodes.grade_documents)
    graph.add_node("generate", nodes.generate)
    graph.add_node("check_answer", nodes.check_answer)
    graph.add_node("respond", nodes.respond)

    graph.add_edge(START, "analyze_query")
    graph.add_conditional_edges("analyze_query", route_after_analyze, ["rewrite_query", "respond"])
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        partial(
            route_after_grade,
            min_relevant=cfg.min_relevant_passages,
            max_attempts=1 + cfg.max_rewrite_retries,
        ),
        ["generate", "rewrite_query"],
    )
    graph.add_conditional_edges("generate", route_after_generate, ["check_answer", "respond"])
    graph.add_conditional_edges(
        "check_answer",
        partial(route_after_check, max_generations=1 + cfg.max_regenerations),
        ["respond", "generate"],
    )
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer)


def state_serializer() -> JsonPlusSerializer:
    # Register our state models explicitly: LangGraph is phasing out deserializing unregistered types.
    allowed = [(model.__module__, model.__name__) for model in STATE_MODELS]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


def build_checkpointer(path: Path) -> SqliteSaver:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn, serde=state_serializer())


def run_turn(
    graph: CompiledStateGraph[Any], question: str, *, thread_id: str, on_event: EventCallback | None = None
) -> dict[str, Any]:
    """Run one user turn on a thread; `on_event` receives status/token/final events as they happen."""
    config: Any = {"configurable": {"thread_id": thread_id}}
    turn = {"question": question, "history": [Turn(role="user", content=question)]}
    for mode, chunk in graph.stream(turn, config, stream_mode=["custom", "updates"]):
        if on_event is None:
            continue
        if mode == "custom" and isinstance(chunk, dict):
            on_event(chunk)
        elif mode == "updates" and isinstance(chunk, dict):
            for node in chunk:
                on_event({"type": "node", "node": str(node)})
    return dict(graph.get_state(config).values)
