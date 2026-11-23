# LangGraph's overloads contain partially unknown types (e.g. CachePolicy[Unknown]) and ship no stubs for some
# modules; relax only those checks in this wiring module.
# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
"""Graph assembly, the SQLite checkpointer, and a helper that runs one conversational turn."""

from __future__ import annotations

import functools
import sqlite3
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from openinference.instrumentation import using_attributes
from openinference.instrumentation.langchain import LangChainInstrumentor, get_current_span
from opentelemetry import trace

from rag import logs
from rag.graph.edges import route_after_analyze, route_after_check, route_after_generate, route_after_grade
from rag.graph.nodes import GraphDeps, Nodes
from rag.graph.state import STATE_MODELS, GraphState, Turn
from rag.graph.stream import TurnStream
from rag.prompts import analyze, check, generate, grade, rewrite

PROMPT_VERSIONS = {
    "analyze": analyze.VERSION,
    "rewrite": rewrite.VERSION,
    "grade": grade.VERSION,
    "generate": generate.VERSION,
    "check": check.VERSION,
}

__all__ = ["GraphDeps", "build_checkpointer", "build_graph", "run_turn", "state_serializer", "turn_stream"]

EventCallback = Callable[[dict[str, Any]], None]


class _NodeFn(Protocol):
    def __call__(self, state: GraphState) -> dict[str, Any]: ...


def _in_node_span(fn: _NodeFn) -> _NodeFn:
    """Run a node with its LangChain-instrumented span as the active OTel span.

    The instrumentor creates node spans from callbacks without activating them in the worker thread, so spans
    opened inside the node (retriever, reranker, LLM SDK calls) would otherwise start new traces.
    """

    @functools.wraps(fn)
    def wrapper(state: GraphState) -> dict[str, Any]:
        # Without instrumentation (tracing off / Phoenix down) there is no node span, and asking for it warns.
        if not LangChainInstrumentor().is_instrumented_by_opentelemetry:
            return fn(state)
        node_span = get_current_span()
        if node_span is None:
            return fn(state)
        with trace.use_span(node_span, end_on_exit=False):
            return fn(state)

    return wrapper


def build_graph(
    deps: GraphDeps, *, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> CompiledStateGraph[Any]:
    nodes = Nodes(deps)
    cfg = deps.settings
    graph = StateGraph(GraphState)
    for name, fn in (
        ("analyze_query", nodes.analyze_query),
        ("rewrite_query", nodes.rewrite_query),
        ("retrieve", nodes.retrieve),
        ("rerank", nodes.rerank),
        ("grade_documents", nodes.grade_documents),
        ("generate", nodes.generate),
        ("check_answer", nodes.check_answer),
        ("respond", nodes.respond),
    ):
        graph.add_node(name, _in_node_span(fn))

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
    graph: CompiledStateGraph[Any],
    question: str,
    *,
    thread_id: str,
    on_event: EventCallback | None = None,
    user_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one user turn on a thread; `on_event` receives status/token/final events as they happen.

    Every span of the turn carries session.id = thread_id (Phoenix groups a conversation's turns by it).
    """
    config: Any = {"configurable": {"thread_id": thread_id}}
    turn = {"question": question, "history": [Turn(role="user", content=question)]}
    trace_metadata = {**(metadata or {}), "prompt_versions": PROMPT_VERSIONS}
    with using_attributes(session_id=thread_id, user_id=user_id, metadata=trace_metadata):
        for mode, chunk in graph.stream(turn, config, stream_mode=["custom", "updates"]):
            if on_event is None:
                continue
            if mode == "custom" and isinstance(chunk, dict):
                on_event(chunk)
            elif mode == "updates" and isinstance(chunk, dict):
                for node in chunk:
                    on_event({"type": "node", "node": str(node)})
    return dict(graph.get_state(config).values)


def turn_stream(
    graph: CompiledStateGraph[Any],
    question: str,
    *,
    thread_id: str,
    timeout_s: float,
    user_id: str = "",
    metadata: dict[str, Any] | None = None,
    on_finish: Callable[[], None] | None = None,
) -> TurnStream:
    """`run_turn` in a worker thread with an end-to-end deadline; iterate `.events()`, then read `.result`."""

    def run(on_event: EventCallback) -> dict[str, Any]:
        # Bound inside the worker: log lines of the graph's nodes carry the conversation id.
        with logs.bind(thread_id=thread_id):
            return run_turn(
                graph, question, thread_id=thread_id, on_event=on_event, user_id=user_id, metadata=metadata
            )

    return TurnStream(run, timeout_s=timeout_s, on_finish=on_finish)
