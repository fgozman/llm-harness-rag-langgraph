import pytest

from rag.graph.edges import route_after_analyze, route_after_check, route_after_generate, route_after_grade
from rag.graph.state import AnswerCheck, GradeReport, GraphState, PassageGrade, QueryAnalysis


def analysis(intent: str) -> QueryAnalysis:
    return QueryAnalysis.model_validate({"intent": intent, "is_complex": False, "direct_reply": None})


def grades(*relevant: bool, sufficient: bool = False) -> GradeReport:
    return GradeReport(
        grades=[PassageGrade(index=i + 1, relevant=r, reason="") for i, r in enumerate(relevant)],
        sufficient=sufficient,
        missing_information="",
    )


def check(grounded: bool = True, addresses: bool = True, abstention: bool = False) -> AnswerCheck:
    return AnswerCheck(
        grounded=grounded,
        addresses_question=addresses,
        is_abstention=abstention,
        supported_claims=[],
        unsupported_claims=[],
        critique="",
    )


@pytest.mark.parametrize(
    ("intent", "expected"),
    [("retrieval", "rewrite_query"), ("chitchat", "respond"), ("out_of_scope", "respond")],
)
def test_after_analyze(intent: str, expected: str) -> None:
    assert route_after_analyze(GraphState(question="q", analysis=analysis(intent))) == expected


@pytest.mark.parametrize(
    ("relevant", "attempts", "expected"),
    [
        ((True, True, False), 1, "generate"),  # enough relevant passages
        ((True, False, False), 1, "rewrite_query"),  # too few -> retry with feedback
        ((True, False, False), 2, "rewrite_query"),
        ((True, False, False), 3, "generate"),  # cap reached (1 initial + 2 retries): proceed anyway
        ((), 3, "generate"),  # nothing found even after retries: generate abstains
        ((), 1, "rewrite_query"),
    ],
)
def test_after_grade_with_retry_cap(relevant: tuple[bool, ...], attempts: int, expected: str) -> None:
    state = GraphState(question="q", grades=grades(*relevant), retrieval_attempts=attempts)
    assert route_after_grade(state, min_relevant=2, max_attempts=3) == expected


@pytest.mark.parametrize(
    ("relevant", "sufficient", "expected"),
    [
        ((True, False, False), True, "generate"),  # one passage fully answers a single-fact question
        (
            (False, False, False),
            True,
            "rewrite_query",
        ),  # "sufficient" without any relevant passage is ignored
    ],
)
def test_after_grade_single_sufficient_passage(
    relevant: tuple[bool, ...], sufficient: bool, expected: str
) -> None:
    state = GraphState(question="q", grades=grades(*relevant, sufficient=sufficient), retrieval_attempts=1)
    assert route_after_grade(state, min_relevant=2, max_attempts=3) == expected


def test_after_generate_skips_check_for_deterministic_abstention() -> None:
    assert route_after_generate(GraphState(question="q", abstained=True)) == "respond"
    assert route_after_generate(GraphState(question="q", abstained=False)) == "check_answer"


@pytest.mark.parametrize(
    ("result", "generations", "expected"),
    [
        (check(), 1, "respond"),
        (check(abstention=True, grounded=False), 1, "respond"),  # an honest "not found" needs no rewrite
        (check(grounded=False), 1, "generate"),  # regenerate once with the critique
        (check(addresses=False), 1, "generate"),
        (check(grounded=False), 2, "respond"),  # cap: return a cautious answer instead
    ],
)
def test_after_check_with_regeneration_cap(result: AnswerCheck, generations: int, expected: str) -> None:
    state = GraphState(question="q", check=result, generation_count=generations)
    assert route_after_check(state, max_generations=2) == expected
