from datetime import UTC, date, datetime

import pytest

from rag.ingestion.dates import DocumentDate, infer_document_date

MTIME = datetime(2026, 11, 22, 18, 0, tzinfo=UTC).timestamp()


def infer(name: str, text: str = "") -> DocumentDate:
    return infer_document_date(name, text, MTIME)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("amd-20260926.htm", date(2026, 9, 26)),
        ("report_2026_11_04.pdf", date(2026, 11, 4)),
        ("minutes 2026.03.15.docx", date(2026, 3, 15)),
        ("vellmark_incident_review_2026-09.docx", date(2026, 9, 1)),
        # Several dates: the latest is when the document was issued (filed after the period it covers).
        ("UBER_10-Q_period-2026-09-30_filed-2026-11-04.htm", date(2026, 11, 4)),
    ],
)
def test_date_from_file_name(name: str, expected: date) -> None:
    assert infer(name, "Written on January 2, 2020.") == DocumentDate(expected, "name")


@pytest.mark.parametrize(
    "name",
    [
        "2026-21920.html",  # Federal Register document number, not a date
        "2608.01147.pdf",  # arXiv id
        "gao-27-108458.pdf",
        "notes-2026-13-01.md",  # month 13
        "x-20260231.md",  # 31 February
    ],
)
def test_names_without_a_valid_date_fall_through_to_content(name: str) -> None:
    assert infer(name, "November 19, 2026\n\nCongressional Committees") == DocumentDate(
        date(2026, 11, 19), "content"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SAN FRANCISCO--(BUSINESS WIRE)--Nov. 4, 2026--Uber today announced", date(2026, 11, 4)),
        ("arXiv:2608.01147v1 [cs.IR] 5 Aug 2026", date(2026, 8, 5)),
        ("For the quarterly period ended September 30, 2026", date(2026, 9, 30)),
        ("Federal Register document 2026-21920, published 2026-11-20 by Justice", date(2026, 11, 20)),
        ("Effective Sept. 1, 2026 the rule applies", date(2026, 9, 1)),
        ("Signed 3rd March 2026", date(2026, 3, 3)),
        # Month-year (cover pages, slide decks) counts as the 1st of the month.
        ("Report to Congressional Committees\nNovember 2026\nFEDERAL REAL PROPERTY", date(2026, 11, 1)),
        ("Prepared for the Board, October 2026\nTern inspection on 14 January 2027", date(2026, 10, 1)),
    ],
)
def test_first_date_in_opening_text(text: str, expected: date) -> None:
    assert infer("document.pdf", text) == DocumentDate(expected, "content")


def test_invalid_content_dates_are_skipped() -> None:
    assert infer("d.pdf", "February 30, 2026 was a typo; issued March 2, 2026.") == DocumentDate(
        date(2026, 3, 2), "content"
    )


def test_compact_digit_runs_in_content_are_not_dates() -> None:
    assert infer("d.pdf", "Account 20261104 closed.").source == "mtime"


def test_only_the_opening_text_is_searched() -> None:
    late = "x " * 2000 + "November 19, 2026"
    assert infer("d.pdf", late).source == "mtime"


def test_falls_back_to_modification_time() -> None:
    assert infer("notes.txt", "no dates here") == DocumentDate(date(2026, 11, 22), "mtime")


def test_timestamp_is_utc_midnight_of_the_date() -> None:
    assert DocumentDate(date(2026, 11, 4), "name").timestamp == datetime(2026, 11, 4, tzinfo=UTC).timestamp()


def test_month_year_needs_a_capitalised_month() -> None:
    # "may" is a common word; only "May 2026" (capitalised) is read as a month.
    assert infer("d.pdf", "results may 2026 vary").source == "mtime"
    assert infer("d.pdf", "Issued May 2026").value == date(2026, 5, 1)


def test_full_date_wins_over_month_year_inside_it() -> None:
    assert infer("d.pdf", "Signed 5 May 2026").value == date(2026, 5, 5)
