import zipfile
from datetime import UTC, date, datetime
from pathlib import Path
from xml.etree import ElementTree

import pymupdf
import pytest

from rag.config import NumericDateOrder
from rag.ingestion.dates import DocumentDate, infer_document_date, metadata_date

MTIME = datetime(2026, 11, 22, 18, 0, tzinfo=UTC).timestamp()


def infer(
    name: str, text: str = "", *, created: date | None = None, numeric_order: NumericDateOrder | None = None
) -> DocumentDate:
    return infer_document_date(name, text, MTIME, created=lambda: created, numeric_order=numeric_order)


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
        ("Federal Register document 2026-21920, published 2026-11-20 by Justice", date(2026, 11, 20)),
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


# --- dates that are not the issue date ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "For the quarterly period ended September 30, 2026",
        "For the three months ended September 30, 2026",
        "Effective Sept. 1, 2026 the rule applies",
        "Effective as of 2026-09-01, the rule applies",
        "Effective date: 1 September 2026",  # nor the "September 2026" inside it
        "Smith et al., cited 2026-10-02",
    ],
)
def test_dates_after_effective_period_ended_or_cited_are_skipped(text: str) -> None:
    assert infer("d.pdf", text).source == "mtime"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("For the fiscal year ending December 31, 2026. Filed Feb 2, 2027", date(2027, 2, 2)),
        ("Balance as of December 31, 2025. Released March 3, 2026", date(2026, 3, 3)),
        ("Effective on and after 1 May 2026. Issued 3 April 2026", date(2026, 4, 3)),
        ("Effective date of this policy is 1 May 2026; issued 3 April 2026", date(2026, 4, 3)),
        # An issue cue beats an earlier uncued date.
        ("Meeting of 2 March 2026, minutes published 9 March 2026", date(2026, 3, 9)),
        ("Effective immediately. Date: 2026-04-03", date(2026, 4, 3)),
    ],
)
def test_issue_cues_win_and_skip_cues_lose(text: str, expected: date) -> None:
    assert infer("d.pdf", text) == DocumentDate(expected, "content")


def test_effective_date_label_is_a_skip_cue_not_an_issue_cue() -> None:
    assert infer("d.pdf", "Effective date: 2026-05-01").source == "mtime"


def test_the_issue_date_is_found_past_a_skipped_one() -> None:
    text = "For the quarterly period ended September 30, 2026\nFiled November 4, 2026"
    assert infer("d.pdf", text) == DocumentDate(date(2026, 11, 4), "content")


# --- numeric dates -------------------------------------------------------------------------------------


def test_numeric_dates_are_skipped_without_an_order() -> None:
    assert infer("d.pdf", "Issued 05/06/2026").source == "mtime"


@pytest.mark.parametrize(
    ("text", "order", "expected"),
    [
        ("Issued 05/06/2026", "dmy", date(2026, 6, 5)),
        ("Issued 05/06/2026", "mdy", date(2026, 5, 6)),
        ("Issued 5.6.2026.", "dmy", date(2026, 6, 5)),
        ("Issued 12-31-2026", "mdy", date(2026, 12, 31)),
    ],
)
def test_numeric_dates_follow_the_configured_order(
    text: str, order: NumericDateOrder, expected: date
) -> None:
    assert infer("d.pdf", text, numeric_order=order) == DocumentDate(expected, "content")


@pytest.mark.parametrize(
    "text",
    [
        "Issued 12-31-2026",  # month 31 in dmy
        "Issued 05/06-2026",  # mixed separators
        "Version 1.05.06.2026",  # part of a longer dotted number
        "Ref 05/06/20261",
        "v1.2.2020",  # version string
        "Tel 01.02.2026",  # phone number
    ],
)
def test_invalid_or_embedded_numeric_dates_are_skipped(text: str) -> None:
    assert infer("d.pdf", text, numeric_order="dmy").source == "mtime"


# --- file metadata ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "created",
    [
        date(1970, 1, 1),  # epoch placeholder
        date(1980, 1, 1),
        date(2013, 12, 23),  # python-docx template default
        date(2026, 12, 25),  # after the file was last modified
    ],
)
def test_implausible_creation_dates_fall_back_to_mtime(created: date) -> None:
    assert infer("d.pdf", "no dates here", created=created).source == "mtime"


def test_metadata_is_read_only_when_name_and_text_give_no_date() -> None:
    def unread() -> date | None:
        raise AssertionError("metadata read although the name dates the file")

    assert (
        infer_document_date("d_2026-01-05.pdf", "", MTIME, created=unread, numeric_order=None).source
        == "name"
    )


def test_metadata_date_comes_after_content_and_before_mtime() -> None:
    created = date(2026, 10, 12)
    assert infer("d.pdf", "no dates here", created=created) == DocumentDate(created, "metadata")
    assert infer("d.pdf", "Issued November 4, 2026", created=created).source == "content"
    assert infer("d_2026-01-05.pdf", created=created).source == "name"


def _pdf(path: Path, creation_date: str) -> Path:
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.set_metadata({"creationDate": creation_date})  # pyright: ignore[reportUnknownMemberType]
        pdf.save(path)  # pyright: ignore[reportUnknownMemberType]
    return path


def _docx(path: Path, core_xml: str | None) -> Path:
    with zipfile.ZipFile(path, "w") as package:
        package.writestr("word/document.xml", "<w:document/>")
        if core_xml is not None:
            package.writestr("docProps/core.xml", core_xml)
    return path


_CORE = (
    '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
    'xmlns:dcterms="http://purl.org/dc/terms/">{}</cp:coreProperties>'
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("D:20261104093000+01'00'", date(2026, 11, 4)),
        ("20261104", date(2026, 11, 4)),  # "D:" prefix missing
        ("", None),
        ("x", None),
    ],
)
def test_pdf_creation_date(tmp_path: Path, value: str, expected: date | None) -> None:
    assert metadata_date(_pdf(tmp_path / "d.pdf", value)) == expected


@pytest.mark.parametrize(
    ("core", "expected"),
    [
        (_CORE.format("<dcterms:created>2026-10-12T09:00:00Z</dcterms:created>"), date(2026, 10, 12)),
        (_CORE.format(""), None),
        (None, None),  # core properties are optional
    ],
)
def test_docx_creation_date(tmp_path: Path, core: str | None, expected: date | None) -> None:
    assert metadata_date(_docx(tmp_path / "d.docx", core)) == expected


def test_malformed_docx_metadata_raises(tmp_path: Path) -> None:
    with pytest.raises(ElementTree.ParseError):
        metadata_date(_docx(tmp_path / "d.docx", "<not xml"))


def test_other_formats_have_no_metadata_date(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("x")
    assert metadata_date(path) is None
