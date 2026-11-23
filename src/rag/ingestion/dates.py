"""Document date: from the file name, else from the opening text, else the PDF/DOCX creation date (when
plausible), else the file's modification time (unreliable: a copy or git checkout resets it).

The date answers "when was this document issued", which is what date filters ("documents from Q3 2026") need.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal, cast
from xml.etree import ElementTree

from rag.config import NumericDateOrder

DateSource = Literal["name", "content", "metadata", "mtime"]

# Dates are only searched near the start of a document: that's where letters, filings, press releases and
# papers state their issue date; deeper text mostly cites other dates.
CONTENT_WINDOW_CHARS = 3000

_YEAR = r"(?<!\d)((?:19|20)\d{2})"
_MONTH_NUM = r"(0[1-9]|1[0-2])"
_DAY_NUM = r"(0[1-9]|[12]\d|3[01])"
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip
_MONTH_NAME = (
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?"
    r"|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)

# File names: separated or compact full dates, and year-month.
_NAME_FULL = re.compile(_YEAR + r"[-_.]?" + _MONTH_NUM + r"[-_.]?" + _DAY_NUM + r"(?!\d)")
_NAME_MONTH = re.compile(_YEAR + r"[-_]" + _MONTH_NUM + r"(?![-_.]?\d)")
# Content: ISO dates and written dates; compact digit runs are too often IDs to trust here.
_CONTENT_ISO = re.compile(_YEAR + r"-" + _MONTH_NUM + r"-" + _DAY_NUM + r"(?!\d)")
_CONTENT_MDY = re.compile(r"\b" + _MONTH_NAME + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+" + _YEAR, re.I)
# Month-year ("November 2026" on report covers / slides) -> 1st of the month. Case-sensitive: "may" is a word.
_CONTENT_MY = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    + _YEAR
    + r"(?!\d)"
)
_CONTENT_DMY = re.compile(r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?\s+" + _MONTH_NAME + r",?\s+" + _YEAR, re.I)
# All-numeric with the year last and one separator throughout (05/06/2026, 5.6.2026, 05-06-2026); not part
# of a longer token (v1.2.2020, 1.05.06.2026).
_CONTENT_NUMERIC = re.compile(r"(?<![\w/.-])(\d{1,2})([/.-])(\d{1,2})\2((?:19|20)\d{2})(?![/.-]?\d)")

# Words right before a date that say what it is. Issue cues mark the document's own date; skip cues mark a
# date about something else (when a rule applies, the period covered, a balance date, a reference, a phone).
_ISSUE_CUE = re.compile(
    r"\b(?:issued|dated|(?<!effective )date|published|filed|signed|released|updated|revised)(?:\s+on)?"
    r"\s*[:,]?\s*$",
    re.I,
)
_SKIP_CUE = re.compile(
    # effective, effective as of, effective on and after, effective date of this policy is
    r"(?:\beffective(?:\s+\w+){0,5}"
    r"|\b(?:periods?|quarters?|years?|months?|weeks?)\s+(?:ended|ending)"
    r"|\bas\s+(?:of|at)|\bthrough|\buntil|\bon\s+and\s+after"
    r"|\b(?:cited|accessed|retrieved)(?:\s+on)?"
    r"|\b(?:tel|phone|fax|mobile)\.?)"
    r"\s*[:,]?\s*$",
    re.I,
)
_CUE_WINDOW_CHARS = 60
_ISSUE, _PLAIN = 0, 1  # candidate tiers: lower wins, reading order within a tier

# PDF date strings: "D:YYYYMMDD..." (the "D:" is often missing). DOCX core.xml: W3CDTF "YYYY-MM-DDThh:mm:ssZ".
_PDF_DATE = re.compile(r"^(?:D:)?((?:19|20)\d{2})(\d{2})(\d{2})")
_W3C_DATE = re.compile(r"^((?:19|20)\d{2})-(\d{2})-(\d{2})")
_DOCX_CORE = "docProps/core.xml"
# Creation dates before this are placeholders (1970/1980 epochs written by some PDF tools), not real ones.
_EARLIEST_CREATED = date(1990, 1, 1)
# Template defaults every file made from the template inherits: python-docx stamps 2013-12-23 unless the
# creator sets core_properties.created.
_TEMPLATE_CREATED = frozenset({date(2013, 12, 23)})
_DCTERMS_CREATED = "{http://purl.org/dc/terms/}created"


@dataclass(frozen=True, slots=True)
class DocumentDate:
    value: date
    source: DateSource

    @property
    def timestamp(self) -> float:
        return datetime(self.value.year, self.value.month, self.value.day, tzinfo=UTC).timestamp()


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:  # e.g. 31 February
        return None


def _month(name: str) -> int:
    return _MONTHS[name[:3].lower()]


def date_from_name(name: str) -> date | None:
    found = [_safe_date(int(y), int(m), int(d)) for y, m, d in _NAME_FULL.findall(name)]
    found += [_safe_date(int(y), int(m), 1) for y, m in _NAME_MONTH.findall(name)]
    valid = [d for d in found if d is not None]
    # Several dates (e.g. period end and filing date): the latest is when the document was issued.
    return max(valid, default=None)


def _numeric(m: re.Match[str], order: NumericDateOrder) -> date | None:
    first, second, year = int(m[1]), int(m[3]), int(m[4])
    day, month = (first, second) if order == "dmy" else (second, first)
    return _safe_date(year, month, day)


def date_from_content(text: str, *, numeric_order: NumericDateOrder | None) -> date | None:
    head = text[:CONTENT_WINDOW_CHARS]
    # (start, end, parsed date or None when invalid)
    found: list[tuple[int, int, date | None]] = []
    for m in _CONTENT_ISO.finditer(head):
        found.append((m.start(), m.end(), _safe_date(int(m[1]), int(m[2]), int(m[3]))))
    for m in _CONTENT_MDY.finditer(head):
        found.append((m.start(), m.end(), _safe_date(int(m[3]), _month(m[1]), int(m[2]))))
    for m in _CONTENT_DMY.finditer(head):
        found.append((m.start(), m.end(), _safe_date(int(m[3]), _month(m[2]), int(m[1]))))
    for m in _CONTENT_MY.finditer(head):
        found.append((m.start(), m.end(), date(int(m[2]), _month(m[1]), 1)))
    if numeric_order is not None:
        for m in _CONTENT_NUMERIC.finditer(head):
            found.append((m.start(), m.end(), _numeric(m, numeric_order)))
    # Rank by the words right before each date: issue cues first, then uncued dates (reading order within a
    # tier). Dates after a skip cue are dropped, with any shorter match inside them (the "September 2026" of
    # "effective 1 September 2026").
    ranked: list[tuple[int, int, int, date | None]] = []  # (tier, start, end, parsed)
    skipped: list[tuple[int, int]] = []
    for start, end, parsed in found:
        before = head[max(0, start - _CUE_WINDOW_CHARS) : start]
        if _ISSUE_CUE.search(before):
            ranked.append((_ISSUE, start, end, parsed))
        elif _SKIP_CUE.search(before):
            skipped.append((start, end))
        else:
            ranked.append((_PLAIN, start, end, parsed))
    candidates = [
        (tier, start, parsed)
        for tier, start, end, parsed in ranked
        if parsed is not None and not any(start < s_end and s_start < end for s_start, s_end in skipped)
    ]
    # A full date ("5 May 2026") starts before the month-year inside it and wins within its tier.
    return min(candidates)[2] if candidates else None


def _match_date(pattern: re.Pattern[str], value: str) -> date | None:
    m = pattern.match(value.strip())
    return _safe_date(int(m[1]), int(m[2]), int(m[3])) if m else None


def _pdf_created(path: Path) -> date | None:
    import pymupdf

    with pymupdf.open(path) as pdf:
        # pymupdf leaves the dict's value type unannotated; its values are strings (None if encrypted).
        metadata = cast("dict[str, str] | None", pdf.metadata)  # pyright: ignore[reportUnknownMemberType]
    # Absent or not a PDF date string (the field is free text in practice): no metadata date.
    return _match_date(_PDF_DATE, metadata.get("creationDate") or "") if metadata else None


def _docx_created(path: Path) -> date | None:
    with zipfile.ZipFile(path) as package:
        if _DOCX_CORE not in package.namelist():  # core properties are optional in OOXML
            return None
        root = ElementTree.fromstring(package.read(_DOCX_CORE))
    created = root.find(_DCTERMS_CREATED)
    return _match_date(_W3C_DATE, created.text or "") if created is not None else None


def metadata_date(path: Path) -> date | None:
    """The creation date stored in a PDF or DOCX file, if it has one; None for other formats.

    A file that can't be opened or whose metadata isn't well-formed XML raises (the file is broken).
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf_created(path)
    if suffix == ".docx":
        return _docx_created(path)
    return None


def _plausible_created(created: date, mtime_day: date) -> bool:
    # A file can't be created after it was last modified; one day of slack because the metadata date is the
    # author's local day while mtime is read in UTC.
    return (
        created >= _EARLIEST_CREATED
        and created not in _TEMPLATE_CREATED
        and created <= mtime_day + timedelta(days=1)
    )


def infer_document_date(
    file_name: str,
    text: str,
    mtime: float,
    *,
    created: Callable[[], date | None],
    numeric_order: NumericDateOrder | None,
) -> DocumentDate:
    """`created` reads the file's metadata creation date (see `metadata_date`). It is only called when the
    name and the text give no date: a file isn't opened again (or failed on bad metadata) for nothing."""
    if (from_name := date_from_name(file_name)) is not None:
        return DocumentDate(from_name, "name")
    if (from_content := date_from_content(text, numeric_order=numeric_order)) is not None:
        return DocumentDate(from_content, "content")
    mtime_day = datetime.fromtimestamp(mtime, tz=UTC).date()
    if (from_metadata := created()) is not None and _plausible_created(from_metadata, mtime_day):
        return DocumentDate(from_metadata, "metadata")
    return DocumentDate(mtime_day, "mtime")
