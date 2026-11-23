"""Document date: from the file name, else from the opening text, else the file's modification time.

The date answers "when was this document issued", which is what date filters ("documents from Q3 2026") need.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Literal

DateSource = Literal["name", "content", "mtime"]

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


def date_from_content(text: str) -> date | None:
    head = text[:CONTENT_WINDOW_CHARS]
    candidates: list[tuple[int, date]] = []
    for m in _CONTENT_ISO.finditer(head):
        if parsed := _safe_date(int(m[1]), int(m[2]), int(m[3])):
            candidates.append((m.start(), parsed))
    for m in _CONTENT_MDY.finditer(head):
        if parsed := _safe_date(int(m[3]), _month(m[1]), int(m[2])):
            candidates.append((m.start(), parsed))
    for m in _CONTENT_DMY.finditer(head):
        if parsed := _safe_date(int(m[3]), _month(m[2]), int(m[1])):
            candidates.append((m.start(), parsed))
    for m in _CONTENT_MY.finditer(head):
        candidates.append((m.start(), date(int(m[2]), _month(m[1]), 1)))
    # First in reading order; a full date ("5 May 2026") starts before the month-year inside it and wins.
    return min(candidates)[1] if candidates else None


def infer_document_date(file_name: str, text: str, mtime: float) -> DocumentDate:
    if (from_name := date_from_name(file_name)) is not None:
        return DocumentDate(from_name, "name")
    if (from_content := date_from_content(text)) is not None:
        return DocumentDate(from_content, "content")
    return DocumentDate(datetime.fromtimestamp(mtime, tz=UTC).date(), "mtime")
