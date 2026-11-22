from datetime import date

from corpus_builder.config import FilingPeriod
from corpus_builder.sec import (
    Filing,
    exhibit_filename,
    filings_from_submissions,
    has_financial_report,
    select_earnings_8k,
    select_period_10q,
    select_pre_cutoff_10q,
)

# Columnar shape of data.sec.gov/submissions (filings.recent), as observed 2026-11-22.
SUBMISSIONS = {
    "name": "Uber Technologies, Inc",
    "filings": {
        "recent": {
            "accessionNumber": [
                "A-8K-NOV",
                "A-10Q-NOV",
                "A-8K-AUG",
                "A-10Q-AUG",
                "A-8K-OTHER",
                "A-10Q-MAY",
                "A-10K",
            ],
            "filingDate": [
                "2026-11-04",
                "2026-11-04",
                "2026-08-05",
                "2026-08-05",
                "2026-07-15",
                "2026-05-06",
                "2026-02-12",
            ],
            "reportDate": [
                "2026-11-04",
                "2026-09-30",
                "2026-08-05",
                "2026-06-30",
                "2026-07-15",
                "2026-03-31",
                "2025-12-31",
            ],
            "form": ["8-K", "10-Q", "8-K", "10-Q", "8-K", "10-Q", "10-K"],
            "items": ["2.02,9.01", "", "2.02,9.01", "", "5.02", "", ""],
            "primaryDocument": [
                "u-1104.htm",
                "uber-20260930.htm",
                "u-0805.htm",
                "uber-20260630.htm",
                "x.htm",
                "uber-20260331.htm",
                "k.htm",
            ],
            "primaryDocDescription": ["8-K", "10-Q", "8-K", "10-Q", "8-K", "10-Q", "10-K"],
        }
    },
}

AUG = FilingPeriod(label="2026-08", filed_from=date(2026, 8, 1), filed_to=date(2026, 8, 31))
NOV = FilingPeriod(label="2026-11", filed_from=date(2026, 11, 1), filed_to=date(2026, 11, 30))


def filings() -> list[Filing]:
    return filings_from_submissions(SUBMISSIONS)


def test_columnar_submissions_become_filings() -> None:
    f = filings()[1]
    assert (f.form, f.filing_date, f.report_date, f.accession, f.primary_document) == (
        "10-Q",
        date(2026, 11, 4),
        date(2026, 9, 30),
        "A-10Q-NOV",
        "uber-20260930.htm",
    )


def test_period_selection_by_filing_month() -> None:
    aug, nov = select_period_10q(filings(), AUG), select_period_10q(filings(), NOV)
    assert aug is not None and aug.accession == "A-10Q-AUG"
    assert nov is not None and nov.accession == "A-10Q-NOV"
    empty = FilingPeriod(label="x", filed_from=date(2026, 9, 1), filed_to=date(2026, 9, 30))
    assert select_period_10q(filings(), empty) is None


def test_pre_cutoff_is_latest_10q_before_cutoff() -> None:
    pre = select_pre_cutoff_10q(filings(), date(2026, 7, 1))
    assert pre is not None and pre.accession == "A-10Q-MAY"


def test_earnings_8k_requires_item_202_within_window() -> None:
    ten_q = select_period_10q(filings(), AUG)
    assert ten_q is not None
    eight_k = select_earnings_8k(filings(), ten_q, window_days=21)
    assert eight_k is not None and eight_k.accession == "A-8K-AUG"
    # The 5.02 8-K in July is not an earnings release.
    assert select_earnings_8k(filings(), ten_q, window_days=0) is not None
    nothing = Filing("10-Q", date(2026, 10, 1), date(2026, 9, 30), "Z", "z.htm", "")
    assert select_earnings_8k(filings(), nothing, window_days=21) is None


# Excerpt of an EDGAR -index-headers.html (SGML headers HTML-escaped), as observed 2026-11-22.
INDEX_HEADERS = """<html><body><pre>
&lt;DOCUMENT&gt;
&lt;TYPE&gt;8-K
&lt;SEQUENCE&gt;1
&lt;FILENAME&gt;uber-20261104.htm
&lt;/DOCUMENT&gt;
&lt;DOCUMENT&gt;
&lt;TYPE&gt;EX-99.1
&lt;SEQUENCE&gt;2
&lt;FILENAME&gt;uberq326earningspressrelea.htm
&lt;DESCRIPTION&gt;EX-99.1
&lt;/DOCUMENT&gt;
&lt;DOCUMENT&gt;
&lt;TYPE&gt;EX-101.SCH
&lt;FILENAME&gt;uber-20261104.xsd
&lt;/DOCUMENT&gt;
</pre></body></html>"""


def test_exhibit_filename_from_index_headers() -> None:
    assert exhibit_filename(INDEX_HEADERS, "EX-99.1") == "uberq326earningspressrelea.htm"
    assert exhibit_filename(INDEX_HEADERS, "EX-99.2") is None


def test_financial_report_detection() -> None:
    listing = {"directory": {"item": [{"name": "a.htm"}, {"name": "Financial_Report.xlsx"}]}}
    assert has_financial_report(listing)
    assert not has_financial_report({"directory": {"item": [{"name": "a.htm"}]}})
