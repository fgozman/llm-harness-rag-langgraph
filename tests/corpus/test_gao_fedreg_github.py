from datetime import date
from typing import Any

from corpus_builder.fedreg import select_documents
from corpus_builder.gao import GaoItem, parse_feed, render_report
from corpus_builder.github_releases import Release, parse_releases, render_release_notes

GAO_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Reports News from the GAO</title>
<item><title>AI and Cybersecurity: Technologies Can Magnify Cyberattacks</title>
<link>https://www.gao.gov/products/gao-27-108458</link>
<description>
What GAO Found

AI makes it possible to create   high-quality phishing.



			Incident
			June 2026: AI-powered malware lab
</description>
<pubDate>Fri, 20 Nov 2026 07:05:04 -0500</pubDate><guid>/products/gao-27-108458</guid></item>
<item><title>Old report</title><link>https://www.gao.gov/products/gao-26-100001</link>
<description>x</description><pubDate>Mon, 15 Jun 2026 07:00:00 -0400</pubDate></item>
</channel></rss>"""


def test_gao_feed_items_and_date_filter() -> None:
    items = parse_feed(GAO_FEED)
    assert [i.report_number for i in items] == ["GAO-27-108458", "GAO-26-100001"]
    assert items[0].published == date(2026, 11, 20)
    assert [i.report_number for i in items if i.published >= date(2026, 7, 1)] == ["GAO-27-108458"]


def test_gao_report_text_has_provenance_and_normalized_paragraphs() -> None:
    item: GaoItem = parse_feed(GAO_FEED)[0]
    text = render_report(item)
    assert text.startswith("AI and Cybersecurity: Technologies Can Magnify Cyberattacks\n")
    assert "GAO-27-108458" in text and "https://www.gao.gov/products/gao-27-108458" in text
    assert "highlights" in text.lower()
    assert "create high-quality phishing." in text
    assert "\n\n\n" not in text
    assert "Incident\nJune 2026: AI-powered malware lab" in text


def fr_doc(number: str, pages: int) -> dict[str, Any]:
    return {
        "document_number": number,
        "title": f"Rule {number}",
        "publication_date": "2026-11-20",
        "page_length": pages,
        "body_html_url": f"https://www.federalregister.gov/x/{number}.html",
        "html_url": f"https://www.federalregister.gov/d/{number}",
        "agencies": [{"name": "Transportation Department"}],
    }


def test_federal_register_selection_respects_page_bounds_and_limit() -> None:
    docs = [fr_doc("a", 2), fr_doc("b", 10), fr_doc("c", 200), fr_doc("d", 30), fr_doc("e", 5)]
    chosen = select_documents(docs, min_pages=3, max_pages=80, limit=2)
    assert [d["document_number"] for d in chosen] == ["b", "d"]


RELEASES = [
    {
        "tag_name": "1.2.24",
        "name": "1.2.24",
        "published_at": "2026-11-10T10:00:00Z",
        "body": "Fixed X.",
        "html_url": "u1",
        "draft": False,
    },
    {
        "tag_name": "prebuilt==1.1.3",
        "name": "",
        "published_at": "2026-09-01T10:00:00Z",
        "body": None,
        "html_url": "u2",
        "draft": False,
    },
    {"tag_name": "draft", "name": "d", "published_at": None, "body": "", "html_url": "u3", "draft": True},
    {
        "tag_name": "1.0.0",
        "name": "1.0.0",
        "published_at": "2026-06-30T23:59:59Z",
        "body": "Old.",
        "html_url": "u4",
        "draft": False,
    },
]


def test_releases_parsed_filtered_and_pagination_stops_at_cutoff() -> None:
    releases, reached_cutoff = parse_releases(RELEASES, since=date(2026, 7, 1))
    assert [r.tag for r in releases] == ["1.2.24", "prebuilt==1.1.3"]
    assert reached_cutoff
    _, more = parse_releases(RELEASES[:2], since=date(2026, 7, 1))
    assert not more


def test_release_notes_markdown() -> None:
    releases = [
        Release("1.2.24", "1.2.24", date(2026, 11, 10), "Fixed X.", "u1"),
        Release("prebuilt==1.1.3", "", date(2026, 9, 1), "", "u2"),
    ]
    md = render_release_notes("langchain-ai/langgraph", releases, since=date(2026, 7, 1), license_id="MIT")
    assert md.startswith("# langchain-ai/langgraph release notes since 2026-07-01")
    assert "## 1.2.24 (2026-11-10)" in md and "Fixed X." in md
    assert "## prebuilt==1.1.3 (2026-09-01)" in md and "_No release notes._" in md
    assert md.index("1.2.24") < md.index("prebuilt==1.1.3")
