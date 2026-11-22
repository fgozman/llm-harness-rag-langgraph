from datetime import date

import pytest

from corpus_builder.arxiv import ArxivRecord, OaiError, accept, oai_set, parse_atom_published, parse_oai_page
from corpus_builder.config import ArxivConfig


def oai_record(arxiv_id: str, categories: str, license_url: str) -> str:
    return f"""<record><header><identifier>oai:arXiv.org:{arxiv_id}</identifier>
    <datestamp>2026-08-03</datestamp>
    <setSpec>cs:cs:IR</setSpec></header><metadata>
    <arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>{arxiv_id}</id><created>2026-08-02</created>
    <authors><author><keyname>Liu</keyname><forenames>Han</forenames></author></authors>
    <title>Paper   {arxiv_id}
      title</title><categories>{categories}</categories><license>{license_url}</license>
    <abstract>Abstract.</abstract></arXiv></metadata></record>"""


CC_BY = "http://creativecommons.org/licenses/by/4.0/"
OAI_PAGE = f"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
{oai_record("2608.01234", "cs.IR cs.CL", CC_BY)}
{oai_record("2404.16555", "cs.IR", CC_BY)}
{oai_record("2609.00001", "cs.IR", "http://arxiv.org/licenses/nonexclusive-distrib/1.0/")}
{oai_record("2610.00002", "cs.CL cs.IR", CC_BY)}
{oai_record("2611.00003", "cs.IR", "http://creativecommons.org/publicdomain/zero/1.0/")}
{oai_record("2611.00004", "cs.IR", "http://creativecommons.org/licenses/by-sa/4.0/")}
<resumptionToken cursor="0" completeListSize="2600">token-123</resumptionToken>
</ListRecords></OAI-PMH>"""


def test_parse_oai_page_records_and_token() -> None:
    records, token = parse_oai_page(OAI_PAGE)
    assert token == "token-123"
    first = records[0]
    assert first == ArxivRecord(
        arxiv_id="2608.01234", title="Paper 2608.01234 title", categories=["cs.IR", "cs.CL"], license=CC_BY
    )


def test_last_page_has_no_token() -> None:
    _, token = parse_oai_page(OAI_PAGE.replace("token-123", ""))
    assert token is None


def test_accept_filters_by_submission_month_primary_category_and_license() -> None:
    records, _ = parse_oai_page(OAI_PAGE)
    cfg = ArxivConfig(submitted_on_or_after=date(2026, 8, 1))
    accepted = [r.arxiv_id for r in records if accept(r, cfg)]
    # 2404.* first submitted before the cutoff; 2609 not CC BY; 2610 primary is cs.CL; by-sa is not CC BY.
    assert accepted == ["2608.01234", "2611.00003"]
    assert [
        r.arxiv_id for r in records if accept(r, cfg.model_copy(update={"require_primary_category": False}))
    ] == [
        "2608.01234",
        "2610.00002",
        "2611.00003",
    ]


ATOM = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/abs/2608.01234v2</id><published>2026-08-05T10:00:00Z</published></entry>
  <entry><id>http://arxiv.org/abs/2611.00003v1</id><published>2026-11-19T18:57:05Z</published></entry>
</feed>"""


def test_atom_published_dates_by_unversioned_id() -> None:
    assert parse_atom_published(ATOM) == {"2608.01234": date(2026, 8, 5), "2611.00003": date(2026, 11, 19)}


def test_oai_set_spec_for_category() -> None:
    assert oai_set("cs.IR") == "cs:cs:IR"
    assert oai_set("stat.ML") == "stat:stat:ML"


def test_oai_error_response_raises() -> None:
    error = """<?xml version="1.0"?><OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
    <error code="badArgument">Invalid set</error></OAI-PMH>"""
    with pytest.raises(OaiError, match="badArgument"):
        parse_oai_page(error)


def test_no_records_match_is_not_an_error() -> None:
    empty = """<?xml version="1.0"?><OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
    <error code="noRecordsMatch">none</error></OAI-PMH>"""
    assert parse_oai_page(empty) == ([], None)
