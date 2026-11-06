from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.lingo_feed import ArticleIngester, ArticleIngestionError
from src.raw_db import RawDB


class FakeDocument:
    def __init__(self, entries):
        self.tokens = [
            SimpleNamespace(
                text=text,
                lemma_=lemma,
                is_alpha=is_alpha,
                sent=SimpleNamespace(text=sentence),
            )
            for text, lemma, is_alpha, sentence in entries
        ]

    def __iter__(self):
        return iter(self.tokens)


class FakeNLP:
    def __init__(self, entries):
        self.entries = entries
        self.received_text = None

    def __call__(self, text):
        self.received_text = text
        return FakeDocument(self.entries)


class FakeRanker:
    ranks = {"medio": 1000, "desarrollo": 1200, "común": 5000, "el": 5, "rara": 9000}

    def rank(self, word):
        return self.ranks.get(word)


def test_local_article_selects_band_and_counts_repetitions(tmp_path):
    article = tmp_path / "article.txt"
    article.write_text("Contenido de prueba", encoding="utf-8")
    nlp = FakeNLP(
        [
            ("El", "el", True, "El desarrollo es común."),
            ("desarrollo", "desarrollo", True, "El desarrollo es común."),
            ("común", "común", True, "El desarrollo es común."),
            ("desarrollo", "desarrollo", True, "Otro desarrollo y medio."),
            ("medio", "medio", True, "Otro desarrollo y medio."),
            ("rara", "rara", True, "Una palabra rara."),
            ("123", "", False, "123"),
        ]
    )
    db = RawDB(tmp_path / "raw.db")
    ingester = ArticleIngester(raw_db=db, nlp=nlp, ranker=FakeRanker(), language="es")

    report = ingester.ingest(article)

    assert report.candidates == 3
    assert report.inserted == 3
    rows = {item.word: item for item in db.get_unprocessed()}
    assert set(rows) == {"medio", "desarrollo", "común"}
    assert rows["desarrollo"].metadata == {"frequency_rank": 1200, "occurrences": 2}
    assert rows["desarrollo"].context == "El desarrollo es común."
    assert rows["medio"].source_type == "article"
    assert rows["medio"].language == "es"

    assert ingester.ingest(article).inserted == 0


def test_url_article_strips_html_and_reports_network_errors(tmp_path):
    session = Mock()
    session.get.return_value.text = (
        "<html><style>ignore</style><body>Texto <b>útil</b>"
        "<script>also ignore</script></body></html>"
    )
    session.get.return_value.raise_for_status.return_value = None
    nlp = FakeNLP([])
    ingester = ArticleIngester(
        raw_db=RawDB(tmp_path / "raw.db"),
        nlp=nlp,
        ranker=FakeRanker(),
        language="es",
        session=session,
    )

    ingester.ingest("https://example.com/article")
    assert nlp.received_text == "Texto útil"

    import requests

    session.get.side_effect = requests.ConnectionError("offline")
    with pytest.raises(ArticleIngestionError, match="offline"):
        ingester.ingest("https://example.com/article")
