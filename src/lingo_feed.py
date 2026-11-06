from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol

import requests

from src.raw_db import RawDB


class ArticleIngestionError(RuntimeError):
    pass


class Ranker(Protocol):
    def rank(self, word: str) -> int | None: ...


class WordFrequencyRanker:
    def __init__(self, language: str, max_rank: int = 5000):
        self.language = language
        self.max_rank = max_rank
        self._ranks: dict[str, int] | None = None

    def rank(self, word: str) -> int | None:
        if self._ranks is None:
            from wordfreq import top_n_list

            self._ranks = {}
            for index, entry in enumerate(top_n_list(self.language, self.max_rank), start=1):
                self._ranks.setdefault(entry.casefold(), index)
        return self._ranks.get(word.casefold())


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        return " ".join(" ".join(self.parts).split())


@dataclass(frozen=True)
class IngestionReport:
    source: str
    candidates: int
    inserted: int


class ArticleIngester:
    def __init__(
        self,
        *,
        raw_db: RawDB,
        nlp: Any,
        ranker: Ranker,
        language: str,
        min_rank: int = 1000,
        max_rank: int = 5000,
        timeout: float = 30,
        session: requests.Session | None = None,
    ):
        if min_rank < 1 or max_rank < min_rank:
            raise ValueError("Invalid frequency band")
        self.raw_db = raw_db
        self.nlp = nlp
        self.ranker = ranker
        self.language = language
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.timeout = timeout
        self.session = session or requests.Session()

    def ingest(self, source: str | Path) -> IngestionReport:
        text, source_label = self._read_source(source)
        if not text.strip():
            raise ArticleIngestionError(f"Article is empty: {source_label}")
        document = self.nlp(text)
        occurrences = Counter(
            token.lemma_.strip().casefold()
            for token in document
            if token.is_alpha and token.lemma_.strip()
        )
        seen: set[str] = set()
        candidates = 0
        inserted = 0
        for token in document:
            if not token.is_alpha:
                continue
            lemma = token.lemma_.strip().casefold()
            if not lemma or lemma in seen:
                continue
            rank = self.ranker.rank(lemma)
            if rank is None or not self.min_rank <= rank <= self.max_rank:
                continue
            seen.add(lemma)
            candidates += 1
            context = self._sentence(token)
            item_id = self.raw_db.add_item(
                lemma,
                context,
                source_type="article",
                language=self.language,
                source=source_label,
                metadata={"frequency_rank": rank, "occurrences": occurrences[lemma]},
            )
            if item_id is not None:
                inserted += 1
        return IngestionReport(source_label, candidates, inserted)

    def _read_source(self, source: str | Path) -> tuple[str, str]:
        source_text = str(source)
        if source_text.startswith(("http://", "https://")):
            try:
                response = self.session.get(source_text, timeout=self.timeout)
                response.raise_for_status()
                return self._html_to_text(response.text), source_text
            except requests.RequestException as exc:
                raise ArticleIngestionError(f"Could not retrieve article {source_text}: {exc}") from exc
        path = Path(source).expanduser()
        try:
            return path.read_text(encoding="utf-8"), str(path.resolve())
        except OSError as exc:
            raise ArticleIngestionError(f"Could not read article {path}: {exc}") from exc

    @staticmethod
    def _html_to_text(html: str) -> str:
        parser = _TextExtractor()
        parser.feed(html)
        return parser.text()

    @staticmethod
    def _sentence(token: Any) -> str:
        try:
            return token.sent.text.strip()
        except (AttributeError, ValueError):
            return token.text.strip()
