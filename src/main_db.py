from __future__ import annotations

import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CardRecord:
    id: int
    word: str
    language: str
    anki_note_id: int
    source_type: str


class MainDB:
    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _migrate(self) -> None:
        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version > self.SCHEMA_VERSION:
                raise RuntimeError(f"Main DB schema {version} is newer than this application")
            if version == 0:
                connection.executescript(
                    """
                    CREATE TABLE cards (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        word_key TEXT NOT NULL UNIQUE,
                        word TEXT NOT NULL,
                        language TEXT NOT NULL,
                        anki_note_id INTEGER NOT NULL UNIQUE,
                        source_type TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    PRAGMA user_version = 1;
                    """
                )

    @staticmethod
    def word_key(word: str, language: str) -> str:
        normalized = unicodedata.normalize("NFKC", word).strip().casefold()
        if not normalized:
            raise ValueError("Card word cannot be empty")
        return f"{language.casefold()}:{normalized}"

    def contains(self, word: str, language: str) -> bool:
        return self.get_card(word, language) is not None

    def get_card(self, word: str, language: str) -> CardRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM cards WHERE word_key = ?", (self.word_key(word, language),)
            ).fetchone()
        if row is None:
            return None
        return CardRecord(
            id=row["id"],
            word=row["word"],
            language=row["language"],
            anki_note_id=row["anki_note_id"],
            source_type=row["source_type"],
        )

    def record_card(self, word: str, language: str, anki_note_id: int, source_type: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO cards (word_key, word, language, anki_note_id, source_type)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self.word_key(word, language), word.strip(), language, anki_note_id, source_type),
            )
            return cursor.lastrowid
