from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RawItem:
    id: int
    word: str
    context: str
    source_type: str
    source: str
    start_seconds: float
    end_seconds: float
    audio_path: str | None
    metadata: dict[str, Any]
    language: str


class RawDB:
    SCHEMA_VERSION = 2

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
                raise RuntimeError(f"Raw DB schema {version} is newer than this application")
            if version == 0:
                connection.executescript(
                    """
                    CREATE TABLE raw_items (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        word TEXT NOT NULL,
                        context TEXT NOT NULL DEFAULT '',
                        source_type TEXT NOT NULL,
                        source TEXT NOT NULL DEFAULT '',
                        start_seconds REAL NOT NULL DEFAULT -1,
                        end_seconds REAL NOT NULL DEFAULT -1,
                        audio_path TEXT,
                        metadata TEXT NOT NULL DEFAULT '{}',
                        item_kind TEXT NOT NULL DEFAULT 'candidate',
                        language TEXT NOT NULL DEFAULT '',
                        processed INTEGER NOT NULL DEFAULT 0,
                        processing_error TEXT,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(word, context, source_type, source, start_seconds, end_seconds, language)
                    );
                    CREATE INDEX raw_items_unprocessed_idx
                        ON raw_items(processed, item_kind, id);
                    PRAGMA user_version = 2;
                    """
                )
            elif version == 1:
                connection.executescript(
                    """
                    CREATE TABLE raw_items_v2 (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        word TEXT NOT NULL,
                        context TEXT NOT NULL DEFAULT '',
                        source_type TEXT NOT NULL,
                        source TEXT NOT NULL DEFAULT '',
                        start_seconds REAL NOT NULL DEFAULT -1,
                        end_seconds REAL NOT NULL DEFAULT -1,
                        audio_path TEXT,
                        metadata TEXT NOT NULL DEFAULT '{}',
                        item_kind TEXT NOT NULL DEFAULT 'candidate',
                        language TEXT NOT NULL DEFAULT '',
                        processed INTEGER NOT NULL DEFAULT 0,
                        processing_error TEXT,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(word, context, source_type, source, start_seconds, end_seconds, language)
                    );
                    INSERT INTO raw_items_v2 (
                        id, word, context, source_type, source, start_seconds, end_seconds,
                        audio_path, metadata, item_kind, language, processed, processing_error,
                        created_at
                    )
                    SELECT
                        id, word, context, source_type, source, start_seconds, end_seconds,
                        audio_path, metadata, item_kind, 'en', processed, processing_error,
                        created_at
                    FROM raw_items;
                    DROP TABLE raw_items;
                    ALTER TABLE raw_items_v2 RENAME TO raw_items;
                    CREATE INDEX raw_items_unprocessed_idx
                        ON raw_items(processed, item_kind, id);
                    PRAGMA user_version = 2;
                    """
                )

    def add_item(
        self,
        word: str,
        context: str = "",
        *,
        source_type: str,
        language: str,
        source: str = "",
        start_seconds: float = -1,
        end_seconds: float = -1,
        audio_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
        item_kind: str = "candidate",
        processed: bool = False,
    ) -> int | None:
        word = word.strip()
        if not word:
            raise ValueError("Raw item word cannot be empty")
        language = language.strip()
        if not language:
            raise ValueError("Raw item language cannot be empty")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO raw_items (
                    word, context, source_type, source, start_seconds, end_seconds,
                    audio_path, metadata, item_kind, language, processed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                (
                    word,
                    context.strip(),
                    source_type,
                    source,
                    start_seconds,
                    end_seconds,
                    str(audio_path) if audio_path else None,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    item_kind,
                    language,
                    int(processed),
                ),
            )
            return cursor.lastrowid if cursor.rowcount else None

    def get_unprocessed(
        self, limit: int | None = None, language: str | None = None
    ) -> list[RawItem]:
        sql = """
            SELECT * FROM raw_items
            WHERE processed = 0 AND item_kind = 'candidate'
        """
        parameters: list[Any] = []
        if language is not None:
            sql += " AND language = ?"
            parameters.append(language)
        sql += " ORDER BY id"
        if limit is not None:
            if limit < 1:
                raise ValueError("Limit must be positive")
            sql += " LIMIT ?"
            parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._to_item(row) for row in rows]

    def list_items(self, item_kind: str | None = None) -> list[RawItem]:
        sql = "SELECT * FROM raw_items"
        parameters: tuple[Any, ...] = ()
        if item_kind is not None:
            sql += " WHERE item_kind = ?"
            parameters = (item_kind,)
        sql += " ORDER BY id"
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._to_item(row) for row in rows]

    def mark_processed(self, item_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE raw_items SET processed = 1, processing_error = NULL WHERE id = ?",
                (item_id,),
            )

    def record_error(self, item_id: int, message: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE raw_items SET processing_error = ? WHERE id = ?",
                (message[:2000], item_id),
            )

    @staticmethod
    def _to_item(row: sqlite3.Row) -> RawItem:
        return RawItem(
            id=row["id"],
            word=row["word"],
            context=row["context"],
            source_type=row["source_type"],
            source=row["source"],
            start_seconds=row["start_seconds"],
            end_seconds=row["end_seconds"],
            audio_path=row["audio_path"],
            metadata=json.loads(row["metadata"]),
            language=row["language"],
        )
