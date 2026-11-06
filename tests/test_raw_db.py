import sqlite3

from src.raw_db import RawDB


def test_migration_and_candidate_lifecycle(tmp_path):
    db = RawDB(tmp_path / "nested" / "raw.db")
    item_id = db.add_item(
        "hablar",
        "Quiero hablar contigo.",
        source_type="article",
        language="es",
        source="sample.txt",
        metadata={"rank": 1200},
    )

    assert item_id is not None
    assert db.add_item(
        "hablar",
        "Quiero hablar contigo.",
        source_type="article",
        language="es",
        source="sample.txt",
    ) is None
    item = db.get_unprocessed()[0]
    assert item.word == "hablar"
    assert item.language == "es"
    assert item.metadata == {"rank": 1200}

    db.record_error(item_id, "Anki is stopped")
    db.mark_processed(item_id)
    assert db.get_unprocessed() == []

    with sqlite3.connect(db.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        error = connection.execute(
            "SELECT processing_error FROM raw_items WHERE id = ?", (item_id,)
        ).fetchone()[0]
    assert error is None


def test_same_word_can_exist_once_per_language(tmp_path):
    db = RawDB(tmp_path / "raw.db")
    en_id = db.add_item("Gift", "A gift for you.", source_type="article", language="en")
    de_id = db.add_item("Gift", "Ein Gift.", source_type="article", language="de")

    assert en_id is not None
    assert de_id is not None
    assert db.add_item("Gift", "A gift for you.", source_type="article", language="en") is None
    assert [item.language for item in db.get_unprocessed()] == ["en", "de"]
    assert [item.language for item in db.get_unprocessed(language="de")] == ["de"]


def test_schema_v1_is_migrated_with_english_backfill(tmp_path):
    path = tmp_path / "raw.db"
    with sqlite3.connect(path) as connection:
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
                processed INTEGER NOT NULL DEFAULT 0,
                processing_error TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(word, context, source_type, source, start_seconds, end_seconds)
            );
            CREATE INDEX raw_items_unprocessed_idx
                ON raw_items(processed, item_kind, id);
            INSERT INTO raw_items (word, context, source_type, source)
            VALUES ('hello', 'Hello world.', 'article', 'old.txt');
            PRAGMA user_version = 1;
            """
        )

    db = RawDB(path)

    item = db.get_unprocessed()[0]
    assert item.word == "hello"
    assert item.language == "en"
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
    assert db.add_item("hello", "Hello world.", source_type="article", language="de") is not None


def test_transcript_rows_are_not_card_candidates(tmp_path):
    db = RawDB(tmp_path / "raw.db")
    db.add_item(
        "Una frase completa",
        source_type="audio",
        language="es",
        item_kind="transcript",
        processed=True,
        start_seconds=1.25,
        end_seconds=3.5,
    )

    assert db.get_unprocessed() == []
    transcript = db.list_items("transcript")[0]
    assert transcript.start_seconds == 1.25
    assert transcript.end_seconds == 3.5
