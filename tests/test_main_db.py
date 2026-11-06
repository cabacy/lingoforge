import sqlite3

import pytest

from src.main_db import MainDB


def test_cards_are_deduplicated_case_insensitively(tmp_path):
    db = MainDB(tmp_path / "main.db")
    note_id = 12345
    db.record_card("Canción", "es", note_id, "article")

    assert db.contains(" canción ", "es")
    assert db.get_card("CANCIÓN", "es").anki_note_id == note_id
    assert not db.contains("canción", "fr")
    with pytest.raises(sqlite3.IntegrityError):
        db.record_card("canción", "es", 67890, "dialogue")

    with sqlite3.connect(db.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
