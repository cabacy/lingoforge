from unittest.mock import Mock

import pytest
import requests

from src.anki_bridge import AnkiAPIError, AnkiBridge, AnkiConnectionError


def response(payload):
    mock = Mock()
    mock.json.return_value = payload
    mock.raise_for_status.return_value = None
    return mock


def test_add_note_uses_anki_connect_and_logs_operations(tmp_path, caplog):
    caplog.set_level("INFO", logger="src.anki_bridge")
    session = Mock()
    session.post.return_value = response({"result": 42, "error": None})
    bridge = AnkiBridge(session=session)
    audio = tmp_path / "word.mp3"
    audio.touch()

    note_id = bridge.add_note(
        deck="LingoForge",
        model="Basic",
        front="hola",
        back="hello",
        tags=["lingoforge", "article"],
        audio_path=audio,
    )

    assert note_id == 42
    payload = session.post.call_args.kwargs["json"]
    assert payload["action"] == "addNote"
    assert payload["params"]["note"]["audio"][0]["filename"] == "word.mp3"
    assert "action=addNote started" in caplog.text
    assert "action=addNote succeeded" in caplog.text


def test_anki_api_error_is_clear():
    session = Mock()
    session.post.return_value = response({"result": None, "error": "deck missing"})
    with pytest.raises(AnkiAPIError, match="deck missing"):
        AnkiBridge(session=session).create_deck("LingoForge")


def test_unreachable_anki_is_wrapped():
    session = Mock()
    session.post.side_effect = requests.ConnectionError("connection refused")
    with pytest.raises(AnkiConnectionError, match="Anki is running"):
        AnkiBridge(session=session).ping()
