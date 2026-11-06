import sqlite3
from unittest.mock import Mock

import requests

from src.anki_bridge import AnkiConnectionError
from src.card_generator import (
    CardGenerator,
    GTTSSpeech,
    LanguagePipeline,
    LLMClient,
    LLMError,
    LLMTranslator,
)
from src.main_db import MainDB
from src.raw_db import RawDB


class FakeAnki:
    def __init__(self):
        self.notes = []
        self.deleted = []

    def create_deck(self, name):
        return name

    def add_note(self, **note):
        self.notes.append(note)
        return 1000 + len(self.notes)

    def delete_notes(self, note_ids):
        self.deleted.extend(note_ids)


class FakeTranslator:
    def __init__(self):
        self.calls = []

    def translate(self, word, context):
        self.calls.append(word)
        return f"translation of {word}"


class FakeSpeech:
    def generate(self, text, destination):
        destination.write_bytes(b"audio")


def pipeline_for(translator=None, speech=None, deck="Test"):
    return LanguagePipeline(
        translator=translator or FakeTranslator(),
        speech=speech or FakeSpeech(),
        deck=deck,
    )


def generator(tmp_path, anki=None, speech=None):
    raw = RawDB(tmp_path / "raw.db")
    main = MainDB(tmp_path / "main.db")
    anki = anki or FakeAnki()
    return raw, main, anki, CardGenerator(
        raw_db=raw,
        main_db=main,
        anki=anki,
        pipelines={"es": pipeline_for(speech=speech)},
        output_dir=tmp_path / "output",
    )


def test_pipeline_creates_card_and_never_duplicates_it(tmp_path):
    raw, main, anki, pipeline = generator(tmp_path)
    first_id = raw.add_item(
        "canción",
        "La <canción> es bonita.",
        source_type="article",
        language="es",
        source="sample.txt",
    )
    second_id = raw.add_item(
        "Canción",
        "Otra canción.",
        source_type="dialogue",
        language="es",
        source="dialog.txt",
    )

    summary = pipeline.process_all()

    assert (summary.created, summary.duplicates, summary.failed) == (1, 1, 0)
    second_summary = pipeline.process_all()
    assert second_summary.total == 0
    assert len(anki.notes) == 1
    assert main.contains("CANCIÓN", "es")
    assert raw.get_unprocessed() == []
    assert "La &lt;canción&gt; es bonita." in anki.notes[0]["back"]
    assert anki.notes[0]["audio_path"].is_file()

    duplicate_id = raw.add_item(
        "canción", "Nuevo contexto", source_type="audio", language="es"
    )
    duplicate_summary = pipeline.process_all()
    assert duplicate_summary.duplicates == 1
    assert len(anki.notes) == 1
    assert raw.get_unprocessed() == []
    assert first_id != second_id != duplicate_id


def test_each_language_is_routed_to_its_own_deck_and_translator(tmp_path):
    raw = RawDB(tmp_path / "raw.db")
    main = MainDB(tmp_path / "main.db")
    anki = FakeAnki()
    en_translator, de_translator = FakeTranslator(), FakeTranslator()
    pipeline = CardGenerator(
        raw_db=raw,
        main_db=main,
        anki=anki,
        pipelines={
            "en": pipeline_for(translator=en_translator, deck="English"),
            "de": pipeline_for(translator=de_translator, deck="Deutsch"),
        },
        output_dir=tmp_path / "output",
    )
    raw.add_item("house", "The house is big.", source_type="article", language="en")
    raw.add_item("Haus", "Das Haus ist groß.", source_type="article", language="de")

    summary = pipeline.process_all()

    assert (summary.created, summary.failed) == (2, 0)
    decks = {note["deck"] for note in anki.notes}
    assert decks == {"English", "Deutsch"}
    assert en_translator.calls == ["house"]
    assert de_translator.calls == ["Haus"]
    assert main.contains("house", "en")
    assert main.contains("Haus", "de")
    assert {lang for note in anki.notes for lang in note["tags"] if lang in {"en", "de"}} == {
        "en",
        "de",
    }


def test_item_without_configured_language_fails_with_clear_error(tmp_path):
    raw, _, _, pipeline = generator(tmp_path)
    raw.add_item("Bonjour", source_type="article", language="fr")

    summary = pipeline.process_all()

    assert summary.failed == 1
    assert "No pipeline configured for language 'fr'" in summary.errors[0]
    assert len(raw.get_unprocessed()) == 1


def test_language_filter_processes_only_that_language(tmp_path):
    raw, _, anki, pipeline = generator(tmp_path)
    pipeline.pipelines["de"] = pipeline_for(deck="Deutsch")
    raw.add_item("house", source_type="article", language="en")
    raw.add_item("Haus", source_type="article", language="de")

    summary = pipeline.process_all(language="de")

    assert summary.created == 1
    assert anki.notes[0]["deck"] == "Deutsch"
    assert [item.language for item in raw.get_unprocessed()] == ["en"]


def test_anki_failure_leaves_item_retryable(tmp_path):
    raw = RawDB(tmp_path / "raw.db")
    main = MainDB(tmp_path / "main.db")
    anki = Mock()
    anki.add_note.side_effect = AnkiConnectionError("Anki is stopped")
    pipeline = CardGenerator(
        raw_db=raw,
        main_db=main,
        anki=anki,
        pipelines={"es": pipeline_for()},
        output_dir=tmp_path,
    )
    raw.add_item("hola", source_type="article", language="es")

    summary = pipeline.process_all()

    assert summary.failed == 1
    assert "Anki is stopped" in summary.errors[0]
    assert len(raw.get_unprocessed()) == 1
    assert not main.contains("hola", "es")


def test_database_failure_rolls_back_new_anki_note(tmp_path, monkeypatch):
    raw, main, anki, pipeline = generator(tmp_path)
    raw.add_item("hola", source_type="article", language="es")

    def fail_record(*args):
        raise sqlite3.OperationalError("database locked")

    monkeypatch.setattr(main, "record_card", fail_record)
    summary = pipeline.process_all()

    assert summary.failed == 1
    assert anki.deleted == [1001]
    assert len(raw.get_unprocessed()) == 1


def test_source_fragment_is_used_instead_of_tts(tmp_path):
    speech = Mock()
    raw, _, anki, pipeline = generator(tmp_path, speech=speech)
    fragment = tmp_path / "fragment.mp3"
    fragment.write_bytes(b"source audio")
    raw.add_item("hola", source_type="audio", language="es", audio_path=fragment)

    summary = pipeline.process_all()

    assert summary.created == 1
    assert anki.notes[0]["audio_path"] == fragment.resolve()
    speech.generate.assert_not_called()


def test_llm_translator_asks_for_native_language_translation():
    client = Mock()
    client.complete.return_value = "дом"
    translator = LLMTranslator(client, "de", "ru")

    assert translator.translate("Haus", "Das Haus ist groß.") == "дом"
    prompt = client.complete.call_args.args[1]
    assert "German" in prompt
    assert "Russian" in prompt


def test_ollama_response_and_network_error():
    session = Mock()
    session.post.return_value.json.return_value = {"message": {"content": " hello "}}
    session.post.return_value.raise_for_status.return_value = None
    client = LLMClient("ollama", model="test", session=session)

    assert client.complete("system", "user") == "hello"
    assert session.post.call_args.kwargs["json"]["model"] == "test"

    session.post.side_effect = requests.ConnectionError("offline")
    try:
        client.complete("system", "user")
    except LLMError as exc:
        assert "request failed" in str(exc)
    else:
        raise AssertionError("Expected LLMError")


def test_gtts_wraps_external_errors(monkeypatch):
    speech = GTTSSpeech("es")
    monkeypatch.setattr("gtts.gTTS", Mock(side_effect=RuntimeError("offline")))
    try:
        speech.generate("hola", Mock())
    except Exception as exc:
        assert "gTTS audio generation failed" in str(exc)
    else:
        raise AssertionError("Expected SpeechError")
