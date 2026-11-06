import yaml

from src.cli import main
from src.raw_db import RawDB


def config_file(tmp_path, languages=None):
    if languages is None:
        languages = {
            "es": {
                "spacy_model": "es_core_news_sm",
                "deck": "Test",
                "frequency_band": {"min_rank": 1000, "max_rank": 5000},
            }
        }
    config = {
        "native_language": "ru",
        "languages": languages,
        "paths": {
            "raw_db": "output/raw.db",
            "main_db": "output/main.db",
            "output": "output",
            "last_dialogue": "output/last.txt",
        },
        "anki": {"model": "Basic"},
        "translation": {"provider": "llm"},
        "llm": {"provider": "ollama"},
        "dialogue": {"record_seconds": 1},
        "network_timeout": 1,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_cards_generate_command_with_empty_database(tmp_path, capsys):
    result = main(["--config", str(config_file(tmp_path)), "cards", "generate"])

    assert result == 0
    assert "0 created, 0 duplicates, 0 failed" in capsys.readouterr().out


def test_voicetutor_analyze_file_command(tmp_path, monkeypatch, capsys):
    dialogue = tmp_path / "dialog.txt"
    dialogue.write_text("Learner: Quiero aprovechar el día.", encoding="utf-8")

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

        def complete(self, system, user):
            return (
                '{"error_analysis": [], "vocabulary": [{"word": "aprovechar", '
                '"context": "Quiero aprovechar el día.", "translation": "использовать с пользой"}]}'
            )

    monkeypatch.setattr("src.cli.LLMClient", FakeLLM)
    result = main(
        [
            "--config",
            str(config_file(tmp_path)),
            "voicetutor",
            "analyze",
            "--file",
            str(dialogue),
        ]
    )

    assert result == 0
    assert "1 new Raw DB rows" in capsys.readouterr().out
    rows = RawDB(tmp_path / "output" / "raw.db").get_unprocessed()
    assert [row.word for row in rows] == ["aprovechar"]
    assert [row.language for row in rows] == ["es"]


def test_lang_is_required_when_several_languages_are_configured(tmp_path, capsys):
    config = config_file(
        tmp_path,
        languages={
            "en": {"spacy_model": "en_core_web_sm"},
            "de": {"spacy_model": "de_core_news_sm"},
        },
    )
    result = main(["--config", str(config), "voicetutor", "analyze"])

    assert result == 1
    assert "--lang is required; configured languages: de, en" in capsys.readouterr().err


def test_unknown_lang_is_rejected(tmp_path, capsys):
    result = main(
        ["--config", str(config_file(tmp_path)), "cards", "generate", "--lang", "fr"]
    )

    assert result == 1
    assert "Unsupported language 'fr'; configured languages: es" in capsys.readouterr().err
