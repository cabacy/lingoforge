from pathlib import Path

import pytest

from src.listenlab import TranscriptSegment
from src.raw_db import RawDB
from src.voicetutor import DialogueError, VoiceTutor


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append((system, user))
        return next(self.responses)


class FakeTranscriber:
    def __init__(self, texts):
        self.texts = iter(texts)

    def transcribe(self, path):
        return [TranscriptSegment(0, 1, next(self.texts))]


def tutor(tmp_path, llm, transcriber=None):
    return VoiceTutor(
        raw_db=RawDB(tmp_path / "raw.db"),
        llm=llm,
        last_dialogue=tmp_path / "output" / "last_dialogue.txt",
        language="es",
        native_language="ru",
        transcriber=transcriber,
    )


def test_text_dialogue_is_saved(tmp_path):
    llm = FakeLLM(["¡Hola! ¿Cómo estás?"])
    voice = tutor(tmp_path, llm)
    answers = iter(["Hola", "/quit"])

    path = voice.start(input_func=lambda prompt: next(answers), print_func=lambda text: None)

    assert path.read_text(encoding="utf-8") == (
        "Learner: Hola\nTutor: ¡Hola! ¿Cómo estás?\n"
    )
    assert "Learner: Hola" in llm.prompts[0][1]


def test_voice_dialogue_uses_whisper(tmp_path):
    llm = FakeLLM(["Mucho gusto."])
    transcriber = FakeTranscriber(["Hola", "/quit"])
    voice = tutor(tmp_path, llm, transcriber)

    def record(path):
        Path(path).write_bytes(b"voice")

    voice.start(voice=True, recorder=record, print_func=lambda text: None)

    assert "Learner: Hola" in voice.last_dialogue.read_text(encoding="utf-8")


def test_analyze_parses_json_and_saves_vocabulary(tmp_path):
    response = """```json
    {"error_analysis": ["Use 'estoy' for location."],
     "vocabulary": [{"word": "aprovechar", "context": "Quiero aprovechar el día.",
                     "translation": "использовать с пользой"}]}
    ```"""
    dialogue = tmp_path / "dialog.txt"
    dialogue.write_text("Learner: Quiero aprovechar el día.", encoding="utf-8")
    llm = FakeLLM([response, response])
    voice = tutor(tmp_path, llm)

    report = voice.analyze(dialogue)

    assert report.analysis.error_analysis == ["Use 'estoy' for location."]
    assert report.inserted == 1
    row = voice.raw_db.get_unprocessed()[0]
    assert row.word == "aprovechar"
    assert row.language == "es"
    assert row.source_type == "dialogue"
    assert row.metadata["translation"] == "использовать с пользой"
    assert "Spanish" in llm.prompts[0][0]
    assert "Russian" in llm.prompts[0][0]
    assert voice.analyze(dialogue).inserted == 0


def test_analyze_rejects_malformed_json(tmp_path):
    dialogue = tmp_path / "dialog.txt"
    dialogue.write_text("Learner: Hola", encoding="utf-8")
    voice = tutor(tmp_path, FakeLLM(["not json"]))

    with pytest.raises(DialogueError, match="JSON object"):
        voice.analyze(dialogue)
