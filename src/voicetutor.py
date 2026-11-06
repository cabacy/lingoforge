from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.card_generator import LLMClient, LLMError
from src.languages import language_name
from src.listenlab import WhisperTranscriber
from src.raw_db import RawDB


class DialogueError(RuntimeError):
    pass


@dataclass(frozen=True)
class VocabularyItem:
    word: str
    context: str
    translation: str


@dataclass(frozen=True)
class DialogueAnalysis:
    error_analysis: list[str]
    vocabulary: list[VocabularyItem]


@dataclass(frozen=True)
class AnalysisReport:
    analysis: DialogueAnalysis
    inserted: int


class VoiceTutor:
    def __init__(
        self,
        *,
        raw_db: RawDB,
        llm: LLMClient,
        last_dialogue: str | Path,
        language: str,
        native_language: str,
        transcriber: WhisperTranscriber | None = None,
        record_seconds: int = 10,
    ):
        self.raw_db = raw_db
        self.llm = llm
        self.last_dialogue = Path(last_dialogue).expanduser()
        self.language = language
        self.native_language = native_language
        self.transcriber = transcriber
        self.record_seconds = record_seconds

    def start(
        self,
        *,
        voice: bool = False,
        input_func: Callable[[str], str] = input,
        print_func: Callable[[str], None] = print,
        recorder: Callable[[Path], None] | None = None,
    ) -> Path:
        if voice and self.transcriber is None:
            raise DialogueError("Voice mode requires a Whisper transcriber")
        self.last_dialogue.parent.mkdir(parents=True, exist_ok=True)
        turns: list[str] = []
        print_func("VoiceTutor started. Enter /quit to finish." if not voice else "VoiceTutor started. Speak /quit or press Ctrl-C to finish.")
        while True:
            if voice:
                audio_path = self.last_dialogue.parent / "voice-turn.wav"
                (recorder or self._record_voice)(audio_path)
                user_text = " ".join(
                    segment.text for segment in self.transcriber.transcribe(audio_path)
                ).strip()
                print_func(f"You: {user_text}")
            else:
                user_text = input_func("You: ").strip()
            if user_text.casefold() == "/quit":
                break
            if not user_text:
                continue
            conversation = "\n".join(turns + [f"Learner: {user_text}"])
            try:
                response = self.llm.complete(
                    (
                        f"You are a patient {language_name(self.language)} language tutor "
                        "for a B1 learner. "
                        "Reply in the target language, keep answers brief, and naturally "
                        "correct serious mistakes without disrupting the conversation."
                    ),
                    conversation,
                )
            except LLMError as exc:
                raise DialogueError(f"Dialogue response failed: {exc}") from exc
            turns.extend([f"Learner: {user_text}", f"Tutor: {response}"])
            self.last_dialogue.write_text("\n".join(turns) + "\n", encoding="utf-8")
            print_func(f"Tutor: {response}")
        return self.last_dialogue

    def analyze(self, file: str | Path | None = None) -> AnalysisReport:
        path = Path(file).expanduser() if file else self.last_dialogue
        try:
            dialogue = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise DialogueError(f"Could not read dialogue {path}: {exc}") from exc
        if not dialogue:
            raise DialogueError(f"Dialogue is empty: {path}")
        system_prompt = (
            f"You analyze a {language_name(self.language)} learner's dialogue. "
            "Return only valid JSON "
            f"with this exact shape: {{\"error_analysis\": [\"{language_name(self.native_language)} explanation\"], "
            "\"vocabulary\": [{\"word\": \"target-language word or phrase\", "
            "\"context\": \"sentence from the dialogue\", "
            f"\"translation\": \"concise {language_name(self.native_language)} translation\"}}]}}}}. "
            "Extract useful new B1+ vocabulary, not every word. Do not use Markdown."
        )
        try:
            response = self.llm.complete(system_prompt, dialogue)
            analysis = self._parse_analysis(response)
        except (LLMError, ValueError, TypeError, KeyError) as exc:
            raise DialogueError(f"Could not analyze dialogue: {exc}") from exc

        inserted = 0
        for item in analysis.vocabulary:
            item_id = self.raw_db.add_item(
                item.word,
                item.context,
                source_type="dialogue",
                language=self.language,
                source=str(path.resolve()),
                metadata={"translation": item.translation} if item.translation else {},
            )
            if item_id is not None:
                inserted += 1
        return AnalysisReport(analysis, inserted)

    def _record_voice(self, destination: Path) -> None:
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "pulse",
            "-i",
            "default",
            "-t",
            str(self.record_seconds),
            str(destination),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise DialogueError(
                f"Could not record microphone audio with ffmpeg/PulseAudio: {exc}"
            ) from exc

    @staticmethod
    def _parse_analysis(response: str) -> DialogueAnalysis:
        text = response.strip()
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("LLM response did not contain a JSON object")
        payload: dict[str, Any] = json.loads(text[start : end + 1])
        errors = payload["error_analysis"]
        vocabulary = payload["vocabulary"]
        if not isinstance(errors, list) or not all(isinstance(item, str) for item in errors):
            raise ValueError("error_analysis must be a list of strings")
        if not isinstance(vocabulary, list):
            raise ValueError("vocabulary must be a list")
        items = []
        for entry in vocabulary:
            if not isinstance(entry, dict) or not str(entry.get("word", "")).strip():
                raise ValueError("Each vocabulary entry needs a non-empty word")
            items.append(
                VocabularyItem(
                    word=str(entry["word"]).strip(),
                    context=str(entry.get("context", "")).strip(),
                    translation=str(entry.get("translation", "")).strip(),
                )
            )
        return DialogueAnalysis(errors, items)
