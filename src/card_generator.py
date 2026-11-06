from __future__ import annotations

import hashlib
import html
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import requests

from src.anki_bridge import AnkiBridge
from src.languages import language_name
from src.main_db import MainDB
from src.raw_db import RawDB, RawItem

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class TranslationError(RuntimeError):
    pass


class SpeechError(RuntimeError):
    pass


class Translator(Protocol):
    def translate(self, word: str, context: str) -> str: ...


class SpeechGenerator(Protocol):
    def generate(self, text: str, destination: Path) -> None: ...


class LLMClient:
    def __init__(
        self,
        provider: str,
        *,
        model: str | None = None,
        timeout: float = 30,
        openai_client: Any = None,
        session: requests.Session | None = None,
    ):
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self.openai_client = openai_client
        self.session = session or requests.Session()

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            if self.provider == "openai":
                if self.openai_client is None:
                    from openai import OpenAI

                    self.openai_client = OpenAI()
                response = self.openai_client.chat.completions.create(
                    model=self.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=messages,
                    temperature=0.2,
                )
                content = response.choices[0].message.content
                if not content:
                    raise LLMError("OpenAI returned an empty response")
                return content.strip()
            if self.provider == "ollama":
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                response = self.session.post(
                    f"{base_url.rstrip('/')}/api/chat",
                    json={
                        "model": self.model or os.getenv("OLLAMA_MODEL", "llama3.2"),
                        "messages": messages,
                        "stream": False,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                content = response.json()["message"]["content"].strip()
                if not content:
                    raise LLMError("Ollama returned an empty response")
                return content
            raise LLMError(f"Unsupported LLM provider: {self.provider}")
        except LLMError:
            raise
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.provider} request failed: {exc}") from exc
        except Exception as exc:
            if self.provider == "openai":
                raise LLMError(f"OpenAI request failed: {exc}") from exc
            raise


class LLMTranslator:
    def __init__(self, client: LLMClient, target_language: str, native_language: str):
        self.client = client
        self.target_language = target_language
        self.native_language = native_language

    def translate(self, word: str, context: str) -> str:
        prompt = (
            f"Translate the {language_name(self.target_language)} word or phrase "
            f"into concise {language_name(self.native_language)}. "
            "Return only the translation, without labels or explanations.\n"
            f"Word or phrase: {word}\nContext: {context}"
        )
        try:
            return self.client.complete(
                "You are a precise bilingual dictionary for language learners.", prompt
            )
        except LLMError as exc:
            raise TranslationError(str(exc)) from exc


class LocalDictionaryTranslator:
    def __init__(self, path: str | Path, language: str):
        self.path = Path(path).expanduser()
        self.language = language
        try:
            entries = json.loads(self.path.read_text(encoding="utf-8"))
            if language in entries and isinstance(entries[language], dict):
                entries = entries[language]
            self.entries = {str(key).casefold(): str(value) for key, value in entries.items()}
        except (OSError, ValueError, AttributeError) as exc:
            raise TranslationError(f"Could not load local dictionary {self.path}: {exc}") from exc

    def translate(self, word: str, context: str) -> str:
        try:
            return self.entries[word.casefold()]
        except KeyError as exc:
            raise TranslationError(f"No local dictionary translation for {word!r}") from exc


class GTTSSpeech:
    def __init__(self, language: str):
        self.language = language

    def generate(self, text: str, destination: Path) -> None:
        try:
            from gtts import gTTS

            gTTS(text=text, lang=self.language).save(str(destination))
        except Exception as exc:
            raise SpeechError(f"gTTS audio generation failed: {exc}") from exc


@dataclass
class ProcessingSummary:
    created: int = 0
    duplicates: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.created + self.duplicates + self.failed


@dataclass(frozen=True)
class LanguagePipeline:
    translator: Translator
    speech: SpeechGenerator
    deck: str


class CardGenerator:
    def __init__(
        self,
        *,
        raw_db: RawDB,
        main_db: MainDB,
        anki: AnkiBridge,
        pipelines: dict[str, LanguagePipeline],
        output_dir: str | Path,
        model: str = "Basic",
    ):
        if not pipelines:
            raise ValueError("CardGenerator needs at least one language pipeline")
        self.raw_db = raw_db
        self.main_db = main_db
        self.anki = anki
        self.pipelines = pipelines
        self.output_dir = Path(output_dir).expanduser()
        self.model = model

    def process_all(
        self, limit: int | None = None, language: str | None = None
    ) -> ProcessingSummary:
        summary = ProcessingSummary()
        for item in self.raw_db.get_unprocessed(limit, language=language):
            try:
                created = self.process_item(item)
                if created:
                    summary.created += 1
                else:
                    summary.duplicates += 1
            except Exception as exc:
                summary.failed += 1
                message = f"Raw item {item.id} ({item.word}): {exc}"
                summary.errors.append(message)
                logger.error(message)
                try:
                    self.raw_db.record_error(item.id, str(exc))
                except Exception:
                    logger.exception("Could not record processing error for raw item %s", item.id)
        return summary

    def process_item(self, item: RawItem) -> bool:
        pipeline = self.pipelines.get(item.language)
        if pipeline is None:
            configured = ", ".join(sorted(self.pipelines))
            raise TranslationError(
                f"No pipeline configured for language {item.language!r} "
                f"(configured: {configured})"
            )
        if self.main_db.contains(item.word, item.language):
            self.raw_db.mark_processed(item.id)
            logger.info("Skipping duplicate word=%r raw_item=%s", item.word, item.id)
            return False

        translation = item.metadata.get("translation") or pipeline.translator.translate(
            item.word, item.context
        )
        audio_path = self._audio_for(item, pipeline)
        self.anki.create_deck(pipeline.deck)
        note_id = self.anki.add_note(
            deck=pipeline.deck,
            model=self.model,
            front=html.escape(item.word),
            back=self._format_back(item, translation),
            tags=["lingoforge", item.source_type, item.language],
            audio_path=audio_path,
        )
        try:
            self.main_db.record_card(item.word, item.language, note_id, item.source_type)
        except Exception as exc:
            try:
                self.anki.delete_notes([note_id])
            except Exception as rollback_exc:
                raise RuntimeError(
                    f"Could not record Anki note {note_id} and rollback failed: {rollback_exc}"
                ) from exc
            raise
        self.raw_db.mark_processed(item.id)
        logger.info("Created Anki note=%s word=%r", note_id, item.word)
        return True

    def _audio_for(self, item: RawItem, pipeline: LanguagePipeline) -> Path:
        if item.audio_path:
            path = Path(item.audio_path).expanduser()
            if not path.is_file():
                raise SpeechError(f"Source audio fragment is missing: {path}")
            return path.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(
            f"{item.language}:{item.word}:{item.id}".encode()
        ).hexdigest()[:16]
        destination = self.output_dir / f"pronunciation-{digest}.mp3"
        pipeline.speech.generate(item.word, destination)
        return destination.resolve()

    @staticmethod
    def _format_back(item: RawItem, translation: str) -> str:
        parts = [f"<b>{html.escape(translation)}</b>"]
        if item.context:
            parts.append(f"<i>{html.escape(item.context)}</i>")
        source = html.escape(item.source)
        if item.start_seconds >= 0:
            source += f" ({item.start_seconds:.2f}s–{item.end_seconds:.2f}s)"
        if source:
            parts.append(f"<small>Source: {source}</small>")
        return "<br>".join(parts)
