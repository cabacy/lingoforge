from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from src.raw_db import RawDB


class MediaIngestionError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class SelectedCandidate:
    word: str
    context: str
    start: float
    end: float
    reasons: set[str]


@dataclass(frozen=True)
class MediaReport:
    source: str
    transcript_segments: int
    candidates: int
    inserted: int


class WhisperTranscriber:
    def __init__(
        self,
        model: str = "small",
        *,
        language: str | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_name = model
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def transcribe(self, path: str | Path) -> list[TranscriptSegment]:
        try:
            if self._model is None:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self.model_name, device=self.device, compute_type=self.compute_type
                )
            segments, _ = self._model.transcribe(str(path), language=self.language)
            return [
                TranscriptSegment(float(segment.start), float(segment.end), segment.text.strip())
                for segment in segments
                if segment.text.strip()
            ]
        except Exception as exc:
            raise MediaIngestionError(f"Whisper transcription failed: {exc}") from exc


class FFmpegProcessor:
    @staticmethod
    def extract_audio(source: Path, destination: Path) -> None:
        try:
            import ffmpeg

            destination.parent.mkdir(parents=True, exist_ok=True)
            (
                ffmpeg.input(str(source))
                .output(str(destination), format="mp3", ac=1, ar=16000)
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as exc:
            raise MediaIngestionError(f"ffmpeg audio extraction failed: {exc}") from exc

    @staticmethod
    def slice(source: Path, destination: Path, start: float, end: float) -> None:
        try:
            import ffmpeg

            destination.parent.mkdir(parents=True, exist_ok=True)
            (
                ffmpeg.input(str(source), ss=max(0, start), to=end)
                .output(str(destination), format="mp3")
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as exc:
            raise MediaIngestionError(f"ffmpeg audio slicing failed: {exc}") from exc


class MediaIngester:
    def __init__(
        self,
        *,
        raw_db: RawDB,
        nlp: Any,
        transcriber: WhisperTranscriber,
        processor: FFmpegProcessor,
        output_dir: str | Path,
        data_dir: str | Path,
        language: str,
        video_extensions: set[str] | None = None,
        idioms: set[str] | None = None,
        slang: set[str] | None = None,
        grammar_markers: set[str] | None = None,
        timeout: float = 30,
        session: requests.Session | None = None,
    ):
        self.raw_db = raw_db
        self.nlp = nlp
        self.transcriber = transcriber
        self.processor = processor
        self.output_dir = Path(output_dir).expanduser()
        self.data_dir = Path(data_dir).expanduser()
        self.language = language
        self.video_extensions = video_extensions or {"mp4", "mkv", "mov", "avi", "webm"}
        self.idioms = idioms or set()
        self.slang = slang or set()
        self.grammar_markers = grammar_markers or set()
        self.timeout = timeout
        self.session = session or requests.Session()

    def ingest(self, source: str | Path) -> MediaReport:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        media_path, source_label = self._obtain_source(source)
        extension = media_path.suffix.lower().lstrip(".")
        is_video = extension in self.video_extensions
        source_type = "video" if is_video else "audio"
        if is_video:
            audio_path = self.output_dir / f"extracted-{self._digest(source_label)}.mp3"
            self.processor.extract_audio(media_path, audio_path)
        else:
            audio_path = media_path

        segments = self.transcriber.transcribe(audio_path)
        if not segments:
            raise MediaIngestionError(f"No speech transcribed from {source_label}")
        for segment in segments:
            self.raw_db.add_item(
                segment.text,
                source_type=source_type,
                language=self.language,
                source=source_label,
                start_seconds=segment.start,
                end_seconds=segment.end,
                item_kind="transcript",
                processed=True,
            )

        candidates = self._select_candidates(segments)
        inserted = 0
        for candidate in candidates:
            fragment_name = f"fragment-{self._digest(source_label + candidate.word + str(candidate.start))}.mp3"
            fragment = self.output_dir / fragment_name
            self.processor.slice(audio_path, fragment, candidate.start, candidate.end)
            item_id = self.raw_db.add_item(
                candidate.word,
                candidate.context,
                source_type=source_type,
                language=self.language,
                source=source_label,
                start_seconds=candidate.start,
                end_seconds=candidate.end,
                audio_path=fragment,
                metadata={"reasons": sorted(candidate.reasons)},
            )
            if item_id is not None:
                inserted += 1
        return MediaReport(source_label, len(segments), len(candidates), inserted)

    def _obtain_source(self, source: str | Path) -> tuple[Path, str]:
        source_text = str(source)
        if source_text.startswith(("http://", "https://")):
            suffix = Path(urlparse(source_text).path).suffix or ".media"
            self.data_dir.mkdir(parents=True, exist_ok=True)
            destination = self.data_dir / f"download-{self._digest(source_text)}{suffix}"
            try:
                response = self.session.get(source_text, timeout=self.timeout)
                response.raise_for_status()
                destination.write_bytes(response.content)
                return destination, source_text
            except (requests.RequestException, OSError) as exc:
                raise MediaIngestionError(f"Could not download media {source_text}: {exc}") from exc
        path = Path(source).expanduser()
        if not path.is_file():
            raise MediaIngestionError(f"Media file does not exist: {path}")
        return path.resolve(), str(path.resolve())

    def _select_candidates(self, segments: list[TranscriptSegment]) -> list[SelectedCandidate]:
        documents = [self.nlp(segment.text) for segment in segments]
        occurrences = Counter(
            token.lemma_.strip().casefold()
            for document in documents
            for token in document
            if token.is_alpha and not token.is_stop and token.lemma_.strip()
        )
        repeated = {word for word, count in occurrences.items() if count >= 3}
        added_repeated: set[str] = set()
        selected: dict[tuple[str, float, float], SelectedCandidate] = {}

        def add(segment: TranscriptSegment, word: str, reason: str) -> None:
            key = (word, segment.start, segment.end)
            if key not in selected:
                selected[key] = SelectedCandidate(
                    word, segment.text, segment.start, segment.end, set()
                )
            selected[key].reasons.add(reason)

        for segment, document in zip(segments, documents, strict=True):
            lowered = segment.text.casefold()
            for token in document:
                lemma = token.lemma_.strip().casefold()
                if lemma in repeated and lemma not in added_repeated:
                    add(segment, lemma, "repeated_3_plus")
                    added_repeated.add(lemma)
                if any("Sub" in value for value in token.morph.get("Mood")):
                    add(segment, segment.text, "subjunctive_construction")
            for phrase in self.idioms:
                if phrase in lowered:
                    add(segment, phrase, "idiom")
            token_texts = {token.text.casefold() for token in document}
            for phrase in self.slang:
                if phrase in token_texts:
                    add(segment, phrase, "slang")
            if any(marker in lowered for marker in self.grammar_markers):
                add(segment, segment.text, "grammar_construction")
        return list(selected.values())

    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()[:20]
