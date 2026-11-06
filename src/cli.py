from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.anki_bridge import AnkiBridge
from src.card_generator import (
    CardGenerator,
    GTTSSpeech,
    LanguagePipeline,
    LLMClient,
    LLMTranslator,
    LocalDictionaryTranslator,
)
from src.languages import resolve_language
from src.lingo_feed import ArticleIngester, WordFrequencyRanker
from src.listenlab import FFmpegProcessor, MediaIngester, WhisperTranscriber
from src.main_db import MainDB
from src.raw_db import RawDB
from src.voicetutor import VoiceTutor

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML object: {config_path}")
    load_dotenv(config_path.parent / ".env")
    return config, config_path.parent


def _path(config: dict[str, Any], root: Path, name: str) -> Path:
    path = Path(config["paths"][name]).expanduser()
    return path if path.is_absolute() else root / path


def _raw_db(config: dict[str, Any], root: Path) -> RawDB:
    return RawDB(_path(config, root, "raw_db"))


def _llm(config: dict[str, Any]) -> LLMClient:
    return LLMClient(
        config["llm"]["provider"], timeout=float(config.get("network_timeout", 30))
    )


def _transcriber(config: dict[str, Any], language: str) -> WhisperTranscriber:
    whisper = config["whisper"]
    return WhisperTranscriber(
        whisper.get("model", "small"),
        language=language,
        device=whisper.get("device", "cpu"),
        compute_type=whisper.get("compute_type", "int8"),
    )


def _native_language(config: dict[str, Any]) -> str:
    native = config.get("native_language")
    if not native:
        raise ValueError("Configuration must define 'native_language'")
    return str(native)


def _translator(config: dict[str, Any], root: Path, language: str):
    translation_provider = config["translation"]["provider"]
    if translation_provider == "dictionary":
        return LocalDictionaryTranslator(_path(config, root, "dictionary"), language)
    if translation_provider == "llm":
        return LLMTranslator(_llm(config), language, _native_language(config))
    raise ValueError(f"Unsupported translation provider: {translation_provider}")


def _card_generator(config: dict[str, Any], root: Path) -> CardGenerator:
    languages = config.get("languages")
    if not isinstance(languages, dict) or not languages:
        raise ValueError("Configuration must define at least one entry under 'languages'")
    pipelines = {
        code: LanguagePipeline(
            translator=_translator(config, root, code),
            speech=GTTSSpeech(code),
            deck=str(lang_config.get("deck", f"LingoForge::{code}")),
        )
        for code, lang_config in languages.items()
    }
    return CardGenerator(
        raw_db=_raw_db(config, root),
        main_db=MainDB(_path(config, root, "main_db")),
        anki=AnkiBridge(
            os.getenv("ANKI_CONNECT_URL", "http://localhost:8765"),
            timeout=float(config.get("network_timeout", 30)),
        ),
        pipelines=pipelines,
        output_dir=_path(config, root, "output"),
        model=config["anki"].get("model", "Basic"),
    )


def _voice_tutor(
    config: dict[str, Any], root: Path, language: str, voice: bool = False
) -> VoiceTutor:
    return VoiceTutor(
        raw_db=_raw_db(config, root),
        llm=_llm(config),
        last_dialogue=_path(config, root, "last_dialogue"),
        language=language,
        native_language=_native_language(config),
        transcriber=_transcriber(config, language) if voice else None,
        record_seconds=int(config["dialogue"].get("record_seconds", 10)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lingoforge", description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML configuration path")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Ingest learning material")
    ingest_commands = ingest.add_subparsers(dest="ingest_command", required=True)
    article = ingest_commands.add_parser("article", help="Ingest an article URL or UTF-8 file")
    article.add_argument("source")
    article.add_argument("--lang", help="Target language code from the configuration")
    media = ingest_commands.add_parser("media", help="Ingest a direct media URL or local file")
    media.add_argument("source")
    media.add_argument("--lang", help="Target language code from the configuration")

    cards = commands.add_parser("cards", help="Generate Anki cards")
    cards_commands = cards.add_subparsers(dest="cards_command", required=True)
    generate = cards_commands.add_parser("generate", help="Process pending Raw DB items")
    generate.add_argument("--limit", type=int, help="Maximum number of pending items")
    generate.add_argument("--lang", help="Process only items of this language")

    tutor = commands.add_parser("voicetutor", help="Practice and analyze dialogues")
    tutor_commands = tutor.add_subparsers(dest="tutor_command", required=True)
    start = tutor_commands.add_parser("start", help="Start an interactive dialogue")
    start.add_argument("--voice", action="store_true", help="Record voice turns with ffmpeg")
    start.add_argument("--record-seconds", type=int, help="Duration of each voice turn")
    start.add_argument("--lang", help="Target language code from the configuration")
    analyze = tutor_commands.add_parser("analyze", help="Analyze the last or a saved dialogue")
    analyze.add_argument("--file", help="Dialogue text file (defaults to the last dialogue)")
    analyze.add_argument("--lang", help="Target language code from the configuration")
    return parser


def _run(args: argparse.Namespace) -> int:
    config, root = load_config(args.config)
    if args.command == "ingest":
        import spacy

        language = resolve_language(config, args.lang)
        language_config = config["languages"][language]
        nlp = spacy.load(language_config["spacy_model"])
        if args.ingest_command == "article":
            band = language_config["frequency_band"]
            ingester = ArticleIngester(
                raw_db=_raw_db(config, root),
                nlp=nlp,
                ranker=WordFrequencyRanker(language, int(band["max_rank"])),
                language=language,
                min_rank=int(band["min_rank"]),
                max_rank=int(band["max_rank"]),
                timeout=float(config.get("network_timeout", 30)),
            )
            report = ingester.ingest(args.source)
            print(
                f"Article ingested: {report.inserted} new candidates "
                f"({report.candidates} selected) from {report.source}"
            )
        else:
            ingester = MediaIngester(
                raw_db=_raw_db(config, root),
                nlp=nlp,
                transcriber=_transcriber(config, language),
                processor=FFmpegProcessor(),
                output_dir=_path(config, root, "output"),
                data_dir=_path(config, root, "data"),
                language=language,
                video_extensions=set(config["media"]["video_extensions"]),
                idioms=set(language_config.get("idioms") or ()),
                slang=set(language_config.get("slang") or ()),
                grammar_markers=set(language_config.get("grammar_markers") or ()),
                timeout=float(config.get("network_timeout", 30)),
            )
            report = ingester.ingest(args.source)
            print(
                f"Media ingested: {report.transcript_segments} transcript segments, "
                f"{report.inserted} new candidates ({report.candidates} selected)"
            )
        return 0

    if args.command == "cards":
        language = resolve_language(config, args.lang) if args.lang else None
        summary = _card_generator(config, root).process_all(args.limit, language=language)
        print(
            f"Cards processed: {summary.created} created, {summary.duplicates} duplicates, "
            f"{summary.failed} failed"
        )
        for error in summary.errors:
            print(f"Error: {error}", file=sys.stderr)
        return 1 if summary.failed else 0

    if args.tutor_command == "start":
        language = resolve_language(config, args.lang)
        voice = _voice_tutor(config, root, language, voice=args.voice)
        if args.record_seconds:
            voice.record_seconds = args.record_seconds
        path = voice.start(voice=args.voice)
        print(f"Dialogue saved to {path}")
        return 0

    language = resolve_language(config, args.lang)
    report = _voice_tutor(config, root, language).analyze(args.file)
    print("Error analysis:")
    for explanation in report.analysis.error_analysis:
        print(f"- {explanation}")
    if not report.analysis.error_analysis:
        print("- No errors reported")
    print(
        f"Vocabulary: {len(report.analysis.vocabulary)} extracted, "
        f"{report.inserted} new Raw DB rows"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return _run(args)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
