from __future__ import annotations

from typing import Any

LANGUAGE_NAMES = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
}


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code.casefold(), code)


def resolve_language(config: dict[str, Any], lang: str | None) -> str:
    languages = config.get("languages")
    if not isinstance(languages, dict) or not languages:
        raise ValueError("Configuration must define at least one entry under 'languages'")
    if lang is None:
        if len(languages) == 1:
            return next(iter(languages))
        codes = ", ".join(sorted(languages))
        raise ValueError(f"--lang is required; configured languages: {codes}")
    if lang not in languages:
        codes = ", ".join(sorted(languages))
        raise ValueError(f"Unsupported language {lang!r}; configured languages: {codes}")
    return lang
