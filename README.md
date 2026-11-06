# LingoForge

LingoForge is a local-first Python CLI that turns articles, audio/video, and AI dialogues into deduplicated Anki cards with translations, audio, and source context. It supports multiple target languages (English and German by default) with translations in your native language (Russian by default).

## Prerequisites

- Python 3.11+
- Anki with the **Anki-Connect** add-on. In Anki, open **Tools → Add-ons → Get Add-ons**, enter code `2055492159`, and restart Anki. Keep Anki open while generating cards.
- ffmpeg on your `PATH` (`ffmpeg -version`). Voice recording additionally requires PulseAudio.
- For the default local LLM provider, Ollama and a model such as `ollama pull llama3.2`.

## Setup

From this project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
cp .env.example .env
```

Install the spaCy model for each language listed in `config.yaml`. Edit `.env` only if you need different service URLs/models or want OpenAI. Keep real API keys out of version control. The default Ollama setup needs no API key. The first Whisper run downloads its configured model; subsequent runs can use the local cache.

For OpenAI, set `llm.provider: openai` in `config.yaml` and put your key in `.env` as `OPENAI_API_KEY`. `OPENAI_MODEL` defaults to `gpt-4o-mini`. `OLLAMA_MODEL` and `OLLAMA_BASE_URL` configure Ollama. `ANKI_CONNECT_URL` defaults to `http://localhost:8765`.

## Commands

Run commands from this directory after activating the virtual environment. `--config PATH` can be placed before any command to use another YAML file. Commands that work in a single target language take `--lang CODE`; it is required when the configuration lists more than one language.

```bash
# Ingest an article from a local UTF-8 file or an http(s) URL
python -m src.cli ingest article data/article.txt --lang en
python -m src.cli ingest article https://example.com/article.html --lang de

# Ingest a local audio/video file or a direct media-file URL
python -m src.cli ingest media data/podcast.mp3 --lang en
python -m src.cli ingest media data/video.mp4 --lang de
python -m src.cli ingest media https://example.com/podcast.mp3 --lang en

# Create Anki cards for all pending Raw DB candidates (all languages, or one)
python -m src.cli cards generate
python -m src.cli cards generate --limit 20
python -m src.cli cards generate --lang de

# Start and save a text dialogue; enter /quit to finish
python -m src.cli voicetutor start --lang en

# Record each voice turn with ffmpeg/PulseAudio and transcribe with Whisper
python -m src.cli voicetutor start --voice --record-seconds 10 --lang de

# Analyze the last dialogue or a saved dialogue
python -m src.cli voicetutor analyze --lang en
python -m src.cli voicetutor analyze --file ~/dialog.txt --lang en
```

Ingestion and analysis write candidates to the Raw DB, tagged with their target language. Run `cards generate` afterward to send them through the one shared card pipeline; each language gets its own translator, TTS voice, and Anki deck. Translations and dialogue error explanations are written in the configured `native_language`. Repeating ingestion or card generation does not create duplicate cards; the same word in two different languages produces two separate cards. A failed item stays pending for retry; the command reports errors and exits nonzero if any item fails.

## Configuration

`config.yaml` contains:

- `native_language`: language code for translations and error explanations, default `ru`.
- `languages`: map of target language codes to per-language settings:
  - `spacy_model`: installed spaCy model for that language.
  - `deck`: destination Anki deck, default `LingoForge::<code>`.
  - `frequency_band.min_rank` / `max_rank`: inclusive article vocabulary ranks, default 1000–5000. `wordfreq` supplies the language frequency list; spaCy supplies lemmas.
  - `idioms`, `slang`, `grammar_markers`: optional phrase lists used by the media phrase-selection heuristics. Empty by default; fill them in per language as you like.
- `paths.raw_db`, `paths.main_db`: separate SQLite databases shared by all languages.
- `paths.data`: downloaded/local media directory; `paths.output`: generated audio, databases, and last dialogue.
- `paths.dictionary`: local JSON dictionary used when `translation.provider` is `dictionary`.
- `paths.last_dialogue`: dialogue saved by `voicetutor start` and read by default by `analyze`.
- `anki.model`: Anki note model. The default `Basic` model must have `Front` and `Back` fields.
- `translation.provider`: `llm` or `dictionary`. Dialogue-provided translations are reused without another translation request.
- `llm.provider`: `ollama` or `openai`, used for dialogue and optionally translation.
- `whisper.model`, `device`, `compute_type`: faster-whisper settings, default `small`, `cpu`, `int8`.
- `media.video_extensions`: suffixes for which ffmpeg extracts audio before transcription.
- `dialogue.record_seconds`: default voice-turn duration.
- `network_timeout`: HTTP timeout in seconds.

A local dictionary is either a flat word-to-native-language map or a map under the language code:

```json
{"de": {"nutzen": "использовать", "Werkzeug": "инструмент"}}
```

gTTS requires network access. Local dictionary and Ollama translation work offline, and cached Whisper/spaCy/wordfreq data is local. Article/media URL retrieval inherently requires network access.

## Tests and lint

External APIs are mocked in the unit tests:

```bash
ruff check
pytest
```

To inspect ingestion without creating cards, query `paths.raw_db` with SQLite. Rows with `item_kind = 'candidate'` are pending/generated card material; `transcript` rows retain media text and `start_seconds`/`end_seconds` timestamps. Every row has a `language` column with its target language code.

Databases created before multi-language support (Raw DB schema version 1) are migrated automatically; their existing rows are assigned language `en`. To correct that backfill, run e.g. `sqlite3 output/raw.db "UPDATE raw_items SET language = 'es';"` before adding new data.

## Known limitations

- No GUI, sync, multi-user mode, or mobile support.
- Article HTML extraction is simple and does not execute JavaScript or remove every navigation element.
- Media URLs must point directly to downloadable media files; site-specific extractors such as YouTube downloaders are not included.
- Media phrase selection uses transparent heuristics (per-language idiom/slang lists from the configuration, subjunctive/grammar markers, and words repeated at least three times), not an LLM. It may produce false positives or miss constructions.
- Voice recording currently uses ffmpeg with PulseAudio (`default` input), common on Linux; other microphone backends are not configured.
- Translation and LLM extraction quality depends on the selected model. gTTS pronunciation is generated speech; media cards reuse their source audio fragment.
- Real card creation requires Anki and Anki-Connect to be running. Generated cards and source fragments are local artifacts under `output/` and are not synchronized.
