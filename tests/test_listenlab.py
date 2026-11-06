from types import SimpleNamespace

from src.listenlab import MediaIngester, TranscriptSegment
from src.raw_db import RawDB


class FakeNLP:
    def __call__(self, text):
        tokens = []
        for raw in text.split():
            word = raw.strip(".,!?¡¿")
            tokens.append(
                SimpleNamespace(
                    text=word,
                    lemma_=word.casefold(),
                    is_alpha=word.isalpha(),
                    is_stop=word.casefold() in {"el", "la", "quiero", "tiene"},
                    morph={"Mood": ["Sub"] if word.casefold() == "venga" else []},
                )
            )
        return tokens


class FakeTranscriber:
    def __init__(self, segments):
        self.segments = segments
        self.paths = []

    def transcribe(self, path):
        self.paths.append(path)
        return self.segments


class FakeProcessor:
    def __init__(self):
        self.extracted = []
        self.slices = []

    def extract_audio(self, source, destination):
        self.extracted.append((source, destination))
        destination.write_bytes(b"audio")

    def slice(self, source, destination, start, end):
        self.slices.append((source, destination, start, end))
        destination.write_bytes(b"fragment")


def segments():
    return [
        TranscriptSegment(0.0, 1.5, "Vale, quiero desarrollo"),
        TranscriptSegment(1.5, 3.0, "El desarrollo tiene desarrollo"),
        TranscriptSegment(3.0, 5.0, "Aunque venga, no importa"),
    ]


def ingester(tmp_path, transcriber, processor):
    db = RawDB(tmp_path / "raw.db")
    return db, MediaIngester(
        raw_db=db,
        nlp=FakeNLP(),
        transcriber=transcriber,
        processor=processor,
        output_dir=tmp_path / "output",
        data_dir=tmp_path / "data",
        language="es",
        idioms={"no importa"},
        slang={"vale"},
        grammar_markers={"aunque"},
    )


def test_audio_ingestion_saves_transcripts_and_timestamped_candidates(tmp_path):
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"audio")
    transcriber = FakeTranscriber(segments())
    processor = FakeProcessor()
    db, media = ingester(tmp_path, transcriber, processor)

    report = media.ingest(audio)

    assert report.transcript_segments == 3
    assert report.candidates == 4
    assert report.inserted == 4
    transcripts = db.list_items("transcript")
    assert len(transcripts) == 3
    assert (transcripts[0].start_seconds, transcripts[0].end_seconds) == (0.0, 1.5)
    assert all(item.language == "es" for item in transcripts)
    candidates = db.get_unprocessed()
    assert all(item.language == "es" for item in candidates)
    assert {item.word for item in candidates} == {
        "desarrollo",
        "vale",
        "no importa",
        "Aunque venga, no importa",
    }
    repeated = next(item for item in candidates if item.word == "desarrollo")
    assert repeated.metadata["reasons"] == ["repeated_3_plus"]
    assert repeated.audio_path.endswith(".mp3")
    assert len(processor.slices) == 4

    assert media.ingest(audio).inserted == 0
    assert len(db.list_items("transcript")) == 3


def test_video_audio_is_extracted_before_transcription_and_slicing(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    transcriber = FakeTranscriber(segments()[:1])
    processor = FakeProcessor()
    _, media = ingester(tmp_path, transcriber, processor)

    media.ingest(video)

    extracted_path = processor.extracted[0][1]
    assert transcriber.paths == [extracted_path]
    assert processor.slices[0][0] == extracted_path
