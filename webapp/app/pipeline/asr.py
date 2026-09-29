from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..config import settings


@dataclass
class Segment:
    start: float
    end: float
    text: str
    translation: str = ""
    audio: Path | None = field(default=None)

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


@dataclass
class Transcript:
    language: str
    language_probability: float
    segments: list[Segment]


@lru_cache(maxsize=1)
def _whisper():
    from faster_whisper import WhisperModel

    return WhisperModel(
        settings.whisper_model, device="cpu", compute_type=settings.whisper_compute_type
    )


def _transcribe_local(audio: Path) -> Transcript:
    model = _whisper()
    segments, info = model.transcribe(str(audio), vad_filter=True, beam_size=1)
    items = [
        Segment(start=s.start, end=s.end, text=s.text.strip())
        for s in segments
        if s.text.strip()
    ]
    return Transcript(info.language, float(info.language_probability), items)


def _transcribe_openai(audio: Path) -> Transcript:
    import httpx

    with audio.open("rb") as fh:
        response = httpx.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            files={"file": (audio.name, fh, "audio/wav")},
            data={"model": "whisper-1", "response_format": "verbose_json",
                  "timestamp_granularities[]": "segment"},
            timeout=600,
        )
    response.raise_for_status()
    payload = response.json()
    items = [
        Segment(start=float(s["start"]), end=float(s["end"]), text=s["text"].strip())
        for s in payload.get("segments", [])
        if s["text"].strip()
    ]
    return Transcript(payload.get("language", "en"), 1.0, items)


def transcribe(audio: Path) -> Transcript:
    if settings.asr_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return _transcribe_openai(audio)
    return _transcribe_local(audio)
