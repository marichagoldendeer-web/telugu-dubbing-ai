import json
import shutil
from pathlib import Path
from typing import Callable

from ..config import LANGUAGE_NAMES, TARGET_LANGUAGES, settings
from . import asr, lipsync, media, translate, tts

Progress = Callable[[str, int, str], None]

MIN_TEMPO = 0.7
MAX_TEMPO = 1.6


def _fit_to_slot(segment: asr.Segment, raw: Path, workdir: Path, index: int) -> Path:
    """Speed the dubbed clip up/down so it lands inside its original time slot."""
    info = media.probe(raw)
    clip_duration = float(info["format"]["duration"])
    slot = segment.duration
    if slot <= 0.05 or clip_duration <= 0.05:
        return raw
    tempo = clip_duration / slot
    if MIN_TEMPO <= tempo <= 1.02:
        return raw
    tempo = min(max(tempo, MIN_TEMPO), MAX_TEMPO)
    return media.time_stretch(raw, workdir / f"seg_{index:04d}_fit.wav", tempo)


def dub(
    video_path: Path,
    target: str,
    workdir: Path,
    progress: Progress,
    lip_sync: bool = True,
) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    segments_dir = workdir / "segments"
    segments_dir.mkdir(exist_ok=True)

    progress("probe", 2, "Reading video")
    info = media.video_info(video_path)
    if not info["has_audio"]:
        raise ValueError("the uploaded video has no audio track to dub")

    progress("extract", 6, "Extracting audio")
    source_wav = media.extract_audio(video_path, workdir / "source.wav")

    progress("asr", 12, "Transcribing and detecting the source language")
    transcript = asr.transcribe(source_wav)
    source_lang = translate.normalize_language(transcript.language)
    if not transcript.segments:
        raise ValueError("no speech was found in the video")

    progress(
        "translate",
        35,
        f"Translating {LANGUAGE_NAMES.get(source_lang, source_lang)} "
        f"-> {TARGET_LANGUAGES[target]['name']}",
    )
    translations = translate.translate([s.text for s in transcript.segments], source_lang, target)
    for segment, text in zip(transcript.segments, translations):
        segment.translation = text

    total = len(transcript.segments)
    clips: list[tuple[float, Path]] = []
    for index, segment in enumerate(transcript.segments):
        if not segment.translation.strip():
            continue
        progress("tts", 45 + int(25 * index / max(total, 1)), f"Voicing line {index + 1}/{total}")
        raw = tts.synthesize(segment.translation, target, segments_dir / f"seg_{index:04d}.wav")
        clips.append((segment.start, _fit_to_slot(segment, raw, segments_dir, index)))

    progress("mix", 72, "Assembling the dubbed audio track")
    dubbed_wav = tts.build_track(clips, info["duration"], workdir / "dubbed.wav")

    output = workdir / "output.mp4"
    warning = ""
    if lip_sync and settings.lipsync_provider != "none":
        progress("lipsync", 78, "Running lip sync (this is the slow part on CPU)")
        try:
            lipsync.apply(video_path, dubbed_wav, output, workdir)
        except lipsync.FaceNotFound as exc:
            warning = str(exc)
            media.mux(video_path, dubbed_wav, output)
    else:
        progress("mux", 90, "Muxing audio and video")
        media.mux(video_path, dubbed_wav, output)

    result = {
        "source_language": source_lang,
        "source_language_name": LANGUAGE_NAMES.get(source_lang, source_lang),
        "language_probability": round(transcript.language_probability, 3),
        "target_language": target,
        "target_language_name": TARGET_LANGUAGES[target]["name"],
        "lip_sync": lip_sync and not warning and settings.lipsync_provider != "none",
        "warning": warning,
        "duration": info["duration"],
        "segments": [
            {
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": s.text,
                "translation": s.translation,
            }
            for s in transcript.segments
        ],
    }
    (workdir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    shutil.rmtree(segments_dir, ignore_errors=True)
    progress("done", 100, "Finished")
    return result
