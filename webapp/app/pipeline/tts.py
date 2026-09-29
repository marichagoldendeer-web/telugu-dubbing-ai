import base64
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import TARGET_LANGUAGES, settings

SAMPLE_RATE = 24000


@lru_cache(maxsize=4)
def _mms(model_id: str):
    from transformers import AutoTokenizer, VitsModel

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def _uroman():
    import uroman as ur

    return ur.Uroman()


def _synth_local(text: str, target: str, out_wav: Path) -> Path:
    import torch

    tokenizer, model = _mms(TARGET_LANGUAGES[target]["mms"])
    if getattr(tokenizer, "is_uroman", False):
        text = _uroman().romanize_string(text, lcode=target)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.inference_mode():
        waveform = model(**inputs).waveform[0].cpu().numpy()
    sf.write(out_wav, waveform, model.config.sampling_rate)
    return out_wav


def _synth_elevenlabs(text: str, target: str, out_wav: Path) -> Path:
    import httpx

    voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel (multilingual)
    response = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={"xi-api-key": settings.elevenlabs_api_key},
        json={"text": text, "model_id": "eleven_multilingual_v2"},
        timeout=300,
    )
    response.raise_for_status()
    mp3 = out_wav.with_suffix(".mp3")
    mp3.write_bytes(response.content)
    from .media import run

    run(["ffmpeg", "-y", "-i", str(mp3), "-ar", str(SAMPLE_RATE), "-ac", "1", str(out_wav)])
    return out_wav


def _synth_sarvam(text: str, target: str, out_wav: Path) -> Path:
    import httpx

    response = httpx.post(
        "https://api.sarvam.ai/text-to-speech",
        headers={"api-subscription-key": settings.sarvam_api_key},
        json={
            "inputs": [text],
            "target_language_code": TARGET_LANGUAGES[target]["sarvam"],
            "speaker": "meera",
            "model": "bulbul:v1",
        },
        timeout=300,
    )
    response.raise_for_status()
    wav_bytes = base64.b64decode(response.json()["audios"][0])
    out_wav.write_bytes(wav_bytes)
    return out_wav


def synthesize(text: str, target: str, out_wav: Path) -> Path:
    if settings.tts_provider == "elevenlabs":
        if not settings.elevenlabs_api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")
        return _synth_elevenlabs(text, target, out_wav)
    if settings.tts_provider == "sarvam":
        if not settings.sarvam_api_key:
            raise RuntimeError("SARVAM_API_KEY is not set")
        return _synth_sarvam(text, target, out_wav)
    return _synth_local(text, target, out_wav)


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    from scipy.signal import resample_poly
    from math import gcd

    divisor = gcd(src_rate, dst_rate)
    return resample_poly(audio, dst_rate // divisor, src_rate // divisor)


def read_mono(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    return _resample(audio.mean(axis=1), rate, SAMPLE_RATE)


def build_track(clips: list[tuple[float, Path]], duration: float, out_wav: Path) -> Path:
    """Place each dubbed clip at its original start time on a silent canvas."""
    canvas = np.zeros(int(duration * SAMPLE_RATE) + SAMPLE_RATE, dtype=np.float32)
    for start, clip in clips:
        audio = read_mono(clip)
        offset = int(max(start, 0.0) * SAMPLE_RATE)
        end = offset + len(audio)
        if end > len(canvas):
            canvas = np.pad(canvas, (0, end - len(canvas)))
        canvas[offset:end] += audio
    peak = float(np.abs(canvas).max())
    if peak > 1.0:
        canvas /= peak
    sf.write(out_wav, canvas, SAMPLE_RATE)
    return out_wav
