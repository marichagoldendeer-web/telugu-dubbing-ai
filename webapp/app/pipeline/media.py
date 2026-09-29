import json
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd[:4])}...\n{proc.stderr[-2000:]}")
    return proc.stdout


def probe(path: Path) -> dict:
    out = run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    return json.loads(out)


def video_info(path: Path) -> dict:
    data = probe(path)
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])
    if video is None:
        raise ValueError("the uploaded file has no video stream")
    num, den = (video.get("r_frame_rate") or "25/1").split("/")
    fps = float(num) / float(den or 1)
    return {
        "duration": float(data["format"].get("duration", 0.0)),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": fps,
        "has_audio": has_audio,
    }


def extract_audio(video: Path, out_wav: Path, sample_rate: int = 16000) -> Path:
    run([
        "ffmpeg", "-y", "-i", str(video), "-vn",
        "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(out_wav),
    ])
    return out_wav


def time_stretch(src: Path, dst: Path, tempo: float) -> Path:
    """Change speed without changing pitch. ffmpeg atempo accepts 0.5-2.0 per filter."""
    filters, remaining = [], tempo
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")
    run(["ffmpeg", "-y", "-i", str(src), "-filter:a", ",".join(filters), str(dst)])
    return dst


def scale_video(src: Path, dst: Path, max_width: int) -> Path:
    run([
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"scale='min({max_width},iw)':-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an", str(dst),
    ])
    return dst


def mux(video: Path, audio: Path, dst: Path) -> Path:
    run([
        "ffmpeg", "-y", "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-shortest", str(dst),
    ])
    return dst
