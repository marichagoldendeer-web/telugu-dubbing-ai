import subprocess
import sys
import time
from pathlib import Path

import httpx

from ..config import settings
from . import media


class FaceNotFound(RuntimeError):
    pass


def wav2lip_ready() -> bool:
    root = settings.vendor_dir / "Wav2Lip"
    return (root / "inference.py").exists() and (
        root / "checkpoints" / settings.wav2lip_checkpoint
    ).exists()


def _wav2lip(video: Path, audio: Path, out_video: Path, workdir: Path) -> Path:
    root = settings.vendor_dir / "Wav2Lip"
    if not wav2lip_ready():
        raise RuntimeError(
            "Wav2Lip is not installed. Run scripts/setup_wav2lip.sh to fetch the model."
        )

    face = video
    info = media.video_info(video)
    if info["width"] > settings.lipsync_max_width:
        face = media.scale_video(video, workdir / "face_input.mp4", settings.lipsync_max_width)

    proc = subprocess.run(
        [
            sys.executable, "inference.py",
            "--checkpoint_path", str(root / "checkpoints" / settings.wav2lip_checkpoint),
            "--face", str(face.resolve()),
            "--audio", str(audio.resolve()),
            "--outfile", str(out_video.resolve()),
            "--resize_factor", str(settings.wav2lip_resize_factor),
            "--wav2lip_batch_size", str(settings.wav2lip_batch_size),
            "--nosmooth",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-3000:]
        if "Face not detected" in tail:
            raise FaceNotFound("no face was detected in the video, skipping lip sync")
        raise RuntimeError(f"Wav2Lip failed:\n{tail}")
    return out_video


def _sync_api(video: Path, audio: Path, out_video: Path) -> Path:
    """sync.so hosted lip sync. Needs publicly reachable URLs for the inputs."""
    headers = {"x-api-key": settings.sync_api_key}
    video_url = _upload_temp(video)
    audio_url = _upload_temp(audio)
    response = httpx.post(
        "https://api.sync.so/v2/generate",
        headers=headers,
        json={
            "model": "lipsync-2",
            "input": [
                {"type": "video", "url": video_url},
                {"type": "audio", "url": audio_url},
            ],
        },
        timeout=120,
    )
    response.raise_for_status()
    job_id = response.json()["id"]
    while True:
        time.sleep(10)
        poll = httpx.get(f"https://api.sync.so/v2/generate/{job_id}", headers=headers, timeout=60)
        poll.raise_for_status()
        body = poll.json()
        if body["status"] == "COMPLETED":
            out_video.write_bytes(httpx.get(body["outputUrl"], timeout=600).content)
            return out_video
        if body["status"] in {"FAILED", "CANCELED", "REJECTED", "TIMED_OUT"}:
            raise RuntimeError(f"sync.so job {body['status']}: {body.get('error')}")


def _upload_temp(path: Path) -> str:
    response = httpx.post(
        "https://tmpfiles.org/api/v1/upload",
        files={"file": (path.name, path.read_bytes())},
        timeout=600,
    )
    response.raise_for_status()
    url = response.json()["data"]["url"]
    return url.replace("tmpfiles.org/", "tmpfiles.org/dl/")


def apply(video: Path, audio: Path, out_video: Path, workdir: Path) -> Path:
    if settings.lipsync_provider == "sync":
        if not settings.sync_api_key:
            raise RuntimeError("SYNC_API_KEY is not set")
        return _sync_api(video, audio, out_video)
    return _wav2lip(video, audio, out_video, workdir)
