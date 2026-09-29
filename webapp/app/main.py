import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import TARGET_LANGUAGES, settings
from .jobs import store
from .pipeline import lipsync

app = FastAPI(title="Dubbing Studio")
STATIC = Path(__file__).parent / "static"

ALLOWED_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


@app.get("/api/config")
def get_config() -> dict:
    return {
        "languages": [{"code": c, "name": v["name"]} for c, v in TARGET_LANGUAGES.items()],
        "providers": {
            "asr": settings.asr_provider,
            "translate": settings.translate_provider,
            "tts": settings.tts_provider,
            "lipsync": settings.lipsync_provider,
        },
        "lipsync_ready": settings.lipsync_provider == "sync" or lipsync.wav2lip_ready(),
    }


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    target_language: str = Form(...),
    lip_sync: bool = Form(True),
) -> dict:
    if target_language not in TARGET_LANGUAGES:
        raise HTTPException(400, f"unsupported target language: {target_language}")
    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"unsupported file type: {suffix or 'unknown'}")

    job = store.create(file.filename or "upload.mp4", target_language, lip_sync)
    source = job.dir / f"source{suffix}"
    with source.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    store.start(job, source)
    return job.to_dict()


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [job.to_dict() for job in store.list()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return JSONResponse(job.to_dict())


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str) -> FileResponse:
    job = store.get(job_id)
    if job is None or not (job.dir / "output.mp4").exists():
        raise HTTPException(404, "output not ready")
    return FileResponse(
        job.dir / "output.mp4",
        media_type="video/mp4",
        filename=f"{Path(job.filename).stem}_{job.target}.mp4",
    )


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
