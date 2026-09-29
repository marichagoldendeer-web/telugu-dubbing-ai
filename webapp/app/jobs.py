import json
import threading
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import settings
from .pipeline import run


@dataclass
class Job:
    id: str
    filename: str
    target: str
    lip_sync: bool
    status: str = "queued"  # queued | running | done | error
    stage: str = "queued"
    progress: int = 0
    message: str = "Waiting to start"
    error: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    result: dict | None = None

    @property
    def dir(self) -> Path:
        return settings.data_dir / "jobs" / self.id

    def to_dict(self) -> dict:
        data = asdict(self)
        data["has_output"] = (self.dir / "output.mp4").exists()
        return data


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, filename: str, target: str, lip_sync: bool) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], filename=filename, target=target, lip_sync=lip_sync)
        job.dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def start(self, job: Job, source: Path) -> None:
        threading.Thread(target=self._run, args=(job, source), daemon=True).start()

    def _run(self, job: Job, source: Path) -> None:
        def progress(stage: str, percent: int, message: str) -> None:
            job.stage, job.progress, job.message = stage, percent, message
            self._persist(job)

        job.status = "running"
        progress("starting", 1, "Starting")
        try:
            job.result = run.dub(source, job.target, job.dir, progress, lip_sync=job.lip_sync)
            job.status = "done"
            job.message = job.result.get("warning") or "Dubbed video is ready"
        except Exception as exc:  # surfaced to the UI
            job.status = "error"
            job.error = str(exc) or exc.__class__.__name__
            job.message = "Failed"
            traceback.print_exc()
        self._persist(job)

    def _persist(self, job: Job) -> None:
        (job.dir / "job.json").write_text(json.dumps(job.to_dict(), ensure_ascii=False, indent=2))


store = JobStore()
