"""Reusable, restart-safe runtime utilities for the Kaggle dubbing pipeline.

This module incorporates the most useful ideas from the Devin prototype without
pulling its CPU-oriented ML dependencies into the existing GPU pipeline:
- exact input resolution and SHA-256 fingerprinting
- isolated per-run directories
- persistent stage state for resume/recovery
- atomic manifest writes
- provider/config knobs without model-specific dependencies
- segment TTS timing calculations
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional


RUN_DIR_NAMES = (
    "input",
    "audio",
    "separation",
    "transcription",
    "translation",
    "tts",
    "alignment",
    "mix",
    "lipsync",
    "output",
    "qc",
    "logs",
)

STAGE_NAMES = (
    "input_validation",
    "audio_extraction",
    "separation",
    "stt",
    "translation",
    "tts",
    "alignment",
    "mixing",
    "lipsync",
    "mux",
    "qc",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into RAM."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_exact_input(
    path: Path,
    allowed_suffixes: Optional[List[str]] = None,
) -> Path:
    """Resolve one explicitly named video; never pick an arbitrary/latest file."""
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Exact input file does not exist: {candidate}")

    if allowed_suffixes:
        allowed = {
            suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
            for suffix in allowed_suffixes
        }
        if candidate.suffix.lower() not in allowed:
            raise ValueError(
                f"Unsupported input type {candidate.suffix!r}; "
                f"allowed: {sorted(allowed)}"
            )
    return candidate


def resolve_single_dataset_video(
    dataset_dir: Path,
    filename: Optional[str] = None,
    allowed_suffixes: Optional[List[str]] = None,
) -> Path:
    """Safely resolve a dataset video.

    If filename is omitted, exactly one matching video must exist. This prevents
    stale/incorrect MP4 selection in Kaggle input datasets.
    """
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    if filename:
        return resolve_exact_input(root / filename, allowed_suffixes)

    allowed = {
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in (allowed_suffixes or [".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"])
    }
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    )

    if len(candidates) != 1:
        names = "\n".join(f" - {path}" for path in candidates[:20])
        raise RuntimeError(
            f"Expected exactly one video in {root}, found {len(candidates)}.\n{names}"
        )

    return candidates[0].resolve()


@dataclass(frozen=True)
class ProviderConfig:
    """Stage provider selection kept independent from model code."""

    asr: str = "local"
    translation: str = "local"
    tts: str = "local"
    lipsync: str = "local"

    @classmethod
    def from_env(cls) -> "ProviderConfig":
        return cls(
            asr=os.getenv("ASR_PROVIDER", "local").strip().lower(),
            translation=os.getenv("TRANSLATE_PROVIDER", "local").strip().lower(),
            tts=os.getenv("TTS_PROVIDER", "local").strip().lower(),
            lipsync=os.getenv("LIPSYNC_PROVIDER", "local").strip().lower(),
        )

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class StageRecord:
    status: str = "pending"
    started_at: str = ""
    ended_at: str = ""
    outputs: List[str] = field(default_factory=list)
    message: str = ""
    error: str = ""


@dataclass
class RunManifest:
    schema_version: int
    run_id: str
    created_at: str
    updated_at: str
    input_path: str
    input_filename: str
    input_sha256: str
    input_size_bytes: int
    config: Dict[str, object]
    stages: Dict[str, StageRecord]

    def to_dict(self) -> dict:
        raw = asdict(self)
        raw["stages"] = {
            name: asdict(record) for name, record in self.stages.items()
        }
        return raw


class RunContext:
    """Persistent, restart-safe state for one dubbing run."""

    _STAGE_DIRS = {
        "input_validation": "input",
        "audio_extraction": "audio",
        "separation": "separation",
        "stt": "transcription",
        "translation": "translation",
        "tts": "tts",
        "alignment": "alignment",
        "mixing": "mix",
        "lipsync": "lipsync",
        "mux": "output",
        "qc": "qc",
    }

    def __init__(self, run_dir: Path, manifest: RunManifest):
        self.run_dir = Path(run_dir)
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.manifest = manifest

    @classmethod
    def create(
        cls,
        input_path: Path,
        work_root: Path = Path("/kaggle/working/dubbing_runs"),
        config: Optional[Dict[str, object]] = None,
        snapshot_input: bool = False,
    ) -> "RunContext":
        source = resolve_exact_input(
            input_path,
            [".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"],
        )
        digest = sha256_file(source)
        run_id = (
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{uuid.uuid4().hex[:8]}"
        )
        run_dir = Path(work_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        for dirname in RUN_DIR_NAMES:
            (run_dir / dirname).mkdir()

        manifest = RunManifest(
            schema_version=1,
            run_id=run_id,
            created_at=utc_now(),
            updated_at=utc_now(),
            input_path=str(source),
            input_filename=source.name,
            input_sha256=digest,
            input_size_bytes=source.stat().st_size,
            config=dict(config or {}),
            stages={name: StageRecord() for name in STAGE_NAMES},
        )

        context = cls(run_dir, manifest)
        context.save_manifest()

        if snapshot_input:
            snapshot = run_dir / "input" / source.name
            shutil.copy2(source, snapshot)
            context.manifest.input_path = str(snapshot)
            context.save_manifest()

        return context

    @classmethod
    def resume(cls, run_dir: Path) -> "RunContext":
        run_path = Path(run_dir).expanduser().resolve()
        manifest_path = run_path / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Run manifest not found: {manifest_path}")

        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        stages = {
            name: StageRecord(**value)
            for name, value in raw.get("stages", {}).items()
        }

        manifest = RunManifest(
            schema_version=int(raw["schema_version"]),
            run_id=str(raw["run_id"]),
            created_at=str(raw["created_at"]),
            updated_at=str(raw["updated_at"]),
            input_path=str(raw["input_path"]),
            input_filename=str(raw["input_filename"]),
            input_sha256=str(raw["input_sha256"]),
            input_size_bytes=int(raw["input_size_bytes"]),
            config=dict(raw.get("config") or {}),
            stages=stages,
        )
        return cls(run_path, manifest)

    def save_manifest(self) -> None:
        self.manifest.updated_at = utc_now()
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                self.manifest.to_dict(),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, self.manifest_path)

    def verify_input_unchanged(self) -> None:
        current_path = Path(self.manifest.input_path)
        digest = sha256_file(current_path)
        if digest != self.manifest.input_sha256:
            raise RuntimeError(
                "Input fingerprint changed since run creation.\n"
                f"expected={self.manifest.input_sha256}\n"
                f"actual={digest}\n"
                f"path={current_path}"
            )

        size = current_path.stat().st_size
        if size != self.manifest.input_size_bytes:
            raise RuntimeError(
                "Input size changed since run creation: "
                f"expected={self.manifest.input_size_bytes}, actual={size}"
            )

    def stage_dir(self, stage: str) -> Path:
        try:
            dirname = self._STAGE_DIRS[stage]
        except KeyError as exc:
            raise KeyError(f"Unknown stage: {stage}") from exc
        return self.run_dir / dirname

    def is_complete(
        self,
        stage: str,
        expected_outputs: Optional[List[Path]] = None,
    ) -> bool:
        record = self.manifest.stages[stage]
        if record.status != "passed":
            return False

        if expected_outputs:
            return all(Path(path).is_file() for path in expected_outputs)

        if record.outputs:
            return all((self.run_dir / path).is_file() for path in record.outputs)

        return True

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        message: str = "",
    ) -> Iterator[StageRecord]:
        if name not in self.manifest.stages:
            raise KeyError(f"Unknown stage: {name}")

        record = self.manifest.stages[name]
        record.status = "running"
        record.started_at = utc_now()
        record.ended_at = ""
        record.error = ""
        record.message = message
        self.save_manifest()

        try:
            yield record
        except Exception as exc:
            record.status = "failed"
            record.ended_at = utc_now()
            record.error = f"{type(exc).__name__}: {exc}"
            self.save_manifest()
            raise
        else:
            record.status = "passed"
            record.ended_at = utc_now()
            self.save_manifest()

    def record_outputs(self, stage: str, paths: List[Path]) -> None:
        record = self.manifest.stages[stage]
        outputs: List[str] = []

        for path in paths:
            absolute = Path(path).expanduser().resolve()
            try:
                relative = absolute.relative_to(self.run_dir.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"Stage output must live inside run directory: {absolute}"
                ) from exc
            outputs.append(str(relative))

        record.outputs = outputs
        self.save_manifest()

    def skip(self, stage: str, message: str) -> None:
        record = self.manifest.stages[stage]
        record.status = "skipped"
        record.message = message
        record.started_at = record.started_at or utc_now()
        record.ended_at = utc_now()
        self.save_manifest()


def fit_audio_to_slot(
    generated_seconds: float,
    target_seconds: float,
    *,
    min_factor: float = 0.70,
    max_factor: float = 1.60,
) -> dict:
    """Calculate an FFmpeg atempo factor for segment-level timing correction.

    atempo speed factor = generated_duration / target_duration.
    """
    if generated_seconds <= 0 or target_seconds <= 0:
        raise ValueError("generated_seconds and target_seconds must be > 0")
    if min_factor <= 0 or max_factor < min_factor:
        raise ValueError("invalid atempo bounds")

    raw_factor = generated_seconds / target_seconds
    factor = min(max(raw_factor, min_factor), max_factor)
    resulting_seconds = generated_seconds / factor
    residual_seconds = resulting_seconds - target_seconds
    tolerance = max(0.08, target_seconds * 0.02)

    return {
        "generated_seconds": round(generated_seconds, 6),
        "target_seconds": round(target_seconds, 6),
        "raw_atempo_factor": round(raw_factor, 6),
        "applied_atempo_factor": round(factor, 6),
        "resulting_seconds": round(resulting_seconds, 6),
        "residual_seconds": round(residual_seconds, 6),
        "clamped": abs(factor - raw_factor) > 1e-9,
        "within_limits": abs(residual_seconds) <= tolerance,
    }


def build_atempo_filter(factor: float) -> str:
    """Build a valid FFmpeg atempo chain for any positive factor."""
    if factor <= 0:
        raise ValueError("factor must be > 0")

    factors: List[float] = []
    remaining = float(factor)

    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0

    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5

    factors.append(remaining)
    return ",".join(f"atempo={value:.8f}" for value in factors)
