#!/usr/bin/env python3
"""R85 preflight: create an exact, restart-safe run for the GPU dubbing pipeline.

This stage intentionally performs no AI inference. It creates the run manifest and
validates the exact input before an expensive GPU stage starts.

Examples:
    python kaggle/r85/runner.py --input /kaggle/input/my-dataset/sample.mp4
    python kaggle/r85/runner.py --dataset /kaggle/input/my-dataset --filename sample.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Allow execution as: python kaggle/r85/runner.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubbing_runtime import (
    ProviderConfig,
    RunContext,
    resolve_exact_input,
    resolve_single_dataset_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Exact input video path.")
    source.add_argument("--dataset", type=Path, help="Dataset directory containing the input video.")
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Exact filename inside --dataset. Required when the dataset contains multiple videos.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("/kaggle/working/dubbing_runs"),
        help="Root directory for isolated run directories.",
    )
    parser.add_argument(
        "--snapshot-input",
        action="store_true",
        help="Copy the source video into the run/input directory.",
    )
    return parser.parse_args()


def probe_media(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=filename,duration,size,format_name:stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {path}: {(result.stderr or result.stdout).strip()}"
        )
    return json.loads(result.stdout)


def main() -> int:
    args = parse_args()

    if args.input is not None:
        input_path = resolve_exact_input(
            args.input,
            [".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"],
        )
    else:
        input_path = resolve_single_dataset_video(
            args.dataset,
            filename=args.filename,
        )

    providers = ProviderConfig.from_env()
    context = RunContext.create(
        input_path,
        work_root=args.work_root,
        config={
            "providers": providers.as_dict(),
            "profile": "gpu-dubbing-r85",
        },
        snapshot_input=args.snapshot_input,
    )

    with context.stage(
        "input_validation",
        message="Validate exact registered input with FFprobe",
    ):
        context.verify_input_unchanged()
        media_info = probe_media(Path(context.manifest.input_path))
        media_info_path = context.stage_dir("input_validation") / "media_info.json"
        media_info_path.write_text(
            json.dumps(media_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        context.record_outputs("input_validation", [media_info_path])

    handoff = {
        "run_id": context.manifest.run_id,
        "run_dir": str(context.run_dir),
        "input_path": context.manifest.input_path,
        "input_filename": context.manifest.input_filename,
        "input_sha256": context.manifest.input_sha256,
        "input_size_bytes": context.manifest.input_size_bytes,
        "media_info": str(context.stage_dir("input_validation") / "media_info.json"),
        "manifest": str(context.manifest_path),
        "providers": providers.as_dict(),
        "next_stage": "audio_extraction",
    }
    handoff_path = context.run_dir / "handoff.json"
    handoff_path.write_text(
        json.dumps(handoff, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("R85 — SAFE DUBBING RUN PREFLIGHT")
    print("=" * 78)
    print(f"RUN_ID={context.manifest.run_id}")
    print(f"RUN_DIR={context.run_dir}")
    print(f"INPUT={context.manifest.input_path}")
    print(f"INPUT_SHA256={context.manifest.input_sha256}")
    print(f"MANIFEST={context.manifest_path}")
    print(f"HANDOFF={handoff_path}")
    print("INPUT_VALIDATION=PASS")
    print("NEXT_STAGE=audio_extraction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
