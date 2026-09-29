#!/usr/bin/env python3
"""Create or inspect a restart-safe dubbing run before expensive GPU work.

Usage:
    python kaggle/prepare_run.py --input /kaggle/input/my-dataset/video.mp4
    python kaggle/prepare_run.py --resume /kaggle/working/dubbing_runs/RUN_ID
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow this script to be run directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dubbing_runtime import ProviderConfig, RunContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Exact input video path.")
    source.add_argument("--resume", type=Path, help="Existing run directory.")
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("/kaggle/working/dubbing_runs"),
        help="Root directory for isolated runs.",
    )
    parser.add_argument(
        "--snapshot-input",
        action="store_true",
        help="Copy the source video into the run directory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    providers = ProviderConfig.from_env()

    if args.resume:
        context = RunContext.resume(args.resume)
        context.verify_input_unchanged()
        print(f"RESUMED_RUN={context.manifest.run_id}")
    else:
        context = RunContext.create(
            args.input,
            work_root=args.work_root,
            config={"providers": providers.as_dict()},
            snapshot_input=args.snapshot_input,
        )
        context.verify_input_unchanged()
        print(f"CREATED_RUN={context.manifest.run_id}")

    manifest = context.manifest.to_dict()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"RUN_DIR={context.run_dir}")
    print(f"INPUT={context.manifest.input_path}")
    print(f"INPUT_SHA256={context.manifest.input_sha256}")
    print("STATUS=READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
