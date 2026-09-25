import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input/r84-r80-dub-input")
REPORT = WORK / "r84_report"
REPORT.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(str(msg), flush=True)

def run(cmd, cwd=None, timeout=1800, check=False):
    log("$ " + " ".join(map(str, cmd)))
    p = subprocess.run(
        cmd, cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=timeout, check=False
    )
    if p.stdout:
        print(p.stdout, end="", flush=True)
    log(f"[returncode={p.returncode}]")
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")
    return p

try:
    log("=" * 78)
    log("R84 — MUSE TALK 1.5 LIP-SYNC FEASIBILITY BENCHMARK")
    log("R84 MISSION=apply lip-sync to the known-good R80 actual-user-source dub")
    log("=" * 78)

    run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"], timeout=120)

    source_candidates = sorted(INPUT.rglob("*.mp4"))
    if not source_candidates:
        raise FileNotFoundError("R84 input dataset contains no MP4")
    dubbed = source_candidates[0]
    log(f"R80_DUB_INPUT={dubbed}")

    # Benchmark only a 12-second real segment first. This is deliberately
    # separate from the production pipeline; success here gates full-length work.
    clip = WORK / "r84_benchmark_input.mp4"
    audio = WORK / "r84_benchmark_audio.wav"
    rc = run([
        "ffmpeg", "-y", "-ss", "0", "-t", "12",
        "-i", str(dubbed),
        "-vf", "scale=-2:720,fps=25",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        str(clip)
    ], timeout=300)
    if rc.returncode != 0 or not clip.exists():
        raise RuntimeError("Could not create 12s benchmark video clip")

    rc = run([
        "ffmpeg", "-y", "-ss", "0", "-t", "12",
        "-i", str(dubbed),
        "-vn", "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", str(audio)
    ], timeout=180)
    if rc.returncode != 0 or not audio.exists():
        raise RuntimeError("Could not extract benchmark English audio")

    # Use a dedicated Python 3.10 environment when Kaggle provides it,
    # because upstream MuseTalk documents Python 3.10 and CUDA 11.7/11.8.
    py310 = shutil.which("python3.10")
    if py310:
        venv = WORK / "musetalk-venv"
        run([py310, "-m", "venv", str(venv)], timeout=180)
        PY = str(venv / "bin" / "python")
        PIP = str(venv / "bin" / "pip")
    else:
        PY = sys.executable
        PIP = [PY, "-m", "pip"]
        log("WARNING: python3.10 not found; using Kaggle Python runtime")

    run([PY, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        timeout=600, check=True)

    repo = WORK / "MuseTalk"
    if not repo.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/TMElyralab/MuseTalk.git", str(repo)],
            timeout=600, check=True)

    # Install the upstream requirements first. Then install the MMLab stack.
    # Torch is installed explicitly only when the venv is fresh.
    run([
        PY, "-m", "pip", "install",
        "torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ], timeout=1800, check=True)

    run([PY, "-m", "pip", "install", "-r", "requirements.txt"], cwd=repo,
        timeout=1800, check=True)

    run([PY, "-m", "pip", "install", "--no-cache-dir", "-U", "openmim"],
        timeout=600, check=True)
    run(["mim", "install", "mmengine"], timeout=1200, check=True)
    run(["mim", "install", "mmcv==2.0.1"], timeout=1800, check=True)
    run(["mim", "install", "mmdet==3.1.0"], timeout=1800, check=True)
    run(["mim", "install", "mmpose==1.1.0"], timeout=1800, check=True)

    # Download the official MuseTalk weights and their required component weights.
    if not (repo / "models" / "musetalkV15" / "unet.pth").exists():
        run(["bash", "download_weights.sh"], cwd=repo, timeout=3600, check=True)

    test_yaml = repo / "configs" / "inference" / "r84_test.yaml"
    test_yaml.write_text(
        'task_0:\n'
        f'  video_path: "{clip}"\n'
        f'  audio_path: "{audio}"\n'
        '  bbox_shift: 0\n',
        encoding="utf-8"
    )

    result_dir = WORK / "r84_musetalk_result"
    result_dir.mkdir(parents=True, exist_ok=True)

    rc = run([
        PY, "-m", "scripts.inference",
        "--inference_config", str(test_yaml),
        "--result_dir", str(result_dir),
        "--unet_model_path", str(repo / "models/musetalkV15/unet.pth"),
        "--unet_config", str(repo / "models/musetalkV15/musetalk.json"),
        "--version", "v15",
        "--ffmpeg_path", shutil.which("ffmpeg") or "ffmpeg"
    ], cwd=repo, timeout=3600)

    outputs = sorted(result_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if rc.returncode != 0 or not outputs:
        raise RuntimeError("MuseTalk inference did not produce an MP4")

    output = outputs[0]
    final = WORK / "r84_musetalk_12s_benchmark.mp4"
    shutil.copy2(output, final)

    # Technical integrity checks.
    probe = run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=codec_type,codec_name,width,height",
        "-of", "json", str(final)
    ], check=True)
    info = json.loads(probe.stdout)
    duration = float(info.get("format", {}).get("duration", 0.0))
    size = int(info.get("format", {}).get("size", 0))
    streams = info.get("streams", [])

    report = {
        "status": "PASS",
        "benchmark": "MuseTalk 1.5",
        "input": str(dubbed),
        "clip_seconds_requested": 12.0,
        "output": str(final),
        "output_duration_seconds": duration,
        "output_size_bytes": size,
        "streams": streams,
        "note": "This is a feasibility benchmark only; human visual lip-sync review is required before full-length integration."
    }
    (REPORT / "r84_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (REPORT / "r84_summary.txt").write_text(
        "R84 MUSE TALK BENCHMARK\n"
        "======================\n"
        "Status: PASS\n"
        "Model: MuseTalk 1.5\n"
        f"Output duration: {duration:.3f}s\n"
        f"Output size: {size} bytes\n"
        "Human visual review is required before full-length integration.\n",
        encoding="utf-8"
    )

    log("R84 MUSE TALK BENCHMARK COMPLETE")
    log(f"R84_OUTPUT={final}")
    log(f"R84_OUTPUT_DURATION={duration:.3f}s")

except Exception as exc:
    (REPORT / "r84_summary.txt").write_text(
        "R84 MUSE TALK BENCHMARK\n======================\n"
        f"Status: FAIL\n{type(exc).__name__}: {exc}\n",
        encoding="utf-8"
    )
    log("=" * 78)
    log("R84 MUSE TALK BENCHMARK FAILED")
    log(f"{type(exc).__name__}: {exc}")
    raise
